"""Replicas served over HTTP, and channels routed to them.

A process that serves engines can serve them to other machines (`serve_engines`): each engine of each channel is a
replica, reached at an address of its own (`replica_address`). It samples tokens in, tokens and logprobs out, and says
what it serves (the adapter, the version it is stamped with, what else is loaded). `RemoteEngine` is an `Engine` over
that interface, beside `VllmEngine` and the scripted engines.

The recorder stays with the episode runner, so that what is recorded is exactly what was sampled; only the engines are
elsewhere. A `RemoteChannel` is one run's channel as a runner samples it: it finds the replicas that serve the channel
(from engine hosts' heartbeats, by the run and the channel's name, or at fixed addresses), asks each what it serves, and
routes each session to one of them, where it stays. A replica serving a checkpoint further behind what the run says the
channel should serve (`rollout_train.serving`) than `max_lag` checkpoints is given no new session; a session already on
it stays. A replica that cannot be reached is routed around: its sessions move to another, from the start of the turn.

    GET    /channels                               every channel's replicas: what each serves
    GET    /channels/NAME/replicas/I               what replica I serves (`Channel.state`)
    POST   /channels/NAME/replicas/I/generate      {prompt, max_tokens, temperature, top_p, stop_token_ids, adapter,
                                                    version} → {tokens, logprobs, finish_reason}; 409 if the adapter is
                                                    not served at that version
    POST   /channels/NAME/replicas/I/adapters      {name, path}: load an adapter from a path on the replica's machine
    DELETE /channels/NAME/replicas/I/adapters/A    remove one
    POST   /channels/NAME/replicas/I/weights       {path}: load full weights from a path on the replica's machine
    POST   /channels/NAME/replicas/I/sleep, /wake
"""

import asyncio
import contextlib
import time
import zlib
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import JsonValue

from rollout_train.inference.channel import Channel, Generation, Limits, Throughput, Unserved
from rollout_train.ledger import Ledger
from rollout_train.presence import Presence, alive
from rollout_train.serving import Serving, parts, qualified, wanted

if TYPE_CHECKING:
    from rollout_train.recorder.renderers import Renderer

ENGINES = "engines"
"""The `kind` of an engine host's heartbeat (`rollout_train.following`)."""
MAX_LAG = 1
"""Checkpoints a replica may be behind what its channel should serve and still be given new sessions, unless a profile
says otherwise: one, the checkpoint before, which it serves while it loads the newest."""


class Unreachable(Unserved):
    """A replica did not answer: it is routed around."""


class NoReplica(Exception):
    """No replica serves a run's channel close enough to what it should serve, and none did for as long as a session
    waits."""


def replica_address(base: str, channel: str, index: int) -> str:
    """Where replica `index` of a channel is reached, on a server at `base` (`serve_engines`)."""
    return f"{base.rstrip('/')}/channels/{channel}/replicas/{index}"


def serve_engines(channels: Mapping[str, Channel]) -> Any:
    """A Starlette application serving each engine of each channel as a replica (needs the `http` extra)."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    def replica(request: Request) -> tuple[Channel, int]:
        channel = channels.get(str(request.path_params["channel"]))
        index = int(request.path_params["index"])
        if channel is None or not 0 <= index < len(channel.engines):
            raise LookupError(f"no replica {request.path_params['channel']}/{index}")
        return channel, index

    def answered(work: Callable[[Request], Awaitable[Response]]) -> Callable[[Request], Awaitable[Response]]:
        async def answer(request: Request) -> Response:
            try:
                return await work(request)
            except LookupError as error:
                return JSONResponse({"error": str(error)}, status_code=404)
            except Exception as error:  # (the caller raises it)
                return JSONResponse({"error": f"{type(error).__name__}: {error}"}, status_code=500)

        return answer

    async def every(request: Request) -> Response:
        listed = {name: {"replicas": len(channel.engines), **channel.state()} for name, channel in channels.items()}
        return JSONResponse({"channels": listed})

    async def state(request: Request) -> Response:
        channel, _ = replica(request)
        return JSONResponse(channel.state())

    async def generate(request: Request) -> Response:
        channel, index = replica(request)
        body = await request.json()
        try:
            generation = await channel.generate(
                body["prompt"],
                max_tokens=int(body["max_tokens"]),
                temperature=float(body["temperature"]),
                top_p=float(body["top_p"]),
                stop_token_ids=body["stop_token_ids"],
                adapter=body.get("adapter"),
                version=body.get("version"),
                replica=index,
            )
        except Unserved as error:
            return JSONResponse({"error": str(error), "state": channel.state()}, status_code=409)
        return JSONResponse(
            {"tokens": generation.tokens, "logprobs": generation.logprobs, "finish_reason": generation.finish_reason}
        )

    async def load_adapter(request: Request) -> Response:
        channel, index = replica(request)
        body = await request.json()
        await channel.engines[index].load_adapter(str(body["name"]), str(body["path"]))
        return JSONResponse({})

    async def remove_adapter(request: Request) -> Response:
        channel, index = replica(request)
        await channel.engines[index].remove_adapter(str(request.path_params["adapter"]))
        return JSONResponse({})

    async def load_weights(request: Request) -> Response:
        channel, index = replica(request)
        await channel.engines[index].load_weights(str((await request.json())["path"]))
        return JSONResponse({})

    async def sleep(request: Request) -> Response:
        channel, index = replica(request)
        await channel.engines[index].sleep()
        return JSONResponse({})

    async def wake(request: Request) -> Response:
        channel, index = replica(request)
        await channel.engines[index].wake()
        return JSONResponse({})

    at = "/channels/{channel}/replicas/{index:int}"
    return Starlette(
        routes=[
            Route("/channels", answered(every)),
            Route(at, answered(state)),
            Route(f"{at}/generate", answered(generate), methods=["POST"]),
            Route(f"{at}/adapters", answered(load_adapter), methods=["POST"]),
            Route(f"{at}/adapters/{{adapter}}", answered(remove_adapter), methods=["DELETE"]),
            Route(f"{at}/weights", answered(load_weights), methods=["POST"]),
            Route(f"{at}/sleep", answered(sleep), methods=["POST"]),
            Route(f"{at}/wake", answered(wake), methods=["POST"]),
        ]
    )


class RemoteEngine:
    """An engine served elsewhere (`serve_engines`), at its replica's address: `Engine` over HTTP. A profile names it
    as a channel's `engine`, each entry of `engines` giving one replica's `address`."""

    processes: Sequence[int] = ()

    def __init__(
        self,
        model: str = "",
        *,
        address: str,
        client: httpx.AsyncClient | None = None,
        timeout: float = 600.0,
        max_model_len: int | None = None,
    ) -> None:
        self.model = model
        self.address = address.rstrip("/")
        self._http = client or httpx.AsyncClient(timeout=timeout)
        self._owned = client is None
        self._max_model_len = max_model_len

    @property
    def max_model_len(self) -> int:
        """The longest sequence it accepts, as the replica says (asked once, if it was not given)."""
        if self._max_model_len is None:
            response = httpx.get(self.address, timeout=30)
            response.raise_for_status()
            self._max_model_len = int(response.json()["max_model_len"])
        return self._max_model_len

    @max_model_len.setter
    def max_model_len(self, value: int) -> None:
        self._max_model_len = value

    async def state(self, within: float = 2.0) -> dict[str, Any]:
        """What the replica serves (`Channel.state`); `Unreachable` if it does not answer `within` seconds."""
        said = await self._call("GET", "", within=within)
        self._max_model_len = int(said["max_model_len"])
        return said

    async def generate(
        self,
        prompt: Sequence[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_token_ids: Sequence[int],
        adapter: str | None,
        version: int | None = None,
    ) -> Generation:
        """Sample on the replica; with `version`, refused (`Unserved`) unless it serves `adapter` at that version."""
        body: dict[str, JsonValue] = {
            "prompt": list(prompt),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop_token_ids": list(stop_token_ids),
            "adapter": adapter,
            "version": version,
        }
        said = await self._call("POST", "/generate", body)
        return Generation(tokens=said["tokens"], logprobs=said["logprobs"], finish_reason=said["finish_reason"])

    async def load_adapter(self, name: str, path: str) -> None:
        await self._call("POST", "/adapters", {"name": name, "path": path})

    async def remove_adapter(self, name: str) -> None:
        await self._call("DELETE", f"/adapters/{name}")

    async def load_weights(self, path: str) -> None:
        await self._call("POST", "/weights", {"path": path})

    async def sleep(self) -> None:
        await self._call("POST", "/sleep")

    async def wake(self) -> None:
        await self._call("POST", "/wake")

    def close(self) -> None:
        if self._owned:
            _closing(self._http)

    async def _call(
        self, method: str, path: str, body: JsonValue = None, *, within: float | None = None
    ) -> dict[str, Any]:
        options: dict[str, Any] = {"timeout": within} if within is not None else {}
        try:
            response = await self._http.request(method, self.address + path, json=body, **options)
        except httpx.TransportError as error:
            raise Unreachable(f"{self.address}: {type(error).__name__}: {error}") from error
        if response.status_code == 409:
            raise Unserved(f"{self.address}: {response.json().get('error')}")
        if response.status_code >= 400:
            raise RuntimeError(f"{self.address}: {response.status_code} {response.text[:300]}")
        return response.json()


Find = Callable[[Serving | None], Awaitable[list[str]]]
"""The addresses of the replicas serving a run's channel, given what it should serve (which may name another run's
channel to serve it: `Serving.served_by`)."""


class RemoteChannel:
    """One run's channel, routed to replicas that serve elsewhere: what the recorder samples from (`Sampler`).

    Each session is given a replica when its first turn begins: of those that answer and are no more than `max_lag`
    checkpoints behind what the run says the channel should serve (`Serving.max_lag`, where the run says, as an eval
    does), the one a hash of the session and the replica's address ranks first, so that sessions spread over replicas
    and keep to them as others come and go. A session stays on its replica while it answers, however far behind it
    falls: its prompts' shared beginnings are cached there, and every token is stamped with the version that sampled
    it. A new session waits while no replica is close enough, for `patience` seconds at most (then `NoReplica`). What
    the replicas serve is asked again every `every` seconds, and at once after a replica refused or did not answer."""

    def __init__(
        self,
        name: str,
        renderer: "Renderer",
        limits: Limits,
        *,
        find: Find,
        wanted: Callable[[], Awaitable[Serving | None]],
        max_lag: int = MAX_LAG,
        every: float = 2.0,
        patience: float = 300.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.renderer = renderer
        self.max_lag = max_lag
        self._limits = limits
        self._find = find
        self._wanted_now = wanted
        self._every = every
        self._patience = patience
        self._http = client or httpx.AsyncClient(timeout=600.0)
        self._owned = client is None
        self._wanted: Serving | None = None
        self._replicas: dict[str, dict[str, Any]] = {}
        """What each replica that answered serves, by address."""
        self._engines: dict[str, RemoteEngine] = {}
        self._sessions: dict[str, str] = {}
        """Each session's replica, by address."""
        self._refreshed = -float("inf")
        self._refreshing = asyncio.Lock()
        self._accepted: int | None = None
        """The longest sequence the replicas accept, as last said."""
        self._throughput = Throughput()
        self._in_flight = 0

    @property
    def limits(self) -> Limits:
        """The profile's limits; the longest turn the trainer can train on, as the run says, unless they say one."""
        said = self._wanted.sequence if self._wanted is not None else None
        return replace(self._limits, sequence=self._limits.sequence or said)

    @property
    def context_limit(self) -> int:
        accepted = [int(state["max_model_len"]) for state in self._replicas.values()]
        self._accepted = min(accepted) if accepted else self._accepted
        sequence = self.limits.sequence
        if self._accepted is None:
            return sequence or 0
        return min(self._accepted, sequence or self._accepted)

    @property
    def bound(self) -> int:
        """How many checkpoints behind a replica may be and still be given a new session."""
        said = self._wanted.max_lag if self._wanted is not None else None
        return self.max_lag if said is None else said

    def lag(self, state: Mapping[str, Any]) -> int:
        """How many checkpoints behind what the channel should serve a replica is (by depth: its version); less than
        none for one that serves something deeper, which the run never asked of it."""
        return self._wanted.depth - int(state["version"]) if self._wanted is not None else 0

    def eligible(self) -> list[str]:
        """The replicas that would be given a new session now, by address."""
        return [address for address, state in self._replicas.items() if 0 <= self.lag(state) <= self.bound]

    async def refresh(self, *, now: bool = False) -> None:
        """Ask again what the channel should serve, where its replicas are and what each serves (unless that was asked
        within `every` seconds and `now` is not said). A replica that does not answer loses its sessions."""
        async with self._refreshing:
            if not now and time.monotonic() - self._refreshed < self._every:
                return
            self._wanted = await self._wanted_now()
            addresses = list(dict.fromkeys(await self._find(self._wanted)))
            states = await asyncio.gather(*(self._engine(each).state() for each in addresses), return_exceptions=True)
            self._replicas = {
                address: state for address, state in zip(addresses, states, strict=True) if isinstance(state, dict)
            }
            self._sessions = {
                session: address for session, address in self._sessions.items() if address in self._replicas
            }
            self._refreshed = time.monotonic()

    async def reaches(self) -> bool:
        """Whether a replica would take a new session now."""
        await self.refresh()
        return bool(self.eligible())

    async def weights(self, session: str) -> tuple[str | None, int]:
        """The session's replica (given one now if it has none), and the adapter it samples from there with the version
        its tokens are stamped with: what the replica serves newest."""
        waited_until = time.monotonic() + self._patience
        now = False
        while True:
            await self.refresh(now=now)
            address = self._sessions.get(session)
            if address is None and (choices := self.eligible()):
                address = max(choices, key=lambda each: zlib.crc32(f"{session} {each}".encode()))
                self._sessions[session] = address
            if address is not None:
                state = self._replicas[address]
                return state.get("adapter"), int(state["version"])
            if time.monotonic() > waited_until:
                raise NoReplica(f"no replica of {self.name} is within {self.bound} checkpoints of what it should serve")
            await asyncio.sleep(self._every)
            now = True

    async def generate(
        self,
        prompt: Sequence[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_token_ids: Sequence[int],
        adapter: str | None,
        session: str = "",
        version: int | None = None,
    ) -> Generation:
        """Sample on the session's replica. `Unserved` if it does not serve `adapter` at `version` any more, or does not
        answer (`Unreachable`: the session is given another replica at its next turn)."""
        address = self._sessions.get(session)
        if address is None:
            raise Unserved(f"session {session} has no replica of {self.name}")
        started = time.monotonic()
        if self._in_flight == 0:
            self._throughput.busy(started)
        self._in_flight += 1
        try:
            generation = await self._engine(address).generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_token_ids=stop_token_ids,
                adapter=adapter,
                version=version,
            )
        except Unreachable:
            self._replicas.pop(address, None)
            self._sessions = {each: there for each, there in self._sessions.items() if there != address}
            raise
        except Unserved:
            self._refreshed = -float("inf")  # (what it serves is asked again before the turn begins again)
            raise
        finally:
            self._in_flight -= 1
            self._throughput.ended(started, idle=self._in_flight == 0)
        self._throughput.counted(len(prompt), len(generation.tokens))
        return generation

    def replicas(self) -> list[dict[str, JsonValue]]:
        """The replicas it routes to, as last asked: each one's address, what it serves, how far behind it is, whether
        it takes new sessions, and how many sessions it has."""
        taking = set(self.eligible())
        held = [address for address in self._sessions.values()]
        return [
            {
                "address": address,
                "adapter": state.get("adapter"),
                "version": state.get("version"),
                "behind": self.lag(state),
                "taking": address in taking,
                "sessions": held.count(address),
            }
            for address, state in self._replicas.items()
        ]

    def take(self) -> dict[str, float]:
        """What passed through since the last call, as `Channel.take` counts it."""
        return self._throughput.take(busy=self._in_flight > 0)

    def close(self) -> None:
        if self._owned:
            _closing(self._http)

    def _engine(self, address: str) -> RemoteEngine:
        if address not in self._engines:
            self._engines[address] = RemoteEngine(address=address, client=self._http)
        return self._engines[address]


@dataclass(frozen=True)
class Route:
    """How a channel whose replicas serve elsewhere is sampled: its model family's renderer, its limits, where its
    replicas are (`addresses`; None: found from engine hosts' heartbeats), and how far behind one may be (`max_lag`)."""

    renderer: "Renderer"
    limits: Limits = field(default_factory=Limits)
    addresses: tuple[str, ...] | None = None
    max_lag: int = MAX_LAG


class Routes:
    """The routed channels of every run a runner plays (`RemoteChannel`), each made when first asked for: implements
    the recorder's `Routes`. A run's channel's replicas are found at the route's addresses, or from the heartbeats
    beside `ledger` of the engine hosts that follow that run's channel (`rollout_train.following`): by the run, or
    the run whose channel the record names (`Serving.served_by`), and the channel's name."""

    def __init__(
        self,
        routes: Mapping[str, Route],
        ledger: Ledger,
        presence: Presence | None,
        *,
        every: float = 2.0,
        patience: float = 300.0,
    ) -> None:
        self.routes = dict(routes)
        self.ledger = ledger
        self.presence = presence
        self._every = every
        self._patience = patience
        self._channels: dict[tuple[str, str], RemoteChannel] = {}
        self._http = httpx.AsyncClient(timeout=600.0)

    def routed(self, channel: str) -> bool:
        return channel in self.routes

    def channel(self, run: str, channel: str) -> RemoteChannel:
        key = (run, channel)
        if key not in self._channels:
            route = self.routes[channel]

            async def find(said: Serving | None) -> list[str]:
                if route.addresses is not None:
                    return list(route.addresses)
                return await self.found(*(parts(said.served_by) if said and said.served_by else (run, channel)))

            async def now() -> Serving | None:
                return await wanted(self.ledger, run, channel)

            self._channels[key] = RemoteChannel(
                channel, route.renderer, route.limits, find=find, wanted=now, max_lag=route.max_lag,
                every=self._every, patience=self._patience, client=self._http,
            )  # fmt: skip
        return self._channels[key]

    async def reaches(self, run: str, channel: str) -> bool:
        return await self.channel(run, channel).reaches()

    async def found(self, run: str, channel: str) -> list[str]:
        """The addresses of the replicas of a run's channel that engine hosts beating now say they serve."""
        if self.presence is None:
            return []
        now = time.time()
        addresses: list[str] = []
        for beat in await self.presence.beats():
            about: Any = beat.about
            if not alive(beat, now) or about.get("kind") != ENGINES or about.get("follows") != run:
                continue
            entries: list[Any] = about.get("channels") or []
            for entry in entries:
                replicas: list[Any] = entry.get("replicas") or [] if entry.get("channel") == channel else []
                addresses += [str(each["address"]) for each in replicas if each.get("address")]
        return addresses

    def channels(self) -> dict[str, RemoteChannel]:
        """Every routed channel made so far, by its name within its run (`RUN/NAME`)."""
        return {qualified(run, name): channel for (run, name), channel in self._channels.items()}

    def close(self) -> None:
        _closing(self._http)


def _closing(client: httpx.AsyncClient) -> None:
    """Close a client from code that cannot wait (`Engine.close`): in the loop running now, if one is."""
    with contextlib.suppress(RuntimeError):  # (no loop running: the client goes with the process)
        task = asyncio.get_running_loop().create_task(client.aclose())
        _CLOSING.add(task)
        task.add_done_callback(_CLOSING.discard)


_CLOSING: set[asyncio.Task[None]] = set()
"""Clients being closed (a task the loop holds no reference to may be collected before it runs)."""
