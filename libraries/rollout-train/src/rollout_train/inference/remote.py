"""Replicas served over HTTP, and channels routed to them.

A process that serves engines can serve them to other machines (`serve_engines`): each engine of each channel is a
replica, with an id of its own (`replica_id`: the server's name, the channel's, the engine's place). It samples tokens
in, tokens and logprobs out, and says what it serves (the adapter, the version its samples are stamped with, what else
is loaded). `RemoteEngine` is an `Engine` over that interface, beside `VllmEngine` and the scripted engines.

The recorder stays with the episode runner, so that what is recorded is exactly what was sampled; only the engines are
elsewhere. A `RemoteChannel` is one run's channel as a runner samples it: it finds the replicas that serve the channel
(from engine hosts' heartbeats, by the run and the channel's name, or at fixed addresses), asks each what it serves, and
routes each session's turns to one of them, worked out from the session's id. A replica serving a checkpoint further
behind what the run says the channel should serve (`rollout_train.serving`) than `max_lag` checkpoints is given no turn,
nor is one that cannot be reached: their sessions go on on another, from the start of the turn.

The runner picks the replica; whatever stands between it and the replica only passes requests on. Every request names
its replica in its path and in the `Rollout-Replica` header, so that a reverse proxy or a tunnel can route it knowing
nothing else of this protocol, and a runner can send every request to one URL (`via`). A request carries what it needs:
the session, and a name of its own (`request`) under which a request sent again is answered once, so a proxy may retry
it. A server may ask for a bearer token, read from the environment or a file and never written down, and be reached
over TLS with a CA bundle and a client certificate of the deployment's (`Connection`). Every answer names the replica
that gave it, and a sample the version that made it: the runner checks both, so a proxy that sends a request to the
wrong replica, or answers from a cache, cannot make a recorded version wrong.

    GET    /replicas                      every replica: what each serves
    GET    /replicas/ID                   what replica ID serves (`Channel.state`, with `replica`)
    POST   /replicas/ID/generate          {prompt, max_tokens, temperature, top_p, stop_token_ids, adapter, version,
                                           session, request} → {tokens, logprobs, finish_reason, replica, version}; 409
                                           if the adapter is not served at that version
    POST   /replicas/ID/adapters          {name, path}: load an adapter from a path on the replica's machine
    DELETE /replicas/ID/adapters/A        remove one
    POST   /replicas/ID/weights           {path}: load full weights from a path on the replica's machine
    POST   /replicas/ID/sleep, /wake
"""

import asyncio
import contextlib
import hmac
import os
import ssl
import time
import zlib
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

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
"""Checkpoints a replica may be behind what its channel should serve and still be given turns, unless a profile says
otherwise: one, the checkpoint before, which it serves while it loads the newest."""
REPLICA = "Rollout-Replica"
"""The header that names a request's replica, beside its path."""
ANSWERED = 4096
"""Answers a server keeps, by the request they answered, for a request sent again."""
EVERY = 2.0
"""Seconds between asks of what a routed channel's replicas serve, unless it is told otherwise."""


class Unreachable(Unserved):
    """A replica did not answer, or someone else answered for it: it is routed around."""


class NoReplica(Exception):
    """No replica serves a run's channel close enough to what it should serve, and none did for as long as a session
    waits."""


def replica_id(server: str, channel: str, index: int) -> str:
    """The id of engine `index` of a channel served by the server named `server` (`serve_engines`)."""
    return f"{server}.{channel}.{index}"


@dataclass(frozen=True)
class Replica:
    """Where a replica is reached: the server (or whatever passes requests on to it, `via`), and its id."""

    address: str
    id: str

    @property
    def url(self) -> str:
        return f"{self.address.rstrip('/')}/replicas/{self.id}"


@dataclass(frozen=True)
class Connection:
    """How a runner reaches replicas: a bearer token read from an environment variable (`token_env`) or a file
    (`token_file`), never written down; TLS verified against a CA bundle (`ca`), and a client certificate and its key
    (`certificate`, `key`) for servers that ask for one. Nothing: plain HTTP, no token."""

    token_env: str | None = None
    token_file: str | None = None
    ca: str | None = None
    certificate: str | None = None
    key: str | None = None

    def token(self) -> str | None:
        if self.token_env is not None:
            return os.environ.get(self.token_env) or None
        if self.token_file is not None:
            return Path(self.token_file).expanduser().read_text().strip() or None
        return None

    def client(self, timeout: float = 600.0) -> httpx.AsyncClient:
        """A client that reaches replicas so."""
        token = self.token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        verify: ssl.SSLContext | bool = True
        if self.ca is not None or self.certificate is not None:
            verify = ssl.create_default_context(cafile=str(Path(self.ca).expanduser()) if self.ca else None)
            if self.certificate is not None:
                key = str(Path(self.key).expanduser()) if self.key else None
                verify.load_cert_chain(str(Path(self.certificate).expanduser()), key)
        return httpx.AsyncClient(timeout=timeout, headers=headers, verify=verify)


def serve_engines(channels: Mapping[str, Channel], name: str, *, token: str | None = None) -> Any:
    """A Starlette application serving each engine of each channel as a replica, ids beginning with the server's
    `name` (needs the `http` extra). With `token`, a request without it as a bearer token is refused (401)."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    replicas = {
        replica_id(name, channel, index): (channel, index)
        for channel, served in channels.items()
        for index in range(len(served.engines))
    }
    answered: OrderedDict[str, asyncio.Future[dict[str, Any]]] = OrderedDict()
    """Each request's answer (or the answer being made), by its name, the newest `ANSWERED`."""

    def state(id: str) -> dict[str, Any]:
        channel, _ = replicas[id]
        return {"replica": id, **channels[channel].state()}

    def handled(work: Callable[[Request, str], Awaitable[Response]]) -> Callable[[Request], Awaitable[Response]]:
        async def handle(request: Request) -> Response:
            if token is not None:
                said = request.headers.get("authorization", "")
                if not hmac.compare_digest(said.encode(), f"Bearer {token}".encode()):
                    return JSONResponse({"error": "a bearer token is needed"}, status_code=401)
            id = str(request.path_params.get("replica", ""))
            if id and id not in replicas:
                return JSONResponse({"error": f"no replica {id}"}, status_code=404)
            try:
                return await work(request, id)
            except Exception as error:  # (the caller raises it)
                return JSONResponse({"error": f"{type(error).__name__}: {error}"}, status_code=500)

        return handle

    async def every(request: Request, _: str) -> Response:
        return JSONResponse({"replicas": [state(id) for id in replicas]})

    async def one(request: Request, id: str) -> Response:
        return JSONResponse(state(id))

    async def generate(request: Request, id: str) -> Response:
        body = await request.json()
        name = body.get("request")
        if name and name in answered:  # (sent again: answered once)
            try:
                return JSONResponse(await asyncio.shield(answered[name]))
            except (Unserved, asyncio.CancelledError) as error:  # (the first was refused, or its caller went away)
                if isinstance(error, asyncio.CancelledError) and not answered[name].cancelled():
                    raise
                return JSONResponse({"error": "it was not sampled: send it anew", **state(id)}, status_code=409)
        made: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        if name:
            answered[name] = made
            while len(answered) > ANSWERED:
                answered.popitem(last=False)
        channel, index = replicas[id]
        try:
            generation = await channels[channel].generate(
                body["prompt"],
                max_tokens=int(body["max_tokens"]),
                temperature=float(body["temperature"]),
                top_p=float(body["top_p"]),
                stop_token_ids=body["stop_token_ids"],
                adapter=body.get("adapter"),
                version=body.get("version"),
                replica=index,
            )
        except Exception as error:
            answered.pop(name, None)
            made.set_exception(error)
            made.exception()  # (retrieved: a request sent again meanwhile is told)
            if isinstance(error, Unserved):
                return JSONResponse({"error": str(error), **state(id)}, status_code=409)
            raise
        except BaseException:  # (its caller went away)
            answered.pop(name, None)
            made.cancel()
            raise
        said = {"tokens": generation.tokens, "logprobs": generation.logprobs, "finish_reason": generation.finish_reason}
        made.set_result(said | {"replica": id, "version": generation.version})
        return JSONResponse(made.result())

    async def load_adapter(request: Request, id: str) -> Response:
        channel, index = replicas[id]
        body = await request.json()
        await channels[channel].engines[index].load_adapter(str(body["name"]), str(body["path"]))
        return JSONResponse({"replica": id})

    async def remove_adapter(request: Request, id: str) -> Response:
        channel, index = replicas[id]
        await channels[channel].engines[index].remove_adapter(str(request.path_params["adapter"]))
        return JSONResponse({"replica": id})

    async def load_weights(request: Request, id: str) -> Response:
        channel, index = replicas[id]
        await channels[channel].engines[index].load_weights(str((await request.json())["path"]))
        return JSONResponse({"replica": id})

    async def sleep(request: Request, id: str) -> Response:
        channel, index = replicas[id]
        await channels[channel].engines[index].sleep()
        return JSONResponse({"replica": id})

    async def wake(request: Request, id: str) -> Response:
        channel, index = replicas[id]
        await channels[channel].engines[index].wake()
        return JSONResponse({"replica": id})

    at = "/replicas/{replica}"
    return Starlette(
        routes=[
            Route("/replicas", handled(every)),
            Route(at, handled(one)),
            Route(f"{at}/generate", handled(generate), methods=["POST"]),
            Route(f"{at}/adapters", handled(load_adapter), methods=["POST"]),
            Route(f"{at}/adapters/{{adapter}}", handled(remove_adapter), methods=["DELETE"]),
            Route(f"{at}/weights", handled(load_weights), methods=["POST"]),
            Route(f"{at}/sleep", handled(sleep), methods=["POST"]),
            Route(f"{at}/wake", handled(wake), methods=["POST"]),
        ]
    )


class RemoteEngine:
    """An engine served elsewhere (`serve_engines`): `Engine` over HTTP. It is reached at `address` (its server, or
    whatever passes requests on to it) as replica `replica`, so: `connection` says with what token and certificates.
    Every answer must name the replica, and every sample the version asked for, or it is refused (`Unreachable`,
    `Unserved`)."""

    processes: Sequence[int] = ()

    def __init__(
        self,
        model: str = "",
        *,
        address: str,
        replica: str,
        connection: Connection | None = None,
        client: httpx.AsyncClient | None = None,
        max_model_len: int | None = None,
    ) -> None:
        self.model = model
        self.replica = Replica(address, replica)
        self._http = client or (connection or Connection()).client()
        self._owned = client is None
        self._max_model_len = max_model_len

    @property
    def max_model_len(self) -> int:
        """The longest sequence it accepts, as the replica says (asked once, if it was not given)."""
        if self._max_model_len is None:
            response = httpx.get(self.replica.url, headers={**self._http.headers, REPLICA: self.replica.id}, timeout=30)
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
        session: str = "",
        request: str | None = None,
    ) -> Generation:
        """Sample on the replica. With `version`, refused (`Unserved`) unless it serves `adapter` at that version, and
        unless its answer says it sampled at it."""
        body: dict[str, JsonValue] = {
            "prompt": list(prompt),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop_token_ids": list(stop_token_ids),
            "adapter": adapter,
            "version": version,
            "session": session,
            "request": request,
        }
        said = await self._call("POST", "/generate", body)
        if version is not None and said.get("version") != version:
            raise Unserved(f"{self.replica.id} sampled at version {said.get('version')}, not {version}")
        return Generation(
            tokens=said["tokens"],
            logprobs=said["logprobs"],
            finish_reason=said["finish_reason"],
            version=said.get("version"),
            replica=self.replica.id,
        )

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
        url, headers = self.replica.url + path, {REPLICA: self.replica.id}
        try:
            response = await self._http.request(method, url, json=body, headers=headers, **options)
        except httpx.TransportError as error:
            raise Unreachable(f"{self.replica.id}: {type(error).__name__}: {error}") from error
        said: Any = _answer(response)
        if response.status_code == 409:
            raise Unserved(f"{self.replica.id}: {said.get('error')}")
        if response.status_code == 404:  # (a server without it answered: it was sent elsewhere)
            raise Unreachable(f"{self.replica.id}: {said.get('error') or 'not found'}")
        if response.status_code >= 400:
            raise RuntimeError(f"{self.replica.id}: {response.status_code} {response.text[:300]}")
        if said.get("replica") != self.replica.id:  # (whatever passed the request on sent it elsewhere)
            raise Unreachable(f"{self.replica.id} was asked, and {said.get('replica')} answered")
        return said


def _answer(response: httpx.Response) -> dict[str, Any]:
    """What a response says, as JSON (nothing, for one that is not an object: a proxy's own page, say)."""
    try:
        said: Any = response.json()
    except ValueError:
        return {}
    return cast(dict[str, Any], said) if isinstance(said, dict) else {}


Find = Callable[[Serving | None], Awaitable[list[Replica]]]
"""The replicas serving a run's channel, given what it should serve (which may name another run's channel to serve it:
`Serving.served_by`)."""


class RemoteChannel:
    """One run's channel, routed to replicas that serve elsewhere: what the recorder samples from (`Sampler`).

    A session's turns go to one replica, worked out from the session's id alone, so that nothing is kept per session:
    of the replicas that answer and are no more than `max_lag` checkpoints behind what the run says the channel should
    serve (`Serving.max_lag`, where the run says, as an eval does), the one a hash of the session and the replica's id
    ranks first. Sessions spread over the replicas, and keep to theirs, where their prompts' shared beginnings are
    cached, as others come and go; a session whose replica stops answering, or falls further behind, goes on on the
    next, from the start of its turn. Every token is stamped with the version that sampled it. A turn waits while no
    replica is close enough, for `patience` seconds at most (then `NoReplica`). What the replicas serve is asked again
    every `every` seconds, and at once after a replica refused or did not answer. With `via`, every request goes to that
    URL, naming its replica, rather than to the replica's own address."""

    def __init__(
        self,
        name: str,
        renderer: "Renderer",
        limits: Limits,
        *,
        find: Find,
        wanted: Callable[[], Awaitable[Serving | None]],
        max_lag: int = MAX_LAG,
        via: str | None = None,
        connection: Connection | None = None,
        every: float | None = None,
        patience: float = 300.0,
    ) -> None:
        self.name = name
        self.renderer = renderer
        self.max_lag = max_lag
        self.via = via
        self._limits = limits
        self._find = find
        self._wanted_now = wanted
        self._every = EVERY if every is None else every
        self._patience = patience
        self._http = (connection or Connection()).client()
        self._wanted: Serving | None = None
        self._replicas: dict[str, dict[str, Any]] = {}
        """What each replica that answered serves, by id."""
        self._engines: dict[str, RemoteEngine] = {}
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
        """How many checkpoints behind a replica may be and still be given a turn."""
        said = self._wanted.max_lag if self._wanted is not None else None
        return self.max_lag if said is None else said

    def lag(self, state: Mapping[str, Any]) -> int:
        """How many checkpoints behind what the channel should serve a replica is (by depth: its version); less than
        none for one that serves something deeper, which the run never asked of it."""
        return self._wanted.depth - int(state["version"]) if self._wanted is not None else 0

    def eligible(self) -> list[str]:
        """The replicas that would be given a turn now, by id."""
        return [id for id, state in self._replicas.items() if 0 <= self.lag(state) <= self.bound]

    def replica_of(self, session: str) -> str | None:
        """The replica a session's turns go to now (none while no replica is close enough): of those eligible, the one a
        hash of the session and its id ranks first."""
        choices = self.eligible()
        return max(choices, key=lambda each: zlib.crc32(f"{session} {each}".encode())) if choices else None

    async def refresh(self, *, now: bool = False) -> None:
        """Ask again what the channel should serve, where its replicas are and what each serves (unless that was asked
        within `every` seconds and `now` is not said). A replica that does not answer is given no turn."""
        async with self._refreshing:
            if not now and time.monotonic() - self._refreshed < self._every:
                return
            self._wanted = await self._wanted_now()
            found = {each.id: each for each in await self._find(self._wanted)}
            for id, replica in found.items():
                if id not in self._engines or self._engines[id].replica.address != (self.via or replica.address):
                    self._engines[id] = RemoteEngine(address=self.via or replica.address, replica=id, client=self._http)
            states = await asyncio.gather(*(self._engines[id].state() for id in found), return_exceptions=True)
            self._replicas = {id: state for id, state in zip(found, states, strict=True) if isinstance(state, dict)}
            self._refreshed = time.monotonic()

    async def reaches(self) -> bool:
        """Whether a replica would take a turn now."""
        await self.refresh()
        return bool(self.eligible())

    async def weights(self, session: str) -> tuple[str | None, int]:
        """The adapter a session's next turn samples from on its replica, and the version its tokens are stamped with:
        what the replica serves newest."""
        waited_until = time.monotonic() + self._patience
        now = False
        while True:
            await self.refresh(now=now)
            if (id := self.replica_of(session)) is not None:
                state = self._replicas[id]
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
        request: str | None = None,
    ) -> Generation:
        """Sample on the session's replica. `Unserved` if it does not serve `adapter` at `version` any more, or does not
        answer (`Unreachable`: it is given no turn until it answers again)."""
        id = self.replica_of(session)
        if id is None:
            raise Unserved(f"no replica of {self.name} would take a turn of session {session}")
        started = time.monotonic()
        if self._in_flight == 0:
            self._throughput.busy(started)
        self._in_flight += 1
        try:
            generation = await self._engines[id].generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_token_ids=stop_token_ids,
                adapter=adapter,
                version=version,
                session=session,
                request=request,
            )
        except Unreachable:
            self._replicas.pop(id, None)
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
        """The replicas it routes to, as last asked: each one's id, what it serves, how far behind it is, and whether
        it is given turns."""
        taking = set(self.eligible())
        return [
            {
                "replica": id,
                "adapter": state.get("adapter"),
                "version": state.get("version"),
                "behind": self.lag(state),
                "taking": id in taking,
            }
            for id, state in self._replicas.items()
        ]

    def take(self) -> dict[str, float]:
        """What passed through since the last call, as `Channel.take` counts it."""
        return self._throughput.take(busy=self._in_flight > 0)

    def close(self) -> None:
        _closing(self._http)


@dataclass(frozen=True)
class Route:
    """How a channel whose replicas serve elsewhere is sampled: its model family's renderer, its limits, where its
    replicas are (`replicas`; None: found from engine hosts' heartbeats), how far behind one may be (`max_lag`), a URL
    every request goes to instead of the replica's own address (`via`), and how they are reached (`connection`)."""

    renderer: "Renderer"
    limits: Limits = field(default_factory=Limits)
    replicas: tuple[Replica, ...] | None = None
    max_lag: int = MAX_LAG
    via: str | None = None
    connection: Connection = field(default_factory=Connection)


class Routes:
    """The routed channels of every run a runner plays (`RemoteChannel`), each made when first asked for: implements
    the recorder's `Routes`. A run's channel's replicas are the route's, or are found from the heartbeats beside
    `ledger` of the engine hosts that follow that run's channel (`rollout_train.following`): by the run, or the run
    whose channel the record names (`Serving.served_by`), and the channel's name."""

    def __init__(
        self,
        routes: Mapping[str, Route],
        ledger: Ledger,
        presence: Presence | None,
        *,
        every: float | None = None,
        patience: float = 300.0,
    ) -> None:
        self.routes = dict(routes)
        self.ledger = ledger
        self.presence = presence
        self._every = EVERY if every is None else every
        self._patience = patience
        self._channels: dict[tuple[str, str], RemoteChannel] = {}

    def routed(self, channel: str) -> bool:
        return channel in self.routes

    def channel(self, run: str, channel: str) -> RemoteChannel:
        key = (run, channel)
        if key not in self._channels:
            route = self.routes[channel]

            async def find(said: Serving | None) -> list[Replica]:
                if route.replicas is not None:
                    return list(route.replicas)
                return await self.found(*(parts(said.served_by) if said and said.served_by else (run, channel)))

            async def now() -> Serving | None:
                return await wanted(self.ledger, run, channel)

            self._channels[key] = RemoteChannel(
                channel, route.renderer, route.limits, find=find, wanted=now, max_lag=route.max_lag, via=route.via,
                connection=route.connection, every=self._every, patience=self._patience,
            )  # fmt: skip
        return self._channels[key]

    async def reaches(self, run: str, channel: str) -> bool:
        return await self.channel(run, channel).reaches()

    async def found(self, run: str, channel: str) -> list[Replica]:
        """The replicas of a run's channel that engine hosts beating now say they serve."""
        if self.presence is None:
            return []
        now = time.time()
        found: list[Replica] = []
        for beat in await self.presence.beats():
            about: Any = beat.about
            if not alive(beat, now) or about.get("kind") != ENGINES or about.get("follows") != run:
                continue
            entries: list[Any] = about.get("channels") or []
            for entry in entries:
                replicas: list[Any] = entry.get("replicas") or [] if entry.get("channel") == channel else []
                found += [
                    Replica(str(each["address"]), str(each["replica"])) for each in replicas if each.get("address")
                ]
        return found

    def channels(self) -> dict[str, RemoteChannel]:
        """Every routed channel made so far, by its name within its run (`RUN/NAME`)."""
        return {qualified(run, name): channel for (run, name), channel in self._channels.items()}

    def close(self) -> None:
        for channel in self._channels.values():
            channel.close()


def _closing(client: httpx.AsyncClient) -> None:
    """Close a client from code that cannot wait (`Engine.close`): in the loop running now, if one is."""
    with contextlib.suppress(RuntimeError):  # (no loop running: the client goes with the process)
        task = asyncio.get_running_loop().create_task(client.aclose())
        _CLOSING.add(task)
        task.add_done_callback(_CLOSING.discard)


_CLOSING: set[asyncio.Task[None]] = set()
"""Clients being closed (a task the loop holds no reference to may be collected before it runs)."""
