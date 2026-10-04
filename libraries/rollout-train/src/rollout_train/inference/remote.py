"""Engines on other machines: vLLM's OpenAI-compatible servers, and channels whose sessions are sampled there.

An engine elsewhere is a stock vLLM server (`vllm serve`), or anything that speaks its API in front of several (a
router, a proxy, a tunnel). The model a request names is the checkpoint it samples from: a LoRA checkpoint is loaded
as an adapter named by the checkpoint's id (`/v1/load_lora_adapter`, which the server allows with
`VLLM_ALLOW_RUNTIME_LORA_UPDATING`), and the base model is served under its own name. A sample is a completion of
the prompt's token ids (`/v1/completions` with `return_token_ids` and `logprobs`), whose answer names the model that
sampled it. `RemoteEngine` is an `Engine` over that API, beside `VllmEngine` and the scripted engines.

The recorder stays with the episode runner, so that what is recorded is exactly what was sampled. A `RemoteChannel` is
one run's channel as a runner samples it: it reads what the run says the channel should serve
(`rollout_train.serving`), and asks for that checkpoint by name; where the server does not have it yet, it asks for the
newest one before it the server has, no more than `max_lag` checkpoints behind, and for the newest again at its next
look. Every token is stamped with the depth of the checkpoint its answer names. Which server samples is the router's
business, or, with a list of servers and no router, worked out from the session's id.

What a request carries is all a server needs: the checkpoint, the prompt's tokens, the session (`session_id`, which a
router may keep to one server) and a name of its own (`request_id`). A bearer token (read from the environment or a
file, never written down) and TLS with a CA bundle and a client certificate are the deployment's (`Connection`).
"""

import asyncio
import contextlib
import os
import ssl
import time
import zlib
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import httpx
from pydantic import JsonValue

from rollout_train.inference.channel import Generation, Limits, Throughput, Unserved
from rollout_train.ledger import Ledger
from rollout_train.record import table
from rollout_train.serving import SERVING, Serving, qualified

if TYPE_CHECKING:
    from rollout_train.recorder.renderers import Renderer

ENGINES = "engines"
"""The `kind` of an engine host's heartbeat (`rollout_train.following`)."""
MAX_LAG = 1
"""Checkpoints behind what its channel should serve a sample may be, unless a profile says otherwise: one, the
checkpoint before, which a server serves while it loads the newest."""
EVERY = 2.0
"""Seconds between asks of what a routed channel should serve and what its servers have, unless it is told otherwise."""


class Unreachable(Unserved):
    """A server did not answer: it is routed around."""


class NoReplica(Exception):
    """No server has a checkpoint of a run's channel close enough to what it should serve, and none did for as long as
    a turn waits."""


@dataclass(frozen=True)
class Connection:
    """How servers are reached: a bearer token read from an environment variable (`token_env`) or a file
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
        """A client that reaches servers so."""
        token = self.token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        verify: ssl.SSLContext | bool = True
        if self.ca is not None or self.certificate is not None:
            verify = ssl.create_default_context(cafile=str(Path(self.ca).expanduser()) if self.ca else None)
            if self.certificate is not None:
                key = str(Path(self.key).expanduser()) if self.key else None
                verify.load_cert_chain(str(Path(self.certificate).expanduser()), key)
        return httpx.AsyncClient(timeout=timeout, headers=headers, verify=verify)


class RemoteEngine:
    """An engine served elsewhere: a vLLM OpenAI-compatible server at `address` (or a router or a proxy in front of
    several), serving `model` under its own name. `Engine` over its API: a request names the adapter it samples from as
    its model (the base model's name for none), and an answer that names another is refused (`Unserved`), as is a model
    the server does not have (`NotLoaded`). Adapters are loaded and removed by name; the server must allow it
    (`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`) and read the path given on its own machine. Full weights cannot be served
    under a name of their own by the server: `load_weights` refuses."""

    processes: Sequence[int] = ()

    def __init__(
        self,
        model: str = "",
        *,
        address: str,
        connection: Connection | None = None,
        client: httpx.AsyncClient | None = None,
        max_model_len: int | None = None,
    ) -> None:
        self.model = model
        self.address = address.rstrip("/")
        self._http = client or (connection or Connection()).client()
        self._owned = client is None
        self.max_model_len = max_model_len or 0
        """The longest sequence the server accepts, as it last said of its model (`models`); 0 until it has."""

    async def models(self, within: float = 2.0) -> dict[str, Any]:
        """The models the server has (`/v1/models`), by name; `Unreachable` if it does not answer `within` seconds."""
        said = await self._call("GET", "/v1/models", within=within)
        listed: list[Any] = said.get("data") or []
        found = {str(card["id"]): card for card in map(_object, listed) if "id" in card}
        with contextlib.suppress(KeyError, TypeError, ValueError, IndexError):
            self.max_model_len = _accepted(said, self.model)
        return found

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
        request: str | None = None,
    ) -> Generation:
        """Complete the prompt's tokens with the model `adapter` names (the base model for none): the tokens sampled,
        the logprob of each, how it ended, and the model the server says sampled it."""
        model = adapter or self.model
        body: dict[str, JsonValue] = {
            "model": model,
            "prompt": list(prompt),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop_token_ids": list(stop_token_ids),
            "logprobs": 0,
            "return_token_ids": True,
            "skip_special_tokens": False,
            "include_stop_str_in_output": True,
        }
        if session:
            body["session_id"] = session
        if request:
            body["request_id"] = request
        said = await self._call("POST", "/v1/completions", body)
        if said.get("model") != model:  # (whatever passed the request on sent it to another model)
            raise Unserved(f"{model} was asked for, and {said.get('model')} answered")
        choices: list[Any] = said["choices"]
        (choice,) = map(_object, choices)
        listed: list[Any] = choice.get("token_ids") or []
        tokens = [int(token) for token in listed]
        logprobs: list[Any] = _object(choice.get("logprobs")).get("token_logprobs") or []
        if len(logprobs) != len(tokens) or any(each is None for each in logprobs):
            raise RuntimeError(f"{self.address} returned a sampled token without its logprob")
        finish = "length" if choice.get("finish_reason") == "length" else "stop"
        return Generation(tokens=tokens, logprobs=[float(each) for each in logprobs], finish_reason=finish, model=model)

    async def load_adapter(self, name: str, path: str) -> None:
        await self._call("POST", "/v1/load_lora_adapter", {"lora_name": name, "lora_path": path}, answer=False)

    async def remove_adapter(self, name: str) -> None:
        await self._call("POST", "/v1/unload_lora_adapter", {"lora_name": name}, answer=False)

    async def load_weights(self, path: str) -> None:
        raise NotImplementedError(
            "a vLLM server serves full weights only under the name it was started with: a full checkpoint is served by "
            "a server started on its files, as a model of its own"
        )

    async def sleep(self) -> None:
        """Free the server's accelerator (it must allow it: `VLLM_SERVER_DEV_MODE=1`)."""
        await self._call("POST", "/sleep?level=2", answer=False)

    async def wake(self) -> None:
        await self._call("POST", "/wake_up", answer=False)

    def close(self) -> None:
        if self._owned:
            _closing(self._http)

    async def _call(
        self, method: str, path: str, body: JsonValue = None, *, within: float | None = None, answer: bool = True
    ) -> dict[str, Any]:
        options: dict[str, Any] = {"timeout": within} if within is not None else {}
        try:
            response = await self._http.request(method, self.address + path, json=body, **options)
        except httpx.TransportError as error:
            raise Unreachable(f"{self.address}: {type(error).__name__}: {error}") from error
        said = _answer(response) if answer or response.status_code >= 400 else {}
        if response.status_code == 404 and path == "/v1/completions":
            raise NotLoaded(f"{self.address}: {_said(said) or 'no such model'}")
        if response.status_code >= 400:
            raise RuntimeError(f"{self.address}: {response.status_code} {_said(said) or response.text[:300]}")
        return said


class NotLoaded(Unserved):
    """The server does not have the model a request names (yet)."""


def _accepted(listing: Mapping[str, Any], model: str) -> int:
    """The longest sequence a `/v1/models` listing says the server accepts for `model` (or its first model)."""
    listed: list[Any] = listing.get("data") or []
    cards = [_object(each) for each in listed]
    card = next((each for each in cards if each.get("id") == model), cards[0])
    return int(card["max_model_len"])


def _answer(response: httpx.Response) -> dict[str, Any]:
    """What a response says, as JSON (nothing, for one that is not an object: a proxy's own page, say)."""
    try:
        return _object(response.json())
    except ValueError:
        return {}


def _object(value: Any) -> dict[str, Any]:
    """A JSON object, as one (nothing, for any other value)."""
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _said(answer: Mapping[str, Any]) -> str:
    error: Any = answer.get("error")
    return str(_object(error).get("message") or error or "")


class RemoteChannel:
    """One run's channel, sampled on servers elsewhere: what the recorder samples from (`Sampler`).

    Each turn asks for the checkpoint the run says the channel should serve (`wanted`), by its id as the model's name;
    or, where its server does not have it yet, the newest one before it that the server has, no more than `max_lag`
    checkpoints behind (`Serving.max_lag`, where the run says, as an eval does: 0). A server that answers that it does
    not have the model is asked for the one before, and for the newest again at the next look. A turn waits while no
    server has a checkpoint close enough, for `patience` seconds at most (then `NoReplica`). Every token is stamped with
    the depth of the checkpoint its answer names; an answer that names another model is refused, and the turn sampled
    again.

    The channel's servers are one URL (a router, a proxy, or a single server), or a list: a session's turns then go to
    one of those that answer and have a checkpoint close enough, worked out from the session's id alone, so that nothing
    is kept per session. What the run says and what each server has are asked again every `every` seconds."""

    def __init__(
        self,
        name: str,
        renderer: "Renderer",
        limits: Limits,
        *,
        model: str,
        servers: Sequence[str],
        wanted: Callable[[], Awaitable[Sequence[Serving]]],
        max_lag: int = MAX_LAG,
        connection: Connection | None = None,
        every: float | None = None,
        patience: float = 300.0,
    ) -> None:
        if not servers:
            raise ValueError(f"channel {name!r} has no server")
        self.name = name
        self.renderer = renderer
        self.model = model
        self.max_lag = max_lag
        self._limits = limits
        self._wanted_now = wanted
        self._every = EVERY if every is None else every
        self._patience = patience
        self._http = (connection or Connection()).client()
        self._engines = {address: RemoteEngine(model, address=address, client=self._http) for address in servers}
        self._said: list[Serving] = []
        """What the run said the channel should serve, each checkpoint once, deepest first."""
        self._has: dict[str, set[str]] = {}
        """The models each server that answered has, by its address."""
        self._refreshed = -float("inf")
        self._refreshing = asyncio.Lock()
        self._throughput = Throughput()
        self._in_flight = 0

    @property
    def limits(self) -> Limits:
        """The profile's limits; the longest turn the trainer can train on, as the run says, unless they say one."""
        said = self._said[0].sequence if self._said else None
        return replace(self._limits, sequence=self._limits.sequence or said)

    @property
    def context_limit(self) -> int:
        accepted = [engine.max_model_len for address, engine in self._engines.items() if address in self._has]
        sequence = self.limits.sequence
        if not accepted:
            return sequence or 0
        return min(min(accepted), sequence or min(accepted))

    @property
    def bound(self) -> int:
        """How many checkpoints behind what the channel should serve a sample may be."""
        said = self._said[0].max_lag if self._said else None
        return self.max_lag if said is None else said

    def name_of(self, said: Serving) -> str:
        """The model a server serves a checkpoint as: its id; the base model's name for none."""
        return said.checkpoint or said.model or self.model

    def choices(self) -> list[Serving]:
        """The checkpoints a turn may sample from now, newest first: what the channel should serve, and those before it
        no more than `bound` behind (a full checkpoint, which a server cannot serve under its own name, is none)."""
        if not self._said:
            return [Serving(self.name, model=self.model)]  # (the run said nothing: the base model)
        newest = self._said[0].depth
        return [said for said in self._said if newest - said.depth <= self.bound and said.kind != "full"]

    def offered(self, address: str) -> Serving | None:
        """What a server would sample a turn from now: the newest of the `choices` it has."""
        has = self._has.get(address)
        if has is None:
            return None
        return next((said for said in self.choices() if self.name_of(said) in has), None)

    def server_of(self, session: str) -> str | None:
        """The server a session's turns go to now (none while none has a checkpoint close enough): of those that do,
        the one a hash of the session and its address ranks first."""
        able = [address for address in self._engines if self.offered(address) is not None]
        return max(able, key=lambda each: zlib.crc32(f"{session} {each}".encode())) if able else None

    async def refresh(self, *, now: bool = False) -> None:
        """Ask again what the channel should serve and what each server has (unless that was asked within `every`
        seconds and `now` is not said). A server that does not answer is given no turn."""
        async with self._refreshing:
            if not now and time.monotonic() - self._refreshed < self._every:
                return
            self._said = sorted(await self._wanted_now(), key=lambda each: -each.depth)
            listed = await asyncio.gather(*(each.models() for each in self._engines.values()), return_exceptions=True)
            self._has = {
                address: set(models)
                for address, models in zip(self._engines, listed, strict=True)
                if isinstance(models, dict)
            }
            self._refreshed = time.monotonic()

    async def reaches(self) -> bool:
        """Whether a server would take a turn now."""
        await self.refresh()
        return any(self.offered(address) is not None for address in self._engines)

    async def weights(self, session: str) -> tuple[str | None, int]:
        """The checkpoint a session's next turn samples from (None: the base model) and the version its tokens are
        stamped with: its depth."""
        waited_until = time.monotonic() + self._patience
        now = False
        while True:
            await self.refresh(now=now)
            if (address := self.server_of(session)) is not None and (said := self.offered(address)) is not None:
                return said.checkpoint, said.depth
            if time.monotonic() > waited_until:
                raise NoReplica(
                    f"no server of {self.name} has a checkpoint within {self.bound} of what it should serve"
                )
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
        """Sample on the session's server, from the checkpoint `adapter` names. `Unserved` if the server does not have
        it any more (`NotLoaded`), answers for another, or does not answer (`Unreachable`)."""
        address = self.server_of(session)
        if address is None:
            raise Unserved(f"no server of {self.name} would take a turn of session {session}")
        started = time.monotonic()
        if self._in_flight == 0:
            self._throughput.busy(started)
        self._in_flight += 1
        try:
            generation = await self._engines[address].generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_token_ids=stop_token_ids,
                adapter=adapter,
                session=session,
                request=request,
            )
        except Unreachable:
            self._has.pop(address, None)
            raise
        except NotLoaded:  # (until the next look, the turn is sampled from the checkpoint before)
            self._has.get(address, set()).discard(adapter or self.model)
            raise
        finally:
            self._in_flight -= 1
            self._throughput.ended(started, idle=self._in_flight == 0)
        self._throughput.counted(len(prompt), len(generation.tokens))
        return generation

    def servers(self) -> list[dict[str, JsonValue]]:
        """The servers it samples on, as last asked: each one's address, the checkpoint it would sample from now and its
        depth, and how far that is behind what the channel should serve."""
        newest = self._said[0].depth if self._said else 0
        listed: list[dict[str, JsonValue]] = []
        for address in self._engines:
            said = self.offered(address)
            depth: JsonValue = said.depth if said is not None else None
            behind: JsonValue = newest - said.depth if said is not None else None
            listed.append({"address": address, "serving": said.checkpoint if said else None, "version": depth,
                           "behind": behind, "answers": address in self._has})  # fmt: skip
        return listed

    def take(self) -> dict[str, float]:
        """What passed through since the last call, as `Channel.take` counts it."""
        return self._throughput.take(busy=self._in_flight > 0)

    def close(self) -> None:
        _closing(self._http)


async def serving_of(ledger: Ledger, run: str, channel: str) -> list[Serving]:
    """Every checkpoint a run has said its channel serves (once each, as the ledger keys them): what a routed channel
    chooses from."""
    return [
        Serving.from_json(record)
        for record in (await ledger.read(table(run, SERVING))).values()
        if isinstance(record, dict) and record.get("channel") == channel
    ]


@dataclass(frozen=True)
class Route:
    """How a channel whose engines serve elsewhere is sampled: its model family's renderer, its limits, its base
    model's name, its servers (a router, a proxy, a server, or a list), how far behind a sample may be (`max_lag`), and
    how the servers are reached (`connection`)."""

    renderer: "Renderer"
    model: str
    servers: tuple[str, ...]
    limits: Limits = field(default_factory=Limits)
    max_lag: int = MAX_LAG
    connection: Connection = field(default_factory=Connection)


class Routes:
    """The routed channels of every run a runner plays (`RemoteChannel`), each made when first asked for, choosing from
    what that run says its channel serves (in `ledger`): implements the recorder's `Routes`."""

    def __init__(
        self, routes: Mapping[str, Route], ledger: Ledger, *, every: float | None = None, patience: float = 300.0
    ) -> None:
        self.routes = dict(routes)
        self.ledger = ledger
        self._every = every
        self._patience = patience
        self._channels: dict[tuple[str, str], RemoteChannel] = {}

    def routed(self, channel: str) -> bool:
        return channel in self.routes

    def channel(self, run: str, channel: str) -> RemoteChannel:
        key = (run, channel)
        if key not in self._channels:
            route = self.routes[channel]

            async def wanted() -> list[Serving]:
                return await serving_of(self.ledger, run, channel)

            self._channels[key] = RemoteChannel(
                channel, route.renderer, route.limits, model=route.model, servers=route.servers, wanted=wanted,
                max_lag=route.max_lag, connection=route.connection, every=self._every, patience=self._patience,
            )  # fmt: skip
        return self._channels[key]

    async def reaches(self, run: str, channel: str) -> bool:
        return await self.channel(run, channel).reaches()

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
