"""Engines on other machines: vLLM's OpenAI-compatible servers, and channels whose sessions are sampled there.

An engine elsewhere is a stock vLLM server (`vllm serve`), or anything that speaks its API in front of several (a
router, a proxy, a tunnel). The model a request names is the checkpoint it samples from: a LoRA checkpoint is loaded
as an adapter named by the checkpoint's id (`/v1/load_lora_adapter`, which the server allows with
`VLLM_ALLOW_RUNTIME_LORA_UPDATING`), and the base model is served under its own name. A sample is a completion of
the prompt's token ids (`/v1/completions` with `return_token_ids` and `logprobs`), whose answer names the model that
sampled it; scoring given tokens is a completion of them with `prompt_logprobs`, its one generated token dropped.
`RemoteEngine` is an `Engine` over that API, beside `VllmEngine` and the scripted engines.

The gateway records what was sampled, exactly as it was sampled. A `RemoteChannel` is one run's channel as the gateway
samples it: it reads what the run says the channel should serve
(`rollout_train.serving`), and asks for that checkpoint by name; where the server does not have it yet, it asks for the
newest one before it the server has, no more than `max_lag` checkpoints behind, and for the newest again at its next
look. Every token is stamped with the depth of the checkpoint its answer names. Which server samples is the router's
business, or, with a list of servers and no router, worked out from the session's id.

What a request carries is all a server needs: the checkpoint, the prompt's tokens, the session (`session_id`, which a
router may keep to one server) and a name of its own (`request_id`). A bearer token (read from the environment or a
file, never written down) and TLS with a CA bundle and a client certificate are the deployment's (`Connection`). A
server reached by its address alone (a pod's public IP) is known by the identity its certificate carries, a URI SAN such
as `spiffe://rollout/pod/NAME`, checked in the handshake in place of the host name (`Connection.identity`).
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
from typing import TYPE_CHECKING, Any, Protocol, cast

import httpx
from pydantic import JsonValue

from rollout_train.http import answer_of, error_of
from rollout_train.inference.channel import (
    MAX_LAG,
    Generation,
    Limits,
    NotLoaded,
    Scores,
    Throughput,
    Unserved,
    most_likely,
    scored_range,
)
from rollout_train.ledger import Ledger
from rollout_train.serving import Serving, qualified, serving_of

if TYPE_CHECKING:
    from rollout_train.recorder.renderers import Renderer

ENGINES = "engines"
"""The `kind` of an engine host's heartbeat (`rollout_train.following`)."""
EVERY = 2.0
"""Seconds between asks of what a routed channel should serve and what its servers have, unless it is told otherwise."""
CONNECTIONS = 1024
"""Connections a client keeps to one server at most: a turn in flight holds one for as long as it samples, so fewer
than a server's sequences would hold turns back in the client (httpx's default is 100)."""

PASSED_ON_UNANSWERED = frozenset({502, 503, 504})
"""What a proxy before a server (a pod's Envoy) answers for a request the server did not: it could not reach it, or
the server closed the connection before answering, or did not answer in time."""


class Unreachable(Unserved):
    """A server did not answer: it is routed around."""


class NoReplica(Exception):
    """No server has a checkpoint of a run's channel close enough to what it should serve, and none did for as long as
    a turn waits."""


class CheckpointServer(Protocol):
    """What a `RemoteChannel` samples on: a server that holds checkpoints by name and samples the one a request names.
    `RemoteEngine` is one (a vLLM server elsewhere); `rollout_train.inference.hosts.HostServer` is another (an engine
    host actor)."""

    address: str
    """How it is known: a URL, or an actor's name."""
    max_model_len: int
    """The longest sequence it accepts, as it last said (`models`); 0 until it has."""

    async def models(self, within: float = 2.0) -> dict[str, Any]:
        """The checkpoints it holds, by name (each a card as vLLM's `/v1/models` lists it); `Unreachable` if it does
        not answer `within` seconds."""
        ...

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
        thinking_budget: int | None = None,
        top: int = 0,
    ) -> Generation:
        """Sample from the checkpoint `adapter` names (None: the model it started with); `NotLoaded` where it does not
        hold it, `Unreachable` where it does not answer. The answer names what sampled it (`Generation.model`). With
        `top`, each sampled token comes with the `top` most likely tokens there (`Engine.generate`)."""
        ...

    async def score(
        self,
        tokens: Sequence[int],
        *,
        start: int,
        end: int | None = None,
        top: int = 0,
        adapter: str | None,
        session: str = "",
        request: str | None = None,
    ) -> Scores:
        """Score tokens with the checkpoint `adapter` names (`Engine.score`), refused as `generate` refuses. The answer
        names what scored them (`Scores.model`)."""
        ...

    def close(self) -> None: ...


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
    identity: str | None = None
    """The URI SAN the server's certificate must carry (`spiffe://rollout/pod/NAME`), checked in the TLS handshake in
    place of the host name: a server whose certificate names another identity, or none, is refused before anything is
    sent to it. A connection with an identity reaches only `https://` addresses: a request to any other is refused
    before it is sent (`httpx.UnsupportedProtocol`). None: the host name is checked, as TLS does."""

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
        if self.ca is not None or self.certificate is not None or self.identity is not None:
            verify = ssl.create_default_context(cafile=str(Path(self.ca).expanduser()) if self.ca else None)
            if self.certificate is not None:
                key = str(Path(self.key).expanduser()) if self.key else None
                verify.load_cert_chain(str(Path(self.certificate).expanduser()), key)
            if self.identity is not None:
                verify = requiring(verify, self.identity)
        hooks: dict[str, list[Callable[[httpx.Request], Awaitable[None]]]] = {}
        if self.identity is not None:
            hooks["request"] = [_https_only(self.identity)]
        limits = httpx.Limits(max_connections=CONNECTIONS, max_keepalive_connections=CONNECTIONS)
        return httpx.AsyncClient(timeout=timeout, headers=headers, verify=verify, event_hooks=hooks, limits=limits)


def https_only(address: str, connection: Connection | None) -> None:
    """Refuse (`ValueError`) to reach `address` with a connection that checks an identity unless it is `https://`."""
    if connection is not None and connection.identity is not None and not address.startswith("https://"):
        raise ValueError(f"{connection.identity} is reached over https only, not at {address}")


def _https_only(identity: str) -> Callable[[httpx.Request], Awaitable[None]]:
    """A request hook that refuses, before anything is sent, a request that is not over `https`: the identity a
    connection checks is checked only in a TLS handshake."""

    async def refused(request: httpx.Request) -> None:
        if request.url.scheme != "https":
            raise httpx.UnsupportedProtocol(
                f"{identity} is reached over https only, not at {request.url.scheme}://{request.url.netloc.decode()}",
                request=request,
            )

    return refused


def requiring(context: ssl.SSLContext, identity: str) -> ssl.SSLContext:
    """`context`, refusing in the handshake a peer whose certificate does not carry `identity` as a URI SAN: the
    server, for a client's context (its host name is then not checked: the identity stands in for it); the client, for
    a server's context that asks for a client certificate (`ssl.CERT_REQUIRED`)."""
    if context.verify_mode != ssl.CERT_REQUIRED:
        raise ValueError("an identity is checked only on a certificate that is verified (ssl.CERT_REQUIRED)")
    context.check_hostname = False

    class Identified(ssl.SSLObject):
        def do_handshake(self) -> None:
            super().do_handshake()
            peer = self.getpeercert()
            names: Any = peer.get("subjectAltName", ()) if peer else ()
            carried = [str(value) for kind, value in names if kind == "URI"]
            if identity not in carried:
                raise ssl.SSLCertVerificationError(
                    f"the peer's certificate is not {identity} (it names {', '.join(carried) or 'no URI'})"
                )

    context.sslobject_class = Identified
    return context


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
        bounds_thinking: bool = False,
    ) -> None:
        self.model = model
        self.address = address.rstrip("/")
        self.bounds_thinking = bounds_thinking
        """Whether the server was started with its reasoning config, so that it bounds thinking (`generate`)."""
        https_only(self.address, connection)
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
        thinking_budget: int | None = None,
        top: int = 0,
    ) -> Generation:
        """Complete the prompt's tokens with the model `adapter` names (the base model for none): the tokens sampled,
        the logprob of each, how it ended, and the model the server says sampled it. With `top`, the `top` most likely
        tokens at each, by id (`return_tokens_as_token_ids`)."""
        model = adapter or self.model
        body: dict[str, JsonValue] = {
            "model": model,
            "prompt": list(prompt),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop_token_ids": list(stop_token_ids),
            "logprobs": top,
            "return_token_ids": True,
            "skip_special_tokens": False,
            "include_stop_str_in_output": True,
        }
        if top:
            body["return_tokens_as_token_ids"] = True
        if thinking_budget is not None:  # (the server forces its reasoning config's close once it is spent)
            body["thinking_token_budget"] = thinking_budget
        choice = await self._completed(body, session, request)
        listed: list[Any] = choice.get("token_ids") or []
        tokens = [int(token) for token in listed]
        said = _object(choice.get("logprobs"))
        logprobs: list[Any] = said.get("token_logprobs") or []
        if len(logprobs) != len(tokens) or any(each is None for each in logprobs):
            raise RuntimeError(f"{self.address} returned a sampled token without its logprob")
        finish = "length" if choice.get("finish_reason") == "length" else "stop"
        tops: list[tuple[list[int], list[float]]] = []
        if top:
            entries: list[Any] = said.get("top_logprobs") or []
            if len(entries) != len(tokens):
                raise RuntimeError(f"{self.address} returned a sampled token without its most likely tokens")
            tops = [most_likely(_by_id(entry), token, top) for token, entry in zip(tokens, entries, strict=True)]
        return Generation(
            tokens=tokens, logprobs=[float(each) for each in logprobs], finish_reason=finish, model=model,
            top_tokens=[ids for ids, _ in tops], top_logprobs=[values for _, values in tops],
        )  # fmt: skip

    async def score(
        self,
        tokens: Sequence[int],
        *,
        start: int,
        end: int | None = None,
        top: int = 0,
        adapter: str | None,
        session: str = "",
        request: str | None = None,
    ) -> Scores:
        """The logprobs the model `adapter` names (the base model for none) gives the tokens at positions `start` to
        `end` of `tokens`, with the `top` most likely tokens at each, and the model the server says scored them: a
        completion of the tokens up to `end` with `prompt_logprobs`, whose one generated token (vLLM generates at least
        one) is dropped. The server caps `top` at its `--max-logprobs`."""
        end = scored_range(len(tokens), start, end)
        prompt = [int(token) for token in tokens[:end]]
        body: dict[str, JsonValue] = {
            "model": adapter or self.model,
            "prompt": list(prompt),
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": top,
            "skip_special_tokens": False,
        }
        choice = await self._completed(body, session, request)
        listed: list[Any] = choice.get("prompt_logprobs") or []
        if len(listed) != end:
            raise RuntimeError(f"{self.address} returned {len(listed)} prompt logprobs for {end} tokens")
        logprobs: list[float] = []
        tops: list[tuple[list[int], list[float]]] = []
        for at in range(start, end):
            entry = _by_id(listed[at])
            if prompt[at] not in entry:
                raise RuntimeError(f"{self.address} returned a scored token without its logprob")
            logprobs.append(entry[prompt[at]])
            if top:
                tops.append(most_likely(entry, prompt[at], top))
        return Scores(
            start, logprobs, top_tokens=[ids for ids, _ in tops], top_logprobs=[values for _, values in tops],
            model=str(body["model"]),
        )  # fmt: skip

    async def _completed(self, body: dict[str, JsonValue], session: str, request: str | None) -> dict[str, Any]:
        """The one choice of a completion of `body`, refused (`Unserved`) where another model answered than the one it
        names."""
        if session:
            body["session_id"] = session
        if request:
            body["request_id"] = request
        said = await self._call("POST", "/v1/completions", body)
        if said.get("model") != body["model"]:  # (whatever passed the request on sent it to another model)
            raise Unserved(f"{body['model']} was asked for, and {said.get('model')} answered")
        choices: list[Any] = said["choices"]
        (choice,) = map(_object, choices)
        return choice

    async def load_adapter(self, name: str, path: str) -> None:
        """Load the adapter at `path` (read on the server's machine) under `name`. One the server holds under that name
        already (loaded before a follower started again) is taken as loaded."""
        try:
            await self._call("POST", "/v1/load_lora_adapter", {"lora_name": name, "lora_path": path}, answer=False)
        except RuntimeError as error:
            if "has already been loaded" not in str(error):
                raise

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
        if response.status_code == 404 and path == "/v1/completions":
            raise NotLoaded(f"{self.address}: {error_of(response).get('message') or 'no such model'}")
        if response.status_code in PASSED_ON_UNANSWERED:  # (the proxy before it: the server did not answer)
            said = error_of(response).get("message") or response.text[:300]
            raise Unreachable(f"{self.address}: {response.status_code} {said}")
        if response.status_code >= 400:
            said = error_of(response).get("message") or response.text[:300]
            raise RuntimeError(f"{self.address}: {response.status_code} {said}")
        return answer_of(response) if answer else {}


def _by_id(entry: Any) -> dict[int, float]:
    """Logprobs by token id, from what vLLM's API returns at a position: a prompt position's `{"ID": {"logprob": …}}`,
    or a sampled token's `{"token_id:ID": LOGPROB}` (`return_tokens_as_token_ids`)."""
    found: dict[int, float] = {}
    for key, value in _object(entry).items():
        found[int(str(key).removeprefix("token_id:"))] = float(_object(value).get("logprob", value))
    return found


def _accepted(listing: Mapping[str, Any], model: str) -> int:
    """The longest sequence a `/v1/models` listing says the server accepts for `model` (or its first model)."""
    listed: list[Any] = listing.get("data") or []
    cards = [_object(each) for each in listed]
    card = next((each for each in cards if each.get("id") == model), cards[0])
    return int(card["max_model_len"])


def _object(value: Any) -> dict[str, Any]:
    """A JSON object, as one (nothing, for any other value)."""
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


class RemoteChannel:
    """One run's channel, sampled on servers elsewhere: what the gateway samples from (`Sampler`).

    Each turn asks for the checkpoint the run says the channel should serve (`wanted`), by its id as the model's name;
    or, where its server does not have it yet, the newest one before it that the server has, no more than `max_lag`
    checkpoints behind (`Serving.max_lag`, where the run says, as an eval does: 0). A server that answers that it does
    not have the model is asked for the one before, and for the newest again at the next look. A turn waits while no
    server has a checkpoint close enough, for `patience` seconds at most (then `NoReplica`). Every token is stamped with
    the depth of the checkpoint its answer names; an answer that names another model is refused, and the turn sampled
    again.

    The channel's servers are one URL (a router, a proxy, or a single server), or a list: a session's turns then go to
    one of those that answer and have a checkpoint close enough, worked out from the session's id alone, so that nothing
    is kept per session. A server is a URL (a vLLM server, reached as `connection` says) or any `CheckpointServer` (an
    engine host's, say), which stays the caller's to close. What the run says and what each server has are asked again
    every `every` seconds. With `discover`, the servers themselves are asked for again at each look (pods that come and
    go: `rollout_train.pods.routing.LeasedServers`), beside those given."""

    def __init__(
        self,
        name: str,
        renderer: "Renderer",
        limits: Limits,
        *,
        model: str,
        servers: Sequence["str | CheckpointServer"],
        wanted: Callable[[], Awaitable[Sequence[Serving]]],
        max_lag: int = MAX_LAG,
        connection: Connection | None = None,
        every: float | None = None,
        patience: float = 300.0,
        discover: Callable[[], Awaitable[Sequence["CheckpointServer"]]] | None = None,
        bounds_thinking: bool = False,
    ) -> None:
        if not servers and discover is None:
            raise ValueError(f"channel {name!r} has no server")
        self.name = name
        self.renderer = renderer
        self.model = model
        self.max_lag = max_lag
        self.bounds_thinking = bounds_thinking
        """Whether its servers bound a generation's thinking themselves (vLLM started with its reasoning config)."""
        self._limits = limits
        self._wanted_now = wanted
        self._every = EVERY if every is None else every
        self._patience = patience
        self._http = (connection or Connection()).client()
        self._engines: dict[str, CheckpointServer] = {
            each.address: each
            for each in (
                RemoteEngine(model, address=server, client=self._http, bounds_thinking=bounds_thinking)
                if isinstance(server, str)
                else server
                for server in servers
            )
        }
        self._given = dict(self._engines)
        self._discover = discover
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
        """The limits it was given; the longest turn the trainer can train on, as the run says, unless they say one."""
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
        no more than `bound` behind. A server that cannot serve a full checkpoint under its own name (a vLLM server)
        never lists it, so it is never offered there."""
        if not self._said:
            return [Serving(self.name, model=self.model)]  # (the run said nothing: the base model)
        newest = self._said[0].depth
        return [said for said in self._said if newest - said.depth <= self.bound]

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
            if self._discover is not None:
                with contextlib.suppress(Exception):  # (the servers it had stay until it can ask again)
                    found = {each.address: each for each in await self._discover()}
                    self._engines = {**self._given, **found}
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
        thinking_budget: int | None = None,
        top: int = 0,
    ) -> Generation:
        """Sample on the session's server, from the checkpoint `adapter` names. `Unserved` if the server does not have
        it any more (`NotLoaded`), answers for another, or does not answer (`Unreachable`)."""

        async def asked(server: CheckpointServer) -> Generation:
            return await server.generate(
                prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop_token_ids=stop_token_ids,
                adapter=adapter, session=session, request=request, thinking_budget=thinking_budget, top=top,
            )  # fmt: skip

        generation = await self._asked(asked, adapter, session)
        self._throughput.counted(len(prompt), len(generation.tokens))
        return generation

    async def score(
        self,
        tokens: Sequence[int],
        *,
        start: int,
        end: int | None = None,
        top: int = 0,
        adapter: str | None,
        session: str = "",
        version: int | None = None,
        request: str | None = None,
    ) -> Scores:
        """Score tokens on the session's server with the checkpoint `adapter` names, refused as `generate` is: every
        token scored is counted as a token in, and none as a token out."""

        async def asked(server: CheckpointServer) -> Scores:
            return await server.score(
                tokens, start=start, end=end, top=top, adapter=adapter, session=session, request=request
            )

        scores = await self._asked(asked, adapter, session)
        self._throughput.counted(scored_range(len(tokens), start, end), 0)
        return scores

    async def _asked[T](
        self, asked: Callable[[CheckpointServer], Awaitable[T]], adapter: str | None, session: str
    ) -> T:
        """What `asked` gets of the session's server, counted as a request in flight. A server that does not answer
        takes no turn until the next look; one that no longer has the checkpoint is asked for the one before. While no
        server would take it, the turn waits for one, looking again, for as long as a turn waits for a replica: one
        server that missed a look or a request does not fail every turn in flight."""
        address = self.server_of(session)
        waited_until = time.monotonic() + self._patience
        while address is None:
            if time.monotonic() > waited_until:
                raise Unserved(f"no server of {self.name} would take a turn of session {session}")
            await asyncio.sleep(self._every)
            await self.refresh(now=True)
            address = self.server_of(session)
        started = time.monotonic()
        if self._in_flight == 0:
            self._throughput.busy(started)
        self._in_flight += 1
        try:
            return await asked(self._engines[address])
        except Unreachable:
            self._has.pop(address, None)
            raise
        except NotLoaded:  # (until the next look, the turn is sampled from the checkpoint before)
            self._has.get(address, set()).discard(adapter or self.model)
            raise
        finally:
            self._in_flight -= 1
            self._throughput.ended(started, idle=self._in_flight == 0)

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
        if callable(close := getattr(self._discover, "close", None)):  # (the servers it found, theirs to close)
            close()


@dataclass(frozen=True)
class Route:
    """How a channel whose engines serve elsewhere is sampled: its model family's renderer, its limits, its base
    model's name, its servers (each the URL of a router, a proxy or a server, or a `CheckpointServer`, such as an engine
    host's `HostServer`), how far behind a sample may be (`max_lag`), and how URLs are reached (`connection`)."""

    renderer: "Renderer"
    model: str
    servers: tuple["str | CheckpointServer", ...]
    limits: Limits = field(default_factory=Limits)
    max_lag: int = MAX_LAG
    connection: Connection = field(default_factory=Connection)
    discover: Callable[[str], Callable[[], Awaitable[Sequence["CheckpointServer"]]]] | None = None
    """Given a run, what its channel asks at each look for servers that come and go (RunPod's pods)."""
    bounds_thinking: bool = False
    """Whether its servers bound a generation's thinking themselves (`Engine.bounds_thinking`)."""


class Routes:
    """The routed channels of every run a runner plays (`RemoteChannel`), each made when first asked for, choosing from
    what that run says its channel serves (in `ledger`): what the gateway samples them through."""

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
                discover=route.discover(run) if route.discover is not None else None,
                bounds_thinking=route.bounds_thinking,
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
