"""A channel: a policy being served, by name.

Task code never sees a channel: a run's binding names one for a model slot, and the recorder samples from it. Whoever
trains publishes new weights to it; whoever deploys decides which engines stand behind it.
"""

import asyncio
import time
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from pydantic import JsonValue

if TYPE_CHECKING:
    from rollout_train.recorder.renderers import Renderer


@dataclass(frozen=True)
class Generation:
    tokens: list[int]
    logprobs: list[float]
    """Of each sampled token, under the distribution it was sampled from."""
    finish_reason: str
    """`stop` (a stop token, included in `tokens`) or `length`."""


class Engine(Protocol):
    """One replica serving a model: in this process, or a client of a server elsewhere."""

    max_model_len: int
    """The longest sequence (prompt and completion) it accepts."""

    async def generate(
        self,
        prompt: Sequence[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_token_ids: Sequence[int],
        adapter: str | None,
    ) -> Generation: ...

    async def load_adapter(self, name: str, path: str) -> None:
        """Register a LoRA adapter under `name`; requests name it to sample from it."""
        ...

    async def remove_adapter(self, name: str) -> None: ...

    async def load_weights(self, path: str) -> None:
        """Serve the full weights in `path` (a checkpoint's files) in place of the model's own, from now on."""
        ...

    async def sleep(self) -> None:
        """Free the accelerator (for a trainer that shares it)."""
        ...

    async def wake(self) -> None: ...

    @property
    def processes(self) -> Sequence[int]:
        """The processes it started on this machine, for whoever must end them if this process is killed."""
        ...

    def close(self) -> None: ...


class Unserved(Exception):
    """The weights a request names are not served at the version its tokens would be stamped with (they were dropped,
    or replaced while a turn was in progress), or its replica cannot be reached: the turn is sampled again, from the
    weights served then."""


@dataclass(frozen=True)
class Limits:
    """What a turn may take, in tokens: the deployment's hardware decides, and code above it receives the outcome
    (a context limit in a model's capability contract, a refusal when a context is full), never these numbers."""

    thinking: int = 1024
    """Tokens of thinking per turn before it is closed by force."""
    answer: int = 400
    """Room for the answer after the thinking."""
    sequence: int | None = None
    """The longest turn (prompt and completion): the smaller of what the engines accept and what the trainer can
    train on. A long prompt leaves less room to think, so that every turn can be trained on."""


class Sampler(Protocol):
    """What the recorder samples from: a `Channel`, whose engines this process publishes to, or a channel routed to
    replicas that serve elsewhere (`rollout_train.inference.remote.RemoteChannel`)."""

    @property
    def name(self) -> str: ...

    @property
    def renderer(self) -> "Renderer": ...

    @property
    def limits(self) -> Limits: ...

    @property
    def context_limit(self) -> int: ...

    async def weights(self, session: str) -> tuple[str | None, int]:
        """The adapter a session's next turn samples from (None: the weights its replica holds), and the version its
        tokens are stamped with."""
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
        version: int | None = None,
    ) -> Generation:
        """Sample on the session's replica; `Unserved` if it does not serve `adapter` at `version` (or is gone)."""
        ...


@dataclass
class Channel:
    name: str
    engines: Sequence[Engine]
    renderer: "Renderer"
    """The model family's token format."""
    limits: Limits = Limits()
    adapter: str | None = None
    """The adapter sampling now (None: the weights the engines hold, the model's own or a full checkpoint's)."""
    serving: str | None = None
    """What is served, by name: the adapter, or the full checkpoint the engines hold (None: the model's own)."""
    version: int = 0
    """How many times weights have been published; recorded with every sampled token."""
    held: str | None = None
    """The full checkpoint the engines hold, by name (None: the model's own)."""
    held_version: int = 0
    """The version the weights the engines hold are served as: a request that names no adapter is stamped with it."""
    _loaded: dict[str, int] = field(default_factory=dict[str, int])
    """The adapters loaded, oldest first, with the version each is served as."""
    _open: asyncio.Event = field(default_factory=asyncio.Event)
    _idle: asyncio.Event = field(default_factory=asyncio.Event)
    _in_flight: int = 0
    _throughput: "Throughput" = field(default_factory=lambda: Throughput())

    def __post_init__(self) -> None:
        if not self.engines:
            raise ValueError(f"channel {self.name!r} has no engine")
        self._open.set()
        self._idle.set()

    @property
    def context_limit(self) -> int:
        """The longest turn the channel takes, and what it tells programs."""
        accepted = min(engine.max_model_len for engine in self.engines)
        return min(accepted, self.limits.sequence or accepted)

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
        replica: int | None = None,
    ) -> Generation:
        """Sample from one of the engines: the same one for a session every time, where its prompts' shared
        beginnings are cached. `version` is the version the caller stamps the tokens with. A request addressed to one
        engine from elsewhere names it (`replica`, by its place among them; `rollout_train.inference.remote`), and is
        refused with `Unserved` if the weights `adapter` names are not served at `version` there: what that caller
        knows of the engine may be out of date, while this process's own callers read what it publishes."""
        while not self._open.is_set():  # (again after waking: the gate may have closed before this task ran)
            await self._open.wait()
        if replica is not None and version is not None and self.version_of(adapter) != version:
            raise Unserved(f"{self.name} does not serve {adapter or 'the weights it holds'} at version {version}")
        engine = self.engines[zlib.crc32(session.encode()) % len(self.engines) if replica is None else replica]
        started = time.monotonic()
        if self._in_flight == 0:
            self._throughput.busy(started)
        self._in_flight += 1
        self._idle.clear()
        try:
            generation = await engine.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_token_ids=stop_token_ids,
                adapter=adapter,
            )
        finally:
            self._in_flight -= 1
            self._throughput.ended(started, idle=self._in_flight == 0)
            if self._in_flight == 0:
                self._idle.set()
        self._throughput.counted(len(prompt), len(generation.tokens))
        return generation

    async def weights(self, session: str) -> tuple[str | None, int]:
        """The adapter a session's next turn samples from (None: the weights the engines hold), and the version its
        tokens are stamped with: the same for every session, as this process publishes to every engine at once."""
        return self.adapter, self.version

    def version_of(self, adapter: str | None) -> int | None:
        """The version an adapter loaded on the engines is served as (None: the weights they hold); None if it is not
        loaded."""
        return self.held_version if adapter is None else self._loaded.get(adapter)

    def state(self) -> dict[str, JsonValue]:
        """What the engines serve, as a replica tells whoever routes to it: the adapter sampling now, what is served
        and its version, the full checkpoint held and its version, each adapter loaded with its version, and the
        longest sequence they accept."""
        return {
            "adapter": self.adapter,
            "serving": self.serving,
            "version": self.version,
            "held": self.held,
            "held_version": self.held_version,
            "loaded": dict[str, JsonValue](self._loaded),
            "max_model_len": min(engine.max_model_len for engine in self.engines),
        }

    async def publish(self, adapter: str, path: str, version: int | None = None, *, full: bool = False) -> int:
        """Serve `adapter` from now on: a LoRA directory every engine can read at `path`, or with `full`, a full
        checkpoint's weights there, which the engines load in place of what they hold. Returns the version it is
        served as: `version` if one is given (the checkpoint's depth, which means the same in every process), or
        one more than the last. An adapter before stays loaded, so that a turn in progress finishes under the
        weights it began with; the one before that is dropped. Full weights replace the engines' at once, and the
        adapters trained on the weights before go with them. Publishing what is being served changes nothing."""
        if adapter == self.serving:
            return self.version
        served_as = self.version + 1 if version is None else version
        if full:
            await self.pause()  # (no turn may be half sampled when the weights under it change)
            try:
                await asyncio.gather(*(engine.load_weights(path) for engine in self.engines))
                self.held, self.held_version = adapter, served_as
            finally:
                self.resume()
            for dropped in self._loaded:
                await asyncio.gather(*(engine.remove_adapter(dropped) for engine in self.engines))
            self._loaded.clear()
            self.adapter = None
        else:
            await asyncio.gather(*(engine.load_adapter(adapter, path) for engine in self.engines))
            self.adapter = adapter
            self._loaded[adapter] = served_as
            while len(self._loaded) > 2:
                dropped = next(iter(self._loaded))
                del self._loaded[dropped]
                await asyncio.gather(*(engine.remove_adapter(dropped) for engine in self.engines))
        self.serving, self.version = adapter, served_as
        return self.version

    async def pause(self) -> None:
        """Hold new requests back, and wait for those in flight to finish."""
        self._open.clear()
        await self._idle.wait()

    def resume(self) -> None:
        self._open.set()

    async def sleep(self) -> None:
        await asyncio.gather(*(engine.sleep() for engine in self.engines))

    async def wake(self) -> None:
        await asyncio.gather(*(engine.wake() for engine in self.engines))

    def take(self) -> dict[str, float]:
        """What passed through since the last call: requests, tokens in and out, and throughput.
        `tokens_per_second` is everything generated over the time the channel was generating."""
        return self._throughput.take(busy=self._in_flight > 0)

    def close(self) -> None:
        for engine in self.engines:
            engine.close()


class Throughput:
    """What passed through a channel until it is taken: requests, tokens in and out, the time requests took, and the
    time the channel was generating (while any request was in flight)."""

    def __init__(self) -> None:
        self._counts = dict.fromkeys(("requests", "prompt_tokens", "generated_tokens", "request_s", "busy_s"), 0.0)
        self._busy_since = 0.0

    def busy(self, since: float) -> None:
        """A request began while none was in flight."""
        self._busy_since = since

    def ended(self, started: float, *, idle: bool) -> None:
        """A request begun at `started` ended; `idle`: none is in flight now."""
        finished = time.monotonic()
        self._counts["request_s"] += finished - started
        if idle:
            self._counts["busy_s"] += finished - self._busy_since

    def counted(self, prompt: int, generated: int) -> None:
        """A request with `prompt` tokens in generated `generated` tokens."""
        self._counts["requests"] += 1
        self._counts["prompt_tokens"] += prompt
        self._counts["generated_tokens"] += generated

    def take(self, *, busy: bool) -> dict[str, float]:
        """What passed through since the last call; `busy`: a request is in flight, and its stretch counts up to now."""
        if busy:
            now = time.monotonic()
            self._counts["busy_s"] += now - self._busy_since
            self._busy_since = now
        counts, spent, each = self._counts, self._counts["busy_s"], self._counts["request_s"]
        taken = {
            "requests": counts["requests"],
            "prompt_tokens": counts["prompt_tokens"],
            "generated_tokens": counts["generated_tokens"],
            "busy_seconds": round(spent, 1),
            "tokens_per_second": round(counts["generated_tokens"] / spent, 1) if spent else 0.0,
            "tokens_per_second_per_stream": round(counts["generated_tokens"] / each, 1) if each else 0.0,
            "mean_concurrency": round(each / spent, 1) if spent else 0.0,
        }
        self._counts = dict.fromkeys(counts, 0.0)
        return taken
