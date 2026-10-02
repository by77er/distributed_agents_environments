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

if TYPE_CHECKING:
    from rollout.recorder.renderers import Renderer


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

    async def sleep(self) -> None:
        """Free the accelerator (for a trainer that shares it)."""
        ...

    async def wake(self) -> None: ...

    @property
    def processes(self) -> Sequence[int]:
        """The processes it started on this machine, for whoever must end them if this process is killed."""
        ...

    def close(self) -> None: ...


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


@dataclass
class Channel:
    name: str
    engines: Sequence[Engine]
    renderer: "Renderer"
    """The model family's token format."""
    limits: Limits = Limits()
    adapter: str | None = None
    """The adapter sampling now (None: the base model)."""
    version: int = 0
    """How many times weights have been published; recorded with every sampled token."""
    _loaded: list[str] = field(default_factory=list[str])
    _open: asyncio.Event = field(default_factory=asyncio.Event)
    _idle: asyncio.Event = field(default_factory=asyncio.Event)
    _in_flight: int = 0
    _busy_since: float = 0.0
    _counts: dict[str, float] = field(default_factory=dict[str, float])

    def __post_init__(self) -> None:
        if not self.engines:
            raise ValueError(f"channel {self.name!r} has no engine")
        self._open.set()
        self._idle.set()
        self._counts = dict.fromkeys(("requests", "prompt_tokens", "generated_tokens", "request_s", "busy_s"), 0.0)

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
    ) -> Generation:
        """Sample from one of the engines: the same one for a session every time, where its prompts' shared
        beginnings are cached."""
        while not self._open.is_set():  # (again after waking: the gate may have closed before this task ran)
            await self._open.wait()
        engine = self.engines[zlib.crc32(session.encode()) % len(self.engines)]
        started = time.monotonic()
        if self._in_flight == 0:
            self._busy_since = started
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
            finished = time.monotonic()
            self._in_flight -= 1
            self._counts["request_s"] += finished - started
            if self._in_flight == 0:
                self._counts["busy_s"] += finished - self._busy_since
                self._idle.set()
        self._counts["requests"] += 1
        self._counts["prompt_tokens"] += len(prompt)
        self._counts["generated_tokens"] += len(generation.tokens)
        return generation

    async def publish(self, adapter: str, path: str) -> int:
        """Serve `adapter` (a LoRA directory every engine can read at `path`) from now on; returns the new version.
        The adapter before stays loaded, so that a turn in progress finishes under the weights it began with; the
        one before that is dropped."""
        await asyncio.gather(*(engine.load_adapter(adapter, path) for engine in self.engines))
        self.adapter, self.version = adapter, self.version + 1
        self._loaded.append(adapter)
        while len(self._loaded) > 2:
            dropped = self._loaded.pop(0)
            await asyncio.gather(*(engine.remove_adapter(dropped) for engine in self.engines))
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
        if self._in_flight:  # a stretch still going counts up to now
            now = time.monotonic()
            self._counts["busy_s"] += now - self._busy_since
            self._busy_since = now
        counts, busy, each = self._counts, self._counts["busy_s"], self._counts["request_s"]
        taken = {
            "requests": counts["requests"],
            "prompt_tokens": counts["prompt_tokens"],
            "generated_tokens": counts["generated_tokens"],
            "busy_seconds": round(busy, 1),
            "tokens_per_second": round(counts["generated_tokens"] / busy, 1) if busy else 0.0,
            "tokens_per_second_per_stream": round(counts["generated_tokens"] / each, 1) if each else 0.0,
            "mean_concurrency": round(each / busy, 1) if busy else 0.0,
        }
        self._counts = dict.fromkeys(counts, 0.0)
        return taken

    def close(self) -> None:
        for engine in self.engines:
            engine.close()
