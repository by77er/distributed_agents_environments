"""A channel: a policy being served, by name.

Task code never sees a channel: a run's binding names one for a model slot, and the gateway samples from it. Whoever
trains publishes new weights to it; whoever deploys decides which engines stand behind it.
"""

import asyncio
import time
import zlib
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rollout_train.recorder.renderers import Renderer


@dataclass(frozen=True)
class Generation:
    tokens: list[int]
    logprobs: list[float]
    """Of each sampled token, under the distribution it was sampled from."""
    finish_reason: str
    """`stop` (a stop token, included in `tokens`) or `length`."""
    model: str | None = None
    """The model that sampled it, where a server elsewhere says (`rollout_train.inference.remote`): the checkpoint, by
    the name it is served as. The gateway checks that it is the checkpoint it stamps the tokens with."""
    top_tokens: list[list[int]] = field(default_factory=list[list[int]])
    """Where a request asked for the `top` most likely tokens at each position: at each sampled token, those tokens,
    most likely first, under the distribution it was sampled from (empty where none were asked for)."""
    top_logprobs: list[list[float]] = field(default_factory=list[list[float]])
    """Their logprobs, beside `top_tokens`."""


@dataclass(frozen=True)
class Scores:
    """A model's logprobs of given tokens: of each position from `start` on, the logprob of the token there given the
    tokens before it, and the `top` most likely tokens there with their logprobs, most likely first. Scores are of the
    model's own distribution (temperature 1)."""

    start: int
    """The first position scored (at least 1: the first token has nothing before it)."""
    logprobs: list[float]
    """Of the token at each position scored, in order."""
    top_tokens: list[list[int]] = field(default_factory=list[list[int]])
    """At each position scored, the most likely tokens, most likely first (empty where none were asked for)."""
    top_logprobs: list[list[float]] = field(default_factory=list[list[float]])
    """Their logprobs, beside `top_tokens`."""
    model: str | None = None
    """The model that scored them, where a server elsewhere says, as `Generation.model`."""

    @property
    def end(self) -> int:
        """The position after the last one scored."""
        return self.start + len(self.logprobs)


def scored_range(length: int, start: int, end: int | None) -> int:
    """The end of the positions `start` to `end` of a sequence of `length` tokens (`end` None: its end), checked:
    `ValueError` for a range that does not lie within it, or that begins at the first token, which has nothing before
    it to be scored by."""
    end = length if end is None else end
    if not 1 <= start <= end <= length:
        raise ValueError(f"positions {start} to {end} are not a range to score in a sequence of {length} tokens")
    return end


def most_likely(entry: Mapping[int, float], token: int, top: int) -> tuple[list[int], list[float]]:
    """The `top` most likely tokens at a position, most likely first, and their logprobs, from the logprobs by token
    that vLLM returns there: those tokens and the position's own `token`, which is among them or one more."""
    ranked = sorted(entry.items(), key=lambda item: -item[1])
    if len(ranked) > top:
        ranked = [(each, value) for each, value in ranked if each != token][:top]
    return [each for each, _ in ranked], [value for _, value in ranked]


class Engine(Protocol):
    """One replica serving a model: in this process, or a client of a server elsewhere."""

    max_model_len: int
    """The longest sequence (prompt and completion) it accepts."""
    bounds_thinking: bool
    """Whether it bounds a generation's thinking itself (`generate`'s `thinking_budget`): it forces the thinking's
    close once the budget is spent, and samples on."""

    async def generate(
        self,
        prompt: Sequence[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_token_ids: Sequence[int],
        adapter: str | None,
        thinking_budget: int | None = None,
        top: int = 0,
    ) -> Generation:
        """Sample a completion of `prompt` from `adapter` (None: the weights it holds). With `thinking_budget` (an
        engine that `bounds_thinking`), at most that many tokens after the last thinking open of `prompt` and the
        completion, those of the prompt counted, before it forces the thinking's close. With `top`, each sampled
        token comes with the `top` most likely tokens there and their logprobs."""
        ...

    async def score(
        self, tokens: Sequence[int], *, start: int, end: int | None = None, top: int = 0, adapter: str | None
    ) -> Scores:
        """The logprobs `adapter` gives the tokens at positions `start` to `end` (None: to the end) of `tokens`, each
        given those before it, with the `top` most likely tokens at each. Nothing is sampled."""
        ...

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
    """The checkpoint a turn began with is not served where it is asked for (not loaded yet, dropped, or another
    answered), or the server cannot be reached: the turn is sampled again, from what is served then."""


class NotLoaded(Unserved):
    """The server does not have the model a request names (yet)."""


MAX_LAG = 1
"""Checkpoints behind what its channel should serve a sample may be, unless the run says otherwise (`max_lag`): one,
the checkpoint before, which a server serves while it loads the newest."""


@dataclass(frozen=True)
class Limits:
    """What a turn may take, in tokens: the deployment's hardware decides, and code above it receives the outcome
    (a context limit in a model's capability contract, a refusal when a context is full), never these numbers."""

    thinking: int | None = None
    """Tokens of thinking per turn before it is closed by force; none: thinking runs until the model closes it, or
    until what the context leaves after the answer's room is spent."""
    answer: int | None = None
    """Room for the answer after the thinking; none: whatever room the turn has left. With neither budget, a turn is
    one generation that may fill what the context leaves (`rollout_train.recorder.sampling`)."""
    sequence: int | None = None
    """The longest turn (prompt and completion): the smaller of what the engines accept and what the trainer can
    train on. A long prompt leaves less room to think, so that every turn can be trained on."""


class Sampler(Protocol):
    """What the gateway samples from: a `Channel`, whose engines this process publishes to, or a channel sampled on
    servers elsewhere (`rollout_train.inference.remote.RemoteChannel`)."""

    @property
    def name(self) -> str: ...

    @property
    def renderer(self) -> "Renderer": ...

    @property
    def limits(self) -> Limits: ...

    @property
    def context_limit(self) -> int: ...

    @property
    def bounds_thinking(self) -> bool:
        """Whether its engines bound a generation's thinking themselves (`Engine.bounds_thinking`)."""
        ...

    async def weights(self, session: str) -> tuple[str | None, int]:
        """The adapter (the checkpoint) a session's next turn samples from (None: the weights the engines hold), and
        the version its tokens are stamped with."""
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
        request: str | None = None,
        thinking_budget: int | None = None,
        top: int = 0,
    ) -> Generation:
        """Sample from the checkpoint `adapter` names, stamped `version`; `Unserved` where it is not served (or the
        server is gone). `request` names the request, for whatever logs it. With `top`, each sampled token comes with
        the `top` most likely tokens there (`Engine.generate`)."""
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
        version: int | None = None,
        request: str | None = None,
    ) -> Scores:
        """Score the tokens at positions `start` to `end` of `tokens` with the checkpoint `adapter` names
        (`Engine.score`), as `generate` samples from it."""
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
    keep: int = MAX_LAG + 1
    """Adapters kept loaded: the one served and those before it a turn may still sample from (a run's
    `max_lag + 1`)."""
    model: str | None = None
    """The model the engines were started with, by the name a request asks for it (`resolved`)."""
    _loaded: list[str] = field(default_factory=list[str])
    _depths: dict[str, int] = field(default_factory=dict[str, int])
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

    @property
    def bounds_thinking(self) -> bool:
        """Whether every one of its engines bounds thinking itself."""
        return all(engine.bounds_thinking for engine in self.engines)

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
        """Sample from one of the engines: the same one for a session every time, where its prompts' shared
        beginnings are cached. `version` and `request` (the version the caller stamps the tokens with, and a name for
        the request) are for samplers elsewhere: this process's own callers read what it publishes."""
        return await self._sampled(
            prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop_token_ids=stop_token_ids,
            adapter=lambda: adapter, session=session, thinking_budget=thinking_budget, top=top,
        )  # fmt: skip

    async def sample(
        self,
        prompt: Sequence[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_token_ids: Sequence[int],
        name: str | None,
        session: str = "",
        thinking_budget: int | None = None,
        top: int = 0,
    ) -> Generation:
        """Sample what is served under `name` (a checkpoint's id, or the model's name; None: the model), as a server
        elsewhere is asked (`rollout_train.inference.remote.CheckpointServer`): `NotLoaded` where it is not served here
        once a load in progress has ended (a turn caught by a full checkpoint's load is sampled again). The answer
        names what sampled it."""
        generation = await self._sampled(
            prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop_token_ids=stop_token_ids,
            adapter=lambda: self.resolved(name), session=session, thinking_budget=thinking_budget, top=top,
        )  # fmt: skip
        return replace(generation, model=name or self.model)

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
        """Score tokens on the session's engine (`Engine.score`), as `generate` samples there."""
        return await self._scored(tokens, start=start, end=end, top=top, adapter=lambda: adapter, session=session)

    async def scored(
        self,
        tokens: Sequence[int],
        *,
        start: int,
        end: int | None = None,
        top: int = 0,
        name: str | None,
        session: str = "",
    ) -> Scores:
        """Score tokens with what is served under `name`, as `sample` samples it. The answer names what scored them."""
        scores = await self._scored(
            tokens, start=start, end=end, top=top, adapter=lambda: self.resolved(name), session=session
        )
        return replace(scores, model=name or self.model)

    def resolved(self, name: str | None) -> str | None:
        """The adapter the engines are asked for to sample what is served under `name`: the adapter itself, or None
        for the full checkpoint they hold or the model's own (`model`, or None); `NotLoaded` for anything else."""
        if name is not None and name in self._loaded:
            return name
        if name is not None and name == self.held:
            return None
        if self.held is None and name in (None, self.model):
            return None
        raise NotLoaded(f"{name or 'the model'} is not served here")

    async def _sampled(
        self,
        prompt: Sequence[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_token_ids: Sequence[int],
        adapter: Callable[[], str | None],
        session: str,
        top: int,
        thinking_budget: int | None = None,
    ) -> Generation:
        """Sample on the session's engine from the adapter `adapter` says, asked once the gate is open (what a load in
        progress left)."""

        async def asked(engine: Engine, chosen: str | None) -> Generation:
            return await engine.generate(
                prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop_token_ids=stop_token_ids,
                adapter=chosen, thinking_budget=thinking_budget, top=top,
            )  # fmt: skip

        generation = await self._asked(asked, adapter, session)
        self._throughput.counted(len(prompt), len(generation.tokens))
        return generation

    async def _scored(
        self,
        tokens: Sequence[int],
        *,
        start: int,
        end: int | None,
        top: int,
        adapter: Callable[[], str | None],
        session: str,
    ) -> Scores:
        """Score on the session's engine with the adapter `adapter` says, as `_sampled` samples: every token scored
        is a token in, and none comes out."""

        async def asked(engine: Engine, chosen: str | None) -> Scores:
            return await engine.score(tokens, start=start, end=end, top=top, adapter=chosen)

        scores = await self._asked(asked, adapter, session)
        self._throughput.counted(scored_range(len(tokens), start, end), 0)
        return scores

    async def _asked[T](
        self, asked: Callable[[Engine, str | None], Awaitable[T]], adapter: Callable[[], str | None], session: str
    ) -> T:
        """What `asked` gets of the session's engine and the adapter `adapter` says, once the gate is open (what a
        load in progress left), counted as a request in flight."""
        while not self._open.is_set():  # (again after waking: the gate may have closed before this task ran)
            await self._open.wait()
        chosen = adapter()
        engine = self.engines[zlib.crc32(session.encode()) % len(self.engines)]
        started = time.monotonic()
        if self._in_flight == 0:
            self._throughput.busy(started)
        self._in_flight += 1
        self._idle.clear()
        try:
            return await asked(engine, chosen)
        finally:
            self._in_flight -= 1
            self._throughput.ended(started, idle=self._in_flight == 0)
            if self._in_flight == 0:
                self._idle.set()

    async def weights(self, session: str) -> tuple[str | None, int]:
        """The adapter a session's next turn samples from (None: the weights the engines hold), and the version its
        tokens are stamped with: the same for every session, as this process publishes to every engine at once."""
        return self.adapter, self.version

    @property
    def loaded(self) -> list[str]:
        """The adapters loaded on the engines, oldest first: the one served, and those before it (`keep` in all)."""
        return list(self._loaded)

    def adapters(self) -> list[tuple[str, int, bool]]:
        """What the engines hold for this channel, oldest first: each adapter, and the full checkpoint held, by name,
        with the version it was published as and whether it is full weights."""
        held = [(self.held, self._depths.get(self.held, 0), True)] if self.held is not None else []
        return [*held, *((name, self._depths.get(name, 0), False) for name in self._loaded)]

    async def publish(self, adapter: str, path: str, version: int | None = None, *, full: bool = False) -> int:
        """Serve `adapter` from now on: a LoRA directory every engine can read at `path`, or with `full`, a full
        checkpoint's weights there, which the engines load in place of what they hold. Returns the version it is
        served as: `version` if one is given (the checkpoint's depth, which means the same in every process), or
        one more than the last. The adapters before stay loaded, `keep` in all with this one, so that a turn in
        progress finishes under the weights it began with; older ones are dropped. Full weights replace the engines'
        at once, and the adapters trained on the weights before go with them. Publishing what is being served
        changes nothing."""
        if adapter == self.serving:
            return self.version
        served_as = self.version + 1 if version is None else version
        if full:
            await self.pause()  # (no turn may be half sampled when the weights under it change)
            try:
                await asyncio.gather(*(engine.load_weights(path) for engine in self.engines))
                self.held = adapter
            finally:
                self.resume()
            await self.dropped()
        else:
            await asyncio.gather(*(engine.load_adapter(adapter, path) for engine in self.engines))
            self.adapter = adapter
            self._loaded.append(adapter)
        self._depths[adapter] = served_as
        await self.keeping(self.keep)
        self.serving, self.version = adapter, served_as
        return self.version

    async def keeping(self, keep: int) -> None:
        """Keep `keep` adapters loaded from now on (at least one), dropping the oldest past it."""
        self.keep = max(keep, 1)
        while len(self._loaded) > self.keep:
            dropped = self._loaded.pop(0)
            self._depths.pop(dropped, None)
            await asyncio.gather(*(engine.remove_adapter(dropped) for engine in self.engines))

    async def dropped(self) -> None:
        """Remove every adapter of this channel from the engines (other channels' on the same engines stay)."""
        for name in self._loaded:
            self._depths.pop(name, None)
            await asyncio.gather(*(engine.remove_adapter(name) for engine in self.engines))
        self._loaded.clear()
        self.adapter = None

    def forget(self, names: Collection[str]) -> None:
        """Take it that the engines no longer hold the adapters `names` (a server elsewhere that started again): they
        are loaded again when they are next published."""
        for name in names:
            if name in self._loaded:
                self._loaded.remove(name)
                self._depths.pop(name, None)
        if self.serving is not None and self.serving in names:
            self.serving, self.adapter = None, None

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
