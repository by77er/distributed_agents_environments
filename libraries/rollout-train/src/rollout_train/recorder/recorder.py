"""The recorder: serves recorded model bindings and keeps what trainers need.

A run's `RecordedModel` binding names a channel (`rollout_train.inference.Channel`). The recorder's endpoint renders the
context to tokens, samples from the channel, parses the result into canonical content and records the turn: prompt
tokens, sampled tokens, behavior logprobs and the weights version that sampled them.

Thinking has a budget: a first phase samples until thinking closes or the budget runs out; then the close is forced
(not sampled, so never trained on) and a second phase samples the answer. A request may cap its own output
(`max_output_tokens`): the answer's room comes first and thinking gets what is left, down to none (the block is closed
before it starts).

**What a session exports** (`Recorder.export`) is a list of `Segment`s: token sequences with the spans the policy
sampled. A turn whose prompt begins with everything an earlier turn held (its prompt and what it sampled) continues
that turn's segment: an append-only conversation is one segment, however many turns it has, and is trained in one
pass. A turn whose context was edited (a compaction, a chat template that drops earlier thinking, an observation
replaced by a shorter form) begins a new segment. A turn that repeats an earlier prompt exactly (a client's retry)
replaces it.

**Harnesses that bring their own loop** reach a session over HTTP (`rollout_train.recorder.compat`): `address` gives the
base URL and the key to hand to one.
"""

import math
import secrets
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from rollout.contracts import (
    CapabilityContract,
    ContextOverflow,
    FinishReason,
    ModelAddress,
    ModelEndpoint,
    SampleRequest,
    SampleResult,
    SessionIdentity,
    Usage,
)
from rollout.harness.runner import RecordedModel, SamplingParameters
from rollout_train.inference import Channel

SERVED_UNDER = "/v1"
"""The path `rollout_train.recorder.compat` serves a recorder under: a recorder's `base_url` ends with it."""


@dataclass(frozen=True)
class Span:
    """Tokens `start` to `end` (exclusive) of a segment were sampled by the policy, at weights `version`."""

    start: int
    end: int
    version: int
    effect_id: str = ""
    """The sample that produced them: the `effect_id` its run's events know it by."""


@dataclass(frozen=True)
class Segment:
    """A piece of a session's trajectory: tokens that only grew, as the policy saw and continued them."""

    tokens: list[int]
    spans: list[Span]
    logprobs: list[float]
    """Behavior logprobs of the tokens inside the spans, in order."""
    channel: str = ""
    """The channel that sampled them. Its policy's versions are what the spans' `version`s count."""

    @property
    def sampled(self) -> int:
        return sum(span.end - span.start for span in self.spans)


@dataclass(frozen=True)
class _Turn:
    effect_id: str
    prompt: "array[int]"
    completion: list[int]
    mask: list[bool]
    """True where the policy sampled the token; False where the recorder forced it."""
    logprobs: list[float]
    version: int
    channel: str


@dataclass
class Recorder:
    channels: Mapping[str, Channel]
    base_url: str | None = None
    """Where `rollout_train.recorder.compat` serves this recorder, as harnesses reach it (None: it is not served)."""
    _turns: dict[str, list[_Turn]] = field(default_factory=dict[str, list[_Turn]])
    _by_effect: dict[str, SampleResult] = field(default_factory=dict[str, SampleResult])
    _keys: dict[str, tuple[str, ModelEndpoint]] = field(default_factory=dict[str, tuple[str, ModelEndpoint]])

    def endpoint(self, binding: RecordedModel) -> "RecordedEndpoint":
        channel = self.channels.get(binding.channel)
        if channel is None:
            raise ValueError(f"no recorded channel {binding.channel!r}")
        return RecordedEndpoint(self, channel, binding.sampling)

    def export(self, session_id: str) -> list[Segment]:
        """The session's segments, oldest first (see the module's description)."""
        turns = self._turns.get(session_id, [])
        segments: list[Segment] = []
        dropped: set[int] = set()  # continued by a later turn, or retried
        for index, turn in enumerate(turns):
            parent: Segment | None = None
            for earlier in range(index - 1, -1, -1):
                if turns[earlier].prompt == turn.prompt:
                    dropped.add(earlier)
                held = segments[earlier].tokens
                size = len(held)
                if size <= len(turn.prompt) and turn.prompt[size - 1] == held[-1] and list(turn.prompt[:size]) == held:
                    parent = segments[earlier]
                    dropped.add(earlier)
                    break
            spans = list(parent.spans) if parent else []
            start: int | None = None
            for offset, sampled in enumerate([*turn.mask, False]):  # (the False closes a span that runs to the end)
                position = len(turn.prompt) + offset
                if sampled and start is None:
                    start = position
                elif not sampled and start is not None:
                    spans.append(Span(start, position, turn.version, turn.effect_id))
                    start = None
            logprobs = list(parent.logprobs) if parent else []
            logprobs += [value for value, sampled in zip(turn.logprobs, turn.mask, strict=True) if sampled]
            segments.append(Segment([*turn.prompt, *turn.completion], spans, logprobs, turn.channel))
        return [segment for index, segment in enumerate(segments) if index not in dropped and segment.spans]

    def sessions(self, run_id: str) -> dict[str, list[Segment]]:
        """What each model slot of a run exports, by slot."""
        of_run = [identity for identity in map(SessionIdentity.parse, self._turns) if identity.owner == run_id]
        return {identity.model_slot: self.export(str(identity)) for identity in of_run}

    def forget(self, run_id: str) -> None:
        def of_run(session: str) -> bool:
            return SessionIdentity.parse(session).owner == run_id

        for session in [session for session in self._turns if of_run(session)]:
            for turn in self._turns.pop(session):
                self._by_effect.pop(turn.effect_id, None)
        for key in [key for key, (session, _) in self._keys.items() if of_run(session)]:
            del self._keys[key]

    async def publish(self, channel: str, adapter: str, path: str, version: int | None = None) -> int:
        """Serve new weights on a channel; returns the version they are served as."""
        return await self.channels[channel].publish(adapter, path, version)

    def served(self, key: str) -> tuple[str, ModelEndpoint] | None:
        """The session a harness's key names, and the endpoint that samples for it."""
        return self._keys.get(key)


class RecordedEndpoint:
    """Implements `ModelEndpoint` for one channel."""

    def __init__(self, recorder: Recorder, channel: Channel, sampling: SamplingParameters) -> None:
        self._recorder = recorder
        self._channel = channel
        self._sampling = sampling

    def describe(self, session_id: str) -> CapabilityContract:
        limits = self._channel.limits
        return CapabilityContract(
            context_limit=self._channel.context_limit, max_output_tokens=limits.thinking + limits.answer
        )

    def address(self, session_id: str, through: ModelEndpoint | None = None) -> ModelAddress:
        """Where a harness outside the run's own loop reaches this session, and the key that names it. Its samples
        go `through` an endpoint wrapping this one, if one is given (a runner's, which reports them to its hooks)."""
        if self._recorder.base_url is None:
            raise RuntimeError("this recorder is not served over HTTP: a harness cannot be given an address")
        key = secrets.token_urlsafe(24)
        self._recorder._keys[key] = (session_id, through or self)  # pyright: ignore[reportPrivateUsage]
        return ModelAddress(base_url=self._recorder.base_url, api_key=key, model=self._channel.name)

    async def cancel(self, effect_id: str) -> None:
        """Nothing to do: the generation stops when the task awaiting `sample` is cancelled."""

    async def sample(self, request: SampleRequest) -> SampleResult:
        recorded = self._recorder._by_effect.get(request.effect_id)  # pyright: ignore[reportPrivateUsage]
        if recorded is not None:  # a retried effect: the recorded result, not a second sample
            return recorded
        channel, renderer, limits = self._channel, self._channel.renderer, self._channel.limits
        prompt = renderer.render(request.context.append, request.tools)
        thinking = renderer.thinking
        budget, answer = limits.thinking, limits.answer
        if request.max_output_tokens is not None:  # the request's own cap: the answer first, thinking with the rest
            answer = min(answer, request.max_output_tokens)
            budget = min(budget, request.max_output_tokens - answer)
        closing = len(renderer.encode(thinking.forced_close)) if thinking is not None and thinking.prompt_opens else 0
        if len(prompt) + closing + answer > channel.context_limit:  # no room left to answer: the caller must compact
            raise ContextOverflow(channel.context_limit)
        if limits.sequence is not None:  # what the prompt leaves, after room for the answer
            budget = max(0, min(budget, limits.sequence - len(prompt) - answer - closing))
        stops = renderer.stop_token_ids()
        completion: list[int] = []
        mask: list[bool] = []
        logprobs: list[float] = []
        adapter, version = channel.adapter, channel.version

        async def generate(context: Sequence[int], room: int, stop: Sequence[int]) -> str:
            generation = await channel.generate(
                context,
                max_tokens=room,
                temperature=self._sampling.temperature,
                top_p=self._sampling.top_p,
                stop_token_ids=stop,
                adapter=adapter,
                session=request.session_id,
            )
            completion.extend(generation.tokens)
            mask.extend([True] * len(generation.tokens))
            logprobs.extend(generation.logprobs)
            return generation.finish_reason

        if thinking is not None and thinking.prompt_opens:
            spent = budget == 0  # with no room to think, the block the prompt opened is closed at once
            if budget > 0:
                spent = await generate(prompt, budget, [*renderer.thinking_end_token_ids(), *stops]) == "length"
            if spent:  # out of budget: close the thinking, unsampled
                forced = renderer.encode(thinking.forced_close)
                completion += forced
                mask += [False] * len(forced)
                logprobs += [math.nan] * len(forced)
            if not completion or completion[-1] not in stops:
                await generate([*prompt, *completion], answer, stops)
        else:
            await generate(prompt, budget + answer, stops)
        message = renderer.parse(completion, request.tools)
        finished = bool(completion) and completion[-1] in stops
        reason = FinishReason.TOOL_USE if message.tool_calls else FinishReason.STOP if finished else FinishReason.LENGTH
        result = SampleResult(
            message=message,
            finish_reason=reason,
            usage=Usage(
                input_tokens=len(prompt),
                output_tokens=len(completion),
                context_used=len(prompt) + len(completion),
                context_limit=channel.context_limit,
            ),
        )
        turn = _Turn(request.effect_id, array("i", prompt), completion, mask, logprobs, version, channel.name)
        self._recorder._turns.setdefault(request.session_id, []).append(turn)  # pyright: ignore[reportPrivateUsage]
        self._recorder._by_effect[request.effect_id] = result  # pyright: ignore[reportPrivateUsage]
        return result
