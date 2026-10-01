"""The recorder in the local profile: serves recorded model bindings and records what trainers need.

Each `Channel` is a policy being trained: an engine (tokens in, tokens and logprobs out), a renderer (the model
family's token format, see `renderers`) and the current LoRA adapter. A run's `RecordedModel` binding names the
channel; the recorder's endpoint renders the context to tokens, samples, parses the result into canonical content and
records the turn: prompt tokens, sampled tokens, a loss mask, behavior logprobs and the adapter that sampled them.

Thinking has a budget: a first phase samples until thinking closes or the budget runs out; then the close is forced
(masked from training) and a second phase samples the answer. Each turn is recorded as its own sequence, because
chat templates of reasoning models drop earlier turns' thinking: re-rendered context differs from what was sampled.
"""

import math
from array import array
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from rollout.core.contracts import (
    CapabilityContract,
    FinishReason,
    SampleRequest,
    SampleResult,
    Usage,
)
from rollout.core.harness.runner import RecordedModel
from rollout.recorder.renderers import Renderer


@dataclass(frozen=True)
class Generation:
    tokens: list[int]
    logprobs: list[float]
    """Of each sampled token, under the distribution it was sampled from."""
    finish_reason: str
    """`stop` (a stop token, included in `tokens`) or `length`."""


class Engine(Protocol):
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


@dataclass
class Channel:
    engine: Engine
    renderer: Renderer
    adapter: str | None = None
    """The LoRA adapter sampling now (None: the base model); trainers move it forward."""
    adapter_version: int = 0
    thinking_budget: int = 512
    answer_tokens: int = 384
    context_limit: int = 32_768


@dataclass(frozen=True)
class RecordedTurn:
    session_id: str
    effect_id: str
    prompt: Sequence[int]
    """Kept as a packed array: a long episode records thousands of prompts of thousands of tokens each."""
    completion: list[int]
    loss_mask: list[bool]
    """True where the policy sampled the token; False where the recorder forced it."""
    logprobs: list[float]
    """Behavior logprobs; NaN where `loss_mask` is False."""
    adapter: str | None
    adapter_version: int
    finish_reason: FinishReason


@dataclass
class Recorder:
    channels: Mapping[str, Channel]
    turns: defaultdict[str, list[RecordedTurn]] = field(
        default_factory=lambda: defaultdict[str, list[RecordedTurn]](list)
    )
    """Recorded turns by session (`{run_id}/{slot}`), in order."""
    _by_effect: dict[str, tuple[RecordedTurn, SampleResult]] = field(
        default_factory=dict[str, tuple[RecordedTurn, SampleResult]]
    )

    def endpoint(self, binding: RecordedModel) -> "RecordedEndpoint":
        channel = self.channels.get(binding.channel)
        if channel is None:
            raise ValueError(f"no recorded channel {binding.channel!r}")
        return RecordedEndpoint(self, channel, binding)

    def sessions_of(self, run_id: str) -> dict[str, list[RecordedTurn]]:
        """The turns of each slot of a run, keyed by slot."""
        prefix = f"{run_id}/"
        return {session[len(prefix) :]: turns for session, turns in self.turns.items() if session.startswith(prefix)}

    def forget(self, run_id: str) -> None:
        for session in [session for session in self.turns if session.startswith(f"{run_id}/")]:
            for turn in self.turns.pop(session):
                self._by_effect.pop(turn.effect_id, None)


class RecordedEndpoint:
    """Implements `ModelEndpoint` for one channel."""

    def __init__(self, recorder: Recorder, channel: Channel, binding: RecordedModel) -> None:
        self._recorder = recorder
        self._channel = channel
        self._sampling = binding.sampling

    def describe(self, session_id: str) -> CapabilityContract:
        return CapabilityContract(
            context_limit=self._channel.context_limit,
            max_output_tokens=self._channel.thinking_budget + self._channel.answer_tokens,
        )

    async def cancel(self, effect_id: str) -> None:
        """Nothing to do: the generation stops when the task awaiting `sample` is cancelled."""

    async def sample(self, request: SampleRequest) -> SampleResult:
        recorded = self._recorder._by_effect.get(request.effect_id)  # pyright: ignore[reportPrivateUsage]
        if recorded is not None:  # a retried effect: the recorded result, not a second sample
            return recorded[1]
        channel, renderer = self._channel, self._channel.renderer
        prompt = renderer.render(request.context.append, request.tools)
        temperature, top_p = self._sampling.temperature, self._sampling.top_p
        stops = renderer.stop_token_ids()
        completion: list[int] = []
        mask: list[bool] = []
        logprobs: list[float] = []
        adapter, version = channel.adapter, channel.adapter_version
        thinking = renderer.thinking
        if thinking is not None and thinking.prompt_opens and channel.thinking_budget > 0:
            first = await channel.engine.generate(
                prompt, max_tokens=channel.thinking_budget, temperature=temperature, top_p=top_p,
                stop_token_ids=[*renderer.thinking_end_token_ids(), *stops], adapter=adapter,
            )  # fmt: skip
            completion += first.tokens
            mask += [True] * len(first.tokens)
            logprobs += first.logprobs
            ended = bool(first.tokens) and first.tokens[-1] in stops
            if first.finish_reason == "length":  # out of budget: close the thinking, unsampled
                forced = renderer.encode(thinking.forced_close)
                completion += forced
                mask += [False] * len(forced)
                logprobs += [math.nan] * len(forced)
            if not ended:
                second = await channel.engine.generate(
                    [*prompt, *completion], max_tokens=channel.answer_tokens, temperature=temperature,
                    top_p=top_p, stop_token_ids=stops, adapter=adapter,
                )  # fmt: skip
                completion += second.tokens
                mask += [True] * len(second.tokens)
                logprobs += second.logprobs
        else:
            only = await channel.engine.generate(
                prompt, max_tokens=channel.thinking_budget + channel.answer_tokens, temperature=temperature,
                top_p=top_p, stop_token_ids=stops, adapter=adapter,
            )  # fmt: skip
            completion, mask, logprobs = list(only.tokens), [True] * len(only.tokens), list(only.logprobs)
        message = renderer.parse(completion, request.tools)
        finished = bool(completion) and completion[-1] in stops
        reason = FinishReason.TOOL_USE if message.tool_calls else FinishReason.STOP if finished else FinishReason.LENGTH
        turn = RecordedTurn(
            session_id=request.session_id,
            effect_id=request.effect_id,
            prompt=array("i", prompt),
            completion=completion,
            loss_mask=mask,
            logprobs=logprobs,
            adapter=adapter,
            adapter_version=version,
            finish_reason=reason,
        )
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
        self._recorder.turns[request.session_id].append(turn)
        self._recorder._by_effect[request.effect_id] = (turn, result)  # pyright: ignore[reportPrivateUsage]
        return result
