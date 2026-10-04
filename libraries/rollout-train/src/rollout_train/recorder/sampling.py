"""One turn, sampled: render the context, sample with a two-phase thinking budget, and parse.

`sample_turn` is what a recorded sample does between receiving a request and keeping its turn, for whoever does it (the
recorder in this process, the gateway in front of remote engines). It is told how to generate (`Generate`: the
engines, the weights and the sampling parameters are the caller's), and returns the reply with the tokens to record.

Thinking has a budget: a first phase samples until thinking closes or the budget runs out; then the close is forced
(not sampled, so never trained on) and a second phase samples the answer. Where the model opens its thinking itself
(Qwen3) rather than the prompt, the first phase also has room to open it, and the close is forced only if it did. A
request may cap its own output (`max_output_tokens`): the answer's room comes first and thinking gets what is left,
down to none (the block is closed before it starts).
"""

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from rollout.contracts import ContextOverflow, FinishReason, SampleRequest, SampleResult, Usage
from rollout_train.inference import Generation, Limits

if TYPE_CHECKING:
    from rollout_train.recorder.renderers import Renderer


class Generate(Protocol):
    """Sample a continuation of `context`: at most `room` tokens, stopping at any of `stop`."""

    async def __call__(self, context: Sequence[int], room: int, stop: Sequence[int]) -> Generation: ...


@dataclass(frozen=True)
class SampledTurn:
    result: SampleResult
    prompt: list[int]
    completion: list[int]
    mask: list[bool]
    """True where the policy sampled the token; False where it was forced."""
    logprobs: list[float]
    """Of every completion token (forced ones: NaN)."""
    phases: list[float]
    """Seconds each generation took, in order."""


async def sample_turn(
    request: SampleRequest, renderer: "Renderer", limits: Limits, context_limit: int, generate: Generate
) -> SampledTurn:
    """Sample one reply to `request`. Raises `ContextOverflow` when the prompt leaves no room to answer."""
    prompt = renderer.render(request.context.append, request.tools)
    thinking = renderer.thinking
    budget, answer = limits.thinking, limits.answer
    if request.max_output_tokens is not None:  # the request's own cap: the answer first, thinking with the rest
        answer = min(answer, request.max_output_tokens)
        budget = min(budget, request.max_output_tokens - answer)
    closing = len(renderer.encode(thinking.forced_close)) if thinking is not None else 0
    if len(prompt) + closing + answer > context_limit:  # no room left to answer: the caller must compact
        raise ContextOverflow(context_limit)
    if limits.sequence is not None:  # what the prompt leaves, after room for the answer
        budget = max(0, min(budget, limits.sequence - len(prompt) - answer - closing))
    stops = renderer.stop_token_ids()
    completion: list[int] = []
    mask: list[bool] = []
    logprobs: list[float] = []
    phases: list[float] = []

    def force() -> None:
        """Close the thinking, unsampled: never trained on."""
        assert thinking is not None
        forced = renderer.encode(thinking.forced_close)
        completion.extend(forced)
        mask.extend([False] * len(forced))
        logprobs.extend([math.nan] * len(forced))

    async def phase(context: Sequence[int], room: int, stop: Sequence[int]) -> str:
        started = time.monotonic()
        generation = await generate(context, room, stop)
        phases.append(time.monotonic() - started)
        completion.extend(generation.tokens)
        mask.extend([True] * len(generation.tokens))
        logprobs.extend(generation.logprobs)
        return generation.finish_reason

    if thinking is not None and thinking.prompt_opens:
        spent = budget == 0  # with no room to think, the block the prompt opened is closed at once
        if budget > 0:
            spent = await phase(prompt, budget, [*renderer.thinking_end_token_ids(), *stops]) == "length"
        if spent:  # out of budget: close the thinking, unsampled
            force()
        if not completion or completion[-1] not in stops:
            await phase([*prompt, *completion], answer, stops)
    elif thinking is not None and budget > 0:  # the model opens its thinking, if it thinks at all
        opening = renderer.encode(thinking.open)
        ended = await phase(prompt, budget + len(opening), [*renderer.thinking_end_token_ids(), *stops])
        if ended == "length" and completion[: len(opening)] == opening:  # still thinking at the budget: close it
            force()
        if not completion or completion[-1] not in stops:
            await phase([*prompt, *completion], answer, stops)
    else:
        await phase(prompt, budget + answer, stops)
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
            context_limit=context_limit,
        ),
    )
    return SampledTurn(result, prompt, completion, mask, logprobs, phases)
