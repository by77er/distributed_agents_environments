"""One turn, sampled: render the context, sample within the channel's budgets, and parse.

`sample_turn` is what the gateway does with a sample between receiving its request and keeping its turn. It is told
how to generate (`Generate`: the engines, the weights and the sampling parameters are the caller's), and returns the
reply with the tokens to record.

A turn's room is what the context leaves after the prompt (the channel's `context_limit`, and `Limits.sequence`), and
within the request's own cap (`max_output_tokens`) where it gives one. The budgets (`Limits.thinking`,
`Limits.answer`) are each optional:

- **Neither**: one generation with the whole room. Nothing is forced: a turn that fills the room while it thinks ends
  there (`length`).
- **A budget set**: thinking is bounded, by its own budget if it has one and always by the room left after the
  answer's. A first phase samples until thinking closes or its bound is reached; then the close is forced (not sampled,
  so never trained on) and a second phase samples the answer: `Limits.answer` tokens, or with no answer budget,
  whatever room is left. Where the model opens its thinking itself (Qwen3) rather than the prompt, the first phase also
  has room to open it, and the close is forced only if it did.
- **A budget set, on an engine that bounds thinking itself** (`bounds_thinking`: vLLM with its reasoning config) and
  thinking the prompt opens: one generation, the engine told how many tokens of thinking it may sample after the
  prompt's open (`thinking`), which then forces the renderer's close and samples the answer on. The same turn as two
  phases, without sampling the prompt and the thinking again: the close is forced at the same place (where the engine
  forced only its end, after a sampled token that begins it, only that end), and an answer longer than its budget is
  cut there, as the second phase would have stopped it.

The answer's room is reserved first: `Limits.answer`, or with no answer budget, `MINIMUM_ANSWER` tokens (both within
the request's cap). A prompt that leaves less than that, and room for a forced close, is refused (`ContextOverflow`).
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

MINIMUM_ANSWER = 256
"""The least room left for an answer where no answer budget is set: a prompt that leaves less is refused, so that its
program compacts rather than receives a reply cut off by the context's end. A tool call takes tens of tokens."""


class Generate(Protocol):
    """Sample a continuation of `context`: at most `room` tokens, stopping at any of `stop`; with `thinking`, at most
    that many tokens after the context's last thinking open before the engine forces the thinking's close."""

    async def __call__(
        self, context: Sequence[int], room: int, stop: Sequence[int], thinking: int | None = None
    ) -> Generation: ...


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
    request: SampleRequest,
    renderer: "Renderer",
    limits: Limits,
    context_limit: int,
    generate: Generate,
    *,
    bounds_thinking: bool = False,
) -> SampledTurn:
    """Sample one reply to `request`; in one generation where the engine bounds thinking itself (`bounds_thinking`) and
    the prompt opens the thinking. Raises `ContextOverflow` when the prompt leaves no room to answer, and
    `RuntimeError` where an engine that bounds thinking did not close it with the renderer's close at its budget."""
    prompt = renderer.render(request.context.append, request.tools)
    thinking = renderer.thinking
    cap = request.max_output_tokens
    space = min(context_limit, limits.sequence or context_limit) - len(prompt)  # everything the context leaves
    room = space if cap is None else min(space, cap)  # within the request's own cap
    answer = limits.answer if limits.answer is None or cap is None else min(limits.answer, cap)
    needed = answer if answer is not None else MINIMUM_ANSWER if cap is None else min(MINIMUM_ANSWER, cap)
    bounded = thinking is not None and (limits.thinking is not None or limits.answer is not None)
    closing = len(renderer.encode(thinking.forced_close)) if bounded and thinking is not None else 0
    if closing + needed > space:  # no room left to answer: the caller must compact
        raise ContextOverflow(context_limit)
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

    async def thought_and_answered(budget: int) -> None:
        """Thinking, its close where the engine forced it, and the answer, in one generation (`bounds_thinking`)."""
        assert thinking is not None
        close = renderer.encode(thinking.forced_close)
        opened = _after_open(prompt, renderer.encode(thinking.open))
        after = answer if answer is not None else room - budget - closing
        started = time.monotonic()
        generation = await generate(prompt, budget + closing + after, stops, thinking=opened + budget)
        phases.append(time.monotonic() - started)
        tokens, sampled = generation.tokens, generation.logprobs
        ends = set(renderer.thinking_end_token_ids())
        natural = next((index for index, token in enumerate(tokens[:budget]) if token in ends), None)
        if natural is not None or len(tokens) <= budget:  # closed in time, or stopped while thinking: all sampled
            cut, forced = (natural + 1 if natural is not None else len(tokens)), 0
        else:
            begun = next((size for size in range(len(close)) if tokens[budget - size : budget] == close[:size]
                          and tokens[budget : budget + len(close) - size] == close[size:]), None)  # fmt: skip
            if begun is None:
                raise RuntimeError("the engine did not force the renderer's thinking close at the thinking's budget")
            cut, forced = budget, len(close) - begun
        completion.extend(tokens[:cut])
        mask.extend([True] * cut)
        logprobs.extend(sampled[:cut])
        completion.extend(tokens[cut : cut + forced])
        mask.extend([False] * forced)
        logprobs.extend([math.nan] * forced)
        if completion and completion[-1] in stops:
            return
        left = answer if answer is not None else room - len(completion)
        replied = tokens[cut + forced :][:left]  # (an answer past its budget: the second phase would have stopped it)
        completion.extend(replied)
        mask.extend([True] * len(replied))
        logprobs.extend(sampled[cut + forced :][: len(replied)])

    async def answering() -> None:
        """The answer, unless the turn already ended: its budget, or everything the turn has left."""
        if completion and completion[-1] in stops:
            return
        left = answer if answer is not None else room - len(completion)
        if left > 0:
            await phase([*prompt, *completion], left, stops)

    if bounded and thinking is not None:
        opening = 0 if thinking.prompt_opens else len(renderer.encode(thinking.open))
        budget = room - closing - needed - opening  # thinking may take what the answer leaves
        budget = max(0, budget if limits.thinking is None else min(limits.thinking, budget))
        if thinking.prompt_opens and bounds_thinking and budget > 0:
            await thought_and_answered(budget)
        elif thinking.prompt_opens:
            spent = budget == 0  # with no room to think, the block the prompt opened is closed at once
            if budget > 0:
                spent = await phase(prompt, budget, [*renderer.thinking_end_token_ids(), *stops]) == "length"
            if spent:  # out of budget: close the thinking, unsampled
                force()
            await answering()
        elif budget > 0:  # the model opens its thinking, if it thinks at all
            ended = await phase(prompt, budget + opening, [*renderer.thinking_end_token_ids(), *stops])
            if ended == "length" and completion[:opening] == renderer.encode(thinking.open):  # still thinking: close it
                force()
            await answering()
        else:  # no room to think: the answer's
            await answering()
    elif limits.thinking is not None and limits.answer is not None:  # no thinking block: one phase, both budgets
        await phase(prompt, min(room, limits.thinking + limits.answer), stops)
    else:  # no budget: everything the context leaves
        await phase(prompt, room, stops)
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


def _after_open(prompt: Sequence[int], opening: Sequence[int]) -> int:
    """How many of the prompt's tokens follow its last thinking open (`ValueError` where it opens none)."""
    for start in range(len(prompt) - len(opening), -1, -1):
        if list(prompt[start : start + len(opening)]) == list(opening):
            return len(prompt) - start - len(opening)
    raise ValueError("the prompt does not open the thinking its renderer says it opens")
