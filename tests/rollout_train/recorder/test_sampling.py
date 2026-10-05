"""A turn's room under each combination of budgets: none, only the answer's, only the thinking's, and both."""

import math
from collections.abc import Sequence
from typing import Any, cast

import pytest

from rollout.contracts import ContextDelta, ContextOverflow, FinishReason, Message, SampleRequest
from rollout.contracts.digests import context_digests
from rollout_train.evals import SuiteEntry
from rollout_train.gateway.keys import Fence, Grant
from rollout_train.gateway.service import contract_of, limits_of
from rollout_train.inference import Generation, Limits
from rollout_train.recorder.renderers import Renderer, ThinkingFormat
from rollout_train.recorder.sampling import MINIMUM_ANSWER, sample_turn
from rollout_train.run_settings import key_of
from rollout_train.testing import PlainRenderer, plain_channel

LIMIT = 1000


class Opened(PlainRenderer):
    """Thinking that the prompt opens (as Qwen3.5's): the model only closes it, with `>`."""

    thinking = ThinkingFormat(open="<", close=">", prompt_opens=True, forced_close=">")  # type: ignore[assignment]

    def thinking_end_token_ids(self) -> list[int]:
        return [ord(">")]


class Opening(Opened):
    """Thinking that the model opens itself (as Qwen3's), with `<`."""

    thinking = ThinkingFormat(open="<", close=">", prompt_opens=False, forced_close=">")  # type: ignore[assignment]


class Scripted:
    """Generates its replies in turn, each cut to the room it is given; notes the rooms."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.rooms: list[int] = []

    async def __call__(self, context: Sequence[int], room: int, stop: Sequence[int]) -> Generation:
        self.rooms.append(room)
        text = self.replies.pop(0)
        tokens = [ord(each) for each in text][:room]
        stopped = bool(tokens) and tokens[-1] in stop
        return Generation(tokens, [-0.5] * len(tokens), "stop" if stopped else "length")


def request(cap: int | None = None, text: str = "Go.") -> SampleRequest:
    messages = [Message.user(text)]
    return SampleRequest(
        effect_id="r_1:0:0",
        arguments_digest="d",
        session_id="r_1/policy",
        context=ContextDelta(append=messages, digest=context_digests(messages)[-1]),
        max_output_tokens=cap,
    )


PROMPT = len(PlainRenderer().render(request().context.append, ()))


async def sampled(renderer: Any, limits: Limits, generate: Scripted, cap: int | None = None, limit: int = LIMIT) -> Any:
    return await sample_turn(request(cap), cast(Renderer, renderer), limits, limit, generate)


async def test_with_no_budget_a_turn_is_one_generation_that_may_fill_the_context() -> None:
    generate = Scripted("thought>answer\n")
    turn = await sampled(Opened(), Limits(), generate)
    assert generate.rooms == [LIMIT - PROMPT] and all(turn.mask)  # (nothing forced)
    assert turn.result.finish_reason is FinishReason.STOP

    long = Scripted("x" * 2 * LIMIT)  # thinking that never closes runs to the end of the context, and stops there
    turn = await sampled(Opened(), Limits(), long)
    assert long.rooms == [LIMIT - PROMPT] and len(turn.completion) == LIMIT - PROMPT and all(turn.mask)
    assert turn.result.finish_reason is FinishReason.LENGTH and turn.result.usage.context_used == LIMIT

    for renderer in (Opening(), PlainRenderer()):  # whoever opens the thinking, or with none at all
        generate = Scripted("answer\n")
        await sampled(renderer, Limits(), generate)
        assert generate.rooms == [LIMIT - PROMPT]


async def test_with_no_budget_the_trainers_longest_turn_and_the_requests_cap_still_bound_it() -> None:
    generate = Scripted("answer\n")
    await sampled(Opened(), Limits(sequence=600), generate)
    assert generate.rooms == [600 - PROMPT]
    capped = Scripted("answer\n")
    await sampled(Opened(), Limits(), capped, cap=100)
    assert capped.rooms == [100]


async def test_with_only_an_answer_budget_thinking_takes_what_the_context_leaves_after_it() -> None:
    generate = Scripted("x" * 2 * LIMIT, "answer\n")
    turn = await sampled(Opened(), Limits(answer=50), generate)
    assert generate.rooms == [LIMIT - PROMPT - 1 - 50, 50]  # thinking, closed by force (1 token), then the answer
    assert turn.mask.count(False) == 1 and math.isnan(turn.logprobs[turn.mask.index(False)])
    assert turn.result.usage.context_used == LIMIT - 50 + len("answer\n")

    opening = Scripted("<" + "x" * 2 * LIMIT, "answer\n")  # a model that opens its thinking has room to open it
    await sampled(Opening(), Limits(answer=50), opening)
    assert opening.rooms == [LIMIT - PROMPT - 1 - 50, 50]


async def test_with_only_a_thinking_budget_the_answer_takes_whatever_is_left() -> None:
    generate = Scripted("x" * 100, "answer\n")
    turn = await sampled(Opened(), Limits(thinking=30), generate)
    assert generate.rooms == [30, LIMIT - PROMPT - 30 - 1]  # thinking, its forced close, then all the rest
    assert turn.mask.count(False) == 1

    closed = Scripted("short>", "answer\n")  # thinking closed in time: nothing forced
    turn = await sampled(Opened(), Limits(thinking=30), closed)
    assert closed.rooms == [30, LIMIT - PROMPT - len("short>")] and all(turn.mask)

    opening = Scripted("<" + "x" * 100, "answer\n")
    await sampled(Opening(), Limits(thinking=30), opening)
    assert opening.rooms == [31, LIMIT - PROMPT - 31 - 1]

    near = Scripted("x" * 2 * LIMIT, "answer\n")  # a long prompt: the budget gives way to the least answer's room
    await sampled(Opened(), Limits(thinking=900), near)
    assert near.rooms == [LIMIT - PROMPT - 1 - MINIMUM_ANSWER, MINIMUM_ANSWER]


async def test_with_both_budgets_a_turn_thinks_and_answers_as_they_say() -> None:
    generate = Scripted("x" * 100, "answer\n")
    turn = await sampled(Opened(), Limits(thinking=30, answer=50), generate)
    assert generate.rooms == [30, 50] and turn.mask.count(False) == 1
    plain = Scripted("answer\n")  # no thinking block: one phase, with both
    await sampled(PlainRenderer(), Limits(thinking=30, answer=50), plain)
    assert plain.rooms == [80]
    capped = Scripted("x" * 100, "answer\n")  # a request's cap: the answer first, the close, then thinking
    await sampled(Opened(), Limits(thinking=30, answer=50), capped, cap=60)
    assert capped.rooms == [60 - 50 - 1, 50]


async def test_a_prompt_that_leaves_less_than_the_least_answer_is_refused() -> None:
    with pytest.raises(ContextOverflow):  # no budget: the least answer's room
        await sampled(Opened(), Limits(), Scripted("answer\n"), limit=PROMPT + MINIMUM_ANSWER - 1)
    generate = Scripted("answer\n")
    await sampled(Opened(), Limits(), generate, limit=PROMPT + MINIMUM_ANSWER)
    assert generate.rooms == [MINIMUM_ANSWER]
    with pytest.raises(ContextOverflow):  # only thinking's: the least answer, and room to close the thinking
        await sampled(Opened(), Limits(thinking=30), Scripted("answer\n"), limit=PROMPT + MINIMUM_ANSWER)
    with pytest.raises(ContextOverflow):  # an answer budget: its own room
        await sampled(Opened(), Limits(answer=50), Scripted("answer\n"), limit=PROMPT + 50)
    small = Scripted("answer\n")  # a request that caps its output below the least answer needs only its cap
    await sampled(Opened(), Limits(), small, cap=10, limit=PROMPT + 20)
    assert small.rooms == [10]


def test_a_contract_without_both_budgets_offers_the_whole_context() -> None:
    assert contract_of(plain_channel()).max_output_tokens == plain_channel().context_limit
    assert contract_of(plain_channel(answer=50)).max_output_tokens == plain_channel().context_limit
    assert contract_of(plain_channel(thinking=30, answer=50)).max_output_tokens == 80
    assert limits_of(Limits(thinking=30), None, 7) == Limits(thinking=30, answer=7)  # (none: the channel's)
    assert limits_of(Limits(), None, None) == Limits()


def test_keys_suite_entries_and_run_settings_carry_no_budget_or_a_number() -> None:
    grant = Grant("train", "r_1", "policy", "policy", Fence("train", 1), 0.0)
    assert (Grant.from_json(grant.to_json()).thinking, Grant.from_json(grant.to_json()).answer) == (None, None)
    old = grant.to_json() | {"thinking": 1024, "answer": 400}  # a key minted with numbers
    assert (Grant.from_json(old).thinking, Grant.from_json(old).answer) == (1024, 400)

    entry = SuiteEntry(environment="games:words", environment_version="1", starts=[])
    assert entry.limits == {} and (entry.thinking_tokens, entry.answer_tokens) == (None, None)

    budget = key_of("channels.policy.thinking_tokens")
    assert budget is not None and budget.default is None and budget.problem(None) is None
    assert budget.problem(1024) is None and "at least 1" in str(budget.problem(0))
