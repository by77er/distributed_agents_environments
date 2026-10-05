"""Distillation's data: which teacher an episode is routed to, a teacher's scores aligned with a segment's sampled
tokens (several spans, a teacher that gives fewer than k, tokens beyond its context), a segment scored
through a scorer, and scored segments and distilled items kept and sent as JSON."""

import asyncio
from collections.abc import Sequence

import pytest
from pydantic import TypeAdapter

from rollout_train.distillation import routes_of, scoring_range, taught, teacher_for, teacher_scores
from rollout_train.inference import Scores
from rollout_train.pods.training import batch_bytes, batch_of
from rollout_train.recorder import Segment, Span, TeacherScores
from rollout_train.trainer import Distilled, Weighted, weight_of

# A conversation of two turns: a prompt (0-3), a sampled answer (4-6), a tool's result (7-8), a sampled answer (9-10).
SEGMENT = Segment(list(range(100, 111)), [Span(4, 7, 0), Span(9, 11, 0)], [-0.1] * 5, channel="policy")


def scores_of(tokens: Sequence[int], start: int, end: int, top: int = 0, *, short_at: int | None = None) -> Scores:
    """What a teacher gives positions `start` to `end`: logprob `-position / 10` for each, and `top` tokens ranked
    `1000 + position * 10 + rank` (fewer at `short_at`), the position's own among them."""
    logprobs = [-position / 10 for position in range(start, end)]
    ids = [[1000 + position * 10 + rank for rank in range(top if position != short_at else 2)]
           for position in range(start, end)]  # fmt: skip
    values = [[-0.01 - rank for rank in range(len(each))] for each in ids]
    return Scores(start, logprobs, ids if top else [], values if top else [])


def test_an_episode_is_routed_to_its_rows_teacher_else_its_environments_else_every_others() -> None:
    teachers = {"gsm8k:env/hard": "olympiad", "gsm8k:env": "math", "*": "general"}
    assert teacher_for(teachers, "gsm8k:env", "hard") == "olympiad"
    assert teacher_for(teachers, "gsm8k:env", "easy") == "math"
    assert teacher_for(teachers, "gsm8k:env") == "math"
    assert teacher_for(teachers, "ifeval:env", "x") == "general"
    assert teacher_for({"gsm8k:env": "math"}, "ifeval:env") is None
    assert routes_of(teachers, "ifeval:env") == "all" and routes_of({"gsm8k:env/hard": "t"}, "gsm8k:env") == "some"
    assert routes_of({"gsm8k:env": "math"}, "ifeval:env") == "none"


def test_a_segment_is_scored_from_its_first_sampled_token_to_its_last_within_the_teachers_context() -> None:
    assert scoring_range(SEGMENT) == (4, 11)
    assert scoring_range(SEGMENT, context=10) == (4, 10)
    assert scoring_range(SEGMENT, context=4) is None  # (nothing sampled within it)
    assert scoring_range(Segment([1, 2], [], [])) is None


def test_the_teachers_scores_line_up_with_the_sampled_tokens_of_every_span() -> None:
    found = teacher_scores(SEGMENT, scores_of(SEGMENT.tokens, 4, 11, top=3), "math", top_k=3)
    assert found.teacher == "math"
    assert found.logprobs == pytest.approx([-0.4, -0.5, -0.6, -0.9, -1.0])  # (the tool's result, 7-8, is not taken)
    assert found.top_tokens[0] == [1040, 1041, 1042] and found.top_tokens[3] == [1090, 1091, 1092]
    assert found.top_logprobs[4] == pytest.approx([-0.01, -1.01, -2.01]) and found.top == 3


def test_a_teacher_that_gives_fewer_than_k_keeps_the_fewer_and_more_than_k_keeps_the_most_likely() -> None:
    found = teacher_scores(SEGMENT, scores_of(SEGMENT.tokens, 4, 11, top=5, short_at=5), "t", top_k=4)
    assert [len(each) for each in found.top_tokens] == [4, 2, 4, 4, 4]
    unranked = Scores(4, [-1.0] * 7, [[7, 8, 9]] * 7, [[-3.0, -0.5, -1.0]] * 7)
    assert teacher_scores(SEGMENT, unranked, "t", top_k=2).top_tokens[0] == [8, 9]  # (most likely first)


def test_tokens_beyond_the_teachers_context_are_unscored() -> None:
    found = teacher_scores(SEGMENT, scores_of(SEGMENT.tokens, 4, 10, top=2), "t", top_k=2)
    assert found.logprobs[:4] == pytest.approx([-0.4, -0.5, -0.6, -0.9]) and found.logprobs[4] is None
    assert found.top_tokens[4] == [] and found.top_logprobs[4] == []
    nothing = teacher_scores(SEGMENT, None, "t", top_k=2)
    assert all(each is None for each in nothing.logprobs) and nothing.top == 0


def test_scores_that_do_not_say_what_was_asked_are_refused() -> None:
    with pytest.raises(ValueError, match="asked for its top 4 tokens and gave none"):
        teacher_scores(SEGMENT, scores_of(SEGMENT.tokens, 4, 11), "t", top_k=4)
    with pytest.raises(ValueError, match="do not pair up"):
        teacher_scores(SEGMENT, Scores(4, [-1.0] * 7, [[1]] * 7, [[-1.0]] * 6), "t", top_k=1)
    assert teacher_scores(SEGMENT, scores_of(SEGMENT.tokens, 4, 11, top=3), "t").top_tokens == []  # (none asked)


def test_a_segment_is_scored_through_a_scorer_within_the_teachers_context() -> None:
    asked: list[tuple[int, int, int]] = []

    async def scorer(tokens: Sequence[int], *, start: int, end: int, top: int) -> Scores:
        asked.append((start, end, top))
        return scores_of(tokens, start, end, top)

    scored = asyncio.run(taught(SEGMENT, scorer, "math", top_k=2, context=10))
    assert asked == [(4, 10, 2)]
    assert scored.teacher is not None and scored.teacher.teacher == "math" and scored.teacher.logprobs[-1] is None
    assert scored.tokens == SEGMENT.tokens and SEGMENT.teacher is None
    empty = asyncio.run(taught(SEGMENT, scorer, "math", context=3))
    assert len(asked) == 1 and empty.teacher is not None and all(v is None for v in empty.teacher.logprobs)


def test_scored_segments_and_distilled_items_are_kept_and_sent_as_json() -> None:
    scores = teacher_scores(SEGMENT, scores_of(SEGMENT.tokens, 4, 10, top=2), "math", top_k=2)
    scored = Segment(SEGMENT.tokens, SEGMENT.spans, SEGMENT.logprobs, "policy", teacher=scores)
    kept = TypeAdapter(list[Segment])
    back = kept.validate_json(kept.dump_json([scored, SEGMENT]))
    assert back[1] == SEGMENT and back[0].teacher is not None and back[0].teacher.top_tokens == scores.top_tokens
    assert back[0].teacher.logprobs[-1] is None
    batch = [Distilled(scored, scores, 0.5, "run/1/1/policy/0"), Distilled(scored, scores)]
    sent = batch_of(batch_bytes(batch))
    assert all(isinstance(each, Distilled) for each in sent) and sent[0].source == "run/1/1/policy/0"
    assert isinstance(sent[0], Distilled) and sent[0].scores.top_tokens == scores.top_tokens
    assert isinstance(batch_of(batch_bytes([Weighted(SEGMENT, 1.0)]))[0], Weighted)
    assert weight_of(batch[0]) == 0.5 and TeacherScores("t", [-1.0]).top == 0
