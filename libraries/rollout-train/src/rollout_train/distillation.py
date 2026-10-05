"""Distillation's data: which teacher scores an episode, and a teacher's scores of a segment's sampled tokens.

- **Routing** (`teacher_for`): an objective's `distillation.teachers` maps routes to teacher channels: an environment
  (`module:name`), one of its rows (`module:name/ROW`), or `*` for any other. The most specific route that names an
  episode's environment and row gives its teacher, so one teacher scores every segment of an episode (MOPD's routing by
  domain); teachers are never combined.
- **Scoring** (`scoring_range`, `teacher_scores`): a segment is scored from its first sampled token to its last, as
  `rollout_train.inference.Engine.score` takes positions, with the teacher's `top` most likely tokens at each where the
  objective reads them (`distillation.top_k`). `teacher_scores` aligns the teacher's `Scores` with the segment's
  sampled tokens (`TeacherScores`, one for each, in the order of its spans): a token the teacher did not score, beyond
  the longest sequence it takes, has no logprob and no top tokens, and adds nothing to the loss; a position where the
  teacher gave fewer than `top_k` tokens keeps the fewer.
- **A segment scored** (`taught`): the two, around one call to a scorer, which returns the segment carrying its
  teacher's scores (`Segment.teacher`).

How a run asks its teachers: after each episode ends, each trained segment is routed by the run's environment and the
episode's row (`teacher_for`), scored by its teacher channel (`taught`, with a scorer that calls the gateway's
`Gateway.score` for that channel, which records the request as a turn that is never trained on), and kept in the
episode's trajectories, where `rollout_train.algorithm.Distillations` reads the scores. A dataset of teacher samples
(`rollout_train.datasets`, `teacher` supervision) holds segments scored the same way. The loop does not call teachers
yet: that wiring is where runs are built.
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Protocol

from rollout_train.inference.channel import Scores
from rollout_train.recorder.segments import Segment, TeacherScores

__all__ = ["EVERY", "Scorer", "routes_of", "scoring_range", "taught", "teacher_for", "teacher_scores"]

EVERY = "*"
"""The route of any environment and row no other route names."""


def teacher_for(teachers: Mapping[str, str], environment: str, row: str | None = None) -> str | None:
    """The teacher channel of an episode of `environment` (`module:name`) and `row` (its key): the route of the row
    (`ENVIRONMENT/ROW`), else of the environment, else `*`; none where no route names it."""
    for route in ([f"{environment}/{row}"] if row is not None else []) + [environment, EVERY]:
        if route in teachers:
            return teachers[route]
    return None


def routes_of(teachers: Mapping[str, str], environment: str) -> str:
    """How far `teachers` routes `environment`: `all` (its own route, or `*`), `some` (routes of some of its rows
    alone), or `none`."""
    if environment in teachers or EVERY in teachers:
        return "all"
    return "some" if any(route.startswith(f"{environment}/") for route in teachers) else "none"


def scoring_range(segment: Segment, context: int | None = None) -> tuple[int, int] | None:
    """The positions to score a segment at, `start` to `end` as `Engine.score` takes them: from its first sampled token
    to its last, no further than `context` tokens (the longest sequence the teacher scores; none: any). None where it
    sampled nothing, or nothing within `context`."""
    if not segment.spans:
        return None
    start, end = segment.spans[0].start, segment.spans[-1].end
    if context is not None:
        end = min(end, context)
    return (start, end) if start < end else None


def teacher_scores(segment: Segment, scores: Scores | None, teacher: str, top_k: int = 0) -> TeacherScores:
    """A teacher's scores of each of a segment's sampled tokens, from what it gave (`scores`, of positions from its
    `start`; none: it scored nothing): its logprob of each, none where it did not score the token, and with `top_k` its
    most likely tokens there, at most `top_k` (fewer where it gave fewer). Raises `ValueError` for scores that ask
    for a top-k and carry none, or whose tokens and logprobs do not pair up."""
    logprobs: list[float | None] = []
    top_tokens: list[list[int]] = []
    top_logprobs: list[list[float]] = []
    if scores is not None and top_k and scores.logprobs and not scores.top_tokens:
        raise ValueError(f"the teacher was asked for its top {top_k} tokens and gave none")
    if scores is not None and len(scores.top_tokens) != len(scores.top_logprobs):
        raise ValueError("the teacher's top tokens and their logprobs do not pair up")
    for span in segment.spans:
        for position in range(span.start, span.end):
            at = position - scores.start if scores is not None else -1
            if scores is None or not 0 <= at < len(scores.logprobs):
                logprobs.append(None)
                if top_k:
                    top_tokens.append([])
                    top_logprobs.append([])
                continue
            logprobs.append(float(scores.logprobs[at]))
            if top_k:
                tokens, values = scores.top_tokens[at], scores.top_logprobs[at]
                if len(tokens) != len(values):
                    raise ValueError(f"at position {position}, {len(tokens)} top tokens and {len(values)} logprobs")
                ranked = sorted(zip(tokens, values, strict=True), key=lambda pair: -pair[1])[:top_k]
                top_tokens.append([int(token) for token, _ in ranked])
                top_logprobs.append([float(value) for _, value in ranked])
    return TeacherScores(teacher, logprobs, top_tokens, top_logprobs)


class Scorer(Protocol):
    """What scores tokens for a teacher: `Engine.score` with its adapter given, or a call to the gateway."""

    async def __call__(self, tokens: Sequence[int], *, start: int, end: int, top: int) -> Scores: ...


async def taught(
    segment: Segment, score: Scorer, teacher: str, *, top_k: int = 0, context: int | None = None
) -> Segment:
    """The segment with `teacher`'s scores of its sampled tokens (`Segment.teacher`), asked of `score` within
    `context` tokens (`scoring_range`); those beyond it are not scored."""
    found = scoring_range(segment, context)
    scores = await score(segment.tokens, start=found[0], end=found[1], top=top_k) if found is not None else None
    return replace(segment, teacher=teacher_scores(segment, scores, teacher, top_k))
