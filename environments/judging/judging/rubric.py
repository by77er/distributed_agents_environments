"""The rubrics a judge scores answers against, versioned with the environment, and the verdict it must give.

A rubric (`Rubric`) has a version, says what the answer was asked to do, and lists its criteria. The judge scores each
criterion and the answer as a whole on a scale of 1 to 10, and replies with a JSON object and nothing else:

    {"criteria": {"accuracy": 8, "clarity": 9, "completeness": 7, "length": 10}, "score": 8, "reason": "..."}

`Rubric.verdict` reads a reply strictly: one JSON object, exactly the keys `criteria`, `score` and `reason`, one whole
number from 1 to 10 for each of the rubric's criteria and for the score, and a reason that says something. Anything
else is a `MalformedVerdict` that says what was wrong, which the episode tells the judge when it asks again.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

__all__ = [
    "EXPLAIN",
    "HIGHEST",
    "LOWEST",
    "RUBRICS",
    "SUMMARIZE",
    "Criterion",
    "MalformedVerdict",
    "Rubric",
    "Verdict",
    "normalized",
]

LOWEST, HIGHEST = 1, 10
"""The scale every criterion and the score are on."""


class MalformedVerdict(ValueError):
    """A judge's reply that is not the verdict a rubric asks for: why, in words the judge is told."""


@dataclass(frozen=True)
class Criterion:
    name: str
    asks: str
    """What the judge looks for, as it is told."""


@dataclass(frozen=True)
class Verdict:
    """A judge's verdict, read: a score for each criterion, the score of the answer as a whole, and why."""

    criteria: Mapping[str, int]
    score: int
    reason: str

    @property
    def normalized(self) -> float:
        """The score on a scale of 0 to 1 (`normalized`)."""
        return normalized(self.score)


def normalized(score: float) -> float:
    """A score (or an average of scores) on a scale of 0 to 1: the lowest score is 0, the highest 1."""
    return (score - LOWEST) / (HIGHEST - LOWEST)


@dataclass(frozen=True)
class Rubric:
    version: str
    """Changed whenever what the judge is told changes: every result records it."""
    task: str
    """What the answer was asked to do, as the judge is told."""
    criteria: tuple[Criterion, ...]

    def instructions(self) -> str:
        """The judge's system prompt: the task, the criteria, the scale and the verdict's form."""
        listed = "\n".join(f"- {each.name}: {each.asks}" for each in self.criteria)
        example = {
            "criteria": {each.name: HIGHEST - 2 for each in self.criteria},
            "score": HIGHEST - 2,
            "reason": "One or two sentences on what decided the score.",
        }
        return (
            f"You judge answers. The answer was asked to {self.task}\n\n"
            f"Score it on each of these criteria, from {LOWEST} (fails it entirely) to {HIGHEST} (could not be "
            f"better):\n{listed}\n\n"
            f"Then give the answer as a whole a score from {LOWEST} to {HIGHEST}, consistent with the criteria, and a "
            "short reason.\n\n"
            "Reply with one JSON object and nothing else, no code fence, in exactly this form:\n"
            f"{json.dumps(example)}"
        )

    def asked(self, request: str, answer: str) -> str:
        """What the judge is shown: the request the answer was given, and the answer."""
        return f"The request:\n{request}\n\nThe answer:\n{answer}"

    def verdict(self, reply: str) -> Verdict:
        """A judge's reply, read strictly; `MalformedVerdict` saying why when it is not the verdict asked for."""
        text = reply.strip()
        if not (text.startswith("{") and text.endswith("}")):
            raise MalformedVerdict("the reply is not one JSON object and nothing else")
        try:
            said: Any = json.loads(text)
        except ValueError as error:
            raise MalformedVerdict(f"the reply is not valid JSON ({error})") from None
        if not isinstance(said, dict):
            raise MalformedVerdict("the reply is not a JSON object")
        verdict = cast(dict[str, Any], said)
        if set(verdict) != {"criteria", "score", "reason"}:
            raise MalformedVerdict(f"the object has keys {sorted(verdict)}, not criteria, score and reason")
        criteria: Any = verdict["criteria"]
        names = [each.name for each in self.criteria]
        if not isinstance(criteria, dict) or set(cast(dict[str, Any], criteria)) != set(names):
            raise MalformedVerdict(f"criteria is not an object scoring exactly {', '.join(names)}")
        scores = {name: _score(f"criteria.{name}", cast(dict[str, Any], criteria)[name]) for name in names}
        reason = verdict["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise MalformedVerdict("reason is not a sentence")
        return Verdict(scores, _score("score", verdict["score"]), reason.strip())


def _score(key: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not LOWEST <= value <= HIGHEST:
        raise MalformedVerdict(f"{key} is not a whole number from {LOWEST} to {HIGHEST}: {json.dumps(value)}")
    return value


EXPLAIN = Rubric(
    "explain-1",
    "explain a concept to a ten-year-old in under 80 words.",
    (
        Criterion("accuracy", "Is everything it says true, with nothing important wrong or misleading?"),
        Criterion(
            "clarity",
            "Would a ten-year-old follow it: everyday words, short sentences, a familiar comparison where one helps?",
        ),
        Criterion("completeness", "Does it explain how or why, not only name the thing?"),
        Criterion("length", f"Is it under 80 words? An answer of 80 words or more scores {LOWEST} here."),
    ),
)
SUMMARIZE = Rubric(
    "summarize-1",
    "summarize a short passage in at most two sentences and under 50 words.",
    (
        Criterion("faithfulness", "Does it say only what the passage says, with nothing invented or changed?"),
        Criterion("coverage", "Does it keep the main events and how the passage ends?"),
        Criterion(
            "concision",
            f"Is it at most two sentences and under 50 words, with no detail that does not matter? Longer scores "
            f"{LOWEST} here.",
        ),
        Criterion("fluency", "Is it clear, correct English?"),
    ),
)
RUBRICS: Mapping[str, Rubric] = {"explain": EXPLAIN, "summarize": SUMMARIZE}
"""By the kind of request."""
