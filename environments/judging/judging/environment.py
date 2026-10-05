"""Judging as an environment: what there is to train on, as rows (`rollout.environment.Environment`).

Two rows, one for each kind of request: `explain` (a concept to a ten-year-old, in under 80 words) and `summarize` (a
short passage, in at most two sentences and under 50 words). A start of a row is one request, drawn with its seed from
the row's training list (`judging.data`). The eval data, `judging-eval`, is every request of the eval lists, which
training never draws: concepts and passages the policy is never trained on.

`environment` asks the judge once per answer; `twice` asks it twice and averages the scores.
"""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import JsonValue

from judging.data import EXPLAIN_EVAL, EXPLAIN_TRAIN, SUMMARIZE_EVAL, SUMMARIZE_TRAIN
from judging.episode import JudgedAnswer
from rollout.environment import Description, Row, Start
from rollout.harness import ProgramReference, register

__all__ = ["ROWS", "Judging", "environment", "request", "twice"]

ROWS = [
    Row("explain", "explain a concept to a ten-year-old in under 80 words", {"kind": "explain"}),
    Row("summarize", "summarize a short passage in at most two sentences and under 50 words", {"kind": "summarize"}),
]


def request(kind: str, item: str | tuple[str, str]) -> dict[str, JsonValue]:
    """A request's parameters: what it is about (a concept, or a passage's title) and what the policy is asked."""
    if kind == "explain":
        assert isinstance(item, str)
        return {"kind": kind, "about": item, "request": f"Explain {item} to a ten-year-old in under 80 words."}
    title, passage = item
    asked = f"Summarize this passage in at most two sentences and under 50 words.\n\n{passage}"
    return {"kind": kind, "about": title, "request": asked}


TRAIN: Mapping[str, Sequence[str | tuple[str, str]]] = {"explain": EXPLAIN_TRAIN, "summarize": SUMMARIZE_TRAIN}
EVAL: Mapping[str, Sequence[str | tuple[str, str]]] = {"explain": EXPLAIN_EVAL, "summarize": SUMMARIZE_EVAL}


@dataclass(frozen=True)
class Judging:
    judgements: int = 1
    """How many times the judge is asked about each answer (1 or 2)."""
    program: ProgramReference = field(default_factory=lambda: ProgramReference(program=register(JudgedAnswer)))
    version = "1"
    description = Description(rewards=(0.0, 1.0), solved=False)

    def rows(self) -> Sequence[Row]:
        return ROWS

    def start(self, row: Row, rng: random.Random) -> JsonValue:
        kind = str(row.parameters["kind"])
        return {**request(kind, rng.choice(TRAIN[kind])), "judgements": self.judgements}

    def evals(self) -> Mapping[str, Sequence[Start]]:
        starts = [
            Start(row.key, row.title, seed, {**request(row.key, item), "judgements": self.judgements})
            for row in ROWS
            for seed, item in enumerate(EVAL[row.key])
        ]
        return {"judging-eval": starts}


environment = Judging()
"""`rollout env check judging.environment:environment`."""
twice = Judging(judgements=2)
"""The judge asked twice about each answer, and the scores averaged."""
