"""The gridworld as an environment: what there is to train on, as rows (`rollout.environment.Environment`).

A row is a layout, a number of agents and a turn budget; a start of it is a seed, which draws the level and the
agents' names: episodes given the same start begin identically. The rows go from the open room to the vault, each
with two, three and four agents. Its curriculum (`gridworld.curriculum.GridCurriculum`) unlocks them in that order as
the ones before are learned. Its eval data is three starts of every row (`gridworld-eval`), which training never
draws.

Its description says what an episode samples, for estimating a run's spend: every agent samples every turn, so a
turn is as many samples as the episode has agents; an episode plays at most its turn budget (an unsolved one plays it
all); and a sample's prompt (the system prompt, the tools and the latest observation) is about `PROMPT_TOKENS`
tokens. The figures are the mean over the rows (`typical`).
"""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import JsonValue

from gridworld.curriculum import GridCurriculum
from gridworld.episode import GridEpisode
from rollout.environment import Description, Row, Start, drawn
from rollout.harness import ProgramReference, register

__all__ = ["PROMPT_TOKENS", "ROWS", "Gridworld", "environment", "typical"]


def _row(layout: str, agents: int, turns: int, title: str) -> Row:
    return Row(f"{layout}-{agents}", title, {"layout": layout, "agents": agents, "turns": turns})


ROWS = [
    _row("open", 2, 20, "two agents, two plates, one open room"),
    _row("open", 3, 24, "three agents, three plates, one open room"),
    _row("open", 4, 28, "four agents, four plates, one open room"),
    _row("door", 2, 40, "two agents: a pair of plates opens the door to the room with the two final plates"),
    _row("door", 3, 40, "three agents: a pair of plates opens the door to the room with the three final plates"),
    _row("door", 4, 50, "four agents: a pair of plates opens the door to the room with the four final plates"),
    _row(
        "gate",
        2,
        48,
        "two agents: one holds a gate open while the other pulls the lever behind it, which opens the room with "
        "the two final plates",
    ),
    _row(
        "gate",
        3,
        50,
        "three agents: one holds a gate open while another pulls the lever behind it, which opens the room with "
        "the three final plates",
    ),
    _row(
        "gate",
        4,
        60,
        "four agents: one holds a gate open while another pulls the lever behind it, which opens the room with "
        "the four final plates",
    ),
    _row(
        "vault",
        2,
        64,
        "two agents: a pair of plates opens a door, a held gate leads to a lever, the lever opens the room with the "
        "two final plates",
    ),
    _row(
        "vault",
        3,
        72,
        "three agents: a pair of plates opens a door, a held gate leads to a lever, the lever opens the room with the "
        "three final plates",
    ),
    _row(
        "vault",
        4,
        80,
        "four agents: a pair of plates opens a door, a held gate leads to a lever, the lever opens the room with the "
        "four final plates",
    ),
]
"""Easiest first: by layout (each adds a step before the final plates), then by agents (more to share the plates
out among). Each budget leaves room over the most turns the scripted team took on a thousand starts of it, the open
rooms the most (time to talk it over)."""

PROMPT_TOKENS = 1500
"""A sample's prompt tokens, on average: measured on Qwen3.5-9B's chat template, from about 1,050 (two agents in an
open room) to about 1,700 (four in the vault)."""


def typical(rows: Sequence[Row]) -> tuple[float, float]:
    """The turns an episode plays at most and the samples of a turn (its agents), on average over `rows`, weighted so
    that their product is the mean samples of an episode."""
    samples = [int(str(row.parameters["turns"])) * int(str(row.parameters["agents"])) for row in rows]
    agents = sum(int(str(row.parameters["agents"])) for row in rows) / len(rows)
    return round(sum(samples) / len(rows) / agents, 1), agents


_TURNS, _AGENTS = typical(ROWS)


@dataclass(frozen=True)
class Gridworld:
    program: ProgramReference = field(default_factory=lambda: ProgramReference(program=register(GridEpisode)))
    version = "2"
    description = Description(
        rewards=(0.0, 1.0),
        solved=True,
        saturated=True,
        duration="turns",
        turns=_TURNS,
        samples_per_turn=_AGENTS,
        prompt_tokens=PROMPT_TOKENS,
    )

    def rows(self) -> Sequence[Row]:
        return ROWS

    def start(self, row: Row, rng: random.Random) -> JsonValue:
        return {**row.parameters, "seed": rng.randrange(1 << 31)}

    def evals(self) -> Mapping[str, Sequence[Start]]:
        return {"gridworld-eval": drawn(self, seeds=[1, 2, 3])}

    def curriculum(self) -> GridCurriculum:
        return GridCurriculum(ROWS)


environment = Gridworld()
"""`rollout env check gridworld.environment:environment`."""
