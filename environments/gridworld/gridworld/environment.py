"""The gridworld as an environment: what there is to train on, as rows (`rollout.environment.Environment`).

A row is a layout, a number of agents and a turn budget; a start of it is a seed, which draws the level and the
agents' names: episodes given the same start begin identically. Its eval data is three starts of every row
(`gridworld-eval`), which training never draws.
"""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import JsonValue

from gridworld.episode import GridEpisode
from rollout.environment import Description, Row, Start, drawn
from rollout.harness import ProgramReference, register

ROWS = [
    Row("open-2", "two agents, two plates, one open room", {"layout": "open", "agents": 2, "turns": 20}),
    Row("open-3", "three agents, three plates, one open room", {"layout": "open", "agents": 3, "turns": 24}),
    Row(
        "door-2",
        "two agents: a pair of plates opens the door to the room with the two final plates",
        {"layout": "door", "agents": 2, "turns": 40},
    ),
    Row(
        "door-3",
        "three agents: a pair of plates opens the door to the room with the three final plates",
        {"layout": "door", "agents": 3, "turns": 40},
    ),
    Row(
        "gate-3",
        "three agents: one holds a gate open while another pulls the lever behind it, which opens the room with "
        "the three final plates",
        {"layout": "gate", "agents": 3, "turns": 50},
    ),
    Row(
        "vault-4",
        "four agents: a pair of plates opens a door, a held gate leads to a lever, the lever opens the room with the "
        "four final plates",
        {"layout": "vault", "agents": 4, "turns": 80},
    ),
]
"""Easiest first: more agents to share out among the plates, and more steps before the final plates."""


@dataclass(frozen=True)
class Gridworld:
    program: ProgramReference = field(default_factory=lambda: ProgramReference(program=register(GridEpisode)))
    version = "2"
    description = Description(rewards=(0.0, 1.0), solved=True, saturated=True, duration="turns")

    def rows(self) -> Sequence[Row]:
        return ROWS

    def start(self, row: Row, rng: random.Random) -> JsonValue:
        return {**row.parameters, "seed": rng.randrange(1 << 31)}

    def evals(self) -> Mapping[str, Sequence[Start]]:
        return {"gridworld-eval": drawn(self, seeds=[1, 2, 3])}


environment = Gridworld()
"""`rollout env check gridworld.environment:environment`."""
