"""Minecraft horizons as an environment: what there is to train on, as rows (`rollout.environment.Environment`).

A row is a task: an objective from a setting against a budget of game time (`minecraft_horizons.tasks`). A start of
it is a world, a layout and the names of a team of one to four, drawn at random: episodes given the same start begin
identically, so a group compares how teams did in the same world with the same time. Training draws worlds from
`worlds` seeds; the eval data (`horizons-held-out`, one start of every task) is played in worlds training never sees.

Rows come shortest budgets first, so the curriculum meets the short games before the long ones. A row counts as
solved when the team ends with at least one of its objective (`solved` in the result): what unlocks rows; how much
more it got is the reward's business.
"""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import JsonValue

from minecraft_horizons.episode import HorizonEpisode
from minecraft_horizons.tasks import TASKS
from minecraft_team.tasks import NAMES, TEAM
from rollout.environment import Description, Row, Start
from rollout.harness import ProgramReference, register

__all__ = ["Horizons", "environment"]

HELD_OUT_WORLDS = 4
"""World seeds the eval data is played in, apart from training's."""


@dataclass(frozen=True)
class Horizons:
    worlds: int = 12
    """How many world seeds training starts are drawn from (a server template is generated for each, once)."""
    seed: int = 0
    only: tuple[str, ...] = ()
    """Restrict the rows to these task ids (all tasks when empty)."""
    program: ProgramReference = field(default_factory=lambda: ProgramReference(program=register(HorizonEpisode)))
    version = "1"
    description = Description(
        rewards=(0.0, None),
        saturated=False,
        duration="turns",
        observations="minecraft",
        turns=sum(task.turns for task in TASKS.values()) / len(TASKS),
        samples_per_turn=(1 + len(TEAM)) / 2,
    )

    def rows(self) -> Sequence[Row]:
        return [
            Row(task.id, task.title, {"task": task.id})
            for task in TASKS.values()
            if not self.only or task.id in self.only
        ]

    def start(self, row: Row, rng: random.Random) -> JsonValue:
        """A world seed of the `worlds`, a layout, and who plays: a name from `NAMES` for each of one to four
        players, all different."""
        return self._start(row, rng, self._seeds("world", self.worlds))

    def evals(self) -> Mapping[str, Sequence[Start]]:
        held_out = self._seeds("held-out-world", HELD_OUT_WORLDS)
        return {
            "horizons-held-out": [
                Start(row.key, row.title, 1, self._start(row, random.Random(1), held_out)) for row in self.rows()
            ]
        }

    def _seeds(self, kind: str, count: int) -> list[int]:
        return [random.Random(f"{kind}-{self.seed}-{index}").randrange(1 << 31) for index in range(count)]

    @staticmethod
    def _start(row: Row, rng: random.Random, seeds: Sequence[int]) -> JsonValue:
        world, layout = rng.choice(seeds), rng.randrange(1 << 30)
        players = rng.randint(1, len(TEAM))
        return {**row.parameters, "world_seed": world, "layout_seed": layout, "names": rng.sample(NAMES, players)}


environment = Horizons()
"""`rollout train minecraft_horizons.environment:environment --preset minecraft-one-gpu`."""
