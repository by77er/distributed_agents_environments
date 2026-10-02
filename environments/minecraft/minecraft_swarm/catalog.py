"""The catalog of the Minecraft swarm: what there is to train on, as rows (`rollout.rollouts.Catalog`).

A row is a task; a start of it is a world and a layout drawn at random: episodes given the same start begin
identically.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import JsonValue

from minecraft_swarm.episode import SwarmEpisode
from minecraft_swarm.tasks import catalog as tasks
from rollout.core.harness import ProgramReference, register
from rollout.rollouts import Row


@dataclass(frozen=True)
class Swarm:
    worlds: int = 12
    """How many world seeds starts are drawn from (a server template is generated for each, once)."""
    seed: int = 0
    only: tuple[str, ...] = ()
    """Restrict the catalog to these task ids (all tasks when empty)."""
    program: ProgramReference = field(default_factory=lambda: ProgramReference(program=register(SwarmEpisode)))

    def rows(self) -> Sequence[Row]:
        return [
            Row(task.id, task.title, {"task": task.id}) for task in tasks() if not self.only or task.id in self.only
        ]

    def start(self, row: Row, rng: random.Random) -> JsonValue:
        seeds = [random.Random(f"world-{self.seed}-{index}").randrange(1 << 31) for index in range(self.worlds)]
        return {**row.parameters, "world_seed": rng.choice(seeds), "layout_seed": rng.randrange(1 << 30)}


catalog = Swarm()
"""`rollout train PROFILE minecraft_swarm.catalog:catalog`."""
