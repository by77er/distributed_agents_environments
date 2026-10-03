"""The catalog of the Minecraft swarm: what there is to train on, as rows (`rollout_train.rollouts.Catalog`).

A row is a task; a start of it is a world, a layout and the names of a team of one to four, drawn at random: episodes
given the same start begin identically.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import JsonValue

from minecraft_swarm.episode import SwarmEpisode
from minecraft_swarm.tasks import NAMES, TEAM, Coordination
from minecraft_swarm.tasks import catalog as tasks
from rollout.catalog import Row
from rollout.harness import ProgramReference, register

TASKS = {task.id: task for task in tasks()}


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
            Row(task.id, task.title, {"task": task.id}, counts_for=() if task.guided else (task.id.removesuffix("u"),))
            for task in tasks()
            if not self.only or task.id in self.only
        ]

    def start(self, row: Row, rng: random.Random) -> JsonValue:
        """A world seed of the `worlds`, a layout, and who plays: a name from `NAMES` for each of one to four players
        (at least two where the kit is dealt in parts: no one can finish alone), all different."""
        seeds = [random.Random(f"world-{self.seed}-{index}").randrange(1 << 31) for index in range(self.worlds)]
        fewest = 2 if TASKS[str(row.parameters["task"])].coordination is Coordination.SPLIT else 1
        world, layout = rng.choice(seeds), rng.randrange(1 << 30)
        players = rng.randint(fewest, len(TEAM))
        return {**row.parameters, "world_seed": world, "layout_seed": layout, "names": rng.sample(NAMES, players)}


catalog = Swarm()
"""`rollout train PROFILE minecraft_swarm.catalog:catalog`."""
