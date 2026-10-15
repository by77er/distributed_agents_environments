"""The Minecraft team as an environment: what there is to train on, as rows (`rollout.environment.Environment`).

A row is a task; a start of it is a world, a layout and the names of a team of one to four, drawn at random: episodes
given the same start begin identically. Its eval data is one start of every task (`teams-every-task`), which training
never draws, each drawn with a seed of its row's own (`eval_seed`): the tasks are evaluated across worlds, layouts and
teams, not all in one.
"""

import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import JsonValue

from minecraft_team.episode import TeamEpisode
from minecraft_team.tasks import NAMES, TASKS, TEAM, Coordination
from rollout.environment import Description, Row, Start, drawn
from rollout.harness import ProgramReference, register


def eval_seed(row: Row) -> int:
    """The seed a row's eval start is drawn with: its own, from its key, so that rows differ and a row added later
    changes no other's."""
    return int.from_bytes(hashlib.sha256(f"eval-{row.key}".encode()).digest()[:4]) >> 1


@dataclass(frozen=True)
class Teams:
    worlds: int = 12
    """How many world seeds starts are drawn from (a server template is generated for each, once)."""
    seed: int = 0
    only: tuple[str, ...] = ()
    """Restrict the rows to these task ids (all tasks when empty)."""
    program: ProgramReference = field(default_factory=lambda: ProgramReference(program=register(TeamEpisode)))
    version = "3"
    description = Description(rewards=(0.0, 1.0), saturated=True, duration="turns", observations="minecraft")

    def rows(self) -> Sequence[Row]:
        return [
            Row(task.id, task.title, {"task": task.id}, counts_for=() if task.guided else (task.id.removesuffix("u"),))
            for task in TASKS.values()
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

    def evals(self) -> Mapping[str, Sequence[Start]]:
        rows = self.rows()
        return {
            "teams-every-task": [start for row in rows for start in drawn(self, seeds=[eval_seed(row)], rows=[row.key])]
        }


environment = Teams()
"""`rollout train minecraft_team.environment:environment --preset minecraft-one-gpu`."""
