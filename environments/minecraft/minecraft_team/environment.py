"""The Minecraft team as an environment: what there is to train on, as rows (`rollout.environment.Environment`).

A row is a task; a start of it is a world, a layout and the names of a team of one to four, drawn at random: episodes
given the same start begin identically. Its eval data is one start of every task (`teams-every-task`), which training
never draws.
"""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import JsonValue

from minecraft_team.episode import TeamEpisode
from minecraft_team.tasks import NAMES, TEAM, Coordination
from minecraft_team.tasks import catalog as tasks
from rollout.environment import Description, Row, Start, drawn
from rollout.harness import ProgramReference, register

TASKS = {task.id: task for task in tasks()}


@dataclass(frozen=True)
class Teams:
    worlds: int = 12
    """How many world seeds starts are drawn from (a server template is generated for each, once)."""
    seed: int = 0
    only: tuple[str, ...] = ()
    """Restrict the rows to these task ids (all tasks when empty)."""
    program: ProgramReference = field(default_factory=lambda: ProgramReference(program=register(TeamEpisode)))
    version = "1"
    description = Description(
        rewards=(0.0, None), saturated=True, duration="minutes of game time", observations="minecraft"
    )  # (a team can hold any number of diamonds)

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

    def evals(self) -> Mapping[str, Sequence[Start]]:
        return {"teams-every-task": drawn(self, seeds=[1])}


environment = Teams()
"""`rollout train PROFILE minecraft_team.environment:environment`."""
