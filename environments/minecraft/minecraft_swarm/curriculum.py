"""Which task to train on next: sampled by learning progress over the tasks unlocked so far.

A task's success rate is an exponential moving average of how often its episodes solved it (the team held a diamond,
or earned the milestone the task is about). Tasks the swarm solves about half the time carry the most signal for
group-relative updates (a group that always or never succeeds teaches nothing), so a task's weight is p(1 - p), plus
a little for every unlocked task so none is forgotten; untried tasks get full weight. Tasks unlock in the catalog's
order as easier ones are solved: the first `start` tasks, and `reach` tasks past the hardest one solved at least half
the time.
"""

import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from minecraft_swarm.tasks import Task


@dataclass
class TaskRecord:
    attempts: int = 0
    success: float = 0.0
    """Moving average of the share of episodes that solved the task."""
    reward: float = 0.0
    """Moving average of the mean reward."""


@dataclass
class Curriculum:
    tasks: Sequence[Task]
    rng: random.Random = field(default_factory=random.Random)
    start: int = 3
    """This many tasks are unlocked from the beginning."""
    reach: int = 4
    """Tasks unlocked past the hardest one solved."""
    smoothing: float = 0.5
    """Weight of the newest group in the moving averages."""
    floor: float = 0.05
    records: dict[str, TaskRecord] = field(default_factory=dict[str, TaskRecord])

    def unlocked(self) -> list[Task]:
        solved = [index for index, task in enumerate(self.tasks) if self.record(task).success >= 0.5]
        count = max(self.start, solved[-1] + 1 + self.reach) if solved else self.start
        return list(self.tasks[:count])

    def sample(self) -> Task:
        candidates = self.unlocked()
        weights = [self.weight(task) for task in candidates]
        return self.rng.choices(candidates, weights=weights, k=1)[0]

    def weight(self, task: Task) -> float:
        record = self.record(task)
        if record.attempts == 0:
            return 1.0
        return 4 * record.success * (1 - record.success) + self.floor

    def update(self, task: Task, rewards: Sequence[float], solved: Sequence[bool]) -> None:
        """Record a group of episodes of `task`: each one's reward and whether it solved the task."""
        record = self.record(task)
        success = sum(solved) / len(solved)
        mean = sum(rewards) / len(rewards)
        first = record.attempts == 0
        record.success = success if first else (1 - self.smoothing) * record.success + self.smoothing * success
        record.reward = mean if first else (1 - self.smoothing) * record.reward + self.smoothing * mean
        record.attempts += 1

    def record(self, task: Task) -> TaskRecord:
        return self.records.setdefault(task.id, TaskRecord())

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({task_id: asdict(record) for task_id, record in self.records.items()}, indent=1))

    def load(self, path: Path) -> None:
        for task_id, record in json.loads(path.read_text()).items():
            self.records[task_id] = TaskRecord(**record)
