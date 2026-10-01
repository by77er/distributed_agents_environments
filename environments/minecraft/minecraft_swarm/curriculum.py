"""Which task to train on next: sampled by learning progress over the tasks unlocked so far.

A task's success rate is an exponential moving average of how often its episodes ended with any diamonds. Tasks the
swarm solves about half the time carry the most signal for group-relative updates (a group that always or never
succeeds teaches nothing), so a task's weight is p(1 - p), plus a little for every unlocked task so none is
forgotten; untried tasks get full weight. Tasks unlock as easier ones are solved: everything up to `reach` above the
hardest task solved at least half the time.
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
    """Moving average of the share of episodes that ended with diamonds."""
    reward: float = 0.0
    """Moving average of the mean reward."""


@dataclass
class Curriculum:
    tasks: Sequence[Task]
    rng: random.Random = field(default_factory=random.Random)
    start: float = 1.0
    """Tasks up to this difficulty are unlocked from the beginning."""
    reach: float = 2.0
    smoothing: float = 0.5
    """Weight of the newest group in the moving averages."""
    floor: float = 0.05
    records: dict[str, TaskRecord] = field(default_factory=dict[str, TaskRecord])

    def unlocked(self) -> list[Task]:
        solved = [task.difficulty for task in self.tasks if self.record(task).success >= 0.5]
        limit = max(self.start, max(solved) + self.reach) if solved else self.start
        return [task for task in self.tasks if task.difficulty <= limit]

    def sample(self) -> Task:
        candidates = self.unlocked()
        weights = [self.weight(task) for task in candidates]
        return self.rng.choices(candidates, weights=weights, k=1)[0]

    def weight(self, task: Task) -> float:
        record = self.record(task)
        if record.attempts == 0:
            return 1.0
        return 4 * record.success * (1 - record.success) + self.floor

    def update(self, task: Task, rewards: Sequence[float]) -> None:
        record = self.record(task)
        success = sum(reward > 0 for reward in rewards) / len(rewards)
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
