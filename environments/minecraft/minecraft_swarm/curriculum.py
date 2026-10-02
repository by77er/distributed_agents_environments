"""Which task to train on next: sampled by how much its groups have to teach, over the tasks unlocked so far.

A group-relative update learns from the differences between a group's episodes: a group whose episodes all scored
the same (every one perfect, or every one nothing) teaches nothing. So a task's weight is the share of its recent
groups whose rewards differed (a moving average), plus a little for every unlocked task so that none is forgotten;
untried tasks get full weight. Whether a task was solved decides only what unlocks: tasks unlock in the catalog's
order, the first `start` tasks, and `reach` tasks past the hardest one solved at least half the time.
"""

import json
import random
from collections.abc import Collection, Sequence
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
    signal: float = 0.0
    """Moving average of whether a group's rewards differed (1) or were all the same (0)."""


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

    def sample(self, pending: Collection[str] = ()) -> Task:
        """The next task. `pending` names tasks whose latest group has not been recorded yet: choosing one again
        would be choosing on what was known before it, so the others come first."""
        unlocked = self.unlocked()
        candidates = [task for task in unlocked if task.id not in pending] or unlocked
        weights = [self.weight(task) for task in candidates]
        return self.rng.choices(candidates, weights=weights, k=1)[0]

    def weight(self, task: Task) -> float:
        record = self.record(task)
        if record.attempts == 0:
            return 1.0
        return record.signal + self.floor

    def update(self, task: Task, rewards: Sequence[float], solved: Sequence[bool]) -> None:
        """Record a group of episodes of `task`: each one's reward and whether it solved the task."""
        record = self.record(task)
        success = sum(solved) / len(solved)
        mean = sum(rewards) / len(rewards)
        signal = 1.0 if max(rewards) > min(rewards) else 0.0
        first = record.attempts == 0
        record.success = success if first else (1 - self.smoothing) * record.success + self.smoothing * success
        record.reward = mean if first else (1 - self.smoothing) * record.reward + self.smoothing * mean
        record.signal = signal if first else (1 - self.smoothing) * record.signal + self.smoothing * signal
        record.attempts += 1

    def failed(self, task: Task) -> None:
        """Record a group of `task` none of whose episodes completed (the task could not be built, say): it is no
        longer untried, and it taught nothing."""
        record = self.record(task)
        record.signal = 0.0 if record.attempts == 0 else (1 - self.smoothing) * record.signal
        record.attempts += 1

    def record(self, task: Task) -> TaskRecord:
        return self.records.setdefault(task.id, TaskRecord())

    def save(self, path: Path) -> None:
        titles = {task.id: task.title for task in self.tasks}
        saved = {
            task_id: {**asdict(record), "title": titles.get(task_id, "")} for task_id, record in self.records.items()
        }
        path.write_text(json.dumps(saved, indent=1))

    def load(self, path: Path) -> None:
        """Records go to the task of the same title: an id is a place in the catalog, which changes when tasks are
        added."""
        by_title = {task.title: task.id for task in self.tasks}
        for task_id, saved in json.loads(path.read_text()).items():
            title = saved.pop("title", None)
            if title is not None and title not in by_title:
                continue  # a task the catalog no longer has
            self.records[by_title[title] if title is not None else task_id] = TaskRecord(**saved)
