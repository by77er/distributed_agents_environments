"""The gridworld's curriculum: rows unlock as the ones before them are learned, and rows whose groups come out mixed
are trained on most.

It is the platform's curriculum (`rollout.curriculum.Curriculum`), folded from the run's group results like any other
(so a run started again rebuilds it from its ledger), deciding two things its own way:

- **What is unlocked.** The rows are in order of difficulty. The first `start` (the open rooms) are unlocked from the
  beginning; a row is learned once at least `least` groups of it are recorded and the moving average of its success
  reaches `success_gate`, or that of its progress (`gridworld.scoring.progress_of`) reaches `progress_gate`; the rows
  up to `reach` past the hardest row learned are unlocked.
- **How often each unlocked row is drawn** (`weight`): a row whose groups' rewards differ teaches something, and one
  solved about half the time teaches most. Its weight is the moving average of whether its groups' rewards differed,
  scaled from half (never solved, or always) to whole (solved half the time), plus `floor`. A settled row (every
  episode solved, or none making any progress) keeps the floor: it is still drawn now and then, so that forgetting it
  shows. A row not yet tried has full weight.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from gridworld.scoring import progress_of
from rollout.curriculum import Curriculum
from rollout.environment import Row

__all__ = ["GridCurriculum"]


@dataclass
class GridCurriculum(Curriculum):
    start: int = 3
    reach: int = 2
    floor: float = 0.1
    least: int = 2
    """Groups of a row recorded before it can count as learned."""
    success_gate: float = 0.5
    progress_gate: float = 0.7
    progress: dict[str, float] = field(default_factory=dict[str, float])
    """The moving average of each row's mean progress, by its key."""

    def learned(self, row: Row) -> bool:
        record = self.record(row)
        if record.attempts < self.least:
            return False
        return record.success >= self.success_gate or self.progress.get(row.key, 0.0) >= self.progress_gate

    def unlocked(self) -> list[Row]:
        learned = [index for index, row in enumerate(self.rows) if self.learned(row)]
        count = max(self.start, learned[-1] + 1 + self.reach) if learned else self.start
        return list(self.rows[:count])

    def weight(self, row: Row) -> float:
        record = self.record(row)
        if record.attempts == 0:
            return 1.0
        balance = 4 * record.success * (1 - record.success)  # (1 at half solved, 0 at none or all)
        return self.floor + record.signal * (0.5 + 0.5 * balance)

    def update(self, row: Row, rewards: Sequence[float], solved: Sequence[bool]) -> None:
        first = self.record(row).attempts == 0
        super().update(row, rewards, solved)
        mean = sum(map(progress_of, rewards, solved)) / len(rewards)
        before = self.progress.get(row.key, 0.0)
        self.progress[row.key] = mean if first else (1 - self.smoothing) * before + self.smoothing * mean
