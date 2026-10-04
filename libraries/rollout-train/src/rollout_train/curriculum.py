"""Which row to train on next: sampled by how much its groups have to teach, over the rows unlocked so far.

A group-relative update learns from the differences between a group's episodes: a group whose episodes all scored
the same (every one perfect, or every one nothing) teaches nothing. So a row's weight is the share of its recent
groups whose rewards differed (a moving average), plus a little for every unlocked row so that none is forgotten;
untried rows get full weight. Whether a row was solved decides only what unlocks: rows unlock in the catalog's
order, the first `start` of them, and `reach` past the hardest one solved at least half the time. A group counts for
its own row and for every row that row `counts_for`.

The evals a run makes of its checkpoints (`rollout_train.evals.Schedule`) are folded in too: the newest of each suite is
kept, for what reads them.
"""

import random
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field

from rollout.catalog import Row
from rollout_train.record import Result


@dataclass
class Record:
    attempts: int = 0
    success: float = 0.0
    """Moving average of the share of episodes that solved the row."""
    reward: float = 0.0
    """Moving average of the mean reward."""
    signal: float = 0.0
    """Moving average of whether a group's rewards differed (1) or were all the same (0)."""
    failures: int = 0
    """Groups in a row none of whose episodes completed."""


FAILED_GROUPS = 3
"""Groups of a row that may fail outright (it could not be set up from the start that was drawn) before the row
counts as tried: the next start may be one it can be set up from."""


@dataclass
class Curriculum:
    rows: Sequence[Row]
    rng: random.Random = field(default_factory=random.Random)
    start: int = 3
    """This many rows are unlocked from the beginning."""
    reach: int = 4
    """Rows unlocked past the hardest one solved."""
    smoothing: float = 0.5
    """Weight of the newest group in the moving averages."""
    floor: float = 0.05
    records: dict[str, Record] = field(default_factory=dict[str, Record])
    evaluations: dict[str, tuple[str, list[Result]]] = field(default_factory=dict[str, tuple[str, list[Result]]])
    """The newest eval of each suite, by the suite's name: the checkpoint that played it, and how it did at each start
    (a `Result` each)."""

    def unlocked(self) -> list[Row]:
        solved = [index for index, row in enumerate(self.rows) if self.record(row).success >= 0.5]
        count = max(self.start, solved[-1] + 1 + self.reach) if solved else self.start
        return list(self.rows[:count])

    def sample(self, pending: Collection[str] = (), rng: random.Random | None = None) -> Row:
        """The next row. `pending` names rows whose latest group has not been recorded yet: choosing one again would
        be choosing on what was known before it, so the others come first."""
        unlocked = self.unlocked()
        candidates = [row for row in unlocked if row.key not in pending] or unlocked
        weights = [self.weight(row) for row in candidates]
        return (rng or self.rng).choices(candidates, weights=weights, k=1)[0]

    def recorded(self, line: Result) -> None:
        """Take a group's result into account: the row of its title, or failing that of its key (a key that is a
        place in a catalog changes when rows are added). A curriculum is the fold of a run's results."""
        by_title = {row.title: row for row in self.rows}
        by_key = {row.key: row for row in self.rows}
        row = by_title.get(line.title) or by_key.get(line.task)
        if row is None:
            return  # a row the catalog no longer has
        for each in [row, *(by_key[key] for key in row.counts_for if key in by_key)]:
            if line.rewards:
                self.update(each, line.rewards, line.solved)
            else:
                self.failed(each)

    def evaluated(self, suite: str, checkpoint: str, results: Sequence[Result]) -> None:
        """Take an eval into account: `checkpoint` played `suite`, and `results` say how it did at each start. A
        curriculum folds a run's evals in the order they were made, so the newest of each suite is kept."""
        self.evaluations[suite] = (checkpoint, list(results))

    def weight(self, row: Row) -> float:
        record = self.record(row)
        if record.attempts == 0:
            return 1.0
        return record.signal + self.floor

    def update(self, row: Row, rewards: Sequence[float], solved: Sequence[bool]) -> None:
        """Record a group of episodes of `row`: each one's reward and whether it solved the row."""
        record = self.record(row)
        record.failures = 0
        success = sum(solved) / len(solved)
        mean = sum(rewards) / len(rewards)
        signal = 1.0 if max(rewards) > min(rewards) else 0.0
        first = record.attempts == 0
        record.success = success if first else (1 - self.smoothing) * record.success + self.smoothing * success
        record.reward = mean if first else (1 - self.smoothing) * record.reward + self.smoothing * mean
        record.signal = signal if first else (1 - self.smoothing) * record.signal + self.smoothing * signal
        record.attempts += 1

    def failed(self, row: Row) -> None:
        """Record a group of `row` none of whose episodes completed. After `FAILED_GROUPS` of them in a row it is no
        longer untried, and it taught nothing."""
        record = self.record(row)
        record.failures += 1
        if record.failures >= FAILED_GROUPS:
            record.signal = 0.0 if record.attempts == 0 else (1 - self.smoothing) * record.signal
            record.attempts += 1
            record.failures = 0

    def record(self, row: Row) -> Record:
        return self.records.setdefault(row.key, Record())
