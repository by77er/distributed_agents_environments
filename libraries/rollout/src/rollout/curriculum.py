"""Which row to train on next: sampled by how much its groups have to teach, over the rows unlocked so far.

A group-relative update learns from the differences between a group's episodes: a group whose episodes all scored
the same (every one perfect, or every one nothing) teaches nothing. So a row's weight is the share of its recent
groups whose rewards differed (a moving average), plus a little for every unlocked row so that none is forgotten;
untried rows get full weight. Whether a row was solved decides only what unlocks: rows unlock in the environment's
order, the first `start` of them, and `reach` past the hardest one solved at least half the time. A group counts for
its own row and for every row that row `counts_for`.

The evals a run makes of its checkpoints (`rollout_train.evals.Schedule`) are folded in too (`evaluated`): the loop
calls it with each entry's results as the eval ends (an entry is an environment of the suite, named by `module:name`),
and again with every eval, in the order they ended, when the run is started again (a curriculum is the fold of a run's
results, its evals' among them). The generic curriculum keeps the newest eval of each suite's entry (`evaluations`) and
decides nothing from it.

An environment may supply a curriculum of its own (`environment.curriculum()`, `curriculum_of`): a `Curriculum` that
decides differently, by overriding `unlocked`, `sample` or `weight`. One that gates on evals overrides `evaluated`:

    @dataclass
    class Gated(Curriculum):
        passed: bool = False

        def evaluated(self, suite, checkpoint, results, entry=None):
            super().evaluated(suite, checkpoint, results, entry)
            self.passed = self.passed or (suite == "held-out" and solved_share(results) >= 0.6)

        def unlocked(self):
            return super().unlocked() if self.passed else super().unlocked()[:10]
"""

import random
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from rollout.environment import Environment, Row

__all__ = ["Curriculum", "GroupResult", "curriculum_of", "solved_share"]


class GroupResult(Protocol):
    """A group's result, as a curriculum reads it (`rollout_train.record.Result` is one)."""

    @property
    def task(self) -> str:
        """The row's key."""
        ...

    @property
    def title(self) -> str: ...

    @property
    def rewards(self) -> Sequence[float]:
        """Of the episodes that completed, as is `solved`."""
        ...

    @property
    def solved(self) -> Sequence[bool]: ...


def solved_share(results: Sequence[GroupResult]) -> float:
    """The share of the episodes of `results` that solved their row (0 when there are none)."""
    solved = [each for result in results for each in result.solved]
    return sum(solved) / len(solved) if solved else 0.0


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
    evaluations: dict[tuple[str, str | None], tuple[str | None, list[GroupResult]]] = field(
        default_factory=dict[tuple[str, str | None], tuple[str | None, list[GroupResult]]]
    )
    """The newest eval of each suite's entry, by the suite's name and the entry's environment: the checkpoint that
    played it (None: the base model), and how it did at each start of that entry."""

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

    def recorded(self, line: GroupResult) -> None:
        """Take a group's result into account: the row of its title, or failing that of its key (a key that is a
        place in an environment changes when rows are added). A curriculum is the fold of a run's results."""
        by_title = {row.title: row for row in self.rows}
        by_key = {row.key: row for row in self.rows}
        row = by_title.get(line.title) or by_key.get(line.task)
        if row is None:
            return  # a row the environment no longer has
        for each in [row, *(by_key[key] for key in row.counts_for if key in by_key)]:
            if line.rewards:
                self.update(each, line.rewards, line.solved)
            else:
                self.failed(each)

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

    def evaluated(
        self, suite: str, checkpoint: str | None, results: Sequence[GroupResult], entry: str | None = None
    ) -> None:
        """Take an eval's entry into account: `suite` played by `checkpoint` (None: the base model), one result per
        start of its entry of the environment `entry` (`module:name`; None where the eval does not say). The generic
        curriculum keeps the newest of each suite's entry and decides nothing from it; one that gates on evals
        overrides this."""
        self.evaluations[(suite, entry)] = (checkpoint, list(results))

    def record(self, row: Row) -> Record:
        return self.records.setdefault(row.key, Record())


def curriculum_of(environment: Environment) -> Curriculum:
    """A curriculum that has recorded nothing: the environment's own (`environment.curriculum()`, if it has one), else
    the generic one over its rows."""
    own = getattr(environment, "curriculum", None)
    if not callable(own):
        return Curriculum(environment.rows())
    made = own()
    if not isinstance(made, Curriculum):
        raise TypeError(f"{type(environment).__name__}.curriculum() made a {type(made).__name__}, not a Curriculum")
    return made
