"""What a training run writes down: three tables in a ledger, by the group's number.

- `groups`: what the run decided to play (the row, and the start every episode of the group is given). Written
  before the group is asked for.
- `steps`: what the run decided to train on (the version it starts from, the one it will make, the batch). Written
  before the trainer is called.
- `iterations`: how the group went and what was done with it (an `Iteration`). Written last: a group with an
  iteration is done with.

A run that is started again reads them and goes on: whatever has a decision and no outcome is taken up where it
was left. The monitor and the report read `iterations`.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from pydantic import JsonValue

from rollout_train.ledger import Ledger, between


@dataclass
class Iteration:
    iteration: int
    """The group's number in the run, from 1."""
    time: float
    """When the line was written, in seconds since the epoch."""
    task: str
    """The row's key."""
    title: str = ""
    rollout_seconds: float = 0.0
    """From the group's decision to its last episode's end."""
    seconds: float = 0.0
    """From the group's decision to this line."""
    rewards: list[float] = field(default_factory=list[float])
    """Of the episodes fit to train on, as are `solved` and `durations`."""
    solved: list[bool] = field(default_factory=list[bool])
    durations: list[float | None] = field(default_factory=list[float | None])
    failed: int = 0
    """Episodes that did not complete, or asked to be left out."""
    failures: list[str] = field(default_factory=list[str])
    notes: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """What the algorithm said of the group."""
    sequences_recorded: int = 0
    sequences_trained: int = 0
    update: Mapping[str, float] | None = None
    """The trainer's statistics, if the group was trained on."""
    skipped: str | None = None
    """Why the group was not trained on, if the algorithm found nothing to train on."""
    error: str | None = None
    """What went wrong, if the trainer's step failed."""
    adapter: str | None = None
    """The policy version the step made, by name: its record has the weights, the batch and the trainer's state."""
    version: int | None = None
    """That version's number."""
    unlocked: int = 0
    """Rows of the catalog unlocked after this group."""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Iteration":
        known = {each.name for each in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


def scope(run: str) -> str:
    """The scope whose fence a run's loop holds."""
    return _RUNS + run


def table(run: str, name: str) -> str:
    """A run's table in the ledger."""
    return f"{scope(run)}/{name}"


GROUPS, STEPS, ITERATIONS = "groups", "steps", "iterations"
_RUNS = "runs/"


async def runs_in(ledger: Ledger) -> list[str]:
    """The runs a ledger has groups of, by name."""
    return [run for each in await ledger.tables() if (run := between(each, _RUNS, f"/{GROUPS}"))]


async def iterations(ledger: Ledger, run: str = "train") -> list[Iteration]:
    """The groups a run is done with, by their numbers."""
    logged = await ledger.read(table(run, ITERATIONS))
    lines = [Iteration.from_json(record) for record in logged.values() if isinstance(record, dict)]
    return sorted(lines, key=lambda line: line.iteration)
