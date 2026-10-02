"""What a training run writes down for each group: one line of `metrics.jsonl`.

The loop writes these and sends them to the job as `iteration` notes; the report and the monitor read them.
"""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Protocol

from pydantic import JsonValue


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
    """From the group's submission to its last episode's end."""
    seconds: float = 0.0
    """From the group's submission to this line."""
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
    """The weights the step produced."""
    version: int | None = None
    """The channel's version once they were published."""
    unlocked: int = 0
    """Rows of the catalog unlocked after this group."""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Iteration":
        known = {each.name for each in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


class Store(Protocol):
    """Where a training run keeps its small state: a directory, or anything else that holds named texts."""

    def read(self, name: str) -> str | None: ...
    def write(self, name: str, text: str) -> None: ...
    def append(self, name: str, line: str) -> None: ...


@dataclass(frozen=True)
class Directory:
    """A `Store` in a directory."""

    path: Path

    def read(self, name: str) -> str | None:
        file = self.path / name
        return file.read_text() if file.exists() else None

    def write(self, name: str, text: str) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / name).write_text(text)

    def append(self, name: str, line: str) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        with (self.path / name).open("a") as file:
            file.write(line + "\n")


METRICS = "metrics.jsonl"


def iterations(store: Store) -> list[Iteration]:
    """The groups a run has logged, in the order they were logged."""
    return [Iteration.from_json(json.loads(line)) for line in (store.read(METRICS) or "").splitlines() if line.strip()]
