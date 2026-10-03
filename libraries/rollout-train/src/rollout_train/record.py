"""What a training run writes down: four tables in a ledger.

- `groups`: what the run decided to play (the row, and the start every episode of the group is given), by the
  group's number. Written before the group is asked for.
- `results`: how each group went (a `Result`), by the group's number. Written when its last episode ends, before
  anything is trained on it.
- `steps`: what the run decided to train on, by the step's number: the groups it covers, the version it starts
  from, the one it will make, the batch. Written before the trainer is called. The version it makes, in the
  policy's table, is its outcome.
- `failures`: the steps whose trainer failed, and why, by the step's number.

A group is done with once it has a result that trains on nothing, or a step that covers it has made its version or
failed. A run that is started again reads the tables and goes on: whatever has a decision and no outcome is taken up
where it was left.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Any, cast

from pydantic import JsonValue

from rollout_train.ledger import Ledger, between
from rollout_train.policies import named, versions_in


@dataclass
class Result:
    group: int
    """The group's number in the run, from 1."""
    time: float
    """When it was written, in seconds since the epoch."""
    task: str
    """The row's key."""
    title: str = ""
    rollout_seconds: float = 0.0
    """From the group's decision to its last episode's end (when its result was written)."""
    rewards: list[float] = field(default_factory=list[float])
    """Of the episodes fit to train on, as are `solved` and `durations`."""
    solved: list[bool] = field(default_factory=list[bool])
    durations: list[float | None] = field(default_factory=list[float | None])
    failed: int = 0
    """Episodes that did not complete, or asked to be left out."""
    failures: list[str] = field(default_factory=list[str])
    notes: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """What the algorithm said of the group."""
    segments_recorded: int = 0
    segments: int = 0
    """Segments the algorithm found to train on: none if it skipped the group."""
    skipped: str | None = None
    """Why the algorithm found nothing to train on, if it did not."""
    unlocked: int = 0
    """Rows of the catalog unlocked after this group."""

    def to_json(self) -> dict[str, Any]:
        """The record as the `results` table keeps it: without what the group's own record and key say (`JOINED`)."""
        return {key: value for key, value in asdict(self).items() if key not in JOINED}

    @classmethod
    def from_json(cls, data: Mapping[str, Any], number: int, group: Mapping[str, Any]) -> "Result":
        """A result as it is kept, with what its group's record (`group`, under `number`) says."""
        known = {each.name for each in fields(cls)} - set(JOINED)
        kept = {key: value for key, value in data.items() if key in known}
        written = float(kept.get("time") or 0.0)
        decided = group.get("decided")
        return cls(
            **kept,
            group=number,
            task=str(group.get("task", "")),
            title=str(group.get("title", "")),
            rollout_seconds=round(written - float(decided), 1) if isinstance(decided, int | float) else 0.0,
        )


JOINED = ("group", "task", "title", "rollout_seconds")
"""What a result is read with from its group (its key, and its record in the `groups` table), not kept twice."""


def scope(run: str) -> str:
    """The scope whose fence a run's loop holds."""
    return _RUNS + run


def table(run: str, name: str) -> str:
    """A run's table in the ledger."""
    return f"{scope(run)}/{name}"


GROUPS, RESULTS, STEPS, FAILURES = "groups", "results", "steps", "failures"
_RUNS = "runs/"


async def runs_in(ledger: Ledger) -> list[str]:
    """The runs a ledger has groups of, by name."""
    return [run for each in await ledger.tables() if (run := between(each, _RUNS, f"/{GROUPS}"))]


async def results(ledger: Ledger, run: str = "train") -> list[Result]:
    """How a run's groups went, by their numbers."""
    logged, decided = await ledger.read(table(run, RESULTS)), await ledger.read(table(run, GROUPS))
    lines = [
        Result.from_json(cast(Mapping[str, Any], record), int(key), cast(Mapping[str, Any], decided.get(key) or {}))
        for key, record in logged.items()
        if isinstance(record, dict)
    ]
    return sorted(lines, key=lambda line: line.group)


@dataclass(frozen=True)
class Trained:
    """What was done with a group: the step that covered it, and the version that step made or why it failed."""

    step: int
    version: str | None = None
    """By name, once made."""
    error: str | None = None


async def trained(ledger: Ledger, run: str = "train") -> dict[int, Trained]:
    """For each group a step covers: that step, and its outcome if it has one."""
    steps = await ledger.read(table(run, STEPS))
    failures: Any = await ledger.read(table(run, FAILURES))
    made: dict[str, set[str]] = {}
    covered: dict[int, Trained] = {}
    for key, record in steps.items():
        intent: Any = record
        policy = str(intent.get("policy"))
        if policy not in made:
            made[policy] = {version.name for version in await versions_in(ledger, policy)}
        name = named(policy, int(intent["number"]))
        error = failures[key].get("error") if key in failures else None
        outcome = Trained(int(key), name if error is None and name in made[policy] else None, error)  # (a failed
        # step made nothing: the next one makes the version it would have)
        listed: list[Any] = intent.get("groups") or []
        for group in listed:
            covered[int(group)] = outcome
    return covered
