"""What a training run writes down: its tables in a ledger.

- `starts`: each time the run was started, where and by what (its host and process, when, the checkpoint it starts
  from, its environment: as `module:name`, its version and what its results say; and what its starter adds: its
  directory, the profile, where the monitor on its machine serves, where its blobs are, its settings), by the number of
  the fence its loop took. Written as the loop starts, so that whatever reads the ledger (the monitor) finds every run
  that shares it, and where each keeps the rest.
- `ends`: how each start ended (`ENDINGS`), under the same number.
- `groups`: what the run decided to play (the row, and the start every episode of the group is given), by the
  group's number. Written before the group is asked for.
- `results`: how each group went (a `Result`), by the group's number. Written when its last episode ends, before
  anything is trained on it.
- `steps`: what the run decided to train on, by the step's number: the groups it covers, the checkpoint it starts
  from, the one it will make, the batch. Written before the trainer is called. The checkpoint it makes, in the
  ledger's `checkpoints` table, is its outcome.
- `failures`: the steps whose trainer failed, and why, by the step's number.
- `evals`: the evals of the checkpoints it made that its schedule names (`rollout_train.evals.Schedule`), by the
  number of the step that made each: the suite, the checkpoint, the eval's run, and how it went. Written when the eval
  has played every start.

A group is done with once it has a result that trains on nothing, or a step that covers it has made its checkpoint or
failed. A run that is started again reads the tables and goes on: whatever has a decision and no outcome is taken up
where it was left.
"""

import asyncio
import contextlib
import os
import secrets
import socket
import time
from collections.abc import AsyncGenerator, Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Any, cast

from pydantic import JsonValue

from rollout.environment import Environment
from rollout_train.checkpoints import checkpoints_in
from rollout_train.ledger import Fence, Fenced, Ledger, between


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
    """Rows of the environment unlocked after this group."""

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


def described(environment: Environment) -> dict[str, JsonValue]:
    """What a run's start says of its environment: its version, and what its results say (`Description`)."""
    return {"version": environment.version, "description": environment.description.to_json()}


JOINED = ("group", "task", "title", "rollout_seconds")
"""What a result is read with from its group (its key, and its record in the `groups` table), not kept twice."""


async def end(ledger: Ledger, run: str, how: str, detail: str | None = None) -> None:
    """Say how this process's start of `run` ended, under the fence it started with. Nothing is said if it wrote no
    start, or if another process has taken the run since: that one says how its own start ends."""
    starts = await ledger.read(table(run, STARTS))
    mine = [int(key) for key, record in starts.items() if isinstance(record, dict) and record.get("process") == PROCESS]
    if not mine:
        return
    number = str(max(mine))
    said: JsonValue = {"how": how, "at": round(time.time(), 1), "detail": detail}
    with contextlib.suppress(Fenced):
        await ledger.append(table(run, ENDS), number, said, Fence(scope(run), max(mine)))


@contextlib.asynccontextmanager
async def ending(ledger: Ledger, run: str) -> AsyncGenerator[None]:
    """Say how the work inside ends (`end`): finished, stopped (cancelled: an interrupt) or failed (it raised)."""
    try:
        yield
    except asyncio.CancelledError:
        await asyncio.shield(end(ledger, run, STOPPED))
        raise
    except (Exception, SystemExit) as error:
        await end(ledger, run, FAILED, f"{type(error).__name__}: {error}"[:500])
        raise
    else:
        await end(ledger, run, FINISHED)


def scope(run: str) -> str:
    """The scope whose fence a run's loop holds."""
    return _RUNS + run


def newest_record(records: Mapping[str, JsonValue]) -> dict[str, Any]:
    """The newest of a table's records keyed by fence number (a run's starts, ends or plans): the one under the
    greatest number. Empty for none, or for one that is no object."""
    return mapping(records[max(records, key=int)]) if records else {}


def mapping(record: JsonValue) -> dict[str, Any]:
    """A record as an object (empty for one that is no object)."""
    return cast(dict[str, Any], record) if isinstance(record, dict) else {}


def start_header(**more: JsonValue) -> dict[str, JsonValue]:
    """What a run's start says of where and when it started: the machine (`host`), this process (`PROCESS`) and the
    time; with `more`."""
    return {"host": socket.gethostname(), "process": PROCESS, "started": round(time.time(), 1), **more}


def table(run: str, name: str) -> str:
    """A run's table in the ledger."""
    return f"{scope(run)}/{name}"


GROUPS, RESULTS, STEPS, FAILURES, STARTS, EVALS = "groups", "results", "steps", "failures", "starts", "evals"
ENDS = "ends"
"""How each start of a run ended, under the same key (its fence's number): `how` (`ENDINGS`), `at`, and `detail`."""
FINISHED, STOPPED, FAILED = "finished", "stopped", "failed"
ENDINGS = (FINISHED, STOPPED, FAILED)
"""It did what it was asked (played its groups, its suite, its step); it was stopped (an interrupt, a stop asked
for); it raised (`detail` says what)."""
PROCESS = f"{socket.gethostname()}/{os.getpid()}/{secrets.token_hex(4)}"
"""This process, as the starts it writes name it (`process`), so that it says how its own start ended."""
_RUNS = "runs/"


async def runs_in(ledger: Ledger) -> list[str]:
    """The runs a ledger has, by name: those that were started or decided a group."""
    return named_runs(await ledger.tables())


def named_runs(tables: Iterable[str]) -> list[str]:
    """The runs that tables (by name) are of: those with a `starts` or a `groups` table."""
    found = {run for each in tables for name in (STARTS, GROUPS) if (run := between(each, _RUNS, f"/{name}"))}
    return sorted(found)


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
    """What was done with a group: the step that covered it, and the checkpoint that step made or why it failed."""

    step: int
    checkpoint: str | None = None
    """By id, once made."""
    error: str | None = None


async def trained(ledger: Ledger, run: str = "train") -> dict[int, Trained]:
    """For each group a step covers: that step, and its outcome if it has one."""
    steps = await ledger.read(table(run, STEPS))
    failures: Any = await ledger.read(table(run, FAILURES))
    made = {checkpoint.id for checkpoint in await checkpoints_in(ledger)}
    covered: dict[int, Trained] = {}
    for key, record in steps.items():
        intent: Any = record
        makes = str(intent.get("makes"))
        error = failures[key].get("error") if key in failures else None
        outcome = Trained(int(key), makes if error is None and makes in made else None, error)  # (a failed step made
        # nothing: the next one makes a checkpoint of its own)
        listed: list[Any] = intent.get("groups") or []
        for group in listed:
            covered[int(group)] = outcome
    return covered
