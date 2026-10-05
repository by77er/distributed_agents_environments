"""Runs asked for from anywhere, the jobs they became, and how each goes.

Whoever wants a run (the monitor's New run form, `rollout train`) asks for it with its run settings
(`rollout_train.run_settings`): a `Launch` records what was asked (`Asked`: the kind of run, its name, its settings, the
preset they came from, and the run it resumes, if it resumes one), the run it is (`run`, its id in the registry), the
job it became (`job`: a Ray job's submission id, or a RayJob's name) and its state. `rollout_train.submitting.submit`
records a launch and starts its job; the job's driver (`rollout_train.jobs`) notes when it runs and how it ends, and
whoever reads a launch that is going reads its job's status too (`rollout_train.submitting.followed`), so a job that
died without saying so is noted failed.

A launch goes asked (recorded), submitted (its job was created: it may wait for its resources), running (its driver
started), stopping (a stop was asked for), and ends ended, failed (with why) or stopped. Every change of a launch's
state compares and sets: it is made only if the launch is where its writer expects, and may go where it is sent
(`MOVES`), so a stop is never overwritten by a driver that started meanwhile.

This is ordinary state, changed in place, not part of the ledger's append-only record: a file beside a ledger of
files (`FileLaunches`), a table in a database ledger's database (`rollout_train.database.DatabaseLaunches`).
"""

import asyncio
import json
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import JsonValue

from rollout.contracts import new_ulid
from rollout_train.ledger import FileLedger, Ledger, locked

__all__ = [
    "MOVES",
    "OPEN",
    "Asked",
    "FileLaunches",
    "Launch",
    "Launches",
    "as_launch",
    "changed",
    "launch_of",
    "launches_of",
    "new_launch",
    "stored",
]

ASKED, SUBMITTED, RUNNING, STOPPING, ENDED, FAILED, STOPPED = (
    "asked",
    "submitted",
    "running",
    "stopping",
    "ended",
    "failed",
    "stopped",
)
"""Where a launch is: recorded; its job created; its driver running; asked to stop; and how it finished."""
OPEN = (ASKED, SUBMITTED, RUNNING, STOPPING)
MOVES: Mapping[str, frozenset[str]] = {
    ASKED: frozenset({SUBMITTED, RUNNING, FAILED, STOPPED}),
    SUBMITTED: frozenset({RUNNING, STOPPING, STOPPED, ENDED, FAILED}),
    RUNNING: frozenset({STOPPING, STOPPED, ENDED, FAILED}),
    STOPPING: frozenset({STOPPED, ENDED, FAILED}),
}
"""Where a launch may go from where it is. A launch that finished (ended, failed, stopped) goes nowhere, nothing goes
back, and a launch asked to stop is not running again: a stop asked for while its job starts stays, and its job is
stopped. A driver may start before its submitter has noted its job (asked to running). A state may also be noted again
(its details changed)."""
TRAIN, EVAL, IMITATE, CHECK = "train", "eval", "imitate", "check"
"""What a launch starts: a training run, an eval of one subject, supervised steps on a dataset, an environment's
check (`rollout_train.run_settings.KINDS`)."""


@dataclass(frozen=True)
class Asked:
    """What a run is asked to be: its kind, its name, its run settings (as given: the schema's defaults are not
    written), the preset they came from (`NAME@N`), and, for a launch that resumes a run, that run's id."""

    kind: str = TRAIN
    name: str = ""
    settings: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    preset: str | None = None
    resumes: str | None = None

    @property
    def environment(self) -> str | None:
        """The environment it plays (`module:name`, or a published one as `NAME@VERSION`), if it plays one."""
        said = self.settings.get("environment")
        return str(said) if said else None


@dataclass(frozen=True)
class Launch:
    id: str
    asked: Asked
    at: float
    """When it was asked for."""
    state: str = ASKED
    run: str | None = None
    """The run it is, by id: registered when it was asked for, or the run it resumes."""
    job: str | None = None
    """Its job: a Ray job's submission id, or a RayJob's name."""
    backend: str | None = None
    """Where its job is: `ray` (Ray's job API) or `kubernetes` (a RayJob)."""
    detail: str | None = None
    """Why it failed, how it ended, or what its run waits for."""
    updated: float = 0.0


def as_launch(data: Mapping[str, Any]) -> Launch:
    """A launch as it was stored. A launch asked for a profile (its `asked` names one) is read as its run settings:
    its environment, start, bookmark, groups, groups a step and seed among them, and an eval's suite and episodes."""
    fields: dict[str, Any] = dict(data)
    asked = dict(cast(Mapping[str, Any], fields["asked"]))
    if "profile" in asked:
        asked = _settled(asked)
    fields["asked"] = Asked(**asked)
    if fields.get("state") == "claimed":
        fields["state"] = SUBMITTED
    known = set(Launch.__dataclass_fields__)
    return Launch(**{key: value for key, value in fields.items() if key in known})


def _settled(asked: Mapping[str, Any]) -> dict[str, Any]:
    """The run settings a launch of a profile asked for."""
    kind = EVAL if asked.get("kind") == EVAL else TRAIN
    settings: dict[str, JsonValue] = {"environment": asked.get("environment")}
    for key in ("start", "bookmark") if kind == TRAIN else ("start",):
        if asked.get(key) is not None:
            settings[key] = asked[key]
    if kind == TRAIN:
        settings |= {key: asked[key] for key in ("groups", "groups_per_step", "seed") if key in asked}
    else:
        settings |= {"eval.suite": asked.get("suite"), "eval.episodes": asked.get("episodes")}
    settings |= dict(cast(Mapping[str, JsonValue], asked.get("settings") or {}))
    return {"kind": kind, "name": str(asked.get("name") or ""), "settings": settings, "resumes": asked.get("resumes")}


class Launches(Protocol):
    async def ask(self, asked: Asked, run: str | None = None) -> Launch:
        """Record a launch, as asked, of the run `run` (by id); the launch."""
        ...

    async def all(self) -> list[Launch]:
        """Every launch, newest first."""
        ...

    async def note(self, id: str, *, expect: Collection[str] | None = None, **changes: Any) -> Launch:
        """Note how a launch goes (its state, job, detail), in one step that compares and sets: the changes are written
        only if the launch is in a state of `expect` (any, if None) and may go to the state they name (`MOVES`).
        Returns the launch as it is then, changed or not: whoever moves it compares the state it gets with the state
        it asked for. Raises `KeyError` when there is no such launch."""
        ...


async def launch_of(launches: Launches, id: str) -> Launch:
    """A launch by id. Raises `KeyError` when there is none."""
    found = next((each for each in await launches.all() if each.id == id), None)
    if found is None:
        raise KeyError(f"there is no launch {id}")
    return found


def changed(launch: Launch, expect: Collection[str] | None, changes: Mapping[str, Any]) -> Launch | None:
    """A launch with `changes`, if they may be made of it as it is (`Launches.note`); else None."""
    if expect is not None and launch.state not in expect:
        return None
    state = changes.get("state", launch.state)
    if state != launch.state and state not in MOVES.get(launch.state, frozenset()):
        return None
    return replace(launch, **changes, updated=round(time.time(), 1))


def launches_of(ledger: Ledger) -> Launches | None:
    """The launches beside a ledger: a file beside a ledger of files, a table in a database ledger's database."""
    if isinstance(ledger, FileLedger):
        return FileLaunches(ledger.directory)
    return getattr(ledger, "launches", None)


def new_launch(asked: Asked, run: str | None = None) -> Launch:
    at = time.time()  # (unrounded: launches asked for at once still list newest first)
    return Launch(f"launch_{new_ulid()}", asked, at, run=run, updated=round(at, 1))


def stored(launch: Launch) -> str:
    """A launch as JSON, as it is stored."""
    return json.dumps(asdict(launch))


class FileLaunches:
    """`Launches` in `launches.json` in a ledger's directory, under the lock the ledger's files are written under."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path = directory / "launches.json"

    async def ask(self, asked: Asked, run: str | None = None) -> Launch:
        made = new_launch(asked, run)

        def change(launches: dict[str, Launch]) -> dict[str, Launch]:
            return {**launches, made.id: made}

        await asyncio.to_thread(self._change, change)
        return made

    async def all(self) -> list[Launch]:
        return await asyncio.to_thread(lambda: sorted(self._read().values(), key=lambda each: -each.at))

    async def note(self, id: str, *, expect: Collection[str] | None = None, **changes: Any) -> Launch:
        noted: list[Launch] = []

        def change(launches: dict[str, Launch]) -> dict[str, Launch]:
            if id not in launches:
                raise KeyError(f"there is no launch {id}")
            made = changed(launches[id], expect, changes)
            noted.append(made or launches[id])
            return {**launches, id: made} if made is not None else launches

        await asyncio.to_thread(self._change, change)
        return noted[0]

    def _change(self, change: Callable[[dict[str, Launch]], dict[str, Launch]]) -> None:
        with locked(self.directory):
            launches = change(self._read())
            staged = self.path.with_suffix(".staged")
            staged.write_text(json.dumps([asdict(each) for each in launches.values()]))
            staged.replace(self.path)

    def _read(self) -> dict[str, Launch]:
        if not self.path.exists():
            return {}
        return {each["id"]: as_launch(each) for each in json.loads(self.path.read_text())}
