"""Training runs asked for from anywhere, and started by a launcher on a machine that can run them.

Whoever wants a run (the monitor's page, say) asks for it: a `Launch` names a profile and an environment, what the run
is called, the checkpoint it starts from, and the settings it changes (`rollout train --set`); or, for an eval, the
suite it plays, the checkpoint that plays it and how many episodes of each start (`rollout eval`,
`rollout_train.evals`). A launcher (`rollout_train.launcher`) on a training machine says in its heartbeat which profiles
it can run, claims a launch asked for one of them, starts `rollout train` (or `rollout eval`), and notes how it goes:
claimed, running (with the process), ended or failed (with why). A launch asked to stop is stopped by its launcher.
Every change of a launch's state compares and sets: it is made only if the launch is where its writer expects, and may
go where it is sent (`MOVES`), so a stop is never overwritten by a launcher that started the run meanwhile.

This is ordinary state, changed in place, not part of the ledger's append-only record: a file beside a ledger of
files (`FileLaunches`), a table in a database ledger's database (`rollout_train.database.DatabaseLaunches`).
"""

import asyncio
import fcntl
import json
import time
from collections.abc import Callable, Collection, Generator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from pydantic import JsonValue

from rollout.contracts import new_ulid
from rollout_train.ledger import FileLedger, Ledger

ASKED, CLAIMED, RUNNING, STOPPING, ENDED, FAILED, STOPPED = (
    "asked",
    "claimed",
    "running",
    "stopping",
    "ended",
    "failed",
    "stopped",
)
"""Where a launch is: asked for; claimed by a launcher; its run going; asked to stop; and how it finished."""
OPEN = (ASKED, CLAIMED, RUNNING, STOPPING)
MOVES: Mapping[str, frozenset[str]] = {
    ASKED: frozenset({CLAIMED, STOPPED}),
    CLAIMED: frozenset({RUNNING, STOPPING, STOPPED, ENDED, FAILED}),
    RUNNING: frozenset({STOPPING, STOPPED, ENDED, FAILED}),
    STOPPING: frozenset({STOPPED, ENDED, FAILED}),
}
"""Where a launch may go from where it is. A launch that finished (ended, failed, stopped) goes nowhere, nothing goes
back, and a launch asked to stop is not running again: a stop asked for while its launcher starts the run stays, and
the launcher signals the run it started. A state may also be noted again (its details changed)."""
RUN, EVAL = "run", "eval"
"""What a launch starts: a training run (`rollout train`), or an eval (`rollout eval`)."""


@dataclass(frozen=True)
class Asked:
    """What a run is asked to be: a training run, or (`kind` `eval`) a suite played by a checkpoint."""

    profile: str
    """The profile, by the name a launcher offers it under."""
    environment: str
    """The environment, as `module:name`."""
    name: str
    """What the run is called."""
    start: str | None = None
    """The checkpoint it trains from (a reference, `rollout_train.registry.resolved`); None: the base model."""
    bookmark: str | None = None
    """A bookmark the run carries forward."""
    groups: int = 100
    groups_per_step: int = 4
    seed: int = 0
    settings: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """What it changes of its profile, by dotted key: `trainer.learning_rate`, `episodes_at_once`, say."""
    kind: str = RUN
    """`run` (a training run) or `eval` (a suite played by `start`, the checkpoint; none: the base model)."""
    suite: str | None = None
    """For an eval: the suite it plays (its environment is the suite's)."""
    episodes: int = 1
    """For an eval: episodes of each of the suite's starts."""


@dataclass(frozen=True)
class Launch:
    id: str
    asked: Asked
    at: float
    """When it was asked for."""
    state: str = ASKED
    launcher: str | None = None
    directory: str | None = None
    """The run's directory, once a launcher has chosen it."""
    pid: int | None = None
    """The process playing it, when its launcher started it itself."""
    job: str | None = None
    """The Ray job playing it, when its launcher submitted it to Ray."""
    detail: str | None = None
    """Why it failed, or how it ended."""
    updated: float = 0.0


def as_launch(data: Mapping[str, Any]) -> Launch:
    fields: dict[str, Any] = dict(data)
    fields["asked"] = Asked(**as_asked(fields["asked"]))
    return Launch(**fields)


def as_asked(given: Mapping[str, Any]) -> dict[str, Any]:
    """What a launch asks, with its environment under `environment` (a launch asked for as a `catalog` says it so)."""
    asked = dict(given)
    if "catalog" in asked:
        named = asked.pop("catalog")
        asked.setdefault("environment", named)
    return asked


class Launches(Protocol):
    async def ask(self, asked: Asked) -> Launch:
        """Ask for a run; the launch, as asked."""
        ...

    async def all(self) -> list[Launch]:
        """Every launch, newest first."""
        ...

    async def claim(self, id: str, launcher: str) -> Launch | None:
        """Claim a launch that is asked for: the launch, claimed, or None if another launcher claimed it first."""
        ...

    async def note(self, id: str, *, expect: Collection[str] | None = None, **changes: Any) -> Launch:
        """Note how a launch goes (its state, directory, process, detail), in one step that compares and sets: the
        changes are written only if the launch is in a state of `expect` (any, if None) and may go to the state they
        name (`MOVES`). Returns the launch as it is then, changed or not: whoever moves it compares the state it gets
        with the state it asked for. Raises `KeyError` when there is no such launch."""
        ...


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


def new_launch(asked: Asked) -> Launch:
    at = time.time()  # (unrounded: launches asked for at once still list newest first)
    return Launch(f"launch_{new_ulid()}", asked, at, updated=round(at, 1))


class FileLaunches:
    """`Launches` in `launches.json` in a ledger's directory, under the lock the ledger's files are written under."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path = directory / "launches.json"

    async def ask(self, asked: Asked) -> Launch:
        made = new_launch(asked)

        def change(launches: dict[str, Launch]) -> dict[str, Launch]:
            return {**launches, made.id: made}

        await asyncio.to_thread(self._change, change)
        return made

    async def all(self) -> list[Launch]:
        return await asyncio.to_thread(lambda: sorted(self._read().values(), key=lambda each: -each.at))

    async def claim(self, id: str, launcher: str) -> Launch | None:
        claimed: list[Launch] = []

        def change(launches: dict[str, Launch]) -> dict[str, Launch]:
            if launches[id].state != ASKED:
                return launches
            claimed.append(replace(launches[id], state=CLAIMED, launcher=launcher, updated=round(time.time(), 1)))
            return {**launches, id: claimed[0]}

        await asyncio.to_thread(self._change, change)
        return claimed[0] if claimed else None

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
        with self._locked():
            launches = change(self._read())
            staged = self.path.with_suffix(".staged")
            staged.write_text(json.dumps([asdict(each) for each in launches.values()]))
            staged.replace(self.path)

    def _read(self) -> dict[str, Launch]:
        if not self.path.exists():
            return {}
        return {each["id"]: as_launch(each) for each in json.loads(self.path.read_text())}

    @contextmanager
    def _locked(self) -> Generator[None]:
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
