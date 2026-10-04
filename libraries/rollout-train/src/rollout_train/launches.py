"""Training runs asked for from anywhere, and started by a launcher on a machine that can run them.

Whoever wants a run (the monitor's page, say) asks for it: a `Launch` names a profile and a catalog, what the run is
called, the version it starts from, and the settings it changes (`rollout train --set`). A launcher
(`rollout_train.launcher`) on a training machine says in its heartbeat which profiles it can run, claims a launch
asked for one of them, starts `rollout train`, and notes how it goes: claimed, running (with the process), ended or
failed (with why). A launch asked to stop is stopped by its launcher.

This is ordinary state, changed in place, not part of the ledger's append-only record: a file beside a ledger of
files (`FileLaunches`), a table in a database ledger's database (`rollout_train.database.DatabaseLaunches`).
"""

import asyncio
import fcntl
import json
import time
from collections.abc import Callable, Generator, Mapping
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


@dataclass(frozen=True)
class Asked:
    """What a run is asked to be."""

    profile: str
    """The profile, by the name a launcher offers it under."""
    catalog: str
    """The catalog, as `module:name`."""
    name: str
    """What the run is called."""
    start: str | None = None
    """The version it trains from (a reference, `rollout_train.registry.resolved`); None: the base model."""
    bookmark: str | None = None
    """A bookmark the run carries forward."""
    groups: int = 100
    groups_per_step: int = 4
    seed: int = 0
    settings: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """What it changes of its profile, by dotted key: `trainer.learning_rate`, `episodes_at_once`, say."""


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
    detail: str | None = None
    """Why it failed, or how it ended."""
    updated: float = 0.0


def as_launch(data: Mapping[str, Any]) -> Launch:
    fields: dict[str, Any] = dict(data)
    fields["asked"] = Asked(**fields["asked"])
    return Launch(**fields)


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

    async def note(self, id: str, **changes: Any) -> Launch:
        """Note how a launch goes (its state, directory, process, detail)."""
        ...


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

    async def note(self, id: str, **changes: Any) -> Launch:
        noted: list[Launch] = []

        def change(launches: dict[str, Launch]) -> dict[str, Launch]:
            noted.append(replace(launches[id], **changes, updated=round(time.time(), 1)))
            return {**launches, id: noted[0]}

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
