"""Which processes are alive, and how their machines are doing: a heartbeat each writes, kept beside the ledger.

Episode runners (`rollout_train.rollouts.scheduler`), engine hosts, gateway replicas, launchers and sandbox pools served
on their own beat every few seconds, each under its name. A beat says when, and what the process says of itself: an
episode runner, its host, the run it serves, its places and how many it plays, its machine (memory, accelerators,
disk), its engines' processes, and what each channel serves and how fast. Each one's newest beat is kept, with the
measurements of its recent ones, so the monitor can show how its machine moved, from anywhere.

One whose newest beat is older than `STALE` seconds is taken to be gone: what a runner had claimed is open to be
claimed again. This is ordinary state, changed in place, not part of the ledger's append-only record: a file beside a
ledger of files (`FilePresence`), a table in a database ledger's database (`rollout_train.database.DatabasePresence`).

A beat's time is the store's, not the runner's: the store stamps a beat when it keeps it, and says how old it is when it
is read (`Beat.age`), by the same clock. A Postgres database stamps and ages beats by its server's clock, so a runner
whose machine's clock is behind or ahead of the reader's is judged by when it last beat all the same. A ledger of files
and a SQLite database serve one machine, whose clock every writer and reader shares.
"""

import asyncio
import contextlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from pydantic import JsonValue

from rollout_train.ledger import FileLedger, Ledger, locked

STALE = 90.0
"""Seconds after its newest beat that a process is taken to be gone."""
KEPT = 240
"""Recent measurements kept per process (an hour, at one beat every fifteen seconds)."""
MEASURED = ("at", "machine", "channels", "playing")
"""What of each beat is kept in its history."""


@dataclass(frozen=True)
class Beat:
    runner: str
    at: float
    """When the store kept it, in seconds since the epoch by the store's clock."""
    about: Mapping[str, JsonValue]
    """What the runner said of itself in its newest beat."""
    history: list[dict[str, JsonValue]] = field(default_factory=list[dict[str, JsonValue]])
    """Its recent beats' measurements (`MEASURED`), oldest first."""
    age: float = 0.0
    """Seconds since the store kept it, by the store's clock, when it was read."""


class Presence(Protocol):
    async def beat(self, runner: str, about: Mapping[str, JsonValue]) -> None:
        """Note that `runner` is alive now, with what it says of itself."""
        ...

    async def beats(self) -> list[Beat]:
        """Every runner's newest beat, by runner."""
        ...


def alive(beat: Beat | None) -> bool:
    """Whether a runner beat within the last `STALE` seconds, by the store's clock."""
    return beat is not None and beat.age <= STALE


def presence_of(ledger: Ledger) -> Presence | None:
    """The heartbeats beside a ledger: a file beside a ledger of files, a table in a database ledger's database."""
    if isinstance(ledger, FileLedger):
        return FilePresence(ledger.directory)
    return getattr(ledger, "presence", None)


async def beating(
    presence: Presence, name: str, about: Callable[[], Mapping[str, JsonValue]], *, every: float = 15.0
) -> None:
    """Beat as `name` now and every `every` seconds until cancelled, with what `about` says (called in a thread: it may
    measure). A beat that fails is missed: the next says the same."""
    while True:
        with contextlib.suppress(Exception):
            await presence.beat(name, await asyncio.to_thread(about))
        await asyncio.sleep(every)


def kept(history: list[dict[str, Any]], at: float, about: Mapping[str, JsonValue]) -> list[dict[str, Any]]:
    """A runner's history with a new beat's measurements added, the oldest removed past `KEPT`."""
    point = {"at": at, **{name: about[name] for name in MEASURED if name in about and name != "at"}}
    return [*history, point][-KEPT:]


class FilePresence:
    """`Presence` in `presence.json` in a ledger's directory, under the lock the ledger's files are written under. Its
    clock is the machine's: a ledger of files serves one machine."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path = directory / "presence.json"

    async def beat(self, runner: str, about: Mapping[str, JsonValue]) -> None:
        def noted() -> None:
            with locked(self.directory):
                beats = self._read()
                at = round(time.time(), 1)
                was = beats.get(runner)
                beats[runner] = Beat(runner, at, dict(about), kept(was.history if was else [], at, about))
                staged = self.path.with_suffix(".staged")
                staged.write_text(json.dumps([_stored(each) for each in beats.values()]))
                staged.replace(self.path)

        await asyncio.to_thread(noted)

    async def beats(self) -> list[Beat]:
        def read() -> list[Beat]:
            now = time.time()
            return [replace(beat, age=now - beat.at) for beat in sorted(self._read().values(), key=_runner)]

        return await asyncio.to_thread(read)

    def _read(self) -> dict[str, Beat]:
        if not self.path.exists():
            return {}
        return {each["runner"]: Beat(**each) for each in json.loads(self.path.read_text())}


def _runner(beat: Beat) -> str:
    return beat.runner


def _stored(beat: Beat) -> dict[str, Any]:
    """A beat as a file keeps it: how old it is is said when it is read."""
    kept = asdict(beat)
    del kept["age"]
    return kept
