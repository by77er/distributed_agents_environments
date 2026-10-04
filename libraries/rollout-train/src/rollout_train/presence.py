"""Which runners are alive, and how their machines are doing: a heartbeat each runner writes, kept beside the ledger.

An episode runner (`rollout_train.rollouts.scheduler`) beats every few seconds. A beat says when, and what the runner
says of itself: its host, the run it serves, its places and how many it plays, its machine (memory, accelerators,
disk), its engines' processes, and what each channel serves and how fast. Each runner's newest beat is kept, with the
measurements of its recent ones, so the monitor can show how its machine moved, from anywhere.

A runner whose newest beat is older than `STALE` seconds is taken to be gone: what it had claimed is open to be
claimed again. This is ordinary state, changed in place, not part of the ledger's append-only record: a file beside a
ledger of files (`FilePresence`), a table in a database ledger's database (`rollout_train.database.DatabasePresence`).
"""

import asyncio
import fcntl
import json
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import JsonValue

from rollout_train.ledger import FileLedger, Ledger

STALE = 90.0
"""Seconds after its newest beat that a runner is taken to be gone."""
KEPT = 240
"""Recent measurements kept per runner (an hour, at one beat every fifteen seconds)."""
MEASURED = ("at", "machine", "channels", "playing")
"""What of each beat is kept in a runner's history."""


@dataclass(frozen=True)
class Beat:
    runner: str
    at: float
    about: Mapping[str, JsonValue]
    """What the runner said of itself in its newest beat."""
    history: list[dict[str, JsonValue]] = field(default_factory=list[dict[str, JsonValue]])
    """Its recent beats' measurements (`MEASURED`), oldest first."""


class Presence(Protocol):
    async def beat(self, runner: str, about: Mapping[str, JsonValue]) -> None:
        """Note that `runner` is alive now, with what it says of itself."""
        ...

    async def beats(self) -> list[Beat]:
        """Every runner's newest beat, by runner."""
        ...


def alive(beat: Beat | None, now: float | None = None) -> bool:
    """Whether a runner beat within the last `STALE` seconds."""
    return beat is not None and (time.time() if now is None else now) - beat.at <= STALE


def presence_of(ledger: Ledger) -> Presence | None:
    """The heartbeats beside a ledger: a file beside a ledger of files, a table in a database ledger's database."""
    if isinstance(ledger, FileLedger):
        return FilePresence(ledger.directory)
    return getattr(ledger, "presence", None)


def kept(history: list[dict[str, Any]], at: float, about: Mapping[str, JsonValue]) -> list[dict[str, Any]]:
    """A runner's history with a new beat's measurements added, the oldest removed past `KEPT`."""
    point = {"at": at, **{name: about[name] for name in MEASURED if name in about and name != "at"}}
    return [*history, point][-KEPT:]


class FilePresence:
    """`Presence` in `presence.json` in a ledger's directory, under the lock the ledger's files are written under."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path = directory / "presence.json"

    async def beat(self, runner: str, about: Mapping[str, JsonValue]) -> None:
        def noted() -> None:
            with self._locked():
                beats = self._read()
                at = round(time.time(), 1)
                was = beats.get(runner)
                beats[runner] = Beat(runner, at, dict(about), kept(was.history if was else [], at, about))
                staged = self.path.with_suffix(".staged")
                staged.write_text(json.dumps([asdict(each) for each in beats.values()]))
                staged.replace(self.path)

        await asyncio.to_thread(noted)

    async def beats(self) -> list[Beat]:
        return await asyncio.to_thread(lambda: sorted(self._read().values(), key=lambda beat: beat.runner))

    def _read(self) -> dict[str, Beat]:
        if not self.path.exists():
            return {}
        return {each["runner"]: Beat(**each) for each in json.loads(self.path.read_text())}

    @contextmanager
    def _locked(self) -> Generator[None]:
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
