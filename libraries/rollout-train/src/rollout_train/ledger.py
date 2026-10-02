"""A ledger: append-only tables and fences, the only state an orchestrator has.

Something that must survive its own death (a training loop, say) keeps nothing in memory that it cannot read back:
it appends what it decided and what happened to tables here, and on starting again reads them and goes on. Two
things make that safe.

- **Keys.** A record is appended under a key, and a table has each key once: appending under a key that is there
  changes nothing and says so. An action that is recorded before it is taken can therefore be taken again after a
  crash without being done twice.
- **Fences.** Whoever means to write takes the fence of a scope: a number higher than any taken before. An append
  that carries an older fence is refused (`Fenced`), so a process that was replaced, and does not know it yet,
  cannot write over its replacement.

`FileLedger` keeps tables as files of JSON lines; `rollout_durable` has one in a database, for several machines.
"""

import contextlib
import fcntl
import json
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import JsonValue


class Fenced(Exception):
    """A writer whose fence is no longer the newest tried to write: another has taken its place."""


@dataclass(frozen=True)
class Fence:
    """The right to write within a scope, until someone takes it again."""

    scope: str
    number: int


class Ledger(Protocol):
    async def take(self, scope: str) -> Fence:
        """Take a scope's fence. Whoever held it can no longer write within the scope."""
        ...

    async def append(self, table: str, key: str, record: JsonValue, fence: Fence) -> bool:
        """Append a record under `key`, unless the table has that key: then nothing changes and False is returned.
        Raises `Fenced` if `fence` is not its scope's newest."""
        ...

    async def read(self, table: str) -> dict[str, JsonValue]:
        """A table's records by key, in the order they were appended."""
        ...

    async def tables(self) -> list[str]:
        """The tables that have records, by name."""
        ...

    async def fences(self) -> dict[str, int]:
        """The newest fence of every scope that has been taken."""
        ...


def between(table: str, before: str, after: str) -> str | None:
    """What a table's name has between `before` and `after`, if it begins and ends so: which run's or which
    policy's table it is, among those of many."""
    middle = table[len(before) : len(table) - len(after)]
    return middle if middle and table == before + middle + after else None


class FileLedger:
    """A `Ledger` in a directory: a table is `<table>.jsonl`, one `{"key", "fence", "record"}` per line. Processes
    on one machine may share it: every operation holds a lock on the directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    async def take(self, scope: str) -> Fence:
        with self._locked():
            fences = self._fences()
            fences[scope] = fences.get(scope, 0) + 1
            (self.directory / FENCES).write_text(json.dumps(fences))
            return Fence(scope, fences[scope])

    async def append(self, table: str, key: str, record: JsonValue, fence: Fence) -> bool:
        with self._locked():
            if self._fences().get(fence.scope, 0) != fence.number:
                raise Fenced(f"{fence.scope} has a newer writer than fence {fence.number}")
            if key in self._read(table):
                return False
            path = self._path(table)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as file:
                file.write(json.dumps({"key": key, "fence": fence.number, "record": record}) + "\n")
            return True

    async def read(self, table: str) -> dict[str, JsonValue]:
        with self._locked():
            return self._read(table)

    async def tables(self) -> list[str]:
        with self._locked():
            return sorted(str(path.relative_to(self.directory))[: -len(".jsonl")] for path in self._files())

    async def fences(self) -> dict[str, int]:
        with self._locked():
            return self._fences()

    def _files(self) -> list[Path]:
        return list(self.directory.rglob("*.jsonl"))

    def _read(self, table: str) -> dict[str, JsonValue]:
        path = self._path(table)
        if not path.exists():
            return {}
        records: dict[str, JsonValue] = {}
        for line in path.read_text().splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue  # a line a dying writer left half written: it was never appended
            records.setdefault(str(entry["key"]), entry["record"])
        return records

    def _path(self, table: str) -> Path:
        return self.directory / f"{table}.jsonl"

    def _fences(self) -> dict[str, int]:
        path = self.directory / FENCES
        return json.loads(path.read_text()) if path.exists() else {}

    @contextlib.contextmanager
    def _locked(self) -> Generator[None]:
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


FENCES = "fences.json"
"""In a `FileLedger`'s directory: the newest fence of every scope."""
