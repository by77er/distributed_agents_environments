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

`FileLedger` keeps tables as files of JSON lines; `rollout_train.database.DatabaseLedger` keeps them in SQL tables that
every run and machine using the database shares. A run's directory says where its ledger is (`LOCATION`), so that
whatever reads the run (the monitor, the report) finds it.
"""

import asyncio
import contextlib
import fcntl
import json
import os
import tempfile
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from pydantic import JsonValue


class Fenced(Exception):
    """A writer whose fence is no longer the newest tried to write: another has taken its place."""


@dataclass(frozen=True)
class Fence:
    """The right to write within a scope, until someone takes it again."""

    scope: str
    number: int


@dataclass(frozen=True)
class Appended:
    """What an append did: whether this call wrote its record (`wrote`), and the record the table holds under the key
    now (its own, or the one appended first)."""

    wrote: bool
    record: JsonValue


class Ledger(Protocol):
    async def take(self, scope: str) -> Fence:
        """Take a scope's fence. Whoever held it can no longer write within the scope."""
        ...

    async def append(self, table: str, key: str, record: JsonValue, fence: Fence) -> bool:
        """Append a record under `key`, unless the table has that key: then nothing changes and False is returned.
        Raises `Fenced` if `fence` is not its scope's newest."""
        ...

    async def read(self, table: str) -> dict[str, JsonValue]:
        """A table's records by key, in the order they were appended: the order their appends took effect in, whichever
        scopes made them."""
        ...

    async def tables(self) -> list[str]:
        """The tables that have records, by name."""
        ...

    async def fences(self) -> dict[str, int]:
        """The newest fence of every scope that has been taken."""
        ...


async def appended(ledger: Ledger, table: str, key: str, record: JsonValue, fence: Fence) -> Appended:
    """Append as `Ledger.append` does, and say what the table holds under `key`: whether this call wrote `record`, and
    if it did not, the record appended first. A ledger that says so in the same call (`append_returning`, as
    `FileLedger` and `DatabaseLedger` do) is asked; any other is read back after an append that wrote nothing."""
    returning = getattr(ledger, "append_returning", None)
    if returning is not None:
        said: Appended = await returning(table, key, record, fence)
        return said
    if await ledger.append(table, key, record, fence):
        return Appended(True, record)
    return Appended(False, (await ledger.read(table)).get(key))


def between(table: str, before: str, after: str) -> str | None:
    """What a table's name has between `before` and `after`, if it begins and ends so: which run's or which
    policy's table it is, among those of many."""
    middle = table[len(before) : len(table) - len(after)]
    return middle if middle and table == before + middle + after else None


class FileLedger:
    """A `Ledger` in a directory: a table is `<table>.jsonl`, one `{"key", "fence", "record"}` per line. Processes
    on one machine may share it: every operation holds a lock on the directory, in a thread (the event loop never
    waits on the lock).

    An append is on disk (`fsync`) before it is acknowledged. A last line left unfinished (by a writer that died
    mid-record, or a full disk) was never acknowledged: the next append removes it before writing (or ends it, where it
    holds a whole record), so that nothing is glued to it. `fences.json` is replaced whole (written beside it and put on
    disk, then renamed over it), so a crash while taking a fence leaves the fences as they were. Which keys a table has
    is kept in memory, and read again only as far as its file grew since (other processes' appends), or whole if the
    file was replaced."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._indexes: dict[str, _Index] = {}

    async def take(self, scope: str) -> Fence:
        return await asyncio.to_thread(self._take, scope)

    async def append(self, table: str, key: str, record: JsonValue, fence: Fence) -> bool:
        return (await self.append_returning(table, key, record, fence)).wrote

    async def append_returning(self, table: str, key: str, record: JsonValue, fence: Fence) -> Appended:
        """`append`, saying what the table holds under `key` too."""
        return await asyncio.to_thread(self._append, table, key, record, fence)

    async def read(self, table: str) -> dict[str, JsonValue]:
        return await asyncio.to_thread(self._under_lock, self._read, table)

    async def tables(self) -> list[str]:
        return await asyncio.to_thread(self._under_lock, self._tables)

    async def fences(self) -> dict[str, int]:
        return await asyncio.to_thread(self._under_lock, self._fences)

    def _take(self, scope: str) -> Fence:
        with self._locked():
            fences = self._fences()
            fences[scope] = fences.get(scope, 0) + 1
            _replaced(self.directory / FENCES, json.dumps(fences).encode())
            return Fence(scope, fences[scope])

    def _append(self, table: str, key: str, record: JsonValue, fence: Fence) -> Appended:
        line = (json.dumps({"key": key, "fence": fence.number, "record": record}) + "\n").encode()
        with self._locked():
            if self._fences().get(fence.scope, 0) != fence.number:
                raise Fenced(f"{fence.scope} has a newer writer than fence {fence.number}")
            path = self._path(table)
            path.parent.mkdir(parents=True, exist_ok=True)
            made = not path.exists()
            with path.open("a+b") as file:
                index = self._indexed(table, file)
                if (offset := index.keys.get(key)) is not None:
                    file.seek(offset)
                    return Appended(False, json.loads(file.readline())["record"])
                try:
                    file.write(line)
                    file.flush()
                    os.fsync(file.fileno())
                except BaseException:
                    with contextlib.suppress(OSError):  # (a full disk: what was written of the line goes)
                        os.ftruncate(file.fileno(), index.size)
                    raise
                index.keys[key] = index.size
                index.size += len(line)
            if made:
                _synced_directory(path.parent)
            return Appended(True, record)

    def _indexed(self, table: str, file: BinaryIO) -> "_Index":
        """The keys of the table whose file `file` is, read as far as the file grew since they were last read (whole,
        if it is another file than it was, or shorter). An unfinished last line is removed, or ended if it holds a
        whole record. Called under the lock."""
        status = os.fstat(file.fileno())
        index = self._indexes.get(table)
        if index is None or index.file != (status.st_dev, status.st_ino) or status.st_size < index.size:
            index = self._indexes[table] = _Index((status.st_dev, status.st_ino))
        if status.st_size == index.size:
            return index
        file.seek(index.size)
        *lines, unfinished = file.read(status.st_size - index.size).split(b"\n")
        offset = index.size
        for each in lines:
            if (found := _key_of(each)) is not None:
                index.keys.setdefault(found, offset)
            offset += len(each) + 1
        if unfinished:
            if (found := _key_of(unfinished)) is not None:  # a whole record whose newline was not written
                file.write(b"\n")
                index.keys.setdefault(found, offset)
                offset += len(unfinished) + 1
            else:  # a line a dying writer left half written: it was never appended
                os.ftruncate(file.fileno(), offset)
            file.flush()
            os.fsync(file.fileno())
        index.size = offset
        return index

    def _tables(self) -> list[str]:
        return sorted(str(path.relative_to(self.directory))[: -len(".jsonl")] for path in self._files())

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

    def _under_lock[**P, T](self, call: Callable[P, T], *arguments: P.args, **options: P.kwargs) -> T:
        with self._locked():
            return call(*arguments, **options)

    @contextlib.contextmanager
    def _locked(self) -> Generator[None]:
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


@dataclass
class _Index:
    """What a `FileLedger` knows of a table's file: which file it is (its device and inode), how far it was read, and
    where each key's line begins."""

    file: tuple[int, int]
    size: int = 0
    keys: dict[str, int] = field(default_factory=dict[str, int])


def _key_of(line: bytes) -> str | None:
    """The key of a line of a table's file; None for a line that holds no whole record."""
    try:
        entry: JsonValue = json.loads(line)
    except ValueError:
        return None
    return str(entry["key"]) if isinstance(entry, dict) and "key" in entry else None


def _replaced(path: Path, data: bytes) -> None:
    """Replace a file whole: written beside it and put on disk, then renamed over it (a crash leaves the old one)."""
    handle, staged = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(staged, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(staged)
        raise
    _synced_directory(path.parent)


def _synced_directory(directory: Path) -> None:
    """Put a directory's entries on disk: a file made or renamed in it is then there after a crash."""
    handle = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


FENCES = "fences.json"
"""In a `FileLedger`'s directory: the newest fence of every scope."""


LOCATION = "ledger.json"
"""In a run's directory: where its ledger is, as `opened` reads it."""


def opened(location: Mapping[str, Any]) -> Ledger:
    """The ledger a location names: `{"directory": …}`, files there; or `{"kind": "module:name", …}`, what that makes
    when called with the other entries (`{"kind": "rollout_train.database:DatabaseLedger", "url": …}`)."""
    if "kind" in location:
        from rollout.names import named

        entries = dict(location)
        return named(str(entries.pop("kind")))(**entries)
    return FileLedger(Path(str(location["directory"])).expanduser())


def of_run(directory: Path) -> Ledger:
    """The ledger of the run in `directory`: where its `LOCATION` file says, or files under `directory/ledger`."""
    path = directory / LOCATION
    if path.exists():
        return opened(json.loads(path.read_text()))
    return FileLedger(directory / "ledger")


def present(ledger: Ledger) -> bool:
    """Whether a ledger has anything to read: a reader makes no ledger directory where there is none."""
    return not isinstance(ledger, FileLedger) or ledger.directory.exists()
