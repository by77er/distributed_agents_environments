"""What runs and checkpoints are called: a run's name, which can be chosen and changed, and bookmarks, names for
checkpoints.

Everything a ledger keeps of a run is under its id (`runs/ID/...`), and a checkpoint is its id
(`rollout_train.checkpoints`), so naming either moves nothing. The registry beside the ledger holds the names: a file
beside a ledger of files (`FileRegistry`), tables in a database ledger's database
(`rollout_train.database.DatabaseRegistry`). It is ordinary state, changed in place.

- A **run** has an id and a name. A name is one no other run has, as its name or as its id, so that either finds one
  run. A run's directory says which run it is (`run.json`); `run_of` finds it, or registers the run the first time it
  is started.
- A **bookmark** names a checkpoint, and is moved to another by whoever moves it: a run told to carry one moves it to
  each checkpoint it makes. A checkpoint needs none: it is shown by where it came from.
- A **dataset's name** (`rollout_train.datasets`) names one dataset, for good: a dataset is never changed, so neither
  is what its name says. A dataset needs none: it is found by its id.
- A **suite's name** (`rollout_train.evals`) points to one version of the suite, by its id (`NAME@NUMBER`): the newest,
  moved there by each edit. A version is never changed; the name moves. A suite never edited needs none: its name is
  its version 1.

A name says neither `/`, `@` nor `:` (they are what a reference to a checkpoint is made of: `resolved`).
"""

import asyncio
import fcntl
import json
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from rollout.contracts import new_ulid
from rollout_train.checkpoints import SHORTEST, checkpoints_in
from rollout_train.layout import RUN
from rollout_train.ledger import FileLedger, Ledger
from rollout_train.record import STEPS, table

BASE = "base"
"""The reference to the base model: no checkpoint."""


@dataclass(frozen=True)
class Entry:
    """A run: its id, and what it is called."""

    id: str
    name: str
    created: float


@dataclass(frozen=True)
class Bookmark:
    name: str
    checkpoint: str
    """By id."""
    moved: float


@dataclass(frozen=True)
class Named:
    """A dataset's name."""

    name: str
    dataset: str
    """By id."""
    named: float


@dataclass(frozen=True)
class SuiteName:
    """A suite's name, and the version it points to."""

    name: str
    version: str
    """By id: `NAME@NUMBER`."""
    moved: float


def version_number(version: str) -> int:
    """The number of a suite's version, from its id (`NAME@NUMBER`); 0 for an id that says none."""
    number = version.rpartition("@")[2]
    return int(number) if number.isdigit() else 0


class Taken(ValueError):
    """A name that cannot be given: another has it, or it is no name."""


class Registry(Protocol):
    async def runs(self) -> list[Entry]:
        """Every run registered, oldest first."""
        ...

    async def create(self, name: str, id: str | None = None) -> Entry:
        """Register a run under `name`, with `id` (a new one by default). Raises `Taken`."""
        ...

    async def rename(self, who: str, name: str) -> Entry:
        """Call the run that `who` is (its name or its id) `name` from now on. Raises `Taken`, and `KeyError` when
        there is no such run."""
        ...

    async def bookmarks(self) -> list[Bookmark]:
        """Every bookmark, by name."""
        ...

    async def bookmark(self, name: str, checkpoint: str) -> Bookmark:
        """Make a bookmark name `checkpoint`, or move it there. Raises `Taken` for a name that cannot be one."""
        ...

    async def unbookmark(self, name: str) -> None:
        """Take a bookmark away (the checkpoint stays). Raises `KeyError` when there is no such bookmark."""
        ...

    async def datasets(self) -> list[Named]:
        """Every dataset's name, by name."""
        ...

    async def name_dataset(self, name: str, dataset: str) -> Named:
        """Call a dataset (by id) `name`. Raises `Taken` for a name that cannot be one, or that another dataset has."""
        ...

    async def suites(self) -> list[SuiteName]:
        """Every suite's name that points to a version, by name."""
        ...

    async def point_suite(self, name: str, version: str, *, forward: bool = False) -> SuiteName:
        """Point a suite's name to one of its versions (by id). With `forward`, only to a later version than the one it
        points to (as edits move it: two at once leave it at the newer); otherwise the name stays as it is."""
        ...


def new_run_id() -> str:
    """A new run's id: `run_` and a ULID."""
    return f"run_{new_ulid()}"


def valid(name: str) -> str:
    """`name`, stripped, if it can be a name; else raises `Taken`."""
    name = name.strip()
    if not name or any(mark in name for mark in "/@:") or name == BASE:
        raise Taken(f"{name!r} cannot be a name: it must say something, and none of '/', '@', ':' (nor be {BASE!r})")
    return name


def checked(name: str, id: str, runs: list[Entry]) -> str:
    """`name`, stripped, if the run `id` can be called it among `runs`; else raises `Taken`."""
    name = valid(name)
    if any(each.id != id and name in (each.id, each.name) for each in runs):
        raise Taken(f"another run is called {name!r}")
    return name


def found(runs: list[Entry], who: str) -> Entry | None:
    """The run that `who` is: its name, or its id."""
    return next((each for each in runs if each.name == who), None) or next(
        (each for each in runs if each.id == who), None
    )


def registry_of(ledger: Ledger) -> Registry | None:
    """The registry beside a ledger: a file beside a ledger of files, tables in a database ledger's database."""
    if isinstance(ledger, FileLedger):
        return FileRegistry(ledger.directory)
    return getattr(ledger, "registry", None)


class FileRegistry:
    """A `Registry` in `registry.json` in a ledger's directory, under the lock the ledger's files are written under."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path = directory / "registry.json"

    async def runs(self) -> list[Entry]:
        return await asyncio.to_thread(lambda: self._read()[0])

    async def create(self, name: str, id: str | None = None) -> Entry:
        def created() -> Entry:
            with self._locked():
                runs, marks = self._read()
                made = id or new_run_id()
                if any(each.id == made for each in runs):
                    raise Taken(f"there is a run {made} already")
                entry = Entry(made, checked(name, made, runs), round(time.time(), 1))
                self._write([*runs, entry], marks)
                return entry

        return await asyncio.to_thread(created)

    async def rename(self, who: str, name: str) -> Entry:
        def renamed() -> Entry:
            with self._locked():
                runs, marks = self._read()
                if (was := found(runs, who)) is None:
                    raise KeyError(f"there is no run {who!r}")
                entry = Entry(was.id, checked(name, was.id, runs), was.created)
                self._write([entry if each == was else each for each in runs], marks)
                return entry

        return await asyncio.to_thread(renamed)

    async def bookmarks(self) -> list[Bookmark]:
        return await asyncio.to_thread(lambda: sorted(self._read()[1], key=lambda mark: mark.name))

    async def bookmark(self, name: str, checkpoint: str) -> Bookmark:
        def moved() -> Bookmark:
            with self._locked():
                runs, marks = self._read()
                mark = Bookmark(valid(name), checkpoint, round(time.time(), 1))
                self._write(runs, [each for each in marks if each.name != mark.name] + [mark])
                return mark

        return await asyncio.to_thread(moved)

    async def unbookmark(self, name: str) -> None:
        def taken() -> None:
            with self._locked():
                runs, marks = self._read()
                if not any(each.name == name for each in marks):
                    raise KeyError(f"there is no bookmark {name!r}")
                self._write(runs, [each for each in marks if each.name != name])

        await asyncio.to_thread(taken)

    async def datasets(self) -> list[Named]:
        return await asyncio.to_thread(lambda: sorted(self._named(), key=lambda each: each.name))

    async def name_dataset(self, name: str, dataset: str) -> Named:
        def given() -> Named:
            with self._locked():
                runs, marks = self._read()
                named = self._named()
                entry = Named(valid(name), dataset, round(time.time(), 1))
                if any(each.name == entry.name and each.dataset != dataset for each in named):
                    raise Taken(f"another dataset is called {entry.name!r}")
                self._write(runs, marks, [each for each in named if each.name != entry.name] + [entry])
                return entry

        return await asyncio.to_thread(given)

    async def suites(self) -> list[SuiteName]:
        return await asyncio.to_thread(lambda: sorted(self._suite_names(), key=lambda each: each.name))

    async def point_suite(self, name: str, version: str, *, forward: bool = False) -> SuiteName:
        def pointed() -> SuiteName:
            with self._locked():
                runs, marks = self._read()
                every = self._suite_names()
                was = next((each for each in every if each.name == name), None)
                if forward and was is not None and version_number(was.version) >= version_number(version):
                    return was
                entry = SuiteName(valid(name), version, round(time.time(), 1))
                self._write(runs, marks, suites=[each for each in every if each.name != entry.name] + [entry])
                return entry

        return await asyncio.to_thread(pointed)

    def _read(self) -> tuple[list[Entry], list[Bookmark]]:
        if not self.path.exists():
            return [], []
        kept: Any = json.loads(self.path.read_text())
        return [Entry(**each) for each in kept["runs"]], [Bookmark(**each) for each in kept["bookmarks"]]

    def _kept(self, part: str) -> list[Any]:
        kept: Any = json.loads(self.path.read_text()) if self.path.exists() else {}
        return list(kept.get(part, []))

    def _named(self) -> list[Named]:
        return [Named(**each) for each in self._kept("datasets")]

    def _suite_names(self) -> list[SuiteName]:
        return [SuiteName(**each) for each in self._kept("suites")]

    def _write(
        self,
        runs: list[Entry],
        marks: list[Bookmark],
        named: list[Named] | None = None,
        *,
        suites: list[SuiteName] | None = None,
    ) -> None:
        staged = self.path.with_suffix(".staged")
        datasets = self._named() if named is None else named
        pointers = self._suite_names() if suites is None else suites
        kept: dict[str, Any] = {"runs": [asdict(each) for each in runs], "bookmarks": [asdict(each) for each in marks]}
        if datasets:
            kept["datasets"] = [asdict(each) for each in datasets]
        if pointers:
            kept["suites"] = [asdict(each) for each in pointers]
        staged.write_text(json.dumps(kept, indent=1))
        staged.replace(self.path)

    @contextmanager
    def _locked(self) -> Generator[None]:
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


async def run_of(directory: Path, ledger: Ledger, registry: Registry | None, name: str | None = None) -> Entry:
    """The run in `directory`: the one its `run.json` names; else, the first time it is started, a run registered
    under `name` (by default the directory's name), which `run.json` then names. Without a registry, the run is the
    directory's name."""
    path = directory / RUN
    if await asyncio.to_thread(path.exists):
        id = str(json.loads(await asyncio.to_thread(path.read_text))["id"])
        return await _registered(registry, id, name or directory.name)
    if registry is None:
        return Entry(directory.name, directory.name, 0.0)
    entry = await registry.create(name or directory.name)

    def noted() -> None:
        directory.mkdir(parents=True, exist_ok=True)
        staged = path.with_suffix(".staged")
        staged.write_text(json.dumps({"id": entry.id}))
        os.replace(staged, path)

    await asyncio.to_thread(noted)
    return entry


async def _registered(registry: Registry | None, id: str, name: str) -> Entry:
    """The run `id`, registered under `name` if it is not yet (or, if that name is another's, under its id)."""
    if registry is None:
        return Entry(id, name, 0.0)
    if (entry := next((each for each in await registry.runs() if each.id == id), None)) is not None:
        return entry
    try:
        return await registry.create(name, id)
    except Taken:
        return await registry.create(id, id)


async def resolved(ledger: Ledger, registry: Registry | None, reference: str) -> str | None:
    """The checkpoint a reference says, by id; None for `base` (the base model). A reference is, in this order:
    `base`; a bookmark's name; `RUN:STEP`, the checkpoint a run (by its name or its id) made at a step; `RUN`, the
    newest checkpoint a run made; or a checkpoint's id, or the start of one (at least `SHORTEST` characters) that no
    other id begins with. Raises `KeyError` for one that says no checkpoint, or more than one."""
    if reference == BASE:
        return None
    marks = {mark.name: mark.checkpoint for mark in await registry.bookmarks()} if registry else {}
    if reference in marks:
        return marks[reference]
    runs = await registry.runs() if registry else []
    who, _, step = reference.partition(":")
    run = found(runs, who)
    checkpoints = await checkpoints_in(ledger)
    if run is not None:
        if step:
            intent: Any = (await ledger.read(table(run.id, STEPS))).get(step)
            made = str(intent.get("makes")) if intent else None
            if made is None or not any(checkpoint.id == made for checkpoint in checkpoints):
                raise KeyError(f"the run {who!r} made no checkpoint at step {step}")
            return made
        ours = [checkpoint for checkpoint in checkpoints if checkpoint.run == run.id]
        if not ours:
            raise KeyError(f"the run {who!r} has made no checkpoint")
        return max(ours, key=lambda checkpoint: (checkpoint.depth, checkpoint.made)).id
    if any(checkpoint.id == reference for checkpoint in checkpoints):
        return reference
    starting = [checkpoint.id for checkpoint in checkpoints if checkpoint.id.startswith(reference)]
    if len(reference) >= SHORTEST and len(starting) == 1:
        return starting[0]
    if len(starting) == 1:
        raise KeyError(f"{reference!r} is too short to say a checkpoint: at least {SHORTEST} characters")
    raise KeyError(f"{reference!r} says {'more than one checkpoint' if starting else 'no checkpoint'}")


async def names(registry: Registry | None) -> dict[str, Any]:
    """Every registered run's name, by id, every bookmark's checkpoint, by name, and the version each suite's name
    points to (by id), by name."""
    if registry is None:
        return {"runs": {}, "bookmarks": {}, "suites": {}}
    return {
        "runs": {each.id: each.name for each in await registry.runs()},
        "bookmarks": {mark.name: mark.checkpoint for mark in await registry.bookmarks()},
        "suites": {each.name: each.version for each in await registry.suites()},
    }
