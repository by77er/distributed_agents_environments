"""What runs and policies are called: each has an id that never changes, and a name that can be chosen and changed.

Everything a ledger keeps of a run or a policy is under its id (`runs/ID/...`, `policies/ID/...`, its versions
`ID@N`, the adapters engines load), so naming it again moves nothing. The registry beside the ledger holds each one's
id and name: a file beside a ledger of files (`FileRegistry`), a table in a database ledger's database
(`rollout_train.database.DatabaseRegistry`). A name is one no other of its kind has, as its name or as its id, so
that either finds one thing; it says neither `/` nor `@` (a version is called `NAME@N`).

A run's directory says which run it is (`run.json`); `run_of` finds it, or registers the run the first time it is
started. `policy_of` does the same for the policy a profile names. A run or policy recorded before there was a
registry keeps its key as its id, and is registered under it as its name the first time it is asked for.
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
from typing import Protocol

from rollout.contracts import new_ulid
from rollout_train.layout import RUN
from rollout_train.ledger import FileLedger, Ledger
from rollout_train.policies import policies_in
from rollout_train.record import runs_in

RUNS, POLICIES = "run", "policy"
KINDS = (RUNS, POLICIES)


@dataclass(frozen=True)
class Entry:
    kind: str
    id: str
    name: str
    created: float


class Taken(ValueError):
    """A name that cannot be given: another of its kind has it, or it is no name."""


class Registry(Protocol):
    async def entries(self, kind: str) -> list[Entry]:
        """Every run (or policy) registered, oldest first."""
        ...

    async def create(self, kind: str, name: str, id: str | None = None) -> Entry:
        """Register a run (or policy) under `name`, with `id` (a new one by default). Raises `Taken`."""
        ...

    async def rename(self, kind: str, who: str, name: str) -> Entry:
        """Call the run (or policy) that `who` is (its name or its id) `name` from now on. Raises `Taken`, and
        `KeyError` when there is no such one."""
        ...


async def find(registry: Registry, kind: str, who: str) -> Entry | None:
    """The run (or policy) that `who` is: its name, or its id."""
    entries = await registry.entries(kind)
    return next((each for each in entries if each.name == who), None) or next(
        (each for each in entries if each.id == who), None
    )


def new_id(kind: str) -> str:
    """A new run's (or policy's) id: its kind and a ULID (`run_01K...`)."""
    return f"{kind}_{new_ulid()}"


def checked(kind: str, name: str, id: str, entries: list[Entry]) -> str:
    """`name`, stripped, if the run (or policy) `id` can be called it among `entries`; else raises `Taken`."""
    if kind not in KINDS:
        raise ValueError(f"{kind!r} is not a kind that is named: {' or '.join(KINDS)}")
    name = name.strip()
    if not name or "/" in name or "@" in name:
        raise Taken(f"{name!r} cannot be a name: it must say something, and neither '/' nor '@'")
    if any(each.id != id and name in (each.id, each.name) for each in entries):
        raise Taken(f"another {kind} is called {name!r}")
    return name


def registry_of(ledger: Ledger) -> Registry | None:
    """The registry beside a ledger: a file beside a ledger of files, a table in a database ledger's database."""
    if isinstance(ledger, FileLedger):
        return FileRegistry(ledger.directory)
    return getattr(ledger, "registry", None)


class FileRegistry:
    """A `Registry` in `registry.json` in a ledger's directory, under the lock the ledger's files are written under."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path = directory / "registry.json"

    async def entries(self, kind: str) -> list[Entry]:
        return await asyncio.to_thread(lambda: [each for each in self._read() if each.kind == kind])

    async def create(self, kind: str, name: str, id: str | None = None) -> Entry:
        def created() -> Entry:
            with self._locked():
                entries = self._read()
                made = id or new_id(kind)
                if any(each.kind == kind and each.id == made for each in entries):
                    raise Taken(f"there is a {kind} {made} already")
                ours = [each for each in entries if each.kind == kind]
                entry = Entry(kind, made, checked(kind, name, made, ours), round(time.time(), 1))
                self._write([*entries, entry])
                return entry

        return await asyncio.to_thread(created)

    async def rename(self, kind: str, who: str, name: str) -> Entry:
        def renamed() -> Entry:
            with self._locked():
                entries = self._read()
                ours = [each for each in entries if each.kind == kind]
                found = next((e for e in ours if e.name == who), None) or next((e for e in ours if e.id == who), None)
                if found is None:
                    raise KeyError(f"there is no {kind} {who!r}")
                entry = Entry(kind, found.id, checked(kind, name, found.id, ours), found.created)
                self._write([entry if each == found else each for each in entries])
                return entry

        return await asyncio.to_thread(renamed)

    def _read(self) -> list[Entry]:
        return [Entry(**each) for each in json.loads(self.path.read_text())] if self.path.exists() else []

    def _write(self, entries: list[Entry]) -> None:
        staged = self.path.with_suffix(".staged")
        staged.write_text(json.dumps([asdict(each) for each in entries], indent=1))
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
    under `name` (by default the directory's name), which `run.json` then names. A run recorded before there was a
    registry, under the directory's name, keeps that as its id. Without a registry, the run is the directory's name."""
    path = directory / RUN
    if await asyncio.to_thread(path.exists):
        id = str(json.loads(await asyncio.to_thread(path.read_text))["id"])
        entry = next((each for each in await registry.entries(RUNS) if each.id == id), None) if registry else None
        return entry or await _registered(registry, RUNS, id, name or directory.name)
    if registry is None:
        return Entry(RUNS, directory.name, directory.name, 0.0)
    if directory.name in await runs_in(ledger):  # (recorded before there was a registry)
        entry = await _registered(registry, RUNS, directory.name, directory.name)
    else:
        entry = await registry.create(RUNS, name or directory.name)

    def noted() -> None:
        directory.mkdir(parents=True, exist_ok=True)
        staged = path.with_suffix(".staged")
        staged.write_text(json.dumps({"id": entry.id}))
        os.replace(staged, path)

    await asyncio.to_thread(noted)
    return entry


async def policy_of(ledger: Ledger, registry: Registry | None, who: str) -> Entry:
    """The policy `who` is (its name or its id); one that is not registered is registered under `who` as its name,
    keeping `who` as its id if the ledger has versions of a policy by that key (recorded before there was a
    registry). Without a registry, the policy is `who`."""
    if registry is None:
        return Entry(POLICIES, who, who, 0.0)
    if (found := await find(registry, POLICIES, who)) is not None:
        return found
    if who in await policies_in(ledger):
        return await _registered(registry, POLICIES, who, who)
    return await registry.create(POLICIES, who)


async def _registered(registry: Registry | None, kind: str, id: str, name: str) -> Entry:
    """The entry of `id`, registered under `name` if it is not yet (or, if that name is another's, under its id)."""
    if registry is None:
        return Entry(kind, id, name, 0.0)
    if (found := next((each for each in await registry.entries(kind) if each.id == id), None)) is not None:
        return found
    try:
        return await registry.create(kind, name, id)
    except Taken:
        return await registry.create(kind, id, id)


async def names(registry: Registry | None) -> dict[str, dict[str, str]]:
    """Every registered run's and policy's name, by kind and id."""
    if registry is None:
        return {kind: {} for kind in KINDS}
    return {kind: {each.id: each.name for each in await registry.entries(kind)} for kind in KINDS}
