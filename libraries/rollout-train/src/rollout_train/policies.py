"""Policies and their versions: what is being trained, apart from the run that trains it.

A **policy** is one line of training, by name. Its **versions** are an append-only table: each says which version it
came from (in this policy or another: a fork), where its weights are, what the trainer goes on from, and what it was
trained on. A version's name, `policy@number`, means the same thing in every process and on every machine: it is
what a request to an engine names, and what is stamped on the tokens a policy samples.

Weights are kept as a `Manifest`: a map from the files of a checkpoint to blobs, each file its own blob, as the
trainer wrote them. Whoever needs them reads the files it needs; nothing is packed, and nothing is gathered in one
place to be divided again.
"""

import asyncio
import os
import shutil
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import JsonValue, TypeAdapter

from rollout.contracts import BlobReference
from rollout.harness.blobs import Blobs
from rollout_train.ledger import Fence, Ledger, between


@dataclass(frozen=True)
class Manifest:
    """The files of a checkpoint, by their paths within it, each kept as a blob."""

    files: Mapping[str, BlobReference]
    layout: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """How the weights are divided among the files, where they are divided: whoever wrote them says, so that a
    reader with the same division reads its own files and no others."""


@dataclass(frozen=True)
class Version:
    policy: str
    number: int
    """From 1, in the order the policy's versions were made."""
    weights: Manifest
    parent: str | None = None
    """The version it was trained from, by name: of this policy, or of another (a fork). None: from the base."""
    state: Manifest | None = None
    """What a trainer goes on from: the optimizer's state, say."""
    batch: BlobReference | None = None
    """What it was trained on: the segments, each as its place in a job's log and its advantage."""
    metrics: Mapping[str, float] = field(default_factory=dict[str, float])
    made: float = 0.0
    """When, in seconds since the epoch."""

    @property
    def name(self) -> str:
        return named(self.policy, self.number)


def named(policy: str, number: int) -> str:
    """A version's name: `policy@number`."""
    return f"{policy}@{number}"


def parsed(name: str) -> tuple[str, int]:
    """The policy and number a version's name says."""
    policy, _, number = name.rpartition("@")
    if not policy or not number.isdigit():
        raise ValueError(f"{name!r} is not a version's name: it should be policy@number")
    return policy, int(number)


_VERSION = TypeAdapter(Version)


class Policies:
    """The versions of every policy, in a ledger, and their files in a blob store."""

    def __init__(self, ledger: Ledger, blobs: Blobs) -> None:
        self.ledger = ledger
        self.blobs = blobs

    async def writer(self, policy: str) -> Fence:
        """Become the one that may add versions to a policy: whoever was is shut out."""
        return await self.ledger.take(scope(policy))

    async def versions(self, policy: str) -> list[Version]:
        """A policy's versions, oldest first."""
        return await versions_in(self.ledger, policy)

    async def head(self, policy: str) -> Version | None:
        """A policy's newest version, if it has one."""
        versions = await self.versions(policy)
        return versions[-1] if versions else None

    async def version(self, name: str) -> Version:
        """The version a name says."""
        policy, number = parsed(name)
        record = (await self.ledger.read(_table(policy))).get(str(number))
        if record is None:
            raise KeyError(f"there is no version {name}")
        return _VERSION.validate_python(record)

    async def add(
        self,
        fence: Fence,
        policy: str,
        number: int,
        *,
        weights: Path,
        state: Path | None = None,
        parent: str | None = None,
        batch: BlobReference | None = None,
        metrics: Mapping[str, float] | None = None,
    ) -> Version:
        """Keep a checkpoint's files and append the version that names them. The append is what makes the version
        exist: a writer that dies before it has made nothing, and one that repeats it (the same number) gets the
        version that is there."""
        version = Version(
            policy,
            number,
            weights=await kept(weights, self.blobs),
            parent=parent,
            state=await kept(state, self.blobs) if state is not None else None,
            batch=batch,
            metrics=dict(metrics or {}),
            made=round(time.time(), 1),
        )
        record: Any = _VERSION.dump_python(version, mode="json")
        if not await self.ledger.append(_table(policy), str(number), record, fence):
            return await self.version(named(policy, number))
        return version

    async def files(self, manifest: Manifest, directory: Path) -> Path:
        """A manifest's files under `directory`, read from the blob store if they are not there. The directory
        appears whole or not at all, so whatever looks for a file in it never finds half a checkpoint."""
        if await asyncio.to_thread(directory.exists):
            return directory
        contents = {relative: await self.blobs.read(reference) for relative, reference in manifest.files.items()}
        await asyncio.to_thread(_written, contents, directory)
        return directory


def _written(contents: Mapping[str, bytes], directory: Path) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=directory.parent, prefix=".fetching-"))
    try:
        for relative, data in contents.items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        os.replace(staging, directory)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def scope(policy: str) -> str:
    """The scope whose fence a policy's writer holds."""
    return _POLICIES + policy


def _table(policy: str) -> str:
    return scope(policy) + _VERSIONS


_POLICIES, _VERSIONS = "policies/", "/versions"


async def versions_in(ledger: Ledger, policy: str) -> list[Version]:
    """A policy's versions as a ledger has them, oldest first (for a reader that has no use for their files)."""
    records = await ledger.read(_table(policy))
    return [_VERSION.validate_python(record) for record in records.values()]


async def policies_in(ledger: Ledger) -> list[str]:
    """The policies a ledger has versions of, by name."""
    return [policy for table in await ledger.tables() if (policy := between(table, _POLICIES, _VERSIONS))]


async def kept(path: Path, blobs: Blobs) -> Manifest:
    """Keep a file, or every file under a directory, in `blobs`, each as a blob of its own."""
    files: dict[str, BlobReference] = {}
    for relative, each in await asyncio.to_thread(_listed, path):
        files[relative] = await blobs.put(await asyncio.to_thread(each.read_bytes), "application/octet-stream")
    return Manifest(files)


def _listed(path: Path) -> list[tuple[str, Path]]:
    if path.is_file():
        return [(path.name, path)]
    return [(str(each.relative_to(path)), each) for each in sorted(path.rglob("*")) if each.is_file()]
