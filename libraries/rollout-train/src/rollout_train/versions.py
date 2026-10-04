"""Versions of a model: each made from the ones before it, each saying where it came from.

A **version** is a node of a graph. It has an id of its own, which never changes and means the same thing in every
process and on every machine: it is what a channel serves and what a request to an engine names. It says what it was
made from (its `parents`: the version it was trained from first, then any others it learned from; none, the base model),
the base model it adapts, which run made it and at which step, where its weights are, what a trainer goes on from, and
what it was trained on. Its `depth` counts the steps from the base model along its first parents: the number stamped on
the tokens it samples, which grows along any line of training.

Nothing about a version is a name. A run trains from a version (or the base model) and goes on from the newest it
made; another run started from any version forks there. A **bookmark** (`rollout_train.registry`) is a name for a
version, which a run can carry forward as it trains; a version is otherwise found by its id (or the start of it), or
by the run and step that made it (`resolved`).

All versions are one append-only table of the ledger (`versions`), each appended under the fence of the run that
made it. Weights are kept as a `Manifest`: a map from the files of a checkpoint to blobs, each file its own blob, as
the trainer wrote them; whoever needs them reads the files it needs.
"""

import asyncio
import os
import secrets
import shutil
import tempfile
import time
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from pydantic import JsonValue, TypeAdapter

from rollout.contracts import BlobReference
from rollout.harness.blobs import Blobs
from rollout_train.ledger import Fence, Ledger

VERSIONS, RELEASED = "versions", "versions/released"
"""The ledger's tables of versions, and of the versions whose files were deleted."""
SHORTEST = 4
"""The fewest characters of an id a version is shown by."""
_ALPHABET = "klmnopqrstuvwxyz"
"""What a version's id is written in: sixteen letters, so that it reads as no word and no number."""


@dataclass(frozen=True)
class Manifest:
    """The files of a checkpoint, by their paths within it, each kept as a blob."""

    files: Mapping[str, BlobReference]
    layout: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """How the weights are divided among the files, where they are divided: whoever wrote them says, so that a
    reader with the same division reads its own files and no others."""


@dataclass(frozen=True)
class Version:
    id: str
    weights: Manifest | None
    """None once it was released (`Versions.thin`)."""
    parents: tuple[str, ...] = ()
    """What it was made from, by id: first the version it was trained from, then any others it learned from (the
    teachers of a distillation, say). None: from the base model."""
    depth: int = 1
    """Steps from the base model along its first parents: its first parent's depth and one."""
    base: str | None = None
    """The model it adapts, by name (`Qwen/Qwen3.5-9B`, say): its first parent's, or the one its line began from."""
    run: str | None = None
    """The run that made it, by id."""
    step: int | None = None
    """The run's step that made it (none for a version made outside a run's steps, such as by imitation)."""
    state: Manifest | None = None
    """What a trainer goes on from: the optimizer's state, say."""
    batch: BlobReference | None = None
    """What it was trained on: the segments, each as its source (`RUN/GROUP/EPISODE/SLOT/INDEX`) and its advantage."""
    metrics: Mapping[str, float] = field(default_factory=dict[str, float])
    made: float = 0.0
    """When, in seconds since the epoch."""
    released: float | None = None
    """When its files were deleted (`Versions.thin`), if they were: its weights and its trainer state are then None.
    Its record stays: where it came from, what it was trained on, and its metrics."""

    @property
    def parent(self) -> str | None:
        """The version it was trained from, if any."""
        return self.parents[0] if self.parents else None


_VERSION = TypeAdapter(Version)


def new_id() -> str:
    """A new version's id: sixteen random letters."""
    return "".join(secrets.choice(_ALPHABET) for _ in range(16))


def short(ids: Iterable[str]) -> dict[str, str]:
    """Each id by the shortest start of it (at least `SHORTEST` characters) that no other id begins with. An id of
    the form `NAME@N` (a version recorded before versions had ids of their own) is shown as `@N`, unless another id
    would be shown so too: then whole."""
    every = sorted(set(ids))
    shown: dict[str, str] = {}
    for index, each in enumerate(every):
        neighbours = every[max(0, index - 1) : index] + every[index + 1 : index + 2]
        shared = max((_common(each, other) for other in neighbours), default=0)
        shown[each] = each[: max(SHORTEST, shared + 1)]
    numbered = {each: f"@{each.rpartition('@')[2]}" for each in every if each.rpartition("@")[2].isdigit()}
    taken = [*numbered.values(), *(shown[each] for each in every if each not in numbered)]
    for each, label in numbered.items():
        shown[each] = label if taken.count(label) == 1 else each
    return shown


def _common(one: str, other: str) -> int:
    return next(
        (index for index, (a, b) in enumerate(zip(one, other, strict=False)) if a != b), min(len(one), len(other))
    )


class Versions:
    """Every version, in a ledger, and their files in a blob store."""

    def __init__(self, ledger: Ledger, blobs: Blobs) -> None:
        self.ledger = ledger
        self.blobs = blobs

    async def all(self) -> list[Version]:
        """Every version, oldest first."""
        return await versions_in(self.ledger)

    async def version(self, id: str) -> Version:
        """The version an id says."""
        found = (await self.ledger.read(VERSIONS)).get(id)
        if found is None:
            raise KeyError(f"there is no version {id}")
        return _as_released(_VERSION.validate_python(found), await self.ledger.read(RELEASED))

    async def head(self, run: str) -> Version | None:
        """The newest version a run made, if it made one."""
        made = [version for version in await self.all() if version.run == run]
        return max(made, key=lambda version: (version.depth, version.made)) if made else None

    async def add(
        self,
        fence: Fence,
        id: str,
        *,
        weights: Path,
        run: str | None,
        base: str | None = None,
        step: int | None = None,
        state: Path | None = None,
        parents: Sequence[str] = (),
        batch: BlobReference | None = None,
        metrics: Mapping[str, float] | None = None,
    ) -> Version:
        """Keep a checkpoint's files and append the version that names them, under `fence` (the run's that makes it).
        Its base is its first parent's; `base` names it for a version made from the base model. The append is what makes
        the version exist: a writer that dies before it has made nothing, and one that repeats it (the same id, decided
        before) gets the version that is there."""
        first = await self.version(parents[0]) if parents else None
        version = Version(
            id,
            weights=await kept(weights, self.blobs),
            parents=tuple(parents),
            depth=(first.depth if first else 0) + 1,
            base=first.base if first is not None and first.base else base,
            run=run,
            step=step,
            state=await kept(state, self.blobs) if state is not None else None,
            batch=batch,
            metrics=dict(metrics or {}),
            made=round(time.time(), 1),
        )
        record: Any = _VERSION.dump_python(version, mode="json")
        if not await self.ledger.append(VERSIONS, id, record, fence):
            return await self.version(id)
        return version

    async def thin(self, fence: Fence, run: str, retention: "Retention", keep: Collection[str] = ()) -> list[str]:
        """Delete the files (weights and trainer state) of the versions `run` made that `retention` does not keep,
        nor `keep` (what is served, what is bookmarked, what another run starts from), and return their ids. A
        release is appended to the ledger before its blobs are deleted, and a blob is deleted only if no version
        still names it, so this may be repeated after a crash at any point."""
        every = await self.all()
        ours = [version for version in every if version.run == run]
        kept_depths = retention.kept([version.depth for version in ours])
        released: list[str] = []
        for version in ours:
            if version.released is None and version.depth not in kept_depths and version.id not in keep:
                record: JsonValue = {"at": round(time.time(), 1)}
                await self.ledger.append(RELEASED, version.id, record, fence)
                released.append(version.id)
        records = await self.ledger.read(VERSIONS)
        remaining = await self.all()
        named = {blob.sha256 for version in remaining for manifest in (version.weights, version.state) if manifest
                 for blob in manifest.files.values()}  # fmt: skip
        for id in await self.ledger.read(RELEASED):
            record = records.get(id)
            made = _VERSION.validate_python(record) if record is not None else None
            for manifest in (made.weights, made.state) if made is not None else ():
                for reference in manifest.files.values() if manifest else ():
                    if reference.sha256 not in named:
                        await self.blobs.delete(reference)
        return released

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


@dataclass(frozen=True)
class Retention:
    """Which of a run's versions keep their files, their weights and their trainer state (what can be served, and
    what a step can go on from): the newest `recent`, and every `every`-th by depth, so that saves thin out with
    age."""

    recent: int = 2
    every: int = 20

    def kept(self, depths: list[int]) -> set[int]:
        newest = sorted(depths)[-self.recent :] if self.recent > 0 else []
        return {*newest, *(depth for depth in depths if self.every > 0 and depth % self.every == 0)}


async def versions_in(ledger: Ledger) -> list[Version]:
    """Every version a ledger has, oldest first (for a reader that has no use for their files)."""
    released = await ledger.read(RELEASED)
    return [
        _as_released(_VERSION.validate_python(record), released) for record in (await ledger.read(VERSIONS)).values()
    ]


def _as_released(version: Version, released: Mapping[str, JsonValue]) -> Version:
    """A version as it is once its files were deleted, if they were."""
    record: Any = released.get(version.id)
    return replace(version, weights=None, state=None, released=float(record["at"])) if record is not None else version


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
