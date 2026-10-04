"""Checkpoints of a model: each made from the ones before it, each saying where it came from.

A **checkpoint** is a node of a graph. It has an id of its own, which never changes and means the same thing in every
process and on every machine: it is what a channel serves and what a request to an engine names. It says what it was
made from (its `parents`: the checkpoint it was trained from first, then any others it learned from; none, the base
model), the base model it adapts, which run made it and at which step, where its weights are, what a trainer goes on
from, and what it was trained on. Its `depth` counts the steps from the base model along its first parents: the number
stamped on the tokens it samples, which grows along any line of training.

Nothing about a checkpoint is a name. A run trains from a checkpoint (or the base model) and goes on from the newest it
made; another run started from any checkpoint forks there. A **bookmark** (`rollout_train.registry`) is a name for a
checkpoint, which a run can carry forward as it trains; a checkpoint is otherwise found by its id (or the start of it),
or by the run and step that made it (`resolved`).

All checkpoints are one append-only table of the ledger (`checkpoints`), each appended under the fence of the run that
made it. Weights are kept as a `Manifest`: a map from the files of a checkpoint to blobs, each file its own blob, as
the trainer wrote them; whoever needs them reads the files it needs.
"""

import asyncio
import json
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
from rollout_train.stores import opened

CHECKPOINTS, RELEASED = "checkpoints", "checkpoints/released"
"""The ledger's tables of checkpoints, and of the checkpoints whose files were deleted."""
SHORTEST = 4
"""The fewest characters of an id a checkpoint is shown by."""
_ALPHABET = "klmnopqrstuvwxyz"
"""What a checkpoint's id is written in: sixteen letters, so that it reads as no word and no number."""


@dataclass(frozen=True)
class Manifest:
    """The files of a checkpoint, by their paths within it, each kept as a blob."""

    files: Mapping[str, BlobReference]
    layout: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """How the weights are divided among the files, where they are divided: whoever wrote them says, so that a
    reader with the same division reads its own files and no others."""


@dataclass(frozen=True)
class Checkpoint:
    id: str
    weights: Manifest | None
    """None once it was released (`Checkpoints.thin`)."""
    parents: tuple[str, ...] = ()
    """What it was made from, by id: first the checkpoint it was trained from, then any others it learned from (the
    teachers of a distillation, say). None: from the base model."""
    depth: int = 1
    """Steps from the base model along its first parents: its first parent's depth and one."""
    base: str | None = None
    """What its weights build on: the model its line began from, by name (`Qwen/Qwen3.5-9B`, say); or, for an
    adapter trained over a full checkpoint, that checkpoint, by id."""
    kind: str = "lora"
    """What its weights are: `lora` (an adapter over its base) or `full` (all of a model's weights)."""
    run: str | None = None
    """The run that made it, by id."""
    step: int | None = None
    """The run's step that made it (none for a checkpoint made outside a run's steps, such as by imitation)."""
    state: Manifest | None = None
    """What a trainer goes on from: the optimizer's state, say."""
    batch: BlobReference | None = None
    """What it was trained on: the segments, each as its source (`RUN/GROUP/EPISODE/SLOT/INDEX`) and its advantage."""
    metrics: Mapping[str, float] = field(default_factory=dict[str, float])
    made: float = 0.0
    """When, in seconds since the epoch."""
    released: float | None = None
    """When its files were deleted (`Checkpoints.thin`), if they were: its weights and its trainer state are then None.
    Its record stays: where it came from, what it was trained on, and its metrics."""

    @property
    def parent(self) -> str | None:
        """The checkpoint it was trained from, if any."""
        return self.parents[0] if self.parents else None


_VERSION = TypeAdapter(Checkpoint)


def new_id() -> str:
    """A new checkpoint's id: sixteen random letters."""
    return "".join(secrets.choice(_ALPHABET) for _ in range(16))


def short(ids: Iterable[str]) -> dict[str, str]:
    """Each id by the shortest start of it (at least `SHORTEST` characters) that no other id begins with. An id of
    the form `NAME@N` (a checkpoint recorded before checkpoints had ids of their own) is shown as `@N`, unless another
    id would be shown so too: then whole."""
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


class Checkpoints:
    """Every checkpoint, in a ledger, and their files in a blob store."""

    def __init__(self, ledger: Ledger, blobs: Blobs) -> None:
        self.ledger = ledger
        self.blobs = blobs
        self._stores: list[Blobs] | None = None

    async def all(self) -> list[Checkpoint]:
        """Every checkpoint, oldest first."""
        return await checkpoints_in(self.ledger)

    async def checkpoint(self, id: str) -> Checkpoint:
        """The checkpoint an id says."""
        found = (await self.ledger.read(CHECKPOINTS)).get(id)
        if found is None:
            raise KeyError(f"there is no checkpoint {id}")
        return _as_released(_VERSION.validate_python(found), await self.ledger.read(RELEASED))

    async def under(self, checkpoint: Checkpoint) -> Checkpoint | None:
        """The full checkpoint whose weights `checkpoint` is served over: itself, if it is full; the full checkpoint it
        builds on, if it is an adapter over one; None for an adapter over a model. Raises `ValueError` if that
        checkpoint was released."""
        found = checkpoint
        if checkpoint.kind != "full":
            if checkpoint.base is None or checkpoint.base not in await self.ledger.read(CHECKPOINTS):
                return None
            found = await self.checkpoint(checkpoint.base)
        if found.weights is None:
            raise ValueError(f"{found.id} was released: its weights were deleted")
        return found

    async def head(self, run: str) -> Checkpoint | None:
        """The newest checkpoint a run made, if it made one."""
        made = [checkpoint for checkpoint in await self.all() if checkpoint.run == run]
        return max(made, key=lambda checkpoint: (checkpoint.depth, checkpoint.made)) if made else None

    async def add(
        self,
        fence: Fence,
        id: str,
        *,
        weights: Path,
        run: str | None,
        base: str | None = None,
        kind: str = "lora",
        step: int | None = None,
        state: Path | None = None,
        parents: Sequence[str] = (),
        batch: BlobReference | None = None,
        metrics: Mapping[str, float] | None = None,
    ) -> Checkpoint:
        """Keep a checkpoint's files and append the checkpoint that names them, under `fence` (the run's that makes it).
        Its base is what its weights build on (`_base`): for an adapter over a full checkpoint, that checkpoint (by
        id); for a merge (full weights from an adapter), the `base` it names; else its first parent's base, or `base`
        for a checkpoint made from the base model. The append is what
        makes the checkpoint exist: a writer that dies before it has made nothing, and one that repeats it (the same id,
        decided before) gets the checkpoint that is there."""
        first = await self.checkpoint(parents[0]) if parents else None
        checkpoint = Checkpoint(
            id,
            weights=await kept(weights, self.blobs),
            parents=tuple(parents),
            depth=(first.depth if first else 0) + 1,
            base=_base(first, kind, base),
            kind=kind,
            run=run,
            step=step,
            state=await kept(state, self.blobs) if state is not None else None,
            batch=batch,
            metrics=dict(metrics or {}),
            made=round(time.time(), 1),
        )
        record: Any = _VERSION.dump_python(checkpoint, mode="json")
        if not await self.ledger.append(CHECKPOINTS, id, record, fence):
            return await self.checkpoint(id)
        return checkpoint

    async def thin(self, fence: Fence, run: str, retention: "Retention", keep: Collection[str] = ()) -> list[str]:
        """Delete the files (weights and trainer state) of the checkpoints `run` made that `retention` does not keep,
        nor `keep` (what is served, what is bookmarked, what another run starts from), and return their ids. A
        release is appended to the ledger before its blobs are deleted, and a blob is deleted only if no checkpoint
        still names it, so this may be repeated after a crash at any point."""
        every = await self.all()
        ours = [checkpoint for checkpoint in every if checkpoint.run == run]
        kept_depths = retention.kept([checkpoint.depth for checkpoint in ours])
        released: list[str] = []
        for checkpoint in ours:
            if checkpoint.released is None and checkpoint.depth not in kept_depths and checkpoint.id not in keep:
                record: JsonValue = {"at": round(time.time(), 1)}
                await self.ledger.append(RELEASED, checkpoint.id, record, fence)
                released.append(checkpoint.id)
        records = await self.ledger.read(CHECKPOINTS)
        remaining = await self.all()
        named = {blob.sha256 for checkpoint in remaining for manifest in (checkpoint.weights, checkpoint.state) if
        manifest
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
        appears whole or not at all, so whatever looks for a file in it never finds half a checkpoint. A file this
        store lacks is read from the store of any run that has it (a checkpoint made by a run that kept its blobs
        elsewhere, or a merge of one), as each run's start says where its store is."""
        if await asyncio.to_thread(directory.exists):
            return directory
        contents = {relative: await self._read(reference) for relative, reference in manifest.files.items()}
        await asyncio.to_thread(_written, contents, directory)
        return directory

    async def _read(self, reference: BlobReference) -> bytes:
        try:
            return await self.blobs.read(reference)
        except Exception:
            for store in await self._elsewhere():
                try:
                    return await store.read(reference)
                except Exception:  # (not in that one either)
                    continue
            raise

    async def _elsewhere(self) -> list[Blobs]:
        """The other runs' blob stores, from where their starts say they are."""
        from rollout_train.record import STARTS  # (record reads checkpoints)

        if self._stores is None:
            where: dict[str, Any] = {}
            for name in await self.ledger.tables():
                if name.startswith("runs/") and name.endswith(f"/{STARTS}"):
                    for record in (await self.ledger.read(name)).values():
                        kept = record.get("blobs") if isinstance(record, dict) else None
                        if isinstance(kept, dict):
                            where[json.dumps(kept, sort_keys=True)] = kept
            self._stores = [opened(each) for each in where.values()]
        return self._stores


def _base(first: Checkpoint | None, kind: str, base: str | None) -> str | None:
    if first is None:
        return base
    if first.kind == "full" and kind == "lora":  # an adapter over full weights builds on those
        return first.id
    if first.kind == "lora" and kind == "full":  # a merge: the model its line began from, as it names it
        return base or first.base
    return first.base or base


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
    """Which of a run's checkpoints keep their files, their weights and their trainer state (what can be served, and
    what a step can go on from): the newest `recent`, and every `every`-th by depth, so that saves thin out with
    age."""

    recent: int = 2
    every: int = 20

    def kept(self, depths: list[int]) -> set[int]:
        newest = sorted(depths)[-self.recent :] if self.recent > 0 else []
        return {*newest, *(depth for depth in depths if self.every > 0 and depth % self.every == 0)}


async def checkpoints_in(ledger: Ledger) -> list[Checkpoint]:
    """Every checkpoint a ledger has, oldest first (for a reader that has no use for their files)."""
    released = await ledger.read(RELEASED)
    return [
        _as_released(_VERSION.validate_python(record), released) for record in (await ledger.read(CHECKPOINTS)).values()
    ]


def _as_released(checkpoint: Checkpoint, released: Mapping[str, JsonValue]) -> Checkpoint:
    """A checkpoint as it is once its files were deleted, if they were."""
    record: Any = released.get(checkpoint.id)
    return (
        replace(checkpoint, weights=None, state=None, released=float(record["at"]))
        if record is not None
        else checkpoint
    )


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
