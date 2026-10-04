"""Blobs: content-addressed storage for bytes too large for events, such as images (see canonical content).

Task code stores bytes with `run.blobs.put` and puts the `BlobReference` it gets in a `Media` block; model adapters
read the bytes back when they render the block. Storing is not an effect: the reference depends only on the bytes, so
a replay that stores the same bytes gets the same reference.

A store keeps the time each blob was last put, whether the put wrote it or found it there, so that a blob just put is
not deleted (`Blobs.delete` with `unused_for`): whoever deletes blobs that nothing names reads the names and then
deletes, and a writer that found a blob stored and is about to name it would otherwise lose it in between.
"""

import asyncio
import contextlib
import hashlib
import os
import secrets
import shutil
import tempfile
import time
from pathlib import Path
from typing import Protocol

from rollout.contracts import BlobReference


class Blobs(Protocol):
    async def put(self, data: bytes, media_type: str) -> BlobReference:
        """Store bytes, or find them already stored; either way return their reference. Either way the blob's time
        is now: it is not deleted for a while (`delete`)."""
        ...

    async def read(self, reference: BlobReference) -> bytes: ...

    async def delete(self, reference: BlobReference, *, unused_for: float = 0.0) -> None:
        """Remove a blob if it is there, and was not put (written or found) in the last `unused_for` seconds.
        Whoever stored the same bytes holds the same blob: delete only what nothing else names, and with `unused_for`
        longer than any writer takes from putting a blob to naming it."""
        ...


def blob_digest(data: bytes) -> str:
    """The SHA-256 that names a blob, in lowercase hexadecimal."""
    return hashlib.sha256(data).hexdigest()


def verified(data: bytes, reference: BlobReference) -> bytes:
    """`data`, if it is the blob `reference` names; raises if a store returned other bytes."""
    if blob_digest(data) != reference.sha256:
        raise ValueError(f"blob {reference.sha256} is corrupt")
    return data


class FileBlobStore:
    """Implements `Blobs` in a directory: one file per blob, named by its SHA-256. A blob's time is its file's
    modification time: a put that finds the file sets it to now. Deleting with `unused_for` moves the file aside first
    and looks at its time again there, so a put that found it just before is seen (the file is put back), and a put
    just after finds no file and writes it again."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    async def put(self, data: bytes, media_type: str) -> BlobReference:
        digest = blob_digest(data)
        path = self._path(digest)
        await asyncio.to_thread(_write_once, path, data)
        return BlobReference(uri=path.as_uri(), sha256=digest, size=len(data), media_type=media_type)

    async def read(self, reference: BlobReference) -> bytes:
        return verified(await asyncio.to_thread(self._path(reference.sha256).read_bytes), reference)

    async def delete(self, reference: BlobReference, *, unused_for: float = 0.0) -> None:
        await asyncio.to_thread(_deleted, self._path(reference.sha256), unused_for)

    async def put_file(self, path: Path, media_type: str) -> BlobReference:
        """Store the file at `path`, or find it already stored, without copying its bytes where the store is on the
        same filesystem: the blob is then a hard link to the file, and both are made read-only, since they are one file
        (and share one modification time: the put's). Elsewhere the file is copied. The file is read in pieces, never
        whole."""
        digest, size = await asyncio.to_thread(_file_digest, path)
        target = self._path(digest)
        await asyncio.to_thread(_linked_once, path, target, put=True)
        return BlobReference(uri=target.as_uri(), sha256=digest, size=size, media_type=media_type)

    async def link(self, reference: BlobReference, target: Path) -> bool:
        """Put the blob at `target`: a hard link to it where `target` is on the store's filesystem, else a copy.
        Returns False, putting nothing, if the store does not have it."""
        source = self._path(reference.sha256)
        if not await asyncio.to_thread(source.exists):
            return False
        await asyncio.to_thread(_linked_once, source, target)
        return True

    def _path(self, digest: str) -> Path:
        return self.directory / digest[:2] / digest


def _file_digest(path: Path) -> tuple[str, int]:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest(), os.fstat(file.fileno()).st_size


def _linked_once(source: Path, target: Path, *, put: bool = False) -> None:
    """`target` as a hard link to `source` (read-only, as both now are), or a copy across filesystems. `put`: `target`
    is a blob, whose time is now whether it is made or found."""
    if _touched(target) if put else target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=target.parent)) / target.name
    try:
        try:
            os.link(source, temporary)
            os.chmod(temporary, 0o444)
        except OSError:  # (another filesystem, or one without hard links)
            shutil.copyfile(source, temporary)
        if put:  # (a linked file keeps the time it was written at)
            os.utime(temporary)
        os.replace(temporary, target)  # atomic: a reader never sees a partial file
    finally:
        shutil.rmtree(temporary.parent, ignore_errors=True)


def _write_once(path: Path, data: bytes) -> None:
    if _touched(path):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent)
    with os.fdopen(handle, "wb") as file:
        file.write(data)
    os.replace(temporary, path)  # atomic: a reader never sees a partial blob


def _touched(path: Path) -> bool:
    """Whether a blob's file is there, its time set to now if it is. One that cannot be set (another user's) is
    written again, as this user's."""
    try:
        os.utime(path)
    except (FileNotFoundError, PermissionError):
        return False
    return True


def _deleted(path: Path, unused_for: float) -> None:
    """Delete a blob's file, if it is there and its time is `unused_for` seconds old or older. The file is moved aside
    before it is deleted, and its time looked at again: a put that found it in between set that time (the file is put
    back), and one after finds no file and writes it again."""
    if unused_for <= 0:
        path.unlink(missing_ok=True)
        return
    try:
        if time.time() - path.stat().st_mtime < unused_for:
            return
        aside = path.with_name(f".{path.name}.{secrets.token_hex(4)}.deleting")
        os.rename(path, aside)
    except FileNotFoundError:
        return
    try:
        if time.time() - aside.stat().st_mtime < unused_for:  # found by a put since it was looked at
            with contextlib.suppress(FileExistsError):  # (written again meanwhile: the same bytes)
                os.link(aside, path)
    finally:
        aside.unlink(missing_ok=True)
