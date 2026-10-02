"""Blobs: content-addressed storage for bytes too large for events, such as images (see canonical content).

Task code stores bytes with `run.blobs.put` and puts the `BlobReference` it gets in a `Media` block; model adapters
read the bytes back when they render the block. Storing is not an effect: the reference depends only on the bytes, so
a replay that stores the same bytes gets the same reference.
"""

import asyncio
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Protocol

from rollout.core.contracts import BlobReference


class Blobs(Protocol):
    async def put(self, data: bytes, media_type: str) -> BlobReference:
        """Store bytes, or find them already stored; either way return their reference."""
        ...

    async def read(self, reference: BlobReference) -> bytes: ...


def blob_digest(data: bytes) -> str:
    """The SHA-256 that names a blob, in lowercase hexadecimal."""
    return hashlib.sha256(data).hexdigest()


def verified(data: bytes, reference: BlobReference) -> bytes:
    """`data`, if it is the blob `reference` names; raises if a store returned other bytes."""
    if blob_digest(data) != reference.sha256:
        raise ValueError(f"blob {reference.sha256} is corrupt")
    return data


class FileBlobStore:
    """Implements `Blobs` in a directory: one file per blob, named by its SHA-256."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    async def put(self, data: bytes, media_type: str) -> BlobReference:
        digest = blob_digest(data)
        path = self._path(digest)
        await asyncio.to_thread(_write_once, path, data)
        return BlobReference(uri=path.as_uri(), sha256=digest, size=len(data), media_type=media_type)

    async def read(self, reference: BlobReference) -> bytes:
        return verified(await asyncio.to_thread(self._path(reference.sha256).read_bytes), reference)

    def _path(self, digest: str) -> Path:
        return self.directory / digest[:2] / digest


def _write_once(path: Path, data: bytes) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent)
    with os.fdopen(handle, "wb") as file:
        file.write(data)
    os.replace(temporary, path)  # atomic: a reader never sees a partial blob
