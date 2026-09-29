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


class FileBlobStore:
    """Implements `Blobs` in a directory: one file per blob, named by its SHA-256."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    async def put(self, data: bytes, media_type: str) -> BlobReference:
        digest = hashlib.sha256(data).hexdigest()
        path = self._path(digest)
        await asyncio.to_thread(_write_once, path, data)
        return BlobReference(uri=path.as_uri(), sha256=digest, size=len(data), media_type=media_type)

    async def read(self, reference: BlobReference) -> bytes:
        data = await asyncio.to_thread(self._path(reference.sha256).read_bytes)
        if hashlib.sha256(data).hexdigest() != reference.sha256:
            raise ValueError(f"blob {reference.sha256} is corrupt")
        return data

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
