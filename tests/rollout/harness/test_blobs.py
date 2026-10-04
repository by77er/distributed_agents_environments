"""A store of files keeps a file as a blob by linking it, not copying it, and puts a blob in place the same way. A
blob put lately, whether written or found, is not deleted by a delete that spares blobs used lately."""

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import pytest

from rollout.harness import blobs
from rollout.harness.blobs import FileBlobStore


async def test_a_file_is_kept_as_a_link_to_it_and_both_are_read_only(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    made = tmp_path / "made" / "weights.bin"
    made.parent.mkdir()
    made.write_bytes(b"weights" * 1000)
    reference = await store.put_file(made, "application/octet-stream")
    assert reference == await store.put(b"weights" * 1000, "application/octet-stream")  # (the same blob either way)
    kept = Path(reference.uri.removeprefix("file://"))
    assert os.path.samefile(kept, made) and made.stat().st_nlink == 2  # noqa: ASYNC240 (one file on disk, not two)
    assert not os.access(made, os.W_OK)  # (changing it would change the blob)
    assert await store.read(reference) == b"weights" * 1000


async def test_a_blob_is_put_in_place_as_a_link_to_it(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    reference = await store.put(b"state", "application/octet-stream")
    target = tmp_path / "fetched" / "state.pt"
    assert await store.link(reference, target)
    assert target.read_bytes() == b"state" and target.stat().st_nlink == 2
    absent = await FileBlobStore(tmp_path / "other").put(b"elsewhere", "application/octet-stream")
    assert not await store.link(absent, tmp_path / "fetched" / "nothing")  # (a blob it does not have)
    assert not (tmp_path / "fetched" / "nothing").exists()


def aged(path: Path, seconds: float) -> None:
    then = time.time() - seconds
    os.utime(path, (then, then))


def put_ago(path: Path) -> float:
    return time.time() - path.stat().st_mtime


def there(path: Path) -> bool:
    return path.exists()


def held(path: Path) -> bytes:
    return path.read_bytes()


async def test_a_blob_put_lately_is_not_deleted_and_one_unused_for_long_enough_is(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    reference = await store.put(b"shared", "text/plain")
    path = Path(reference.uri.removeprefix("file://"))
    aged(path, 7200)
    assert await store.put(b"shared", "text/plain") == reference  # found: its time is now
    await store.delete(reference, unused_for=3600)
    assert await store.read(reference) == b"shared"
    aged(path, 7200)
    await store.delete(reference, unused_for=3600)
    assert not there(path)
    await store.delete(reference, unused_for=3600)  # (one that is not there: nothing)
    assert os.listdir(path.parent) == []  # (nothing left beside it)


async def test_a_put_that_finds_a_blob_as_it_is_being_deleted_keeps_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    reference = await store.put(b"shared", "text/plain")
    path = Path(reference.uri.removeprefix("file://"))
    aged(path, 7200)
    renamed = os.rename

    def a_put_finds_it_first(source: Any, target: Any) -> None:  # between the look at its time and moving it aside
        asyncio.run(FileBlobStore(tmp_path / "blobs").put(b"shared", "text/plain"))
        renamed(source, target)

    monkeypatch.setattr(blobs.os, "rename", a_put_finds_it_first)
    await store.delete(reference, unused_for=3600)
    monkeypatch.undo()
    assert await store.read(reference) == b"shared"
    assert os.listdir(path.parent) == [path.name]


async def test_a_file_kept_by_linking_is_a_blob_put_now_and_outlives_the_blob(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    made = tmp_path / "made" / "weights.bin"
    made.parent.mkdir()
    made.write_bytes(b"written long ago")
    aged(made, 7200)
    reference = await store.put_file(made, "application/octet-stream")
    kept = Path(reference.uri.removeprefix("file://"))
    assert put_ago(kept) < 60  # (one file: the working file's time is the put's too)
    await store.delete(reference, unused_for=3600)
    assert there(kept)  # (put just now)
    aged(kept, 7200)
    assert await store.put_file(made, "application/octet-stream") == reference  # found again: put now
    assert put_ago(kept) < 60
    await store.delete(reference)
    assert not there(kept) and held(made) == b"written long ago"  # (the working file is its own link)
    assert await store.put_file(made, "application/octet-stream") == reference  # and is kept again from it
    assert await store.read(reference) == b"written long ago"
