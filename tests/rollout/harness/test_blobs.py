"""A store of files keeps a file as a blob by linking it, not copying it, and puts a blob in place the same way."""

import os
from pathlib import Path

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
