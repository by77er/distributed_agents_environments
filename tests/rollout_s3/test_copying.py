"""Copying stores of files into a bucket: every blob of every store lands under the store's prefix, a reference made by
a store of files reads the same bytes from the bucket, a copy run again copies nothing (and a dry run nothing at all),
and a file that is not its blob is reported and left out."""

from pathlib import Path

import pytest

from rollout.harness import FileBlobStore
from rollout_s3 import S3BlobStore
from rollout_s3.copying import copy_files, main


async def test_stores_of_files_merge_into_the_bucket_and_their_references_read_there(
    tmp_path: Path, s3_bucket: str
) -> None:
    first, second = FileBlobStore(tmp_path / "first"), FileBlobStore(tmp_path / "second")
    one = await first.put(b"one", "text/plain")
    shared = await first.put(b"in both", "text/plain")
    await second.put(b"in both", "text/plain")
    two = await second.put(b"two" * 100_000, "application/octet-stream")
    store = S3BlobStore.from_url(f"s3://{s3_bucket}/blobs")
    planned = copy_files([first.directory, second.directory], store, dry_run=True)
    assert (planned.copied, planned.bytes, planned.present) == (3, 3 + 7 + 300_000, 0)
    assert "Contents" not in store.client.list_objects_v2(Bucket=s3_bucket)  # (a dry run copies nothing)
    done = copy_files([first.directory, second.directory, tmp_path / "absent"], store)
    assert (done.copied, done.bytes, done.present, done.corrupt) == (3, 3 + 7 + 300_000, 0, [])
    assert [await store.read(each) for each in (one, shared, two)] == [b"one", b"in both", b"two" * 100_000]
    listed = store.client.list_objects_v2(Bucket=s3_bucket)["Contents"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert sorted(str(each.get("Key")) for each in listed) == sorted(
        f"blobs/{each.sha256[:2]}/{each.sha256}" for each in (one, shared, two)
    )
    again = copy_files([first.directory, second.directory], store)
    assert (again.copied, again.present) == (0, 3)


async def test_a_file_that_is_not_its_blob_is_reported_and_not_copied(
    tmp_path: Path, s3_bucket: str, capsys: pytest.CaptureFixture[str]
) -> None:
    files = FileBlobStore(tmp_path / "files")
    good = await files.put(b"good", "text/plain")
    bad = await files.put(b"bad", "text/plain")
    spoiled = files.directory / bad.sha256[:2] / bad.sha256
    spoiled.write_bytes(b"changed in place")
    (files.directory / "notes.txt").write_text("not a blob")  # (anything else in the directory is left alone)
    with pytest.raises(SystemExit):
        main([str(files.directory), "--to", f"s3://{s3_bucket}/blobs"])
    captured = capsys.readouterr()
    assert "1 blobs copied (4 bytes), 0 there already" in captured.out
    assert str(spoiled) in captured.err
    store = S3BlobStore.from_url(f"s3://{s3_bucket}/blobs")
    assert await store.read(good) == b"good"
    with pytest.raises(FileNotFoundError):
        await store.read(bad)
