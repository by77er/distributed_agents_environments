"""Blob stores: content-addressed, idempotent, verified on read; a file store and an S3 store behave the same."""

import asyncio
import hashlib
from pathlib import Path

import boto3
import pytest

from rollout.contracts import BlobReference
from rollout.harness import Blobs, FileBlobStore
from rollout_s3 import S3BlobStore


async def test_bytes_come_back_by_their_reference(blob_store: Blobs) -> None:
    data = b"\x89PNG not really"
    reference = await blob_store.put(data, "image/png")
    assert (reference.sha256, reference.size, reference.media_type) == (
        hashlib.sha256(data).hexdigest(),
        15,
        "image/png",
    )
    assert await blob_store.read(reference) == data
    assert await blob_store.put(data, "image/png") == reference  # storing again changes nothing


async def test_a_missing_blob_is_not_found(blob_store: Blobs) -> None:
    missing = BlobReference(
        uri="nowhere", sha256=hashlib.sha256(b"absent").hexdigest(), size=6, media_type="text/plain"
    )
    with pytest.raises(FileNotFoundError):
        await blob_store.read(missing)


async def test_a_blob_found_by_a_put_is_spared_and_one_unused_long_enough_is_deleted(blob_store: Blobs) -> None:
    if isinstance(blob_store, S3BlobStore):
        blob_store.refresh_after = 0.0  # (every put that finds it sets its time)
    reference = await blob_store.put(b"shared", "text/plain")
    await asyncio.sleep(2.2)
    await blob_store.put(b"shared", "text/plain")  # found: put now
    await blob_store.delete(reference, unused_for=2.0)
    assert await blob_store.read(reference) == b"shared"
    await asyncio.sleep(2.2)
    await blob_store.delete(reference, unused_for=2.0)
    with pytest.raises(FileNotFoundError):
        await blob_store.read(reference)
    await blob_store.delete(reference, unused_for=2.0)  # (one that is not there: nothing)


async def test_a_put_that_finds_a_blob_put_lately_does_not_copy_it(s3_bucket: str) -> None:
    store = S3BlobStore.from_url(f"s3://{s3_bucket}/blobs")
    reference = await store.put(b"often", "text/plain")
    key = reference.uri.removeprefix(f"s3://{s3_bucket}/")
    client = boto3.client("s3")  # pyright: ignore[reportUnknownMemberType]
    before = client.head_object(Bucket=s3_bucket, Key=key)["LastModified"]  # pyright: ignore[reportUnknownMemberType]
    await asyncio.sleep(1.1)
    await store.put(b"often", "text/plain")  # (within REFRESH_AFTER: left as it is)
    assert client.head_object(Bucket=s3_bucket, Key=key)["LastModified"] == before  # pyright: ignore[reportUnknownMemberType]
    store.refresh_after = 0.0
    await store.put(b"often", "text/plain")
    head = client.head_object(Bucket=s3_bucket, Key=key)  # pyright: ignore[reportUnknownMemberType]
    assert head["LastModified"] > before and head["Metadata"] == {"sha256": reference.sha256}


async def test_objects_are_named_by_their_hash_under_the_prefix(s3_bucket: str) -> None:
    store = S3BlobStore.from_url(f"s3://{s3_bucket}/run-blobs")
    reference = await store.put(b"hello", "text/plain")
    await store.put(b"hello", "text/plain")
    digest = reference.sha256
    assert reference.uri == f"s3://{s3_bucket}/run-blobs/{digest[:2]}/{digest}"
    listed = boto3.client("s3").list_objects_v2(Bucket=s3_bucket)  # pyright: ignore[reportUnknownMemberType]
    assert [item["Key"] for item in listed["Contents"]] == [f"run-blobs/{digest[:2]}/{digest}"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    head = boto3.client("s3").head_object(Bucket=s3_bucket, Key=f"run-blobs/{digest[:2]}/{digest}")  # pyright: ignore[reportUnknownMemberType]
    assert head["ContentType"] == "text/plain" and head["Metadata"] == {"sha256": digest}


async def test_a_corrupted_object_is_refused(s3_bucket: str) -> None:
    store = S3BlobStore.from_url(f"s3://{s3_bucket}/blobs")
    reference = await store.put(b"original", "text/plain")
    key = reference.uri.removeprefix(f"s3://{s3_bucket}/")
    boto3.client("s3").put_object(Bucket=s3_bucket, Key=key, Body=b"tampered")  # pyright: ignore[reportUnknownMemberType]
    with pytest.raises(ValueError, match="corrupt"):
        await store.read(reference)


async def test_stores_are_interchangeable_by_hash(tmp_path: Path, s3_bucket: str) -> None:
    """A reference made by the file store reads from the S3 store once the same bytes are there (a migration)."""
    local = FileBlobStore(tmp_path / "blobs")
    reference = await local.put(b"moved to s3", "text/plain")
    remote = S3BlobStore.from_url(f"s3://{s3_bucket}/blobs")
    await remote.put(await local.read(reference), reference.media_type)
    assert await remote.read(reference) == b"moved to s3"


def test_urls_name_a_bucket_and_a_prefix() -> None:
    store = S3BlobStore.from_url("s3://my-bucket/deep/prefix", client=object())  # type: ignore[arg-type]
    assert (store.bucket, store.prefix) == ("my-bucket", "deep/prefix/")
    assert S3BlobStore.from_url("s3://my-bucket", client=object()).prefix == ""  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="s3://bucket/prefix"):
        S3BlobStore.from_url("https://my-bucket/prefix")


async def test_a_store_reads_its_own_credentials_from_the_variables_it_names(
    s3_bucket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WRITER_KEY_ID", "writer")
    monkeypatch.setenv("WRITER_SECRET", "writer-secret")
    store = S3BlobStore(s3_bucket, access_key_id_env="WRITER_KEY_ID", secret_access_key_env="WRITER_SECRET")
    credentials = store.client._request_signer._credentials  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
    assert credentials.access_key == "writer"  # pyright: ignore[reportUnknownMemberType]
    reference = await store.put(b"named", "text/plain")
    assert store.holds(reference) and await store.read(reference) == b"named"
    assert not S3BlobStore("another-bucket", client=store.client).holds(reference)
    unkeyed = S3BlobStore(s3_bucket, access_key_id_env="WRITER_KEY_ID", secret_access_key_env="MISSING_SECRET")
    with pytest.raises(ValueError, match="not set here: MISSING_SECRET"):  # (only once it is used)
        await unkeyed.read(reference)
    with pytest.raises(ValueError, match="named both"):
        S3BlobStore(s3_bucket, access_key_id_env="WRITER_KEY_ID")
