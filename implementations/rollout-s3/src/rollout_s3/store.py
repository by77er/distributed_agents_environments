"""Blobs in S3 or any S3-compatible object store: MinIO, Cloudflare R2, SeaweedFS, Ceph, Google Cloud Storage's XML
API (docs/guide/content.md#media-and-blobs).

Each blob is an object named by its SHA-256 under a prefix, e.g. `s3://bucket/blobs/ab/abcdef…`. Storing is
idempotent (an object that exists is not written again), and reads verify the hash. Since blobs are read by hash,
stores are interchangeable: bytes copied from a `FileBlobStore` into this one serve the same references.

A blob's time is its object's `LastModified`, judged by the service's clock (the `Date` of its answer). A put that finds
the object copies it onto itself, which sets that time to now, unless it was set less than `refresh_after` seconds ago;
deleting with `unused_for` deletes only an object older than that. The look at its time and the delete are two
requests: a put that finds the object between them is not seen.

Connection settings come from boto3's usual sources unless given: credentials (environment variables, `~/.aws`,
instance and pod roles), the region, and the endpoint of an S3-compatible service (`AWS_ENDPOINT_URL_S3` or
`AWS_ENDPOINT_URL`). A store's own credentials may be named instead (`access_key_id_env`, `secret_access_key_env`: the
environment variables they are read from when the store is first used), so that one process holds several stores, each
with its own key: the cluster's bucket and an R2 bucket, say. With a custom endpoint, requests use path-style
addressing (`endpoint/bucket/key`), which every S3-compatible service accepts.
"""

import asyncio
import os
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from rollout.contracts import BlobReference
from rollout.harness.blobs import blob_digest, verified

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

__all__ = ["REFRESH_AFTER", "S3BlobStore"]

REFRESH_AFTER = 60.0
"""Seconds after which a put that finds a blob sets its time again: one put a minute of a blob much put (an image every
episode stores) costs a copy, not every one."""
READ_TIMEOUT = 120.0
"""Seconds a read of a response may wait for its next bytes (botocore's default is 60)."""
READS = 5
"""Times an object's body is read before a stalled or cut-off read fails."""


class S3BlobStore:
    """Implements `Blobs` in an S3 bucket."""

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "blobs/",
        endpoint_url: str | None = None,
        region: str | None = None,
        client: "S3Client | None" = None,
        refresh_after: float = REFRESH_AFTER,
        access_key_id_env: str | None = None,
        secret_access_key_env: str | None = None,
    ) -> None:
        """`client` replaces the boto3 client this store would create (e.g. with custom credentials).
        `access_key_id_env` and `secret_access_key_env` name the environment variables the store's key is read from
        (both, or neither: boto3's usual sources), the first time the store is used, so a process that opens the store
        and never uses it needs no key. Raises `ValueError` where only one is named."""
        if (access_key_id_env is None) != (secret_access_key_env is None):
            raise ValueError("a store's credentials are named both: access_key_id_env and secret_access_key_env")
        self.bucket = bucket
        self.prefix = prefix if not prefix or prefix.endswith("/") else f"{prefix}/"
        self._client = client
        self._connection = (endpoint_url, region, access_key_id_env, secret_access_key_env)
        self.refresh_after = refresh_after

    @property
    def client(self) -> "S3Client":
        """The boto3 client, made the first time it is asked for. Raises `ValueError` where the store's key is named
        and not set here."""
        if self._client is None:
            endpoint_url, region, key_env, secret_env = self._connection
            self._client = _client(endpoint_url, region, _credentials(key_env, secret_env))
        return self._client

    @classmethod
    def from_url(cls, url: str, **options: Any) -> "S3BlobStore":
        """A store for `s3://bucket/prefix`."""
        parsed = urlparse(url)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"expected s3://bucket/prefix, not {url!r}")
        return cls(parsed.netloc, prefix=parsed.path.lstrip("/"), **options)

    def holds(self, reference: BlobReference) -> bool:
        """Whether a reference names an object of this store's bucket and prefix (`s3://BUCKET/PREFIX…`)."""
        return reference.uri.startswith(f"s3://{self.bucket}/{self.prefix}")

    async def put(self, data: bytes, media_type: str) -> BlobReference:
        digest = blob_digest(data)
        key = self._key(digest)
        await asyncio.to_thread(self._put_once, key, data, media_type, digest)
        return BlobReference(uri=f"s3://{self.bucket}/{key}", sha256=digest, size=len(data), media_type=media_type)

    async def read(self, reference: BlobReference) -> bytes:
        return verified(await asyncio.to_thread(self._get, self._key(reference.sha256)), reference)

    async def delete(self, reference: BlobReference, *, unused_for: float = 0.0) -> None:
        await asyncio.to_thread(self._delete, self._key(reference.sha256), unused_for)

    async def with_extension(self, reference: BlobReference, extension: str) -> str:
        """The URI of a copy of a blob's object named by its key and `extension` (`s3://BUCKET/KEY.zip`), made inside
        the bucket the first time it is asked for: for readers that tell an archive by its name, as Ray does a runtime
        environment's `working_dir`. Raises `FileNotFoundError` where the store does not have the blob."""
        key = self._key(reference.sha256)
        await asyncio.to_thread(self._copied, key, f"{key}{extension}", reference.size)
        return f"s3://{self.bucket}/{key}{extension}"

    def _copied(self, key: str, target: str, size: int) -> None:
        from botocore.exceptions import ClientError

        try:
            if int(self.client.head_object(Bucket=self.bucket, Key=target)["ContentLength"]) == size:
                return
        except ClientError as error:
            if not _not_found(error):
                raise
        try:
            self.client.copy_object(Bucket=self.bucket, Key=target, CopySource={"Bucket": self.bucket, "Key": key})
        except ClientError as error:
            if _not_found(error):
                raise FileNotFoundError(f"s3://{self.bucket}/{key}") from error
            raise

    def _key(self, digest: str) -> str:
        return f"{self.prefix}{digest[:2]}/{digest}"

    def _put_once(self, key: str, data: bytes, media_type: str, digest: str) -> None:
        if self._refreshed(key, media_type, digest):
            return  # content-addressed: the object already holds these bytes
        self.client.put_object(
            Bucket=self.bucket, Key=key, Body=data, ContentType=media_type, Metadata={"sha256": digest}
        )

    def _refreshed(self, key: str, media_type: str, digest: str) -> bool:
        """Whether the object is there; if it is, its time is now (copied onto itself, unless it was set less than
        `refresh_after` seconds ago)."""
        from botocore.exceptions import ClientError

        if (age := self._age(key)) is None:
            return False
        if age < self.refresh_after:
            return True
        try:
            self.client.copy_object(
                Bucket=self.bucket, Key=key, CopySource={"Bucket": self.bucket, "Key": key},
                MetadataDirective="REPLACE", ContentType=media_type, Metadata={"sha256": digest},
            )  # fmt: skip
        except ClientError as error:
            if _not_found(error):  # (deleted since it was looked at: it is written again)
                return False
            raise
        return True

    def _delete(self, key: str, unused_for: float) -> None:
        if unused_for > 0 and ((age := self._age(key)) is None or age < unused_for):
            return
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def _age(self, key: str) -> float | None:
        """Seconds since the object's time, by the service's clock; None if there is no such object."""
        from botocore.exceptions import ClientError

        try:
            head = self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as error:
            if _not_found(error):
                return None
            raise
        said = head.get("ResponseMetadata", {}).get("HTTPHeaders", {}).get("date")
        now = parsedate_to_datetime(said) if said else datetime.now(UTC)
        return (now - head["LastModified"]).total_seconds()

    def _get(self, key: str) -> bytes:
        """The object's bytes, read whole. The client's retries cover the request, not reading its body after: a body
        that stalls or is cut off is read again from the start, up to `READS` times, waiting longer before each."""
        from botocore.exceptions import (
            ClientError,
            ConnectionClosedError,
            IncompleteReadError,
            ReadTimeoutError,
            ResponseStreamingError,
        )

        for attempt in range(1, READS + 1):
            try:
                response = self.client.get_object(Bucket=self.bucket, Key=key)
            except ClientError as error:
                if _not_found(error):
                    raise FileNotFoundError(f"s3://{self.bucket}/{key}") from error
                raise
            try:
                return response["Body"].read()
            except (ReadTimeoutError, ResponseStreamingError, IncompleteReadError, ConnectionClosedError):
                if attempt == READS:
                    raise
                time.sleep(min(2.0 ** (attempt - 1), 30.0))
        raise AssertionError("unreachable")


def _credentials(key_env: str | None, secret_env: str | None) -> tuple[str, str] | None:
    """The key the variables named hold (none where neither is named)."""
    if key_env is None and secret_env is None:
        return None
    if key_env is None or secret_env is None:
        raise ValueError("a store's credentials are named both: access_key_id_env and secret_access_key_env")
    key, secret = os.environ.get(key_env, ""), os.environ.get(secret_env, "")
    if not key or not secret:
        missing = [name for name, value in ((key_env, key), (secret_env, secret)) if not value]
        raise ValueError(f"the store's credentials are not set here: {', '.join(missing)}")
    return key, secret


def _client(endpoint_url: str | None, region: str | None, credentials: tuple[str, str] | None = None) -> "S3Client":
    import boto3
    from botocore.config import Config

    endpoint = endpoint_url or os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
    config = Config(
        s3={"addressing_style": "path" if endpoint else "auto"},
        # Newer SDKs add CRC checksums to every request by default, which some S3-compatible services reject; reads
        # verify the SHA-256 anyway.
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
        retries={"mode": "standard", "max_attempts": 5},
        read_timeout=READ_TIMEOUT,
    )
    key, secret = credentials if credentials is not None else (None, None)
    client: S3Client = boto3.client(  # pyright: ignore[reportUnknownMemberType]
        "s3", endpoint_url=endpoint, region_name=region, config=config, aws_access_key_id=key,
        aws_secret_access_key=secret,
    )  # fmt: skip
    return client


def _not_found(error: Any) -> bool:
    code = str(error.response.get("Error", {}).get("Code", ""))
    return code in ("404", "NoSuchKey", "NotFound")
