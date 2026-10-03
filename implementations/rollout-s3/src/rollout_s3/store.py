"""Blobs in S3 or any S3-compatible object store: MinIO, Cloudflare R2, SeaweedFS, Ceph, Google Cloud Storage's XML
API (docs/libraries/rollout/contracts/canonical-content.md). Needs the `s3` extra (boto3).

Each blob is an object named by its SHA-256 under a prefix, e.g. `s3://bucket/blobs/ab/abcdef…`. Storing is
idempotent (an object that exists is not written again), and reads verify the hash. Since blobs are read by hash,
stores are interchangeable: bytes copied from a `FileBlobStore` into this one serve the same references.

Connection settings come from boto3's usual sources unless given: credentials (environment variables, `~/.aws`,
instance and pod roles), the region, and the endpoint of an S3-compatible service (`AWS_ENDPOINT_URL_S3` or
`AWS_ENDPOINT_URL`). With a custom endpoint, requests use path-style addressing (`endpoint/bucket/key`), which every
S3-compatible service accepts.
"""

import asyncio
import os
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from rollout.contracts import BlobReference
from rollout.harness.blobs import blob_digest, verified

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

__all__ = ["S3BlobStore"]


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
    ) -> None:
        """`client` replaces the boto3 client this store would create (e.g. with custom credentials)."""
        self.bucket = bucket
        self.prefix = prefix if not prefix or prefix.endswith("/") else f"{prefix}/"
        self.client = client or _client(endpoint_url, region)

    @classmethod
    def from_url(cls, url: str, **options: Any) -> "S3BlobStore":
        """A store for `s3://bucket/prefix`."""
        parsed = urlparse(url)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"expected s3://bucket/prefix, not {url!r}")
        return cls(parsed.netloc, prefix=parsed.path.lstrip("/"), **options)

    async def put(self, data: bytes, media_type: str) -> BlobReference:
        digest = blob_digest(data)
        key = self._key(digest)
        await asyncio.to_thread(self._put_once, key, data, media_type, digest)
        return BlobReference(uri=f"s3://{self.bucket}/{key}", sha256=digest, size=len(data), media_type=media_type)

    async def read(self, reference: BlobReference) -> bytes:
        return verified(await asyncio.to_thread(self._get, self._key(reference.sha256)), reference)

    async def delete(self, reference: BlobReference) -> None:
        await asyncio.to_thread(self.client.delete_object, Bucket=self.bucket, Key=self._key(reference.sha256))

    def _key(self, digest: str) -> str:
        return f"{self.prefix}{digest[:2]}/{digest}"

    def _put_once(self, key: str, data: bytes, media_type: str, digest: str) -> None:
        if self._exists(key):
            return  # content-addressed: the object already holds these bytes
        self.client.put_object(
            Bucket=self.bucket, Key=key, Body=data, ContentType=media_type, Metadata={"sha256": digest}
        )

    def _exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as error:
            if _not_found(error):
                return False
            raise
        return True

    def _get(self, key: str) -> bytes:
        from botocore.exceptions import ClientError

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as error:
            if _not_found(error):
                raise FileNotFoundError(f"s3://{self.bucket}/{key}") from error
            raise
        return response["Body"].read()


def _client(endpoint_url: str | None, region: str | None) -> "S3Client":
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
    )
    client: S3Client = boto3.client("s3", endpoint_url=endpoint, region_name=region, config=config)  # pyright: ignore[reportUnknownMemberType]
    return client


def _not_found(error: Any) -> bool:
    code = str(error.response.get("Error", {}).get("Code", ""))
    return code in ("404", "NoSuchKey", "NotFound")
