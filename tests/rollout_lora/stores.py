"""Blob stores the trainer's processes are told to keep a full state in, for tests (named by `module:name` in a
location, `rollout_train.stores.opened`): one whose puts take a while, and one whose first puts fail."""

import asyncio
from pathlib import Path

from rollout.contracts import BlobReference
from rollout.harness.blobs import FileBlobStore


class SlowStore(FileBlobStore):
    """Files under `directory`, each put taking `delay` seconds first (an upload over a slow link)."""

    def __init__(self, directory: str, delay: float) -> None:
        super().__init__(Path(directory))
        self.delay = delay

    async def put(self, data: bytes, media_type: str) -> BlobReference:
        await asyncio.sleep(self.delay)
        return await super().put(data, media_type)


class FlakyStore(FileBlobStore):
    """Files under `directory`, whose first `failures` puts in each process fail (a bucket that refused them)."""

    def __init__(self, directory: str, failures: int) -> None:
        super().__init__(Path(directory))
        self.failures = failures

    async def put(self, data: bytes, media_type: str) -> BlobReference:
        if self.failures > 0:
            self.failures -= 1
            raise OSError("the bucket refused the upload")
        return await super().put(data, media_type)
