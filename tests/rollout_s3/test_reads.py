"""Reading an object's body again when it stalls or is cut off, which the client's own retries do not cover."""

from typing import Any, cast

import pytest
from botocore.exceptions import ReadTimeoutError

import rollout_s3.store as store
from rollout_s3 import S3BlobStore


class Body:
    def __init__(self, data: bytes, fails: bool) -> None:
        self.data, self.fails = data, fails

    def read(self) -> bytes:
        if self.fails:
            raise ReadTimeoutError(endpoint_url="None")  # (as a stalled body read raises it)
        return self.data


class Client:
    """Answers every get with a body whose first `stalls` reads time out."""

    def __init__(self, stalls: int) -> None:
        self.stalls, self.gets = stalls, 0

    def get_object(self, **_: Any) -> dict[str, Any]:
        self.gets += 1
        return {"Body": Body(b"weights", fails=self.gets <= self.stalls)}


def test_a_body_that_stalls_is_read_again_and_one_that_always_stalls_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_wait(seconds: float) -> None:
        del seconds

    monkeypatch.setattr(store.time, "sleep", no_wait)
    client = Client(stalls=2)
    blobs = S3BlobStore("b", client=cast(Any, client))
    assert blobs._get("k") == b"weights" and client.gets == 3  # pyright: ignore[reportPrivateUsage]
    stuck = Client(stalls=store.READS)
    with pytest.raises(ReadTimeoutError):
        S3BlobStore("b", client=cast(Any, stuck))._get("k")  # pyright: ignore[reportPrivateUsage]
    assert stuck.gets == store.READS
