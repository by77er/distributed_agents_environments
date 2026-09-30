"""Shared fixtures: a Postgres server for runners sharing a database, and an S3 server for blob stores.

By default both are started in-process for the test session (`pgembed`, `moto`). To test against real services, such
as those in deploy/local/compose.yaml, set ROLLOUT_TEST_POSTGRES (a URL whose user may create databases) and
ROLLOUT_TEST_S3 (an endpoint URL; credentials from AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY).
"""

import os
import socket
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from rollout.core.harness import Blobs, FileBlobStore
from rollout.database import create_database, temporary_postgres


@pytest.fixture(scope="session")
def postgres_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A Postgres server for the whole test session: ROLLOUT_TEST_POSTGRES, or a throwaway one (skipped when
    `pgembed` is not installed)."""
    if external := os.environ.get("ROLLOUT_TEST_POSTGRES"):
        yield external
        return
    pytest.importorskip("pgembed")
    with temporary_postgres(tmp_path_factory.mktemp("postgres")) as url:
        yield url


@pytest.fixture
def postgres(postgres_server: str) -> str:
    """An empty database of its own for one test."""
    return create_database(postgres_server, f"t_{uuid.uuid4().hex[:16]}")


@pytest.fixture(params=["sqlite", "postgres"])
def database(request: pytest.FixtureRequest) -> str | None:
    """Each test using this runs twice: on SQLite (None) and on a Postgres database of its own."""
    if request.param == "sqlite":
        return None
    return request.getfixturevalue("postgres")


@pytest.fixture(scope="session")
def s3_server() -> Iterator[str]:
    """An S3-compatible server for the whole test session: ROLLOUT_TEST_S3, or moto; yields its endpoint URL."""
    if external := os.environ.get("ROLLOUT_TEST_S3"):
        yield external
        return
    server_module = pytest.importorskip("moto.server")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = server_module.ThreadedMotoServer(ip_address="127.0.0.1", port=port, verbose=False)
    server.start()
    yield f"http://127.0.0.1:{port}"
    server.stop()


@pytest.fixture
def s3_bucket(s3_server: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """An empty bucket of its own for one test; the endpoint and credentials are in the environment, as they would
    be in production."""
    import boto3

    monkeypatch.setenv("AWS_ENDPOINT_URL", s3_server)
    if not os.environ.get("ROLLOUT_TEST_S3"):  # moto accepts any credentials; a real service has its own
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    bucket = f"b-{uuid.uuid4().hex[:16]}"
    boto3.client("s3", endpoint_url=s3_server).create_bucket(Bucket=bucket)  # pyright: ignore[reportUnknownMemberType]
    return bucket


@pytest.fixture(params=["file", "s3"])
def blob_store(request: pytest.FixtureRequest, tmp_path: Path) -> Blobs:
    """Each test using this runs twice: with a `FileBlobStore` and with an `S3BlobStore`."""
    if request.param == "file":
        return FileBlobStore(tmp_path / "blobs")
    from rollout.adapters.s3 import S3BlobStore

    return S3BlobStore.from_url(f"s3://{request.getfixturevalue('s3_bucket')}/blobs")
