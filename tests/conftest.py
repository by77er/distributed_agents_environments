"""Shared fixtures: a throwaway Postgres server for the tests that run runners on a shared database."""

import uuid
from collections.abc import Iterator

import pytest

from rollout.database import create_database, temporary_postgres


@pytest.fixture(scope="session")
def postgres_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A Postgres server for the whole test session (skipped when `pgembed` is not installed)."""
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
