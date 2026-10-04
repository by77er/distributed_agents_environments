"""A SQL database for the ledger and what is kept beside it: SQLite for one process, Postgres for many.

Stores write portable SQL through `sql` (named parameters, `ON CONFLICT`, `RETURNING`) and declare their tables with
SQLAlchemy metadata, so the same store code runs on both. `write(..., exclusive=name)` runs writes with the same name
one at a time, across every process sharing the database (a Postgres advisory lock held for the transaction; SQLite
serializes every write anyway).
"""

import hashlib
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.engine import Connection, CursorResult

__all__ = [
    "Connection",
    "Database",
    "create_database",
    "fetch_all",
    "fetch_one",
    "sql",
    "temporary_postgres",
]


def sql(connection: Connection, statement: str, parameters: Mapping[str, Any] | None = None) -> CursorResult[Any]:
    """Execute a statement with `:named` parameters."""
    return connection.execute(sa.text(statement), dict(parameters or {}))


def fetch_all(
    connection: Connection, statement: str, parameters: Mapping[str, Any] | None = None
) -> list[tuple[Any, ...]]:
    """Every row a query returns, as tuples."""
    return [tuple(row) for row in sql(connection, statement, parameters)]


def fetch_one(
    connection: Connection, statement: str, parameters: Mapping[str, Any] | None = None
) -> tuple[Any, ...] | None:
    """The first row a query returns, as a tuple, or None."""
    row = sql(connection, statement, parameters).first()
    return tuple(row) if row is not None else None


class Database:
    def __init__(self, url: str, *, pool_size: int = 5) -> None:
        """`url`: `sqlite:///path` or `postgresql://…` (a Postgres that several processes share)."""
        self.url = url
        self.shared = not url.startswith("sqlite")
        """Whether other processes may use this database at the same time (Postgres)."""
        if self.shared:
            driver_url = url.replace("postgresql://", "postgresql+psycopg://", 1).replace(
                "postgres://", "postgresql+psycopg://", 1
            )
            self.engine = sa.create_engine(
                driver_url, pool_size=pool_size, max_overflow=2 * pool_size, pool_pre_ping=True
            )
        else:
            self.engine = sa.create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})
            event.listen(self.engine, "connect", _configure_sqlite)
            event.listen(self.engine, "begin", _begin_immediately)

    def create(self, metadata: sa.MetaData) -> None:
        """Create the tables that do not exist yet."""
        with self.engine.begin() as connection:
            if self.shared:  # concurrent CREATE TABLE IF NOT EXISTS can still collide in Postgres
                sql(connection, "SELECT pg_advisory_xact_lock(:key)", {"key": _key("schema")})
            metadata.create_all(connection)

    def read[T](self, query: Callable[[Connection], T]) -> T:
        """Run statements outside a transaction: each sees the latest committed state."""
        with self.engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
            return query(connection)

    def write[T](self, change: Callable[[Connection], T], *, exclusive: str | None = None) -> T:
        """Run `change` in one transaction. Writes with the same `exclusive` name run one at a time, across
        processes."""
        with self.engine.begin() as connection:
            if exclusive is not None and self.shared:
                sql(connection, "SELECT pg_advisory_xact_lock(:key)", {"key": _key(exclusive)})
            return change(connection)

    def close(self) -> None:
        self.engine.dispose()


def _key(name: str) -> int:
    """A Postgres advisory lock key for a name."""
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "big", signed=True)


def _configure_sqlite(connection: Any, record: Any) -> None:
    connection.isolation_level = None  # SQLAlchemy's "begin" event starts transactions instead of the driver
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")


def _begin_immediately(connection: Connection) -> None:
    """Take SQLite's write lock when a transaction begins, so read-then-write transactions cannot interleave."""
    if connection.get_execution_options().get("isolation_level") != "AUTOCOMMIT":
        connection.exec_driver_sql("BEGIN IMMEDIATE")


@contextmanager
def temporary_postgres(directory: Path, *, max_connections: int = 400) -> Generator[str]:
    """A throwaway Postgres server in `directory`, for tests and evaluations; yields its URL. Needs `pgembed` (a
    development dependency). Several runners hold several connections each, hence the higher connection limit."""
    from pgembed.postgres_server import get_server  # pyright: ignore[reportMissingTypeStubs]

    configuration = directory / "postgresql.conf"
    server = get_server(directory, cleanup_mode="stop")
    setting = f"max_connections = {max_connections}"
    if setting not in configuration.read_text():
        with configuration.open("a") as file:
            file.write(f"\n{setting}\n")
        server.cleanup()  # restart with the setting
        server = get_server(directory, cleanup_mode="stop")
    try:
        yield str(server.get_uri())
    finally:
        server.cleanup()


def create_database(server_url: str, name: str) -> str:
    """Create an empty database on a Postgres server; returns its URL."""
    url = sa.make_url(server_url)
    engine = sa.create_engine(url.set(drivername="postgresql+psycopg"), isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql(f'CREATE DATABASE "{name}"')
    finally:
        engine.dispose()
    return url.set(database=name).render_as_string(hide_password=False)
