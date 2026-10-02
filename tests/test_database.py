"""The database layer: the same store code on SQLite and Postgres, with exclusion across processes."""

import asyncio
import threading
from pathlib import Path

import pytest
import sqlalchemy as sa

from rollout.core.contracts import Conflict, Text, ToolResult
from rollout.database import Connection, Database, effects_table, fetch_one, recorded, sql

COUNTER = sa.MetaData()
sa.Table("counter", COUNTER, sa.Column("name", sa.Text, primary_key=True), sa.Column("value", sa.Integer))


def open_twice(tmp_path: Path, database: str | None) -> list[Database]:
    """Two databases on separate connection pools: as two processes would have."""
    if database is None:
        return [Database.sqlite(tmp_path / "shared.sqlite") for _ in range(2)]
    return [Database(database) for _ in range(2)]


def test_exclusive_writes_never_interleave(tmp_path: Path, database: str | None) -> None:
    databases = open_twice(tmp_path, database)
    databases[0].create(COUNTER)
    databases[0].write(lambda db: sql(db, "INSERT INTO counter (name, value) VALUES ('n', 0)"))

    def increment(db: Connection) -> None:  # read, then write: loses updates unless writes are exclusive
        row = fetch_one(db, "SELECT value FROM counter WHERE name = 'n'")
        assert row is not None
        sql(db, "UPDATE counter SET value = :value WHERE name = 'n'", {"value": row[0] + 1})

    def work(index: int) -> None:
        for _ in range(25):
            databases[index % 2].write(increment, exclusive="counter")

    threads = [threading.Thread(target=work, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert databases[1].read(lambda db: sql(db, "SELECT value FROM counter").scalar_one()) == 100
    for each in databases:
        each.close()


def test_a_recorded_write_happens_once_per_effect(tmp_path: Path, database: str | None) -> None:
    first, second = open_twice(tmp_path, database)
    tables = sa.MetaData()
    sa.Table("counter", tables, sa.Column("name", sa.Text, primary_key=True), sa.Column("value", sa.Integer))
    effects_table(tables)
    first.create(tables)
    first.write(lambda db: sql(db, "INSERT INTO counter (name, value) VALUES ('n', 0)"))

    def increment(db: Connection) -> ToolResult:
        value = sql(db, "UPDATE counter SET value = value + 1 WHERE name = 'n' RETURNING value").scalar_one()
        return ToolResult(content=[Text(text=str(value))])

    def perform(store: Database, effect_id: str, digest: str) -> ToolResult:
        return store.write(lambda db: recorded(db, effect_id, digest, increment), exclusive="counter")

    once = perform(first, "r:0:1", "digest-a")
    assert perform(second, "r:0:1", "digest-a") == once  # the recorded result: nothing is performed again
    assert perform(second, "r:0:2", "digest-a") != once
    with pytest.raises(Conflict):
        perform(first, "r:0:1", "digest-b")
    assert first.read(lambda db: sql(db, "SELECT value FROM counter").scalar_one()) == 2
    for each in (first, second):
        each.close()


async def test_a_lock_is_held_by_one_holder_at_a_time(tmp_path: Path, database: str | None) -> None:
    databases = open_twice(tmp_path, database) if database else [Database.sqlite(tmp_path / "one.sqlite")] * 2
    inside: list[int] = []
    overlaps: list[int] = []

    async def hold(index: int) -> None:
        for _ in range(5):
            async with databases[index % 2].lock("resource"):
                if inside:
                    overlaps.append(index)
                inside.append(index)
                await asyncio.sleep(0.01)
                inside.remove(index)

    await asyncio.gather(*(hold(index) for index in range(4)))
    assert overlaps == []
    for each in set(databases):
        each.close()
