"""The database layer: the same store code on SQLite and Postgres, with exclusion across processes."""

import asyncio
import threading
from pathlib import Path

import sqlalchemy as sa

from rollout.database import Connection, Database, fetch_one, sql

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
