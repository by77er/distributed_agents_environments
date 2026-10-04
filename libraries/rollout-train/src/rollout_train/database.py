"""`DatabaseLedger`: the ledger in SQL tables, which every run and machine using the database shares (it needs the
`durable` extra: the database is `rollout_durable`'s).

Two tables hold it: `ledger_records` (a row per record: its table's name, its key, the order it was appended in, the
fence it was written under, and the record as JSON; a table has each key once) and `ledger_fences` (the newest fence
of every scope). Taking a fence and appending run in transactions that hold the scope's lock, so a writer that was
replaced is refused (`Fenced`) whichever process it is in. SQLite serves one machine; Postgres serves several.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from pydantic import JsonValue

from rollout_durable.database import Connection, Database, fetch_all, fetch_one, sql
from rollout_train.ledger import Fence, Fenced, Ledger

METADATA = sa.MetaData()
RECORDS = sa.Table(
    "ledger_records",
    METADATA,
    sa.Column("name", sa.Text, primary_key=True),
    sa.Column("key", sa.Text, primary_key=True),
    sa.Column("position", sa.BigInteger, nullable=False),
    sa.Column("fence", sa.BigInteger, nullable=False),
    sa.Column("record", sa.Text, nullable=False),
)
FENCES = sa.Table(
    "ledger_fences",
    METADATA,
    sa.Column("scope", sa.Text, primary_key=True),
    sa.Column("number", sa.BigInteger, nullable=False),
)


class DatabaseLedger:
    """A `Ledger` in a SQL database: `url` is `sqlite:///path` (`~` is the home directory) or `postgresql://…`."""

    def __init__(self, url: str) -> None:
        if url.startswith("sqlite:///~"):
            path = Path(url.removeprefix("sqlite:///")).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{path}"
        self.url = url
        self.database = Database(url)
        self.database.create(METADATA)

    async def take(self, scope: str) -> Fence:
        def taken(connection: Connection) -> int:
            row = sql(
                connection,
                "INSERT INTO ledger_fences (scope, number) VALUES (:scope, 1) "
                "ON CONFLICT (scope) DO UPDATE SET number = ledger_fences.number + 1 RETURNING number",
                {"scope": scope},
            ).one()
            return int(row[0])

        return Fence(scope, await asyncio.to_thread(self.database.write, taken, exclusive=f"ledger:{scope}"))

    async def append(self, table: str, key: str, record: JsonValue, fence: Fence) -> bool:
        text = json.dumps(record)

        def appended(connection: Connection) -> bool:
            newest = fetch_one(
                connection, "SELECT number FROM ledger_fences WHERE scope = :scope", {"scope": fence.scope}
            )
            if (int(newest[0]) if newest else 0) != fence.number:
                raise Fenced(f"{fence.scope} has a newer writer than fence {fence.number}")
            inserted = sql(
                connection,
                "INSERT INTO ledger_records (name, key, position, fence, record) VALUES (:name, :key, "
                "(SELECT COALESCE(MAX(position), 0) + 1 FROM ledger_records WHERE name = :name), :fence, :record) "
                "ON CONFLICT (name, key) DO NOTHING",
                {"name": table, "key": key, "fence": fence.number, "record": text},
            )
            return inserted.rowcount == 1

        return await asyncio.to_thread(self.database.write, appended, exclusive=f"ledger:{fence.scope}")

    async def read(self, table: str) -> dict[str, JsonValue]:
        def rows(connection: Connection) -> list[tuple[Any, ...]]:
            query = "SELECT key, record FROM ledger_records WHERE name = :name ORDER BY position"
            return fetch_all(connection, query, {"name": table})

        found = await asyncio.to_thread(self.database.read, rows)
        return {str(key): json.loads(record) for key, record in found}

    async def tables(self) -> list[str]:
        def names(connection: Connection) -> list[tuple[Any, ...]]:
            return fetch_all(connection, "SELECT DISTINCT name FROM ledger_records ORDER BY name")

        return [str(name) for (name,) in await asyncio.to_thread(self.database.read, names)]

    async def fences(self) -> dict[str, int]:
        def rows(connection: Connection) -> list[tuple[Any, ...]]:
            return fetch_all(connection, "SELECT scope, number FROM ledger_fences ORDER BY scope")

        return {str(scope): int(number) for scope, number in await asyncio.to_thread(self.database.read, rows)}

    def close(self) -> None:
        self.database.close()


async def copy(source: Ledger, target: DatabaseLedger) -> int:
    """Copy every table and fence of `source` (files, or another database) into `target`, which must have none of its
    tables yet; returns how many records. Records keep their keys and their order; each is noted under its scope's
    newest fence (the fence a record was written under is not read back through a ledger). A fence already in the
    target is kept if it is newer. To move to Postgres: copy, then point the profile's `[ledger] url` at it."""
    tables = await source.tables()
    there = set(await target.tables())
    if clash := sorted(there & set(tables)):
        raise ValueError(f"the target already has tables of the source: {', '.join(clash[:3])}")
    fences = await source.fences()
    contents = {name: await source.read(name) for name in tables}

    def scope_of(table: str) -> str:
        return max((scope for scope in fences if table.startswith(scope + "/")), key=len, default="")

    def copied(connection: Connection) -> int:
        for scope, number in fences.items():
            sql(
                connection,
                "INSERT INTO ledger_fences (scope, number) VALUES (:scope, :number) ON CONFLICT (scope) DO UPDATE SET "
                "number = CASE WHEN excluded.number > ledger_fences.number THEN excluded.number "
                "ELSE ledger_fences.number END",
                {"scope": scope, "number": number},
            )
        count = 0
        for name, records in contents.items():
            fence = fences.get(scope_of(name), 0)
            for position, (key, record) in enumerate(records.items(), start=1):
                sql(
                    connection,
                    "INSERT INTO ledger_records (name, key, position, fence, record) "
                    "VALUES (:name, :key, :position, :fence, :record)",
                    {"name": name, "key": key, "position": position, "fence": fence, "record": json.dumps(record)},
                )
                count += 1
        return count

    return await asyncio.to_thread(target.database.write, copied, exclusive="ledger:copy")
