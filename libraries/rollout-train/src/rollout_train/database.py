"""`DatabaseLedger`: the ledger in SQL tables, which every run and machine using the database shares (it needs the
`durable` extra: the database is `rollout_durable`'s).

Two tables hold it: `ledger_records` (a row per record: its table's name, its key, the order it was appended in, the
fence it was written under, and the record as JSON; a table has each key once) and `ledger_fences` (the newest fence
of every scope). Taking a fence and appending run in transactions that hold the scope's lock, so a writer that was
replaced is refused (`Fenced`) whichever process it is in. SQLite serves one machine; Postgres serves several.

`DatabaseRegistry` is the registry (`rollout_train.registry`) beside it, in three tables of the same database: `runs`
(each run's id and name, a name once), `bookmarks` (each bookmark's name and checkpoint) and `dataset_names` (each
dataset's name and id); a database ledger's is its `registry`. `DatabasePresence` holds the runners' heartbeats
(`rollout_train.presence`) in another, `presence`: a row per runner, changed in place; a database ledger's is its
`presence`. `DatabaseLaunches` holds the runs asked for (`rollout_train.launches`) in another, `launches`; a database
ledger's is its `launches`. `DatabaseDesiredSettings` holds what is wanted of each run's settings
(`rollout_train.settings`) in another, `run_settings`: a row per run, changed in place; a database ledger's is its
`desired_settings`. `DatabaseLeases` holds the sandbox pools' leases (`rollout_train.sandboxes`) in another,
`sandboxes`: a row per lease; a database ledger's is its `sandboxes`.
"""

import asyncio
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from pydantic import JsonValue

from rollout.harness.sandboxes import Lease
from rollout_durable.database import Connection, Database, fetch_all, fetch_one, sql
from rollout_train.launches import ASKED, CLAIMED, Asked, Launch, as_launch, new_launch
from rollout_train.ledger import Fence, Fenced, Ledger
from rollout_train.presence import Beat, kept
from rollout_train.registry import Bookmark, Entry, Named, Taken, checked, found, new_run_id, registry_of, valid
from rollout_train.settings import Desired

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
RUNS = sa.Table(
    "runs",
    METADATA,
    sa.Column("id", sa.Text, primary_key=True),
    sa.Column("name", sa.Text, nullable=False, unique=True),
    sa.Column("created", sa.Float(), nullable=False),
)
BOOKMARKS = sa.Table(
    "bookmarks",
    METADATA,
    sa.Column("name", sa.Text, primary_key=True),
    sa.Column("checkpoint", sa.Text, nullable=False),
    sa.Column("moved", sa.Float(), nullable=False),
)
DATASET_NAMES = sa.Table(
    "dataset_names",
    METADATA,
    sa.Column("name", sa.Text, primary_key=True),
    sa.Column("dataset", sa.Text, nullable=False),
    sa.Column("named", sa.Float(), nullable=False),
)
LAUNCHES = sa.Table(
    "launches",
    METADATA,
    sa.Column("id", sa.Text, primary_key=True),
    sa.Column("at", sa.Float(), nullable=False),
    sa.Column("state", sa.Text, nullable=False),
    sa.Column("launch", sa.Text, nullable=False),
)
PRESENCE = sa.Table(
    "presence",
    METADATA,
    sa.Column("runner", sa.Text, primary_key=True),
    sa.Column("at", sa.Float(), nullable=False),
    sa.Column("about", sa.Text, nullable=False),
    sa.Column("history", sa.Text, nullable=False),
)
RUN_SETTINGS = sa.Table(
    "run_settings",
    METADATA,
    sa.Column("run", sa.Text, primary_key=True),
    sa.Column("settings", sa.Text, nullable=False),
    sa.Column("changed", sa.Float(), nullable=False),
)


SANDBOXES = sa.Table(
    "sandboxes",
    METADATA,
    sa.Column("key", sa.Text, primary_key=True),
    sa.Column("pool", sa.Text, nullable=False),
    sa.Column("lease", sa.Text, nullable=False),
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

    @property
    def registry(self) -> "DatabaseRegistry":
        """The registry of run names and bookmarks, in this ledger's database."""
        return DatabaseRegistry(self.database)

    @property
    def launches(self) -> "DatabaseLaunches":
        """The runs asked for, in this ledger's database."""
        return DatabaseLaunches(self.database)

    @property
    def presence(self) -> "DatabasePresence":
        """The runners' heartbeats, in this ledger's database."""
        return DatabasePresence(self.database)

    @property
    def desired_settings(self) -> "DatabaseDesiredSettings":
        """What is wanted of each run's settings, in this ledger's database."""
        return DatabaseDesiredSettings(self.database)

    @property
    def sandboxes(self) -> "DatabaseLeases":
        """The sandbox pools' leases, in this ledger's database."""
        return DatabaseLeases(self.database)

    def close(self) -> None:
        self.database.close()


class DatabaseRegistry:
    """A `Registry` (`rollout_train.registry`) in the `runs` and `bookmarks` tables of a database."""

    def __init__(self, database: Database) -> None:
        self.database = database

    async def runs(self) -> list[Entry]:
        return await asyncio.to_thread(self.database.read, _runs)

    async def create(self, name: str, id: str | None = None) -> Entry:
        made = id or new_run_id()

        def created(connection: Connection) -> Entry:
            runs = _runs(connection)
            if any(each.id == made for each in runs):
                raise Taken(f"there is a run {made} already")
            entry = Entry(made, checked(name, made, runs), round(time.time(), 1))
            sql(connection, "INSERT INTO runs (id, name, created) VALUES (:id, :name, :created)", asdict(entry))
            return entry

        return await asyncio.to_thread(self.database.write, created, exclusive="registry")

    async def rename(self, who: str, name: str) -> Entry:
        def renamed(connection: Connection) -> Entry:
            runs = _runs(connection)
            if (was := found(runs, who)) is None:
                raise KeyError(f"there is no run {who!r}")
            entry = Entry(was.id, checked(name, was.id, runs), was.created)
            sql(connection, "UPDATE runs SET name = :name WHERE id = :id", {"name": entry.name, "id": was.id})
            return entry

        return await asyncio.to_thread(self.database.write, renamed, exclusive="registry")

    async def bookmarks(self) -> list[Bookmark]:
        def rows(connection: Connection) -> list[tuple[Any, ...]]:
            return fetch_all(connection, "SELECT name, checkpoint, moved FROM bookmarks ORDER BY name")

        return [Bookmark(*row) for row in await asyncio.to_thread(self.database.read, rows)]

    async def bookmark(self, name: str, checkpoint: str) -> Bookmark:
        mark = Bookmark(valid(name), checkpoint, round(time.time(), 1))

        def moved(connection: Connection) -> Bookmark:
            sql(
                connection,
                "INSERT INTO bookmarks (name, checkpoint, moved) VALUES (:name, :checkpoint, :moved) "
                "ON CONFLICT (name) DO UPDATE SET checkpoint = excluded.checkpoint, moved = excluded.moved",
                asdict(mark),
            )
            return mark

        return await asyncio.to_thread(self.database.write, moved, exclusive="registry")

    async def unbookmark(self, name: str) -> None:
        def taken(connection: Connection) -> None:
            if sql(connection, "DELETE FROM bookmarks WHERE name = :name", {"name": name}).rowcount == 0:
                raise KeyError(f"there is no bookmark {name!r}")

        await asyncio.to_thread(self.database.write, taken, exclusive="registry")

    async def datasets(self) -> list[Named]:
        def rows(connection: Connection) -> list[tuple[Any, ...]]:
            return fetch_all(connection, "SELECT name, dataset, named FROM dataset_names ORDER BY name")

        return [Named(*row) for row in await asyncio.to_thread(self.database.read, rows)]

    async def name_dataset(self, name: str, dataset: str) -> Named:
        entry = Named(valid(name), dataset, round(time.time(), 1))

        def given(connection: Connection) -> Named:
            row = fetch_one(connection, "SELECT dataset FROM dataset_names WHERE name = :name", {"name": entry.name})
            if row is not None and row[0] != dataset:
                raise Taken(f"another dataset is called {entry.name!r}")
            if row is None:
                sql(connection, "INSERT INTO dataset_names (name, dataset, named) VALUES (:name, :dataset, :named)",
                    asdict(entry))  # fmt: skip
            return entry

        return await asyncio.to_thread(self.database.write, given, exclusive="registry")


class DatabaseLaunches:
    """`Launches` (`rollout_train.launches`) in the `launches` table of a database."""

    def __init__(self, database: Database) -> None:
        self.database = database

    async def ask(self, asked: Asked) -> Launch:
        made = new_launch(asked)

        def added(connection: Connection) -> None:
            sql(connection, "INSERT INTO launches (id, at, state, launch) VALUES (:id, :at, :state, :launch)",
                {"id": made.id, "at": made.at, "state": made.state, "launch": json.dumps(asdict(made))})  # fmt: skip

        await asyncio.to_thread(self.database.write, added, exclusive="launches")
        return made

    async def all(self) -> list[Launch]:
        def rows(connection: Connection) -> list[tuple[Any, ...]]:
            return fetch_all(connection, "SELECT launch FROM launches ORDER BY at DESC")

        return [as_launch(json.loads(launch)) for (launch,) in await asyncio.to_thread(self.database.read, rows)]

    async def claim(self, id: str, launcher: str) -> Launch | None:
        def claimed(connection: Connection) -> Launch | None:
            row = fetch_one(connection, "SELECT launch FROM launches WHERE id = :id AND state = :asked",
                            {"id": id, "asked": ASKED})  # fmt: skip
            if row is None:
                return None
            launch = replace(as_launch(json.loads(row[0])), state=CLAIMED, launcher=launcher, updated=time.time())
            self._write(connection, launch)
            return launch

        return await asyncio.to_thread(self.database.write, claimed, exclusive="launches")

    async def note(self, id: str, **changes: Any) -> Launch:
        def noted(connection: Connection) -> Launch:
            row = fetch_one(connection, "SELECT launch FROM launches WHERE id = :id", {"id": id})
            if row is None:
                raise KeyError(f"there is no launch {id}")
            launch = replace(as_launch(json.loads(row[0])), **changes, updated=round(time.time(), 1))
            self._write(connection, launch)
            return launch

        return await asyncio.to_thread(self.database.write, noted, exclusive="launches")

    @staticmethod
    def _write(connection: Connection, launch: Launch) -> None:
        sql(connection, "UPDATE launches SET state = :state, launch = :launch WHERE id = :id",
            {"id": launch.id, "state": launch.state, "launch": json.dumps(asdict(launch))})  # fmt: skip


class DatabasePresence:
    """`Presence` (`rollout_train.presence`) in the `presence` table of a database."""

    def __init__(self, database: Database) -> None:
        self.database = database

    async def beat(self, runner: str, about: Mapping[str, JsonValue]) -> None:
        at = round(time.time(), 1)

        def noted(connection: Connection) -> None:
            row = fetch_one(connection, "SELECT history FROM presence WHERE runner = :runner", {"runner": runner})
            history = kept(json.loads(row[0]) if row else [], at, about)
            sql(
                connection,
                "INSERT INTO presence (runner, at, about, history) VALUES (:runner, :at, :about, :history) "
                "ON CONFLICT (runner) DO UPDATE SET at = excluded.at, about = excluded.about, "
                "history = excluded.history",
                {"runner": runner, "at": at, "about": json.dumps(dict(about)), "history": json.dumps(history)},
            )

        await asyncio.to_thread(self.database.write, noted, exclusive=f"presence:{runner}")

    async def beats(self) -> list[Beat]:
        def rows(connection: Connection) -> list[tuple[Any, ...]]:
            return fetch_all(connection, "SELECT runner, at, about, history FROM presence ORDER BY runner")

        found = await asyncio.to_thread(self.database.read, rows)
        return [
            Beat(str(runner), float(at), json.loads(about), json.loads(history)) for runner, at, about, history in found
        ]


class DatabaseDesiredSettings:
    """`DesiredSettings` (`rollout_train.settings`) in the `run_settings` table of a database."""

    def __init__(self, database: Database) -> None:
        self.database = database

    async def desired(self, run: str) -> Desired | None:
        def row(connection: Connection) -> tuple[Any, ...] | None:
            return fetch_one(connection, "SELECT settings, changed FROM run_settings WHERE run = :run", {"run": run})

        found = await asyncio.to_thread(self.database.read, row)
        return Desired(run, json.loads(found[0]), float(found[1])) if found else None

    async def want(self, run: str, settings: Mapping[str, JsonValue]) -> Desired:
        def changed(connection: Connection) -> Desired:
            row = fetch_one(connection, "SELECT settings FROM run_settings WHERE run = :run", {"run": run})
            now = Desired(run, {**(json.loads(row[0]) if row else {}), **settings}, round(time.time(), 1))
            sql(
                connection,
                "INSERT INTO run_settings (run, settings, changed) VALUES (:run, :settings, :changed) "
                "ON CONFLICT (run) DO UPDATE SET settings = excluded.settings, changed = excluded.changed",
                {"run": run, "settings": json.dumps(dict(now.settings)), "changed": now.changed},
            )
            return now

        return await asyncio.to_thread(self.database.write, changed, exclusive=f"run_settings:{run}")


class DatabaseLeases:
    """`Leases` (`rollout.harness.sandboxes`) in the `sandboxes` table of a database."""

    def __init__(self, database: Database) -> None:
        self.database = database

    async def get(self, key: str) -> Lease | None:
        def row(connection: Connection) -> tuple[Any, ...] | None:
            return fetch_one(connection, "SELECT lease FROM sandboxes WHERE key = :key", {"key": key})

        found = await asyncio.to_thread(self.database.read, row)
        return Lease.model_validate_json(found[0]) if found else None

    async def put(self, lease: Lease) -> None:
        def written(connection: Connection) -> None:
            sql(
                connection,
                "INSERT INTO sandboxes (key, pool, lease) VALUES (:key, :pool, :lease) "
                "ON CONFLICT (key) DO UPDATE SET pool = excluded.pool, lease = excluded.lease",
                {"key": lease.key, "pool": lease.pool, "lease": lease.model_dump_json()},
            )

        await asyncio.to_thread(self.database.write, written, exclusive=f"sandboxes:{lease.key}")

    async def delete(self, key: str) -> None:
        def deleted(connection: Connection) -> None:
            sql(connection, "DELETE FROM sandboxes WHERE key = :key", {"key": key})

        await asyncio.to_thread(self.database.write, deleted, exclusive=f"sandboxes:{key}")

    async def all(self) -> list[Lease]:
        def rows(connection: Connection) -> list[tuple[Any, ...]]:
            return fetch_all(connection, "SELECT lease FROM sandboxes ORDER BY key")

        return [Lease.model_validate_json(lease) for (lease,) in await asyncio.to_thread(self.database.read, rows)]


def _runs(connection: Connection) -> list[Entry]:
    return [Entry(*row) for row in fetch_all(connection, "SELECT id, name, created FROM runs ORDER BY created, id")]


async def copy(source: Ledger, target: DatabaseLedger) -> int:
    """Copy every table and fence of `source` (files, or another database) into `target`, which must have none of its
    tables yet; returns how many records. Records keep their keys and their order; each is noted under its scope's
    newest fence (the fence a record was written under is not read back through a ledger). A fence already in the
    target is kept if it is newer. The runs, bookmarks and dataset names registered beside `source` are registered
    beside `target` too. To move to Postgres: copy, then point the profile's `[ledger] url` at it."""
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

    count = await asyncio.to_thread(target.database.write, copied, exclusive="ledger:copy")
    if (registered := registry_of(source)) is not None:
        for entry in await registered.runs():
            await target.registry.create(entry.name, entry.id)
        for mark in await registered.bookmarks():
            await target.registry.bookmark(mark.name, mark.checkpoint)
        for each in await registered.datasets():
            await target.registry.name_dataset(each.name, each.dataset)
    return count
