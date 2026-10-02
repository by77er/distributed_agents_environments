"""The durable runner's own tables: runs, their events, conversations, delivered messages and runners.

DBOS keeps the journal (workflow inputs, step results, messages). This store keeps what consumers read: the typed run
events, written as a projection. Events are keyed by (run_id, seq) and inserted with ON CONFLICT DO NOTHING, so a
replayed run, which regenerates identical events, never duplicates one. The store runs on SQLite for one runner, or
on a Postgres shared by several (`rollout.database`).
"""

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import sqlalchemy as sa
from pydantic import JsonValue
from sqlalchemy.schema import CreateColumn

from rollout.core.contracts import RunEvent
from rollout.database import Connection, Database, sql

RUNNING = "running"
"""The status of a run that has not ended; an ended run's status is its outcome's (`RunStatus`)."""

METADATA = sa.MetaData()
RUNS = sa.Table(
    "runs",
    METADATA,
    sa.Column("run_id", sa.Text, primary_key=True),
    sa.Column("specification", sa.Text, nullable=False),
    sa.Column("conversation", sa.Text),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("outcome", sa.Text),
    sa.Column("evicted", sa.Integer, nullable=False, server_default="0"),
    sa.Column("wake_at", sa.Text),
    sa.Column("evictions", sa.Integer, nullable=False, server_default="0"),
    sa.Column("last_activity", sa.Text),
)
sa.Table(
    "events",
    METADATA,
    sa.Column("run_id", sa.Text, primary_key=True),
    sa.Column("seq", sa.Integer, primary_key=True, autoincrement=False),
    sa.Column("event", sa.Text, nullable=False),
)
sa.Table(
    "conversations",
    METADATA,
    sa.Column("address", sa.Text, primary_key=True),
    sa.Column("conversation_key", sa.Text, nullable=False),
    sa.Column("live_run_id", sa.Text),
)
sa.Table(
    "conversation_runs",
    METADATA,
    sa.Column("address", sa.Text, primary_key=True),
    sa.Column("run_id", sa.Text, primary_key=True),
    sa.Column("position", sa.Integer, nullable=False),
)
sa.Table(
    "messages",
    METADATA,
    sa.Column("message_id", sa.Text, primary_key=True),
    sa.Column("address", sa.Text, nullable=False),
)
sa.Table("attempts", METADATA, sa.Column("effect_id", sa.Text, primary_key=True))
sa.Table(
    "runners",
    METADATA,
    sa.Column("runner_id", sa.Text, primary_key=True),
    sa.Column("heartbeat_at", sa.Float(asdecimal=False), nullable=False),
)


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    specification: str
    conversation: str | None
    """The conversation address (`{deployment}/{key}`), if the run serves one."""
    status: str
    outcome: str | None

    @property
    def running(self) -> bool:
        return self.status == RUNNING


class RunStore:
    def __init__(self, database: Database | Path) -> None:
        """A `Database`, or the path of a SQLite file."""
        self._owns_database = isinstance(database, Path)
        self.database = Database.sqlite(database) if isinstance(database, Path) else database
        self.database.create(METADATA)
        if not self.database.shared:  # a SQLite store created before `runs` had all its columns gains the others
            engine = self.database.engine
            existing = {column["name"] for column in sa.inspect(engine).get_columns(RUNS.name)}
            for column in RUNS.columns:
                if column.name not in existing:
                    definition = str(CreateColumn(column).compile(dialect=engine.dialect))
                    self.database.write(lambda db, d=definition: sql(db, f"ALTER TABLE {RUNS.name} ADD COLUMN {d}"))
        self._signals: dict[str, asyncio.Event] = defaultdict(asyncio.Event)

    # Runs

    def create_run(
        self, run_id: str, specification: JsonValue, conversation: str | None, conversation_key: JsonValue = None
    ) -> None:
        def create(db: Connection) -> None:
            sql(
                db,
                "INSERT INTO runs (run_id, specification, conversation, status) "
                "VALUES (:run_id, :specification, :conversation, :status) ON CONFLICT DO NOTHING",
                {
                    "run_id": run_id,
                    "specification": json.dumps(specification),
                    "conversation": conversation,
                    "status": RUNNING,
                },
            )
            if conversation is None:
                return
            position = sql(
                db, "SELECT COUNT(*) FROM conversation_runs WHERE address = :address", {"address": conversation}
            ).scalar_one()
            sql(
                db,
                "INSERT INTO conversation_runs (address, run_id, position) VALUES (:address, :run_id, :position) "
                "ON CONFLICT DO NOTHING",
                {"address": conversation, "run_id": run_id, "position": position},
            )
            sql(
                db,
                "INSERT INTO conversations (address, conversation_key, live_run_id) VALUES (:address, :key, :run_id) "
                "ON CONFLICT (address) DO UPDATE SET live_run_id = excluded.live_run_id",
                {"address": conversation, "key": json.dumps(conversation_key), "run_id": run_id},
            )

        self.database.write(create, exclusive=f"conversation positions:{conversation}")

    def finish_run(self, run_id: str, status: str, outcome: JsonValue) -> None:
        def finish(db: Connection) -> None:
            sql(
                db,
                "UPDATE runs SET status = :status, outcome = :outcome WHERE run_id = :run_id",
                {"status": status, "outcome": json.dumps(outcome), "run_id": run_id},
            )
            sql(db, "UPDATE conversations SET live_run_id = NULL WHERE live_run_id = :run_id", {"run_id": run_id})

        self.database.write(finish)
        self._notify(run_id)

    def run(self, run_id: str) -> RunRecord | None:
        row = self.database.read(
            lambda db: sql(
                db,
                "SELECT run_id, specification, conversation, status, outcome FROM runs WHERE run_id = :run_id",
                {"run_id": run_id},
            ).first()
        )
        return RunRecord(*row) if row else None

    def read_all_run_ids(self) -> list[str]:
        return self.database.read(lambda db: list(sql(db, "SELECT run_id FROM runs").scalars()))

    # Eviction

    def evict(self, run_id: str, wake_at: str | None) -> None:
        self.database.write(
            lambda db: sql(
                db,
                "UPDATE runs SET evicted = 1, wake_at = :wake_at, evictions = evictions + 1 WHERE run_id = :run_id",
                {"wake_at": wake_at, "run_id": run_id},
            )
        )

    def wake(self, run_id: str, at: str) -> None:
        self.database.write(
            lambda db: sql(
                db,
                "UPDATE runs SET evicted = 0, wake_at = NULL, last_activity = :at WHERE run_id = :run_id",
                {"at": at, "run_id": run_id},
            )
        )

    def touch(self, run_id: str, at: str) -> None:
        """Record that the run was messaged: it must not be evicted until it suspends again."""
        self.database.write(
            lambda db: sql(
                db, "UPDATE runs SET last_activity = :at WHERE run_id = :run_id", {"at": at, "run_id": run_id}
            )
        )

    def last_activity(self, run_id: str) -> str | None:
        """When the run was last messaged or woken."""
        return self.database.read(
            lambda db: sql(db, "SELECT last_activity FROM runs WHERE run_id = :run_id", {"run_id": run_id}).scalar()
        )

    def evictions(self) -> int:
        """How many times runs were evicted, in total."""
        return int(self.database.read(lambda db: sql(db, "SELECT COALESCE(SUM(evictions), 0) FROM runs").scalar_one()))

    def is_evicted(self, run_id: str) -> bool:
        evicted = self.database.read(
            lambda db: sql(db, "SELECT evicted FROM runs WHERE run_id = :run_id", {"run_id": run_id}).scalar()
        )
        return bool(evicted)

    def due_for_waking(self, now: str) -> list[str]:
        """Evicted runs whose wait times out by `now`."""
        return self.database.read(
            lambda db: list(
                sql(
                    db,
                    "SELECT run_id FROM runs WHERE status = :running AND evicted = 1 AND wake_at IS NOT NULL "
                    "AND wake_at <= :now",
                    {"running": RUNNING, "now": now},
                ).scalars()
            )
        )

    def last_events(self, run_ids: list[str]) -> list[RunEvent]:
        """The latest event of each of these runs that is running and resident."""
        if not run_ids:
            return []
        rows = self.database.read(
            lambda db: db.execute(
                sa.text(
                    "SELECT e.event FROM events e JOIN runs r ON r.run_id = e.run_id "
                    "WHERE r.status = :running AND r.evicted = 0 AND r.run_id IN :run_ids "
                    "AND e.seq = (SELECT MAX(seq) FROM events WHERE run_id = e.run_id)"
                ).bindparams(sa.bindparam("run_ids", expanding=True)),
                {"running": RUNNING, "run_ids": run_ids},
            ).scalars()
        )
        return [RunEvent.model_validate_json(row) for row in rows]

    # Conversations and messages

    def live_run(self, address: str) -> str | None:
        return self.database.read(
            lambda db: sql(
                db, "SELECT live_run_id FROM conversations WHERE address = :address", {"address": address}
            ).scalar()
        )

    def conversation_key(self, address: str) -> JsonValue:
        stored = self.database.read(
            lambda db: sql(
                db, "SELECT conversation_key FROM conversations WHERE address = :address", {"address": address}
            ).scalar()
        )
        return json.loads(stored) if stored is not None else None

    def conversation_runs(self, address: str) -> list[str]:
        return self.database.read(
            lambda db: list(
                sql(
                    db,
                    "SELECT run_id FROM conversation_runs WHERE address = :address ORDER BY position",
                    {"address": address},
                ).scalars()
            )
        )

    def is_claimed(self, message_id: str) -> bool:
        return self.database.read(
            lambda db: sql(db, "SELECT 1 FROM messages WHERE message_id = :id", {"id": message_id}).first() is not None
        )

    def claim_message(self, message_id: str, address: str) -> bool:
        """Record a message as delivered; False if it already was (a retry)."""
        inserted = self.database.write(
            lambda db: (
                sql(
                    db,
                    "INSERT INTO messages (message_id, address) VALUES (:id, :address) ON CONFLICT DO NOTHING",
                    {"id": message_id, "address": address},
                ).rowcount
            )
        )
        return inserted == 1

    # Attempt markers

    def mark_attempt(self, effect_id: str) -> bool:
        """Record that a guarded effect is starting; False if an earlier attempt already started it."""
        inserted = self.database.write(
            lambda db: (
                sql(
                    db, "INSERT INTO attempts (effect_id) VALUES (:id) ON CONFLICT DO NOTHING", {"id": effect_id}
                ).rowcount
            )
        )
        return inserted == 1

    # Runners sharing the database

    def heartbeat(self, runner_id: str, at: float) -> None:
        self.database.write(
            lambda db: sql(
                db,
                "INSERT INTO runners (runner_id, heartbeat_at) VALUES (:id, :at) "
                "ON CONFLICT (runner_id) DO UPDATE SET heartbeat_at = excluded.heartbeat_at",
                {"id": runner_id, "at": at},
            )
        )

    def stale_runners(self, before: float) -> list[str]:
        """Runners whose last heartbeat is older than `before`."""
        return self.database.read(
            lambda db: list(
                sql(db, "SELECT runner_id FROM runners WHERE heartbeat_at < :before", {"before": before}).scalars()
            )
        )

    def forget_runner(self, runner_id: str) -> None:
        self.database.write(lambda db: sql(db, "DELETE FROM runners WHERE runner_id = :id", {"id": runner_id}))

    # Events

    def append(self, event: RunEvent) -> None:
        self.database.write(
            lambda db: sql(
                db,
                "INSERT INTO events (run_id, seq, event) VALUES (:run_id, :seq, :event) ON CONFLICT DO NOTHING",
                {"run_id": event.run_id, "seq": event.seq, "event": event.model_dump_json()},
            )
        )
        self._notify(event.run_id)

    def events(self, run_id: str, from_seq: int = 0) -> list[RunEvent]:
        rows = self.database.read(
            lambda db: list(
                sql(
                    db,
                    "SELECT event FROM events WHERE run_id = :run_id AND seq >= :seq ORDER BY seq",
                    {"run_id": run_id, "seq": from_seq},
                ).scalars()
            )
        )
        return [RunEvent.model_validate_json(row) for row in rows]

    async def changed(self, run_id: str, wait_seconds: float) -> None:
        """Wait until the run records something, or `wait_seconds` pass (other processes write without notifying)."""
        signal = self._signals[run_id]
        try:
            async with asyncio.timeout(wait_seconds):
                await signal.wait()
        except TimeoutError:
            pass

    def _notify(self, run_id: str) -> None:
        signal = self._signals.pop(run_id, None)
        if signal is not None:
            signal.set()

    def close(self) -> None:
        """Close the database if this store opened it (a shared `Database` is closed by its owner)."""
        if self._owns_database:
            self.database.close()
