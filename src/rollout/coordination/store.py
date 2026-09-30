"""The coordination store: participants, an outbox of messages, and a board of posts (SQLite or Postgres).

Tool calls only write here; a `Relay` delivers the outbox through the runner. Every write a tool makes is recorded
against its `effect_id` in the same transaction, so a repeated call returns the recorded result and a known
`effect_id` with different arguments is rejected (docs/contracts/effects.md). Writes run one at a time, across every
process sharing the database, so a claim or a check-then-insert is never interleaved with another write.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import sqlalchemy as sa

from rollout.core.contracts import Conflict, ToolResult
from rollout.database import Connection, Database, fetch_all, fetch_one, sql

METADATA = sa.MetaData()
sa.Table(
    "participants",
    METADATA,
    sa.Column("name", sa.Text, primary_key=True),
    sa.Column("parent", sa.Text),
    sa.Column("purpose", sa.Text, nullable=False),
    sa.Column("created_at", sa.Text, nullable=False),
)
sa.Table(
    "outbox",
    METADATA,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("key", sa.Text, unique=True, nullable=False),
    sa.Column("recipient", sa.Text, nullable=False),
    sa.Column("sender", sa.Text, nullable=False),
    sa.Column("text", sa.Text, nullable=False),
    sa.Column("priority", sa.Text, nullable=False),
    sa.Column("created_at", sa.Text, nullable=False),
    sa.Column("delivered_at", sa.Text),
)
sa.Table(
    "posts",
    METADATA,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("channel", sa.Text, nullable=False),
    sa.Column("kind", sa.Text, nullable=False),
    sa.Column("title", sa.Text, nullable=False),
    sa.Column("body", sa.Text, nullable=False),
    sa.Column("author", sa.Text, nullable=False),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("claimed_by", sa.Text),
    sa.Column("result", sa.Text),
    sa.Column("created_at", sa.Text, nullable=False),
)
sa.Table(
    "subscriptions",
    METADATA,
    sa.Column("channel", sa.Text, primary_key=True),
    sa.Column("participant", sa.Text, primary_key=True),
)
sa.Table(
    "effects",
    METADATA,
    sa.Column("effect_id", sa.Text, primary_key=True),
    sa.Column("arguments_digest", sa.Text, nullable=False),
    sa.Column("result", sa.Text, nullable=False),
)
WRITES = "coordination writes"


@dataclass(frozen=True)
class Delivery:
    """A message waiting in the outbox."""

    id: int
    key: str
    recipient: str
    sender: str
    text: str
    priority: str  # low | normal | high


@dataclass(frozen=True)
class Participant:
    name: str
    parent: str | None
    purpose: str
    created_at: str


@dataclass(frozen=True)
class Post:
    id: int
    channel: str
    kind: str  # note | task
    title: str
    body: str
    author: str
    status: str  # open | claimed | done (notes stay open)
    claimed_by: str | None
    result: str | None
    created_at: str


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class CoordinationStore:
    def __init__(self, database: Database | Path) -> None:
        """A `Database` (shared with other processes, for Postgres), or the path of a SQLite file."""
        self._owns_database = isinstance(database, Path)
        self.database = Database.sqlite(database) if isinstance(database, Path) else database
        self.database.create(METADATA)
        self._outbox_listeners: list[Callable[[], None]] = []

    def recorded(
        self, effect_id: str, arguments_digest: str, perform: Callable[[Connection], ToolResult]
    ) -> ToolResult:
        """Run a write once per effect: `perform` and the record of its result commit in one transaction."""

        def once(db: Connection) -> ToolResult:
            row = fetch_one(db, "SELECT arguments_digest, result FROM effects WHERE effect_id = :id", {"id": effect_id})
            if row is not None:
                recorded_digest, recorded_result = row
                if recorded_digest != arguments_digest:
                    raise Conflict(f"effect {effect_id} was recorded with different arguments")
                return ToolResult.model_validate_json(recorded_result)
            result = perform(db)
            sql(
                db,
                "INSERT INTO effects (effect_id, arguments_digest, result) VALUES (:id, :digest, :result)",
                {"id": effect_id, "digest": arguments_digest, "result": result.model_dump_json()},
            )
            return result

        result = self.database.write(once, exclusive=WRITES)
        self._wake_relay()
        return result

    def read[T](self, query: Callable[[Connection], T]) -> T:
        return self.database.read(query)

    def write[T](self, change: Callable[[Connection], T]) -> T:
        """A write outside any tool call (e.g. by an operator)."""
        value = self.database.write(change, exclusive=WRITES)
        self._wake_relay()
        return value

    # Participants

    def participant(self, name: str) -> Participant | None:
        row = self.read(
            lambda db: fetch_one(
                db, "SELECT name, parent, purpose, created_at FROM participants WHERE name = :name", {"name": name}
            )
        )
        return Participant(*row) if row else None

    def participants(self) -> list[Participant]:
        rows = self.read(
            lambda db: fetch_all(db, "SELECT name, parent, purpose, created_at FROM participants ORDER BY created_at")
        )
        return [Participant(*row) for row in rows]

    # Board

    def posts(self, channel: str | None = None, status: str | None = None, limit: int = 20) -> list[Post]:
        clauses: list[str] = []
        if channel:
            clauses.append("channel = :channel")
        if status:
            clauses.append("status = :status")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.read(
            lambda db: fetch_all(
                db,
                "SELECT id, channel, kind, title, body, author, status, claimed_by, result, created_at "
                f"FROM posts {where} ORDER BY id DESC LIMIT :limit",
                {"channel": channel, "status": status, "limit": limit},
            )
        )
        return [Post(*row) for row in rows]

    # Outbox

    def pending(self) -> list[Delivery]:
        rows = self.read(
            lambda db: fetch_all(
                db,
                "SELECT id, key, recipient, sender, text, priority FROM outbox WHERE delivered_at IS NULL ORDER BY id",
            )
        )
        return [Delivery(*row) for row in rows]

    def delivered(self, delivery_id: int) -> None:
        self.database.write(
            lambda db: sql(db, "UPDATE outbox SET delivered_at = :at WHERE id = :id", {"at": now(), "id": delivery_id})
        )

    def on_outbox(self, listener: Callable[[], None]) -> None:
        self._outbox_listeners.append(listener)

    def _wake_relay(self) -> None:
        for listener in self._outbox_listeners:
            listener()

    def close(self) -> None:
        """Close the database if this store opened it (a shared `Database` is closed by its owner)."""
        if self._owns_database:
            self.database.close()


def enqueue(database: Connection, key: str, recipient: str, sender: str, text: str, priority: str = "normal") -> None:
    """Add a message to the outbox, inside the caller's transaction; a repeated key is ignored."""
    sql(
        database,
        "INSERT INTO outbox (key, recipient, sender, text, priority, created_at) "
        "VALUES (:key, :recipient, :sender, :text, :priority, :at) ON CONFLICT DO NOTHING",
        {"key": key, "recipient": recipient, "sender": sender, "text": text, "priority": priority, "at": now()},
    )
