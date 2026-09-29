"""The coordination store: participants, an outbox of messages, and a board of posts (SQLite).

Tool calls only write here; a `Relay` delivers the outbox through the runner. Every write a tool makes is recorded
against its `effect_id` in the same transaction, so a repeated call returns the recorded result and a known
`effect_id` with different arguments is rejected (docs/contracts/effects.md).
"""

import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rollout.core.contracts import Conflict, ToolResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS participants (
    name TEXT PRIMARY KEY, parent TEXT, purpose TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY, key TEXT UNIQUE NOT NULL, recipient TEXT NOT NULL, sender TEXT NOT NULL,
    text TEXT NOT NULL, priority TEXT NOT NULL, created_at TEXT NOT NULL, delivered_at TEXT);
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY, channel TEXT NOT NULL, kind TEXT NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL,
    author TEXT NOT NULL, status TEXT NOT NULL, claimed_by TEXT, result TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS subscriptions (
    channel TEXT NOT NULL, participant TEXT NOT NULL, PRIMARY KEY (channel, participant));
CREATE TABLE IF NOT EXISTS effects (
    effect_id TEXT PRIMARY KEY, arguments_digest TEXT NOT NULL, result TEXT NOT NULL);
"""


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
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.database = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.database.execute("PRAGMA journal_mode=WAL")
        self.database.executescript(SCHEMA)
        self._outbox_listeners: list[Callable[[], None]] = []

    def recorded(
        self, effect_id: str, arguments_digest: str, perform: Callable[[sqlite3.Connection], ToolResult]
    ) -> ToolResult:
        """Run a write once per effect: `perform` and the record of its result commit in one transaction."""
        with self._lock:
            row = self.database.execute(
                "SELECT arguments_digest, result FROM effects WHERE effect_id = ?", (effect_id,)
            ).fetchone()
            if row is not None:
                if row[0] != arguments_digest:
                    raise Conflict(f"effect {effect_id} was recorded with different arguments")
                return ToolResult.model_validate_json(row[1])
            self.database.execute("BEGIN IMMEDIATE")
            try:
                result = perform(self.database)
                self.database.execute(
                    "INSERT INTO effects (effect_id, arguments_digest, result) VALUES (?, ?, ?)",
                    (effect_id, arguments_digest, result.model_dump_json()),
                )
                self.database.execute("COMMIT")
            except BaseException:
                self.database.execute("ROLLBACK")
                raise
        self._wake_relay()
        return result

    def read[T](self, query: Callable[[sqlite3.Connection], T]) -> T:
        with self._lock:
            return query(self.database)

    def write[T](self, change: Callable[[sqlite3.Connection], T]) -> T:
        """A write outside any tool call (e.g. by an operator)."""
        with self._lock:
            self.database.execute("BEGIN IMMEDIATE")
            try:
                value = change(self.database)
                self.database.execute("COMMIT")
            except BaseException:
                self.database.execute("ROLLBACK")
                raise
        self._wake_relay()
        return value

    # Participants

    def participant(self, name: str) -> Participant | None:
        row = self.read(
            lambda db: db.execute(
                "SELECT name, parent, purpose, created_at FROM participants WHERE name = ?", (name,)
            ).fetchone()
        )
        return Participant(*row) if row else None

    def participants(self) -> list[Participant]:
        rows = self.read(
            lambda db: db.execute("SELECT name, parent, purpose, created_at FROM participants ORDER BY created_at")
        ).fetchall()
        return [Participant(*row) for row in rows]

    # Board

    def posts(self, channel: str | None = None, status: str | None = None, limit: int = 20) -> list[Post]:
        clauses: list[str] = []
        values: list[str | int] = []
        if channel:
            clauses.append("channel = ?")
            values.append(channel)
        if status:
            clauses.append("status = ?")
            values.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.read(
            lambda db: db.execute(
                "SELECT id, channel, kind, title, body, author, status, claimed_by, result, created_at "
                f"FROM posts {where} ORDER BY id DESC LIMIT ?",
                (*values, limit),
            ).fetchall()
        )
        return [Post(*row) for row in rows]

    # Outbox

    def pending(self) -> list[Delivery]:
        rows = self.read(
            lambda db: db.execute(
                "SELECT id, key, recipient, sender, text, priority FROM outbox WHERE delivered_at IS NULL ORDER BY id"
            ).fetchall()
        )
        return [Delivery(*row) for row in rows]

    def delivered(self, delivery_id: int) -> None:
        self.read(lambda db: db.execute("UPDATE outbox SET delivered_at = ? WHERE id = ?", (now(), delivery_id)))

    def on_outbox(self, listener: Callable[[], None]) -> None:
        self._outbox_listeners.append(listener)

    def _wake_relay(self) -> None:
        for listener in self._outbox_listeners:
            listener()

    def close(self) -> None:
        with self._lock:
            self.database.close()


def enqueue(
    database: sqlite3.Connection, key: str, recipient: str, sender: str, text: str, priority: str = "normal"
) -> None:
    """Add a message to the outbox, inside the caller's transaction; a repeated key is ignored."""
    database.execute(
        "INSERT OR IGNORE INTO outbox (key, recipient, sender, text, priority, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (key, recipient, sender, text, priority, now()),
    )
