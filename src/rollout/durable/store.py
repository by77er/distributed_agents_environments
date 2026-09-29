"""The durable runner's own tables: runs, their events, conversations and delivered messages (SQLite).

DBOS keeps the journal (workflow inputs, step results, messages). This store keeps what consumers read: the typed run
events, written as a projection. Events are keyed by (run_id, seq) and written with INSERT OR IGNORE, so a replayed
run, which regenerates identical events, never duplicates one.
"""

import asyncio
import json
import sqlite3
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from rollout.core.contracts import RunEvent


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    specification: str
    conversation: str | None
    """The conversation address (`{deployment}/{key}`), if the run serves one."""
    status: str
    outcome: str | None


class RunStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._database = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._database.execute("PRAGMA journal_mode=WAL")
        self._database.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, specification TEXT NOT NULL, conversation TEXT,
                status TEXT NOT NULL, outcome TEXT);
            CREATE TABLE IF NOT EXISTS events (
                run_id TEXT NOT NULL, seq INTEGER NOT NULL, event TEXT NOT NULL, PRIMARY KEY (run_id, seq));
            CREATE TABLE IF NOT EXISTS conversations (
                address TEXT PRIMARY KEY, conversation_key TEXT NOT NULL, live_run_id TEXT);
            CREATE TABLE IF NOT EXISTS conversation_runs (
                address TEXT NOT NULL, run_id TEXT NOT NULL, position INTEGER NOT NULL, PRIMARY KEY (address, run_id));
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY, address TEXT NOT NULL);
            """
        )
        self._signals: dict[str, asyncio.Event] = defaultdict(asyncio.Event)

    # Runs

    def create_run(
        self, run_id: str, specification: JsonValue, conversation: str | None, conversation_key: JsonValue = None
    ) -> None:
        with self._lock:
            self._database.execute(
                "INSERT OR IGNORE INTO runs (run_id, specification, conversation, status) VALUES (?, ?, ?, 'running')",
                (run_id, json.dumps(specification), conversation),
            )
            if conversation is not None:
                position = self._database.execute(
                    "SELECT COUNT(*) FROM conversation_runs WHERE address = ?", (conversation,)
                ).fetchone()[0]
                self._database.execute(
                    "INSERT OR IGNORE INTO conversation_runs (address, run_id, position) VALUES (?, ?, ?)",
                    (conversation, run_id, position),
                )
                self._database.execute(
                    "INSERT INTO conversations (address, conversation_key, live_run_id) VALUES (?, ?, ?) "
                    "ON CONFLICT (address) DO UPDATE SET live_run_id = excluded.live_run_id",
                    (conversation, json.dumps(conversation_key), run_id),
                )

    def finish_run(self, run_id: str, status: str, outcome: JsonValue) -> None:
        with self._lock:
            self._database.execute(
                "UPDATE runs SET status = ?, outcome = ? WHERE run_id = ?", (status, json.dumps(outcome), run_id)
            )
            self._database.execute("UPDATE conversations SET live_run_id = NULL WHERE live_run_id = ?", (run_id,))
        self._notify(run_id)

    def run(self, run_id: str) -> RunRecord | None:
        with self._lock:
            row = self._database.execute(
                "SELECT run_id, specification, conversation, status, outcome FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return RunRecord(*row) if row else None

    def unfinished_runs(self) -> list[str]:
        with self._lock:
            return [row[0] for row in self._database.execute("SELECT run_id FROM runs WHERE status = 'running'")]

    # Conversations and messages

    def live_run(self, address: str) -> str | None:
        with self._lock:
            row = self._database.execute(
                "SELECT live_run_id FROM conversations WHERE address = ?", (address,)
            ).fetchone()
        return row[0] if row else None

    def conversation_key(self, address: str) -> JsonValue:
        with self._lock:
            row = self._database.execute(
                "SELECT conversation_key FROM conversations WHERE address = ?", (address,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def conversation_runs(self, address: str) -> list[str]:
        with self._lock:
            rows = self._database.execute(
                "SELECT run_id FROM conversation_runs WHERE address = ? ORDER BY position", (address,)
            ).fetchall()
        return [row[0] for row in rows]

    def claim_message(self, message_id: str, address: str) -> bool:
        """Record a message as delivered; False if it already was (a retry)."""
        with self._lock:
            cursor = self._database.execute(
                "INSERT OR IGNORE INTO messages (message_id, address) VALUES (?, ?)", (message_id, address)
            )
        return cursor.rowcount == 1

    # Events

    def append(self, event: RunEvent) -> None:
        with self._lock:
            self._database.execute(
                "INSERT OR IGNORE INTO events (run_id, seq, event) VALUES (?, ?, ?)",
                (event.run_id, event.seq, event.model_dump_json()),
            )
        self._notify(event.run_id)

    def events(self, run_id: str, from_seq: int = 0) -> list[RunEvent]:
        with self._lock:
            rows = self._database.execute(
                "SELECT event FROM events WHERE run_id = ? AND seq >= ? ORDER BY seq", (run_id, from_seq)
            ).fetchall()
        return [RunEvent.model_validate_json(row[0]) for row in rows]

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
        with self._lock:
            self._database.close()
