"""Notes: the assistant's memory across conversations, as an imported tool set backed by SQLite.

Saving a note is a side effect, so the store deduplicates by `effect_id` (docs/contracts/effects.md): a repeated
effect returns the recorded result, and a known `effect_id` with different arguments is rejected as a conflict.
"""

import asyncio
import json
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import JsonValue

from rollout.core.contracts import Conflict, RetryClass, Text, ToolAnnotations, ToolResult, ToolSpecification

SPECIFICATIONS = [
    ToolSpecification(
        name="save_note",
        description="Save a note for later conversations: a decision, a finding, a preference, a to-do.",
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "A short, searchable title."},
                "body": {"type": "string"},
            },
            "required": ["title", "body"],
            "additionalProperties": False,
        },
        retry_class=RetryClass.SIDE_EFFECTING,
    ),
    ToolSpecification(
        name="search_notes",
        description="Search saved notes by words in their title or body. Returns the most recent matches.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        annotations=ToolAnnotations(read_only_hint=True),
        retry_class=RetryClass.IDEMPOTENT,
    ),
    ToolSpecification(
        name="list_notes",
        description="List the most recent notes.",
        input_schema={
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50}},
            "additionalProperties": False,
        },
        annotations=ToolAnnotations(read_only_hint=True),
        retry_class=RetryClass.IDEMPOTENT,
    ),
]


class NotesStore:
    """Implements `ToolSet`. One store per project; every conversation shares it."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._database = sqlite3.connect(path, check_same_thread=False)
        with self._database:
            self._database.executescript(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY, title TEXT NOT NULL, body TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS effects (
                    effect_id TEXT PRIMARY KEY, arguments_digest TEXT NOT NULL, result TEXT NOT NULL);
                """
            )

    def specifications(self) -> Sequence[ToolSpecification]:
        return SPECIFICATIONS

    async def call(
        self, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
        return await asyncio.to_thread(self._call, name, arguments, effect_id, arguments_digest)

    def notes(self) -> list[tuple[int, str, str]]:
        """Every note as (id, title, body), oldest first; for tests and evaluations."""
        with self._lock:
            return list(self._database.execute("SELECT id, title, body FROM notes ORDER BY id"))

    def _call(self, name: str, arguments: Mapping[str, JsonValue], effect_id: str, arguments_digest: str) -> ToolResult:
        with self._lock:
            if name == "save_note":
                return self._save(str(arguments["title"]), str(arguments["body"]), effect_id, arguments_digest)
            if name == "search_notes":
                words = str(arguments["query"]).split()
                clause = " AND ".join("(title LIKE ? OR body LIKE ?)" for _ in words) or "1"
                values = [value for word in words for value in (f"%{word}%", f"%{word}%")]
                rows = self._database.execute(
                    f"SELECT id, title, body, created_at FROM notes WHERE {clause} ORDER BY id DESC LIMIT 20", values
                ).fetchall()
                return _rows(rows)
            if name == "list_notes":
                limit = arguments.get("limit")
                count = int(limit) if isinstance(limit, int) else 20
                rows = self._database.execute(
                    "SELECT id, title, body, created_at FROM notes ORDER BY id DESC LIMIT ?", (count,)
                ).fetchall()
                return _rows(rows)
            return ToolResult(content=[Text(text=f"unknown tool {name!r}")], is_error=True)

    def _save(self, title: str, body: str, effect_id: str, arguments_digest: str) -> ToolResult:
        recorded = self._database.execute(
            "SELECT arguments_digest, result FROM effects WHERE effect_id = ?", (effect_id,)
        ).fetchone()
        if recorded is not None:
            if recorded[0] != arguments_digest:
                raise Conflict(f"effect {effect_id} was recorded with different arguments")
            return ToolResult.model_validate_json(recorded[1])
        with self._database:  # one transaction: the note and its effect record
            cursor = self._database.execute(
                "INSERT INTO notes (title, body, created_at) VALUES (?, ?, ?)",
                (title, body, datetime.now(UTC).isoformat(timespec="seconds")),
            )
            result = ToolResult(content=[Text(text=f"Saved note {cursor.lastrowid}: {title}")])
            self._database.execute(
                "INSERT INTO effects (effect_id, arguments_digest, result) VALUES (?, ?, ?)",
                (effect_id, arguments_digest, result.model_dump_json()),
            )
        return result


def _rows(rows: list[tuple[int, str, str, str]]) -> ToolResult:
    if not rows:
        return ToolResult(content=[Text(text="(no notes)")])
    text = "\n\n".join(f"#{row[0]} {row[1]} ({row[3]})\n{row[2]}" for row in rows)
    return ToolResult(content=[Text(text=text)], structured=json.loads(json.dumps([list(row) for row in rows])))
