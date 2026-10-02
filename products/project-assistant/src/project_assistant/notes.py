"""Notes: the assistant's memory across conversations, as an imported tool set backed by SQLite.

Saving a note is a side effect, so the store deduplicates by `effect_id`
(docs/libraries/rollout/contracts/effects.md): a repeated effect returns the recorded result, and a known `effect_id`
with different arguments is rejected as a conflict.
"""

import asyncio
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from pydantic import JsonValue

from rollout.contracts import RetryClass, Text, ToolResult, ToolSpecification
from rollout_durable.database import Connection, Database, effects_table, fetch_all, recorded, sql

METADATA = sa.MetaData()
sa.Table(
    "notes",
    METADATA,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("title", sa.Text, nullable=False),
    sa.Column("body", sa.Text, nullable=False),
    sa.Column("created_at", sa.Text, nullable=False),
)
effects_table(METADATA)

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
        retry_class=RetryClass.IDEMPOTENT,
    ),
]


class NotesStore:
    """Implements `ToolSet`. One store per project; every conversation shares it."""

    deduplicates = True
    """Saves are performed at most once per effect_id, so they need no attempt marker."""

    def __init__(self, path: Path) -> None:
        self._database = Database.sqlite(path)
        self._database.create(METADATA)

    def specifications(self) -> Sequence[ToolSpecification]:
        return SPECIFICATIONS

    async def call(
        self, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
        return await asyncio.to_thread(self._call, name, arguments, effect_id, arguments_digest)

    def notes(self) -> list[tuple[int, str, str]]:
        """Every note as (id, title, body), oldest first; for tests and evaluations."""
        rows = self._database.read(lambda db: fetch_all(db, "SELECT id, title, body FROM notes ORDER BY id"))
        return [(int(note_id), str(title), str(body)) for note_id, title, body in rows]

    def close(self) -> None:
        self._database.close()

    def _call(self, name: str, arguments: Mapping[str, JsonValue], effect_id: str, arguments_digest: str) -> ToolResult:
        if name == "save_note":
            title, body = str(arguments["title"]), str(arguments["body"])
            return self._database.write(
                lambda db: recorded(db, effect_id, arguments_digest, lambda db: _save(db, title, body))
            )
        if name == "search_notes":
            words = {f"word{index}": f"%{word}%" for index, word in enumerate(str(arguments["query"]).split())}
            clause = " AND ".join(f"(title LIKE :{key} OR body LIKE :{key})" for key in words) or "1 = 1"
            query = f"SELECT id, title, body, created_at FROM notes WHERE {clause} ORDER BY id DESC LIMIT 20"
            return _rows(self._database.read(lambda db: fetch_all(db, query, words)))
        if name == "list_notes":
            limit = arguments.get("limit")
            count = int(limit) if isinstance(limit, int) else 20
            query = "SELECT id, title, body, created_at FROM notes ORDER BY id DESC LIMIT :count"
            return _rows(self._database.read(lambda db: fetch_all(db, query, {"count": count})))
        return ToolResult(content=[Text(text=f"unknown tool {name!r}")], is_error=True)


def _save(db: Connection, title: str, body: str) -> ToolResult:
    note_id = sql(
        db,
        "INSERT INTO notes (title, body, created_at) VALUES (:title, :body, :at) RETURNING id",
        {"title": title, "body": body, "at": datetime.now(UTC).isoformat(timespec="seconds")},
    ).scalar_one()
    return ToolResult(content=[Text(text=f"Saved note {note_id}: {title}")])


def _rows(rows: list[tuple[Any, ...]]) -> ToolResult:
    if not rows:
        return ToolResult(content=[Text(text="(no notes)")])
    text = "\n\n".join(f"#{row[0]} {row[1]} ({row[3]})\n{row[2]}" for row in rows)
    return ToolResult(content=[Text(text=text)], structured=json.loads(json.dumps([list(row) for row in rows])))
