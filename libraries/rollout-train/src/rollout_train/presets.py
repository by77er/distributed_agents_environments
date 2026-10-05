"""Presets: named, versioned run settings, kept beside the ledger as ordinary state.

A preset (`Preset`) is a set of run settings under a name (`rollout_train.run_settings`): any key but `name`, the
environment and the start among them. Each save is a new version (1, 2, …), and nothing is changed in place: a
preset's name points to its newest version, and `NAME@N` names one version. Saving appends the next version by
compare-and-set on the newest number, so two editors saving at once make two versions, never one lost. Deleting a
preset appends a version that says so: its name then points to nothing, and its earlier versions stay readable for
the runs that name them. Saving it again makes the next version.

A run records a full copy of its settings and, as provenance only, the preset version they came from
(`rollout_train.run_settings.recorded`): editing the preset later changes no run.

Presets are kept beside the ledger, as launches and desired settings are: in a directory beside a ledger of files
(`FilePresets`: a file per version, made only if it is not there), or in a table of a database ledger's database
(`DatabasePresets`: `presets`, a row per version). `presets_of` finds the store beside a ledger.
"""

import asyncio
import json
import os
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import JsonValue

from rollout_train.registry import valid
from rollout_train.run_settings import is_trainers, key_of

if TYPE_CHECKING:
    from rollout_train.sql import Connection, Database

__all__ = ["DatabasePresets", "FilePresets", "Preset", "Presets", "parsed", "presets_of"]


@dataclass(frozen=True)
class Preset:
    name: str
    version: int
    """1, 2, …: each save is a new version."""
    settings: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    saved: float = 0.0
    note: str = ""
    deleted: bool = False
    """A version that says the preset was deleted: its name points to nothing from here on."""

    @property
    def id(self) -> str:
        """`NAME@N`: what a run records as where its settings came from."""
        return f"{self.name}@{self.version}"


class Presets(Protocol):
    async def all(self) -> list[Preset]:
        """Every preset's newest version, by name, leaving out the deleted ones."""
        ...

    async def versions(self, name: str) -> list[Preset]:
        """Every version of a preset saved, oldest first (deletions left out)."""
        ...

    async def get(self, reference: str) -> Preset | None:
        """A preset by name (its newest version, none if it was deleted) or one version (`NAME@N`)."""
        ...

    async def save(self, name: str, settings: Mapping[str, JsonValue], note: str = "") -> Preset:
        """Save settings as the preset's next version. Raises `ValueError` for a name that cannot be one, or a key a
        preset cannot hold."""
        ...

    async def delete(self, name: str) -> Preset:
        """Mark a preset deleted (its versions stay readable). Raises `KeyError` for a preset there is none of."""
        ...


def parsed(reference: str) -> tuple[str, int | None]:
    """`NAME` or `NAME@N`, as the name and the version (none: the newest)."""
    name, at, number = reference.partition("@")
    if not at:
        return name, None
    if not number.isdigit() or int(number) < 1:
        raise ValueError(f"{reference!r}: a preset's version is NAME@N, N a whole number from 1")
    return name, int(number)


def checked(name: str, settings: Mapping[str, JsonValue]) -> tuple[str, dict[str, JsonValue]]:
    """The name and settings of a preset about to be saved: a valid name, and only keys a run takes (a trainer's own
    settings among them), never `name`."""
    name = valid(name)
    if "name" in settings:
        raise ValueError("a preset holds no run's name: it is given at each launch")
    for key in settings:
        if key_of(key) is None and not is_trainers(key):
            raise ValueError(f"a preset holds run settings, and {key!r} is not one")
    return name, json.loads(json.dumps(dict(settings)))


def _newest(versions: list[Preset]) -> Preset | None:
    return max(versions, key=lambda each: each.version) if versions else None


def _get(versions: list[Preset], reference: str) -> Preset | None:
    _, number = parsed(reference)
    if number is None:
        newest = _newest(versions)
        return newest if newest is not None and not newest.deleted else None
    found = next((each for each in versions if each.version == number), None)
    return found if found is not None and not found.deleted else None


class FilePresets:
    """`Presets` in a directory: `NAME/N.json` for each version, each written beside its place and linked into it
    only if no version N is there (the compare-and-set), so writers need no lock."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    async def all(self) -> list[Preset]:
        def found() -> list[Preset]:
            if not self.directory.exists():
                return []
            newest = [_newest(self._versions(each.name)) for each in sorted(self.directory.iterdir()) if each.is_dir()]
            return [each for each in newest if each is not None and not each.deleted]

        return await asyncio.to_thread(found)

    async def versions(self, name: str) -> list[Preset]:
        return [each for each in await asyncio.to_thread(self._versions, name) if not each.deleted]

    async def get(self, reference: str) -> Preset | None:
        name, _ = parsed(reference)
        return _get(await asyncio.to_thread(self._versions, name), reference)

    async def save(self, name: str, settings: Mapping[str, JsonValue], note: str = "") -> Preset:
        name, settings = checked(name, settings)
        return await asyncio.to_thread(self._append, name, settings, note, False)

    async def delete(self, name: str) -> Preset:
        if await self.get(name) is None:
            raise KeyError(f"there is no preset {name!r}")
        return await asyncio.to_thread(self._append, name, {}, "", True)

    def _versions(self, name: str) -> list[Preset]:
        place = self.directory / name
        if not place.is_dir():
            return []
        return sorted(
            (Preset(**json.loads(each.read_text())) for each in place.glob("*.json")), key=lambda each: each.version
        )

    def _append(self, name: str, settings: dict[str, JsonValue], note: str, deleted: bool) -> Preset:
        place = self.directory / name
        place.mkdir(parents=True, exist_ok=True)
        while True:
            newest = _newest(self._versions(name))
            preset = Preset(name, (newest.version if newest else 0) + 1, settings, round(time.time(), 1), note, deleted)
            handle, staged = tempfile.mkstemp(dir=place, prefix=".staged-")
            try:
                with os.fdopen(handle, "w") as file:
                    file.write(json.dumps(asdict(preset)))
                    file.flush()
                    os.fsync(file.fileno())
                try:
                    os.link(staged, place / f"{preset.version}.json")
                except FileExistsError:
                    continue  # (another writer took that number: the next one)
                return preset
            finally:
                os.unlink(staged)


class DatabasePresets:
    """`Presets` in the `presets` table of a database (a row per version, keyed by name and version: inserting a
    version that is there fails, and the next number is tried)."""

    def __init__(self, database: "Database") -> None:
        import sqlalchemy as sa

        self.database = database
        self.metadata = sa.MetaData()
        sa.Table(
            "presets",
            self.metadata,
            sa.Column("name", sa.Text, primary_key=True),
            sa.Column("version", sa.Integer, primary_key=True),
            sa.Column("settings", sa.Text, nullable=False),
            sa.Column("saved", sa.Float(), nullable=False),
            sa.Column("note", sa.Text, nullable=False),
            sa.Column("deleted", sa.Boolean, nullable=False),
        )
        database.create(self.metadata)

    async def all(self) -> list[Preset]:
        def rows(connection: "Connection") -> list[tuple[Any, ...]]:
            from rollout_train.sql import fetch_all

            return fetch_all(
                connection,
                "SELECT p.name, p.version, p.settings, p.saved, p.note, p.deleted FROM presets p "
                "JOIN (SELECT name, MAX(version) AS version FROM presets GROUP BY name) n "
                "ON p.name = n.name AND p.version = n.version ORDER BY p.name",
            )

        newest = [_row(each) for each in await asyncio.to_thread(self.database.read, rows)]
        return [each for each in newest if not each.deleted]

    async def versions(self, name: str) -> list[Preset]:
        return [each for each in await self._versions(name) if not each.deleted]

    async def get(self, reference: str) -> Preset | None:
        name, _ = parsed(reference)
        return _get(await self._versions(name), reference)

    async def save(self, name: str, settings: Mapping[str, JsonValue], note: str = "") -> Preset:
        name, settings = checked(name, settings)
        return await self._append(name, settings, note, False)

    async def delete(self, name: str) -> Preset:
        if await self.get(name) is None:
            raise KeyError(f"there is no preset {name!r}")
        return await self._append(name, {}, "", True)

    async def _versions(self, name: str) -> list[Preset]:
        def rows(connection: "Connection") -> list[tuple[Any, ...]]:
            from rollout_train.sql import fetch_all

            return fetch_all(
                connection,
                "SELECT name, version, settings, saved, note, deleted FROM presets WHERE name = :name ORDER BY version",
                {"name": name},
            )

        return [_row(each) for each in await asyncio.to_thread(self.database.read, rows)]

    async def _append(self, name: str, settings: dict[str, JsonValue], note: str, deleted: bool) -> Preset:
        from sqlalchemy.exc import IntegrityError

        def appended(connection: "Connection") -> Preset:
            from rollout_train.sql import fetch_one, sql

            row = fetch_one(connection, "SELECT MAX(version) FROM presets WHERE name = :name", {"name": name})
            number = (int(row[0]) if row and row[0] is not None else 0) + 1
            preset = Preset(name, number, settings, round(time.time(), 1), note, deleted)
            sql(
                connection,
                "INSERT INTO presets (name, version, settings, saved, note, deleted) "
                "VALUES (:name, :version, :settings, :saved, :note, :deleted)",
                {
                    "name": name, "version": number, "settings": json.dumps(settings), "saved": preset.saved,
                    "note": note, "deleted": deleted,
                },
            )  # fmt: skip
            return preset

        while True:
            try:
                return await asyncio.to_thread(self.database.write, appended, exclusive=f"presets:{name}")
            except IntegrityError:
                continue  # (another writer took that number: the next one)


def _row(row: tuple[Any, ...]) -> Preset:
    name, version, settings, saved, note, deleted = row
    return Preset(str(name), int(version), json.loads(settings), float(saved), str(note), bool(deleted))


def presets_of(ledger: object) -> Presets | None:
    """The presets beside a ledger: a table in a database ledger's database, a directory beside a ledger of files
    (`presets`, in its directory), the service's for a ledger reached through it (`HttpLedger.presets`); none beside
    any other."""
    if (found := getattr(ledger, "presets", None)) is not None:
        return found
    database = getattr(ledger, "database", None)
    if database is not None and hasattr(database, "write"):
        return DatabasePresets(database)
    directory = getattr(ledger, "directory", None)
    if isinstance(directory, Path):
        return FilePresets(directory / "presets")
    return None
