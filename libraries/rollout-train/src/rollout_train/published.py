"""Published environments: the versions of environments imported from their source (`rollout_train.publishing`),
kept beside the ledger as ordinary state.

A version (`EnvironmentVersion`) is a project's source, fetched at a commit and stored in the blob store as one zip,
whose id is the zip's SHA-256: the same source is the same version, wherever it was fetched from. It says where the
source came from (a git URL, the ref asked for, the commit that ref was, a subdirectory), the entry point
(`module:name`) that makes the environment, the blob, the Ray runtime environment its code runs in (`runtime_env`), what
the environment says of itself (`rollout_train.monitor.environments.described`) and what its check found. A version is
recorded once and never changed: recording one that is there returns the one there.

A published environment is named `NAME@VERSION` (`EnvironmentVersion.reference`) wherever a built-in one is named by
`module:name`: in a launch, a run's start, a suite's entry (`is_published` tells the two apart).

Versions are kept beside the ledger, as presets are: a file per version in a directory beside a ledger of files
(`FileEnvironmentVersions`), or a row per version in the `environment_versions` table of a database ledger's database
(`DatabaseEnvironmentVersions`). `environment_versions_of` finds the store beside a ledger.
"""

import asyncio
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import JsonValue

if TYPE_CHECKING:
    from rollout_train.sql import Connection, Database

__all__ = [
    "DatabaseEnvironmentVersions",
    "EnvironmentVersion",
    "EnvironmentVersions",
    "FileEnvironmentVersions",
    "environment_versions_of",
    "is_published",
    "loaded",
    "offered_json",
    "parsed",
    "provenance",
    "short",
]

VERSION = re.compile(r"[0-9a-f]{64}")
"""A version's id: the SHA-256 of its source's zip, in lowercase hexadecimal."""
SHORT = 12
"""Characters of a version's id shown where a few are enough."""


@dataclass(frozen=True)
class EnvironmentVersion:
    name: str
    """The environment's name: its entry point's name in the project's `rollout.environments` group, else the
    project's name."""
    version: str
    """The SHA-256 of its source's zip."""
    source: str
    """The git URL it was fetched from."""
    ref: str | None
    """The branch, tag or commit asked for; none: the default branch."""
    commit: str
    """The commit the ref was when it was fetched."""
    subdirectory: str
    """The project's directory in the repository (empty: its root)."""
    entry_point: str
    """`module:name`: what makes the environment, imported in its runtime environment."""
    blob: Mapping[str, JsonValue]
    """The zip's blob reference (`rollout.contracts.BlobReference`, as JSON)."""
    runtime_env: Mapping[str, JsonValue]
    """The Ray runtime environment its code runs in: `working_dir` (the zip, where Ray fetches it), `env_vars` (the
    project's `src` on the path, for a project laid out so), and `uv` (the dependencies the platform does not hold)."""
    dependencies: Sequence[str] = ()
    """The project's dependencies, as its `pyproject.toml` declares them."""
    description: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """What the environment says of itself: its version, description, rows, eval data and curriculum."""
    check: Sequence[Mapping[str, JsonValue]] = ()
    """What its check found: each finding's check, whether it passed, and what it said."""
    imported: float = 0.0

    @property
    def reference(self) -> str:
        """`NAME@VERSION`: how launches, runs and suites name it."""
        return f"{self.name}@{self.version}"

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "reference": self.reference}


def is_published(environment: str) -> bool:
    """Whether an environment's name is a published version's (`NAME@VERSION`), not a built-in's (`module:name`)."""
    return parsed(environment) is not None


def parsed(reference: str) -> tuple[str, str] | None:
    """`NAME@VERSION` as its name and version; none for anything else."""
    name, at, version = reference.rpartition("@")
    return (name, version) if at and name and ":" not in name and VERSION.fullmatch(version) else None


def as_version(data: Mapping[str, Any]) -> EnvironmentVersion:
    fields: dict[str, Any] = {
        key: value for key, value in data.items() if key in EnvironmentVersion.__dataclass_fields__
    }
    fields["dependencies"] = tuple(fields.get("dependencies") or ())
    return EnvironmentVersion(**fields)


class EnvironmentVersions(Protocol):
    async def all(self) -> list[EnvironmentVersion]:
        """Every version, the newest imported first."""
        ...

    async def get(self, reference: str) -> EnvironmentVersion | None:
        """A version by its id, or by `NAME@VERSION` (none where the name is not its)."""
        ...

    async def record(self, version: EnvironmentVersion) -> EnvironmentVersion:
        """Record a version, unless one of its id is there: the version as it is kept (the one there, if there was
        one)."""
        ...


def _found(version: EnvironmentVersion | None, reference: str) -> EnvironmentVersion | None:
    said = parsed(reference)
    if version is None or (said is not None and said[0] != version.name):
        return None
    return version


def _id(reference: str) -> str | None:
    said = parsed(reference)
    version = said[1] if said is not None else reference
    return version if VERSION.fullmatch(version) else None


class FileEnvironmentVersions:
    """`EnvironmentVersions` in a directory: `VERSION.json` for each, written beside its place and linked into it only
    if no file of its id is there, so writers need no lock."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    async def all(self) -> list[EnvironmentVersion]:
        def found() -> list[EnvironmentVersion]:
            if not self.directory.is_dir():
                return []
            every = [as_version(json.loads(each.read_text())) for each in self.directory.glob("*.json")]
            return sorted(every, key=lambda each: -each.imported)

        return await asyncio.to_thread(found)

    async def get(self, reference: str) -> EnvironmentVersion | None:
        version = _id(reference)
        if version is None:
            return None
        path = self.directory / f"{version}.json"

        def read() -> EnvironmentVersion | None:
            return as_version(json.loads(path.read_text())) if path.is_file() else None

        return _found(await asyncio.to_thread(read), reference)

    async def record(self, version: EnvironmentVersion) -> EnvironmentVersion:
        def written() -> EnvironmentVersion:
            self.directory.mkdir(parents=True, exist_ok=True)
            place = self.directory / f"{version.version}.json"
            handle, staged = tempfile.mkstemp(dir=self.directory, prefix=".staged-")
            try:
                with os.fdopen(handle, "w") as file:
                    file.write(json.dumps(asdict(version)))
                    file.flush()
                    os.fsync(file.fileno())
                try:
                    os.link(staged, place)
                except FileExistsError:
                    return as_version(json.loads(place.read_text()))
                return version
            finally:
                os.unlink(staged)

        return await asyncio.to_thread(written)


class DatabaseEnvironmentVersions:
    """`EnvironmentVersions` in the `environment_versions` table of a database: a row per version, keyed by its id
    (inserting one that is there fails, and the one there is read)."""

    COLUMNS = (
        "version", "name", "source", "ref", "commit_id", "subdirectory", "entry_point", "blob", "runtime_env",
        "dependencies", "description", "checked", "imported",
    )  # fmt: skip

    def __init__(self, database: "Database") -> None:
        import sqlalchemy as sa

        self.database = database
        self.metadata = sa.MetaData()
        sa.Table(
            "environment_versions",
            self.metadata,
            sa.Column("version", sa.Text, primary_key=True),
            sa.Column("name", sa.Text, nullable=False),
            sa.Column("source", sa.Text, nullable=False),
            sa.Column("ref", sa.Text, nullable=True),
            sa.Column("commit_id", sa.Text, nullable=False),
            sa.Column("subdirectory", sa.Text, nullable=False),
            sa.Column("entry_point", sa.Text, nullable=False),
            sa.Column("blob", sa.Text, nullable=False),
            sa.Column("runtime_env", sa.Text, nullable=False),
            sa.Column("dependencies", sa.Text, nullable=False),
            sa.Column("description", sa.Text, nullable=False),
            sa.Column("checked", sa.Text, nullable=False),
            sa.Column("imported", sa.Float(), nullable=False),
        )
        database.create(self.metadata)

    async def all(self) -> list[EnvironmentVersion]:
        def rows(connection: "Connection") -> list[tuple[Any, ...]]:
            from rollout_train.sql import fetch_all

            listed = ", ".join(self.COLUMNS)
            return fetch_all(connection, f"SELECT {listed} FROM environment_versions ORDER BY imported DESC")

        return [_row(each) for each in await asyncio.to_thread(self.database.read, rows)]

    async def get(self, reference: str) -> EnvironmentVersion | None:
        version = _id(reference)
        if version is None:
            return None
        return _found(await asyncio.to_thread(self.database.read, lambda connection: self._read(connection, version)),
                      reference)  # fmt: skip

    async def record(self, version: EnvironmentVersion) -> EnvironmentVersion:
        from sqlalchemy.exc import IntegrityError

        def inserted(connection: "Connection") -> EnvironmentVersion:
            from rollout_train.sql import sql

            found = self._read(connection, version.version)
            if found is not None:
                return found
            listed = ", ".join(self.COLUMNS)
            named = ", ".join(f":{each}" for each in self.COLUMNS)
            sql(connection, f"INSERT INTO environment_versions ({listed}) VALUES ({named})", _columns(version))
            return version

        try:
            return await asyncio.to_thread(self.database.write, inserted, exclusive=f"environments:{version.version}")
        except IntegrityError:  # (another writer recorded it first: theirs is the one kept)
            found = await self.get(version.version)
            assert found is not None
            return found

    def _read(self, connection: "Connection", version: str) -> EnvironmentVersion | None:
        from rollout_train.sql import fetch_one

        listed = ", ".join(self.COLUMNS)
        row = fetch_one(
            connection, f"SELECT {listed} FROM environment_versions WHERE version = :version", {"version": version}
        )
        return _row(row) if row is not None else None


def _columns(version: EnvironmentVersion) -> dict[str, Any]:
    return {
        "version": version.version, "name": version.name, "source": version.source, "ref": version.ref,
        "commit_id": version.commit, "subdirectory": version.subdirectory, "entry_point": version.entry_point,
        "blob": json.dumps(version.blob), "runtime_env": json.dumps(version.runtime_env),
        "dependencies": json.dumps(list(version.dependencies)), "description": json.dumps(version.description),
        "checked": json.dumps(list(version.check)), "imported": version.imported,
    }  # fmt: skip


def _row(row: tuple[Any, ...]) -> EnvironmentVersion:
    version, name, source, ref, commit, subdirectory, entry, blob, runtime, dependencies, description, check, at = row
    return EnvironmentVersion(
        name=str(name), version=str(version), source=str(source), ref=None if ref is None else str(ref),
        commit=str(commit), subdirectory=str(subdirectory), entry_point=str(entry), blob=json.loads(blob),
        runtime_env=json.loads(runtime), dependencies=tuple(json.loads(dependencies)),
        description=json.loads(description), check=json.loads(check), imported=float(at),
    )  # fmt: skip


def environment_versions_of(ledger: object) -> EnvironmentVersions | None:
    """The published versions beside a ledger: a table in a database ledger's database, a directory beside a ledger
    of files (`environment_versions`, in its directory); none beside any other."""
    database = getattr(ledger, "database", None)
    if database is not None and hasattr(database, "write"):
        return DatabaseEnvironmentVersions(database)
    directory = getattr(ledger, "directory", None)
    if isinstance(directory, Path):
        return FileEnvironmentVersions(directory / "environment_versions")
    return None


def short(version: str) -> str:
    """A version's id, in a few characters."""
    return version[:SHORT]


def offered_json(versions: Sequence[EnvironmentVersion]) -> list[dict[str, JsonValue]]:
    """What a launcher says of the published versions it offers: each one's reference, name, source and commit."""
    return [
        {"environment": each.reference, "name": each.name, "source": each.source, "commit": each.commit}
        for each in versions
    ]


async def loaded(environment: str, ledger: object) -> tuple[Any, EnvironmentVersion | None]:
    """An environment by its name, imported here: a built-in one by `module:name`; a published one (`NAME@VERSION`) by
    its version's entry point, which imports where this process runs in the version's runtime environment (a Ray job
    given its `runtime_env`), with the version. Raises `KeyError` for a published version the ledger does not keep,
    and whatever importing raises."""
    from rollout.names import named

    if not is_published(environment):
        return named(environment), None
    versions = environment_versions_of(ledger)
    version = await versions.get(environment) if versions is not None else None
    if version is None:
        raise KeyError(f"there is no published environment {environment}")
    return named(version.entry_point), version


def provenance(version: EnvironmentVersion) -> dict[str, JsonValue]:
    """What a run's start records of the published version it plays: where its source came from and what it ran."""
    return {
        "name": version.name, "version": version.version, "source": version.source, "ref": version.ref,
        "commit": version.commit, "subdirectory": version.subdirectory, "entry_point": version.entry_point,
    }  # fmt: skip
