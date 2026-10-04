"""Rewriting a database ledger's records for blobs and directories that moved: after stores of files were copied into
one store (`rollout_s3.copying`, say), and run directories onto another disk.

`relocated(url, relocation)` rewrites the records of the database ledger at `url`, and the launches and the wanted
settings beside it, as a `Relocation` says:

- a location of one of its stores of files (`{"kind": FILES, "directory": …}`, as a run's start or a dataset records
  it) names the new store instead;
- a blob reference into one of them (`{"uri": "file://…/ab/abcdef…", "sha256": …}`) carries the URI the new store
  gives that blob (blobs are read by their SHA-256: the URI says where the bytes were put);
- a path under one of its directories, or a `file://` URI of one, is under the directory it moved to.

Run it on a copy of the ledger (`sqlite3`'s backup of a live one), then move that copy where it goes with
`rollout ledger copy`. Running it again changes nothing more.

    python -m rollout_train.relocating URL --into LOCATION --uris PREFIX --store DIRECTORY... --path FROM=TO \\
        --home HOME
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlparse

from rollout_train.stores import FILES

__all__ = ["TABLES", "Relocation", "main", "relocated"]

TABLES = {
    "ledger_records": ("name", "key", "record"),
    "launches": ("id", "launch"),
    "run_settings": ("run", "settings"),
}
"""The tables of a database ledger whose JSON is rewritten: the columns that find a row, then the JSON's."""


@dataclass(frozen=True)
class Relocation:
    """What moved: the stores of files in `stores` (as their records name them, `~` standing for `home`) into the store
    `into` names (a location, as a run's start records one), whose blobs' URIs begin with `uris` (`s3://BUCKET/PREFIX/`:
    a blob's is `uris` + `ab/abcdef…`); and the directories in `paths`, each to where it maps."""

    into: Mapping[str, Any]
    uris: str
    stores: frozenset[Path] = frozenset()
    paths: Mapping[str, str] = field(default_factory=dict[str, str])
    home: Path = field(default_factory=Path.home)

    def moved(self, value: Any) -> Any:
        """`value`, rewritten."""
        if isinstance(value, dict):
            table = cast(dict[str, Any], value)
            if (moved := self._location(table)) is not None or (moved := self._reference(table)) is not None:
                return moved
            return {key: self.moved(each) for key, each in table.items()}
        if isinstance(value, list):
            return [self.moved(each) for each in cast(list[Any], value)]
        if isinstance(value, str):
            return self._path(value)
        return value

    def _store(self, said: str) -> bool:
        return Path(str(self.home) + said[1:] if said.startswith("~") else said) in self.stores

    def _location(self, value: dict[str, Any]) -> dict[str, Any] | None:
        if value.get("kind") == FILES and set(value) == {"kind", "directory"} and self._store(str(value["directory"])):
            return dict(self.into)
        return None

    def _reference(self, value: dict[str, Any]) -> dict[str, Any] | None:
        uri, digest = value.get("uri"), value.get("sha256")
        if not (isinstance(uri, str) and uri.startswith("file://") and isinstance(digest, str)):
            return None
        if not self._store(str(Path(unquote(urlparse(uri).path)).parent.parent)):
            return None
        return value | {"uri": f"{self.uris.rstrip('/')}/{digest[:2]}/{digest}"}

    def _path(self, value: str) -> str:
        scheme = "file://" if value.startswith("file://") else ""
        path = value.removeprefix(scheme)
        for before, after in self.paths.items():
            before = before.rstrip("/")
            if path == before or path.startswith(before + "/"):
                return scheme + after.rstrip("/") + path[len(before) :]
        return value


def relocated(url: str, relocation: Relocation) -> dict[str, int]:
    """Rewrite the database ledger at `url` as `relocation` says; returns how many rows of each table changed."""
    import sqlalchemy as sa

    from rollout_train.sql import Connection, Database, fetch_all, sql

    database = Database(url)
    try:
        there = set(sa.inspect(database.engine).get_table_names())

        def rewritten(connection: Connection) -> dict[str, int]:
            changed: dict[str, int] = {}
            for name, columns in TABLES.items():
                if name not in there:
                    continue
                *keys, column = columns
                changed[name] = 0
                for row in fetch_all(connection, f"SELECT {', '.join(columns)} FROM {name}"):
                    before = json.loads(row[-1])
                    if (after := relocation.moved(before)) != before:
                        where = " AND ".join(f"{key} = :{key}" for key in keys)
                        values = dict(zip(keys, row[:-1], strict=True)) | {"value": json.dumps(after)}
                        sql(connection, f"UPDATE {name} SET {column} = :value WHERE {where}", values)
                        changed[name] += 1
            return changed

        return database.write(rewritten)
    finally:
        database.engine.dispose()


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m rollout_train.relocating", description="rewrite a database ledger's records for what moved"
    )
    parser.add_argument("url", help="the database: sqlite:///… or postgresql://…")
    parser.add_argument("--into", required=True, help="the store the blobs moved into, as a location (JSON)")
    parser.add_argument("--uris", required=True, help="how its blobs' URIs begin: s3://BUCKET/PREFIX/")
    parser.add_argument("--store", action="append", default=[], type=Path, help="a store of files moved (repeatable)")
    parser.add_argument("--path", action="append", default=[], help="FROM=TO: what was under FROM is under TO")
    parser.add_argument("--home", type=Path, default=Path.home(), help="what `~` stands for in the records")
    given = parser.parse_args(arguments)
    relocation = Relocation(
        json.loads(given.into), given.uris, frozenset(given.store),
        dict(each.split("=", 1) for each in given.path), given.home,
    )  # fmt: skip
    changed = relocated(given.url, relocation)
    print(", ".join(f"{count} rows of {name} rewritten" for name, count in changed.items()))


if __name__ == "__main__":
    main()
