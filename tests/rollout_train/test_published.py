"""Published environments' versions, kept beside the ledger (a directory beside a ledger of files, a table in a database
ledger's database): recorded once by their id and never changed, read by id or `NAME@VERSION`, the newest first."""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import replace
from pathlib import Path

import pytest

from rollout_train import FileLedger
from rollout_train.database import DatabaseLedger
from rollout_train.published import (
    DatabaseEnvironmentVersions,
    EnvironmentVersions,
    FileEnvironmentVersions,
    environment_versions_of,
    is_published,
    parsed,
    provenance,
)
from tests.rollout_train.sources import a_version


@pytest.fixture(params=["files", "sqlite", "postgres"])
async def versions(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncGenerator[EnvironmentVersions]:
    if request.param == "files":
        found = environment_versions_of(FileLedger(tmp_path / "ledger"))
        assert isinstance(found, FileEnvironmentVersions)
        yield found
        return
    url = f"sqlite:///{tmp_path / 'ledger.db'}" if request.param == "sqlite" else request.getfixturevalue("postgres")
    ledger = DatabaseLedger(url)
    found = environment_versions_of(ledger)
    assert isinstance(found, DatabaseEnvironmentVersions)
    try:
        yield found
    finally:
        ledger.close()


async def test_a_version_is_read_by_its_id_or_its_name_and_id(versions: EnvironmentVersions) -> None:
    version = a_version()
    assert await versions.record(version) == version
    assert await versions.get(version.version) == version
    assert await versions.get(version.reference) == version
    assert await versions.get(f"other@{version.version}") is None
    assert await versions.get("words") is None and await versions.get(a_version("two").version) is None


async def test_a_version_recorded_again_is_the_one_there(versions: EnvironmentVersions) -> None:
    version = a_version()
    await versions.record(version)
    again = await versions.record(replace(version, source="https://example.com/fork.git", imported=9.0))
    assert again == version and await versions.get(version.version) == version


async def test_every_version_is_listed_the_newest_imported_first(versions: EnvironmentVersions) -> None:
    for source, at in (("one", 1.0), ("two", 3.0), ("three", 2.0)):
        await versions.record(a_version(source, imported=at))
    assert [each.imported for each in await versions.all()] == [3.0, 2.0, 1.0]


async def test_two_records_of_one_version_at_once_keep_one(versions: EnvironmentVersions) -> None:
    made = [replace(a_version(), imported=float(each)) for each in range(5)]
    kept = await asyncio.gather(*(versions.record(each) for each in made))
    assert len({each.imported for each in kept}) == 1 and len(await versions.all()) == 1


def test_a_published_name_is_told_from_a_built_in_one() -> None:
    version = a_version()
    assert is_published(version.reference) and parsed(version.reference) == ("words", version.version)
    assert not is_published("gridworld.environment:environment") and not is_published("words@latest")
    assert not is_published(f"module:name@{version.version}")
    assert provenance(version)["entry_point"] == "words:environment" and "runtime_env" not in provenance(version)
