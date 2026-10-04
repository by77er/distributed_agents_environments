"""Presets: named run settings in versions, kept beside the ledger (a directory beside a ledger of files, a table in a
database ledger's database). Each save is the next version; two saves at once are two versions; a deleted preset's
name points to nothing while its versions stay readable."""

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from rollout_train import FileLedger
from rollout_train.database import DatabaseLedger
from rollout_train.presets import DatabasePresets, FilePresets, Presets, parsed, presets_of

ONE_GPU = {
    "environment": "minecraft_team.environment:environment",
    "trainer.provider": "local-lora",
    "channels.policy.provider": "local-vllm",
    "channels.policy.model": "cyankiwi/Qwen3.5-9B-AWQ-4bit",
    "trainer.rank": 32,
    "trainer.learning_rate": 5e-5,
}


@pytest.fixture(params=["files", "sqlite", "postgres"])
async def presets(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncGenerator[Presets]:
    if request.param == "files":
        found = presets_of(FileLedger(tmp_path / "ledger"))
        assert isinstance(found, FilePresets)
        yield found
        return
    url = f"sqlite:///{tmp_path / 'ledger.db'}" if request.param == "sqlite" else request.getfixturevalue("postgres")
    ledger = DatabaseLedger(url)
    found = presets_of(ledger)
    assert isinstance(found, DatabasePresets)
    try:
        yield found
    finally:
        ledger.close()


async def test_each_save_is_the_next_version(presets: Presets) -> None:
    first = await presets.save("minecraft-one-gpu", ONE_GPU, note="from the profile")
    second = await presets.save("minecraft-one-gpu", {**ONE_GPU, "trainer.learning_rate": 3e-5})
    assert (first.version, second.version) == (1, 2) and second.id == "minecraft-one-gpu@2"
    assert first.note == "from the profile" and first.saved > 0
    newest = await presets.get("minecraft-one-gpu")
    assert newest is not None and newest.version == 2 and newest.settings["trainer.learning_rate"] == 3e-5
    one = await presets.get("minecraft-one-gpu@1")
    assert one is not None and one.settings == ONE_GPU
    assert await presets.get("minecraft-one-gpu@3") is None and await presets.get("nothing") is None
    assert [each.version for each in await presets.versions("minecraft-one-gpu")] == [1, 2]


async def test_two_saves_at_once_are_two_versions(presets: Presets) -> None:
    saved = await asyncio.gather(*(presets.save("busy", {"groups": each}) for each in range(6)))
    assert sorted(each.version for each in saved) == [1, 2, 3, 4, 5, 6]
    kept = sorted(int(str(each.settings["groups"])) for each in await presets.versions("busy"))
    assert kept == list(range(6))  # (none lost)


async def test_a_deleted_preset_keeps_its_versions_for_the_runs_that_name_them(presets: Presets) -> None:
    await presets.save("old", ONE_GPU)
    await presets.save("kept", {"groups": 3})
    deleted = await presets.delete("old")
    assert deleted.deleted and deleted.version == 2
    assert await presets.get("old") is None
    assert [each.name for each in await presets.all()] == ["kept"]
    one = await presets.get("old@1")
    assert one is not None and one.settings == ONE_GPU
    assert await presets.get("old@2") is None
    again = await presets.save("old", {"groups": 1})
    assert again.version == 3 and [each.name for each in await presets.all()] == ["kept", "old"]
    with pytest.raises(KeyError):
        await presets.delete("never")


async def test_a_preset_holds_run_settings_and_never_a_name(presets: Presets) -> None:
    with pytest.raises(ValueError, match="no run's name"):
        await presets.save("named", {"name": "team-8"})
    with pytest.raises(ValueError, match="'colour' is not one"):
        await presets.save("odd", {"colour": "red"})
    with pytest.raises(ValueError, match="cannot be a name"):
        await presets.save("a/b", {"groups": 1})
    saved = await presets.save("with-trainer", {"trainer.segments_per_step": 384, "channels.judge.model": "gpt-5"})
    assert saved.version == 1


def test_a_version_is_named_name_at_number() -> None:
    assert parsed("minecraft-one-gpu") == ("minecraft-one-gpu", None)
    assert parsed("minecraft-one-gpu@3") == ("minecraft-one-gpu", 3)
    with pytest.raises(ValueError, match="NAME@N"):
        parsed("minecraft-one-gpu@latest")


def test_no_presets_beside_a_ledger_that_keeps_nothing_beside_it() -> None:
    assert presets_of(object()) is None
