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


def run(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *arguments: str) -> tuple[int, str]:
    """A command's exit status and what it printed (and exited saying)."""
    from rollout_train.cli import main

    monkeypatch.setattr("sys.argv", ["rollout", *arguments])
    status, message = 0, ""
    try:
        main()
    except SystemExit as exited:
        status, message = (exited.code, "") if isinstance(exited.code, int) else (1, str(exited.code or ""))
    return status, capsys.readouterr().out + message


def test_the_preset_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    where = ["--ledger", str(tmp_path / "ledger")]
    said = run(monkeypatch, capsys, "preset", "save", "one-gpu", "--set", "trainer.rank=32", "--set",
               "environment=minecraft_team.environment:environment", "--note", "first", *where)  # fmt: skip
    assert said == (0, "saved one-gpu@1: 2 settings\n")
    (tmp_path / "run.toml").write_text("groups = 12\n[trainer]\nrank = 16\nlearning_rate = 1e-4\n")
    said = run(monkeypatch, capsys, "preset", "save", "one-gpu", "--settings", str(tmp_path / "run.toml"),
               "--set", "trainer.rank=64", *where)  # fmt: skip
    assert said == (0, "saved one-gpu@2: 3 settings\n")  # (the flags over the file)
    status, listed = run(monkeypatch, capsys, "preset", "list", *where)
    assert status == 0 and listed.startswith("one-gpu@2") and "3 settings" in listed
    status, shown = run(monkeypatch, capsys, "preset", "show", "one-gpu", *where)
    assert status == 0 and shown.splitlines() == [
        "one-gpu@2 (versions 1, 2)", "groups = 12", "trainer.learning_rate = 0.0001", "trainer.rank = 64",
    ]  # fmt: skip
    first = run(monkeypatch, capsys, "preset", "show", "one-gpu@1", *where)[1]
    assert first.startswith("one-gpu@1 (versions 1, 2): first") and "trainer.rank = 32" in first
    assert run(monkeypatch, capsys, "preset", "delete", "one-gpu", *where)[0] == 0
    assert run(monkeypatch, capsys, "preset", "show", "one-gpu", *where) == (1, "there is no preset 'one-gpu'")
    assert "trainer.rank = 32" in run(monkeypatch, capsys, "preset", "show", "one-gpu@1", *where)[1]
    status, refused = run(monkeypatch, capsys, "preset", "save", "named", "--set", "name=team-8", *where)
    assert status == 1 and "no run's name" in refused


def test_a_preset_is_saved_from_what_a_runs_start_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from pydantic import JsonValue

    from rollout_train.record import STARTS, scope, table
    from rollout_train.registry import registry_of
    from rollout_train.run_settings import RunSettings, recorded

    ledger = FileLedger(tmp_path / "ledger")
    registry = registry_of(ledger)
    assert registry is not None
    entry = asyncio.run(registry.create("team-8"))
    started: JsonValue = {"run_settings": recorded(RunSettings({"name": "team-8", "groups": 12, "trainer.rank": 16}))}

    async def start() -> None:
        await ledger.append(table(entry.id, STARTS), "1", started, await ledger.take(scope(entry.id)))

    asyncio.run(start())
    where = ["--ledger", str(tmp_path / "ledger")]
    status, _ = run(
        monkeypatch, capsys, "preset", "save", "like-8", "--from-run", "team-8", "--set", "groups=24", *where
    )
    assert status == 0
    saved = asyncio.run(FilePresets(tmp_path / "ledger" / "presets").get("like-8"))
    assert saved is not None and saved.settings["groups"] == 24 and saved.settings["trainer.rank"] == 16
    assert "name" not in saved.settings and saved.settings["kind"] == "train"  # (the full copy, less its name)
    asyncio.run(registry.create("unrecorded"))
    status, said = run(monkeypatch, capsys, "preset", "save", "x", "--from-run", "unrecorded", *where)
    assert status == 1 and "records no run settings" in said


async def test_a_runs_settings_layer_over_a_preset_which_gives_only_what_its_kind_takes(tmp_path: Path) -> None:
    from rollout_train.launching import settled
    from rollout_train.run_settings import from_file, from_flags, layered

    kept = presets_of(FileLedger(tmp_path / "ledger"))
    assert kept is not None
    await kept.save("faster", {"trainer.learning_rate": 1e-4, "groups_per_step": 8, "trainer.rank": 16,
                               "channels.policy.model": "m", "channels.policy.thinking_tokens": 64})  # fmt: skip
    (tmp_path / "run.toml").write_text("[trainer]\nrank = 32\n[channels.policy]\nanswer_tokens = 128\n")
    given = layered(from_file(tmp_path / "run.toml"), from_flags(["start=diamonds", "max_lag=3"]), {"groups": 12})
    settings, preset = await settled("train", "fast", given.values, preset="faster", presets=kept)
    assert preset == "faster@1" and (settings.kind, settings["name"]) == ("train", "fast")
    assert (
        settings["trainer.rank"] == 32 and settings["trainer.learning_rate"] == 1e-4
    )  # (the file's over the preset's)
    assert settings["groups_per_step"] == 8 and settings["groups"] == 12 and settings["max_lag"] == 3
    assert settings["start"] == "diamonds" and settings["channels.policy.answer_tokens"] == 128
    evaluated, _ = await settled("eval", "on words", {"eval.suite": "words-v1"}, preset="faster@1", presets=kept)
    assert evaluated["channels.policy.model"] == "m" and evaluated["channels.policy.thinking_tokens"] == 64
    assert not any(key.startswith("trainer.") or key == "groups_per_step" for key in evaluated.values)
    with pytest.raises(KeyError, match="no preset 'slower'"):
        await settled("train", "slow", {}, preset="slower", presets=kept)
