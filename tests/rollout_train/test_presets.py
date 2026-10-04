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
    asyncio.run(registry.create("profile-era"))
    status, said = run(monkeypatch, capsys, "preset", "save", "x", "--from-run", "profile-era", *where)
    assert status == 1 and "records no run settings" in said


PROFILE = """
directory = "{directory}"

[channels.policy]
model = "a-checkpoint"
renderer = "rollout_train.testing:plain_renderer"
engine = "rollout_train.testing:scripted_engine"
thinking_tokens = 64
max_lag = 2

[evals]
suite = "words-v1"
every = 2

[trainer]
kind = "rollout_lora:LoraTrainer"
channel = "policy"
start = "first"
rank = 8
learning_rate = 5e-5
"""


async def test_a_runs_settings_layer_over_its_profile(tmp_path: Path) -> None:
    from rollout_train.cli import _layered, _recorded  # pyright: ignore[reportPrivateUsage]
    from rollout_train.profile import Profile

    path = tmp_path / "profile.toml"
    path.write_text(PROFILE.format(directory=tmp_path / "run"))
    kept = presets_of(FileLedger(tmp_path / "run" / "ledger"))
    assert kept is not None
    await kept.save("faster", {"trainer.learning_rate": 1e-4, "groups_per_step": 8, "trainer.rank": 16})
    (tmp_path / "run.toml").write_text("[trainer]\nrank = 32\n[channels.policy]\nanswer_tokens = 128\n")
    sets = ["trainer.start=diamonds", "max_lag=3", "evals.suite=", "channels.policy.thinking_tokens=none",
            "memory.runs_gib=2"]  # fmt: skip
    layers = await _layered(path, None, "train", "faster", tmp_path / "run.toml", sets, {"groups": 12, "seed": None})
    assert layers.preset == "faster@1"
    # What the profile is loaded with: the run settings it keeps, in its own keys, and its own keys as given.
    assert layers.profile == {
        "trainer.learning_rate": 1e-4, "trainer.rank": 32, "channels.policy.answer_tokens": 128,
        "trainer.start": "diamonds", "channels.policy.max_lag": 3, "evals.suite": "",
        "channels.policy.thinking_tokens": "none", "memory.runs_gib": 2,
    }  # fmt: skip
    described = Profile.load(path, settings=layers.profile)
    assert described.evals is None and described.channels["policy"].thinking_tokens is None
    assert described.trainer is not None and described.trainer.settings["rank"] == 32
    # The run's settings: the profile's, then the preset's, the file's, the flags'.
    settings = layers.settings
    assert settings["start"] == "diamonds" and settings["max_lag"] == 3 and settings["groups"] == 12
    assert settings["groups_per_step"] == 8 and settings["seed"] == 0 and settings["evals.suite"] is None
    assert settings["channels.policy.model"] == "a-checkpoint" and settings["channels.policy.answer_tokens"] == 128
    assert settings["channels.policy.thinking_tokens"] is None and "memory.runs_gib" not in settings.values
    started = _recorded(layers, described)
    assert started["preset"] == "faster@1" and started["fixed"]["trainer.rank"] == 32
    assert started["changeable"]["trainer.learning_rate"] == 1e-4 and started["changeable"]["max_lag"] == 3
    with pytest.raises(SystemExit, match=r"cannot take trainer.provider, limits.spend: those need the cluster config"):
        await _layered(path, None, "train", None, None, ["trainer.provider=tinker-lora", "limits.spend=2"], {})
    with pytest.raises(SystemExit, match="no preset 'slower'"):
        await _layered(path, None, "train", "slower", None, [], {})
    evaluated = await _layered(path, None, "eval", None, None, [], {"start": None, "eval.suite": "words-v1"})
    assert "start" not in evaluated.given and evaluated.settings["eval.suite"] == "words-v1"
    shortened = await _layered(path, None, "train", None, None, ["channels.policy.model=x"], {}, model="small")
    assert shortened.profile == {"channels.policy.model": "small"}  # (--model of the trained channel, over --set)
    checked = await _layered(path, None, "check", None, None, ["groups=2"], {"environment": "e:e"})
    assert checked.given["groups"] == 2 and "trainer.rank" not in checked.settings.values  # (a check trains nothing)
    assert checked.settings["channels.policy.thinking_tokens"] == 64
