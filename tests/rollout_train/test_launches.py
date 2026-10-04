"""Runs asked for from anywhere, a launcher that starts them, and the settings a run changes of its profile."""

import asyncio
import signal
from pathlib import Path
from typing import Any

import pytest

from rollout_train import launcher as launching
from rollout_train.cli import _setting  # pyright: ignore[reportPrivateUsage]
from rollout_train.database import DatabaseLedger
from rollout_train.launcher import LAUNCHER, OUTPUT, Launcher, offered, slug
from rollout_train.launches import (
    ASKED,
    CLAIMED,
    ENDED,
    FAILED,
    RUNNING,
    STOPPED,
    STOPPING,
    Asked,
    as_launch,
    launches_of,
)
from rollout_train.ledger import FileLedger, Ledger
from rollout_train.presence import presence_of
from rollout_train.profile import Profile
from tests.rollout_train.test_profile import PROFILE


def ledgers(tmp_path: Path) -> list[Ledger]:
    return [FileLedger(tmp_path / "files"), DatabaseLedger(f"sqlite:///{tmp_path / 'ledger.db'}")]


@pytest.mark.parametrize("kind", [0, 1], ids=["files", "database"])
async def test_a_launch_is_asked_for_claimed_once_and_noted_as_it_goes(tmp_path: Path, kind: int) -> None:
    launches = launches_of(ledgers(tmp_path)[kind])
    assert launches is not None and await launches.all() == []
    asked = Asked(
        "one-gpu", "minecraft_team.environment:environment", "diamonds", settings={"trainer.learning_rate": 3e-5}
    )
    first = await launches.ask(asked)
    second = await launches.ask(Asked("one-gpu", "c:c", "later"))
    assert first.state == ASKED and first.id.startswith("launch_") and first.asked == asked
    assert [each.id for each in await launches.all()] == [second.id, first.id]  # (newest first)
    claimed = await launches.claim(first.id, "launcher/a")
    assert claimed is not None and claimed.state == CLAIMED and claimed.launcher == "launcher/a"
    assert await launches.claim(first.id, "launcher/b") is None  # (claimed once)
    running = await launches.note(first.id, state=RUNNING, directory="/runs/d", pid=12)
    assert (running.state, running.directory, running.pid, running.launcher) == (RUNNING, "/runs/d", 12, "launcher/a")
    (again,) = [each for each in await launches.all() if each.id == first.id]
    assert again == running and again.asked.settings == {"trainer.learning_rate": 3e-5}


def test_a_launch_asked_for_as_a_catalog_reads_as_its_environment() -> None:
    written = {"id": "launch_1", "at": 1.0, "asked": {"profile": "one-gpu", "catalog": "c:c", "name": "older"}}
    assert as_launch(written).asked == Asked("one-gpu", "c:c", "older")


def profiles(tmp_path: Path) -> Path:
    directory = tmp_path / "profiles"
    directory.mkdir()
    (directory / "small.toml").write_text(PROFILE.format(directory=tmp_path / "run"))
    (directory / "not-a-profile.toml").write_text("nonsense = [")
    without = PROFILE.format(directory=tmp_path / "run").split("[trainer]")[0]
    (directory / "serving-only.toml").write_text(without)
    return directory


def test_a_launcher_offers_the_profiles_that_train_with_the_settings_a_launch_may_change(tmp_path: Path) -> None:
    (small,) = offered(profiles(tmp_path))  # (not one that is no profile, nor one that trains nothing)
    assert small["profile"] == "small" and small["model"] == "a-checkpoint"
    settings = small["settings"]
    assert settings["trainer.segment_tokens"] == 900 and settings["trainer.bookmark"] == "best"
    assert settings["episodes_at_once"] == 6 and settings["channels.policy.thinking_tokens"] == 64
    assert small["weights"] is None  # (its trainer says what it makes only once made)
    assert slug("Diamonds, unguided!") == "diamonds-unguided"


class EveryWeight:
    """A trainer that says what it makes before it is made."""

    weights = "full"


def test_a_launcher_says_what_a_profiles_trainer_makes_where_the_trainer_says() -> None:
    assert launching._weights(f"{__name__}:EveryWeight") == "full"  # pyright: ignore[reportPrivateUsage]
    assert launching._weights("no_such_module:Trainer") is None  # pyright: ignore[reportPrivateUsage]


def test_a_run_changes_settings_of_its_profile_by_dotted_key(tmp_path: Path) -> None:
    path = profiles(tmp_path) / "small.toml"
    changed = Profile.load(path, settings={"trainer.segment_tokens": 500, "episodes_at_once": 2, "trainer.start": "x"})
    assert changed.trainer is not None and changed.trainer.settings["segment_tokens"] == 500
    assert changed.episodes_at_once == 2 and changed.trainer.start == "x"
    with pytest.raises(Exception, match="has no episodes_at_onc"):
        Profile.load(path, settings={"episodes_at_onc": 2})  # (a key a profile does not have is an error)
    assert _setting("trainer.learning_rate=3e-5") == ("trainer.learning_rate", 3e-5)
    assert _setting("trainer.start=curriculum-9:20") == ("trainer.start", "curriculum-9:20")  # (not TOML: text)
    assert _setting('trainer.bookmark="best"') == ("trainer.bookmark", "best")
    assert _setting("x=[1, 2]") == ("x", [1, 2]) and _setting("flag=true") == ("flag", True)
    with pytest.raises(SystemExit):
        _setting("no-value")


class Process:
    """A started `rollout train`, as the launcher sees it: it ends with `code` once told to, or on an interrupt."""

    def __init__(self, code: int) -> None:
        self.pid, self.code, self.signals = 4242, code, list[int]()
        self.done = asyncio.Event()

    def send_signal(self, number: int) -> None:
        self.signals.append(number)
        self.done.set()

    async def wait(self) -> int:
        await self.done.wait()
        return self.code


async def test_a_launcher_starts_what_it_is_asked_for_and_notes_how_it_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    launches, heartbeats = launches_of(ledger), presence_of(ledger)
    assert launches is not None and heartbeats is not None
    started: list[tuple[list[str], Process]] = []
    codes = iter([0, 3, 0])

    async def spawn(*command: str, stdout: Any, **_: Any) -> Process:
        stdout.write(b"loading...\nValueError: the environment has no rows\n")
        stdout.flush()
        started.append((list(command), Process(next(codes))))
        return started[-1][1]

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    found = Launcher("launcher/here", launches, heartbeats, profiles(tmp_path), ["c:c"], tmp_path / "runs", every=0.01)
    elsewhere = await launches.ask(Asked("another-profile", "c:c", "not mine"))
    ended = await launches.ask(
        Asked("small", "c:c", "Ends well", start="kpqx", settings={"trainer.segment_tokens": 500})
    )
    serving = asyncio.create_task(found.serve())
    try:
        async with asyncio.timeout(5):
            while not started:  # noqa: ASYNC110 (the launcher starts it)
                await asyncio.sleep(0.01)
            command, process = started[0]
            assert command[1:5] == ["-m", "rollout_train.cli", "train", str(profiles_dir(tmp_path) / "small.toml")]
            assert "--name" in command and command[command.index("--name") + 1] == "Ends well"
            assert command[command.index("--set") : command.index("--set") + 2] == [
                "--set",
                "trainer.segment_tokens=500",
            ]
            assert 'trainer.start="kpqx"' in command and found.at_once == 1
            failing = await launches.ask(Asked("small", "c:c", "Fails"))
            await asyncio.sleep(0.05)
            assert len(started) == 1  # (one at a time: it waits for room)
            process.done.set()
            while len(started) < 2:  # noqa: ASYNC110
                await asyncio.sleep(0.01)
            started[1][1].done.set()
            stopped = await launches.ask(Asked("small", "c:c", "Stopped"))
            while len(started) < 3:  # noqa: ASYNC110
                await asyncio.sleep(0.01)
            await launches.note(stopped.id, state=STOPPING)
            while (await state(launches, stopped.id)) != STOPPED:  # noqa: ASYNC110
                await asyncio.sleep(0.01)
            while (await state(launches, failing.id)) != FAILED:  # noqa: ASYNC110
                await asyncio.sleep(0.01)
    finally:
        serving.cancel()
    assert started[2][1].signals == [signal.SIGINT]
    by = {each.id: each for each in await launches.all()}
    assert by[ended.id].state == ENDED and by[ended.id].pid == 4242 and by[ended.id].launcher == "launcher/here"
    assert "the environment has no rows" in str(by[failing.id].detail)
    directory = Path(str(by[ended.id].directory))
    assert directory.parent == tmp_path / "runs" and directory.name.startswith("ends-well-")
    assert (directory / OUTPUT).exists() and by[elsewhere.id].state == ASKED  # (a profile it does not offer)
    (beat,) = await heartbeats.beats()
    assert beat.runner == "launcher/here" and beat.about["kind"] == LAUNCHER and beat.about["environments"] == ["c:c"]
    listed: Any = beat.about["profiles"]
    assert [each["profile"] for each in listed] == ["small"]


def profiles_dir(tmp_path: Path) -> Path:
    return tmp_path / "profiles"


async def state(launches: Any, id: str) -> str:
    return next(each.state for each in await launches.all() if each.id == id)


async def test_a_run_that_cannot_start_is_a_failed_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    launches, heartbeats = launches_of(ledger), presence_of(ledger)
    assert launches is not None and heartbeats is not None

    async def refused(*_: Any, **__: Any) -> Any:
        raise PermissionError("no such interpreter")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", refused)
    asked = await launches.ask(Asked("small", "c:c", "Never"))
    found = Launcher("launcher/here", launches, heartbeats, profiles(tmp_path), [], tmp_path / "runs")
    found._offered = launching.offered(found.profiles)  # pyright: ignore[reportPrivateUsage]
    await found._step()  # pyright: ignore[reportPrivateUsage]
    (failed,) = await launches.all()
    assert failed.id == asked.id and failed.state == FAILED and failed.detail == "PermissionError: no such interpreter"
