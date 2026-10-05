"""Pausing a run and resuming it: a paused loop decides nothing and runners claim none of its episodes while what plays
plays out; resumed, it goes on, in place while its process beats, or launched again in its own directory once it is
gone, going on from the ledger."""

import asyncio
import contextlib
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import JsonValue

from rollout.harness.blobs import FileBlobStore
from rollout.testing import until
from rollout_train import Checkpoints, Files, Step, train
from rollout_train import loop as loop_module
from rollout_train.cli import _train  # pyright: ignore[reportPrivateUsage]
from rollout_train.launcher import LAUNCHER, Launcher
from rollout_train.launches import ENDED, launches_of
from rollout_train.ledger import FileLedger, Ledger
from rollout_train.presence import FilePresence, presence_of
from rollout_train.record import ENDS, FINISHED, GROUPS, RESULTS, STARTS, STEPS, STOPPED, scope, table
from rollout_train.registry import registry_of
from rollout_train.resuming import IN_PLACE, LAUNCHED, pause, resume
from rollout_train.rollouts.scheduler import CLAIMS, EPISODES
from rollout_train.settings import PAUSED, desired_settings_of, paused
from rollout_train.testing import Policy, ScriptedEngine, plain_channel
from rollout_train.trainer import Item
from tests.rollout_train.rollouts.games import GATES, Gated, Words
from tests.rollout_train.support import ENVIRONMENT, Counting, Notes, Running, Steps, ask, here, runner, served


def wanted_of(ledger: Ledger, run: str = "train") -> Callable[[], Awaitable[Mapping[str, JsonValue]]]:
    """What a run's loop reads of its desired settings, as `rollout train` reads them."""
    store = desired_settings_of(ledger)
    assert store is not None

    async def desired() -> Mapping[str, JsonValue]:
        found = await store.desired(run)
        return found.settings if found else {}

    return desired


async def test_a_paused_loop_decides_no_group_or_step_while_what_plays_plays_out_and_resumed_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loop_module, "PAUSE_LOOK", 0.02)
    channel = plain_channel(always=[("yes\n", "stop"), ("no\n", "stop")])
    engine = cast(ScriptedEngine, channel.engines[0])
    answer, go = engine.generate, asyncio.Event()

    async def held(*arguments: Any, **options: Any) -> Any:  # (the first group's episodes play until the test says)
        await go.wait()
        return await answer(*arguments, **options)

    engine.generate = held
    recorder = Policy(channel)
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    store = desired_settings_of(ledger)
    assert store is not None
    running, notes = Running(), Notes()
    async with here(ledger, recorder, blobs, hooks=[running], places=4):
        work = train(
            Words(), Counting(), Checkpoints(ledger, blobs), base="words-base", channel="policy",
            directory=tmp_path / "v", publish=recorder.publish, groups=4, groups_per_step=1, episodes_at_once=4, seed=1,
            desired=wanted_of(ledger), hooks=[notes],
        )  # fmt: skip
        training = asyncio.create_task(work)
        await until(lambda: len(running.running) == 4)  # (the first group's four, playing)
        await store.want("train", {PAUSED: True})  # (between two groups, no step near)
        await asyncio.sleep(0.1)
        go.set()
        await until(lambda: ledger.read(table("train", RESULTS)))
        await asyncio.sleep(0.3)
        assert sorted(await ledger.read(table("train", GROUPS))) == ["1", "2"]  # (none decided since)
        assert await ledger.read(table("train", STEPS)) == {}  # (a step was due: none was decided)
        assert sorted(await ledger.read(table("train", EPISODES))) == [f"1/{number}" for number in range(1, 5)]
        assert all(key.startswith("1/") for key in await ledger.read(table("train", CLAIMS)))  # (none of the second)
        assert "paused" in notes.kinds and not training.done()
        await store.want("train", {PAUSED: False})
        async with asyncio.timeout(10):
            await training
    assert sorted(await ledger.read(table("train", RESULTS)), key=int) == ["1", "2", "3", "4"]
    assert await ledger.read(table("train", STEPS)) and notes.kinds.index("resumed") > notes.kinds.index("paused")


async def test_a_runner_claims_nothing_of_a_paused_run_or_its_evals_and_says_so_in_its_beats(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    heartbeats = FilePresence(ledger.directory)
    played, _, _ = runner(tmp_path, "yes", places=2, presence=heartbeats, about=lambda: {"run": "train"}, beating=60)
    groups: dict[int, tuple[JsonValue, int]] = {1: ({"word": "yes", "gate": "paused-a"}, 2)}
    groups[2] = ({"word": "yes", "gate": "paused-b"}, 2)
    await ask(ledger, "train", groups, Gated)
    await ask(ledger, "train-eval-1", {1: ({"word": "yes", "gate": "paused-b"}, 1)}, Gated)
    await ask(ledger, "trees", {1: ({"word": "yes", "gate": "paused-c"}, 1)}, Gated)  # (another run, not paused)
    fence = await ledger.take(scope("train-eval-1"))  # (an eval the run's schedule asked for)
    await ledger.append(table("train-eval-1", STARTS), str(fence.number), {"kind": "eval", "by": "train"}, fence)
    store = desired_settings_of(ledger)
    assert store is not None and not await paused(ledger, "train-eval-1")

    async def claims(run: str) -> list[str]:
        return sorted(await ledger.read(table(run, CLAIMS)))

    async with served(played):
        await until(lambda: _counted(claims("train"), 2))
        await store.want("train", {PAUSED: True})
        assert await paused(ledger, "train-eval-1")  # (with the run that asked for it)
        await a_start(ledger, "an-eval-1", kind="eval", part_of="an-eval")  # (a part of an eval of its own)
        assert not await paused(ledger, "an-eval-1")
        await store.want("an-eval", {PAUSED: True})
        assert await paused(ledger, "an-eval-1")  # (with the eval it is a part of)
        await until(lambda: _paused_in_beat(heartbeats))  # (within a look: no step, no group boundary waited for)
        GATES.setdefault("paused-a", asyncio.Event()).set()
        await until(lambda: _counted(ledger.read(table("train", EPISODES)), 2))  # (what played, played out)
        GATES.setdefault("paused-b", asyncio.Event()).set()
        await asyncio.sleep(0.2)
        assert await claims("train") == ["1/1/1", "1/2/1"] and await claims("train-eval-1") == []
        assert await claims("trees") == ["1/1/1"]  # (the room left went to a run not paused)
        assert await played.open() == []
        (beat,) = await heartbeats.beats()
        assert beat.about["paused"] == ["train", "train-eval-1"]
        await store.want("train", {PAUSED: False})
        await until(lambda: _counted(ledger.read(table("train", EPISODES)), 4))
        await until(lambda: _counted(ledger.read(table("train-eval-1", EPISODES)), 1))
        GATES.setdefault("paused-c", asyncio.Event()).set()
        await until(lambda: _counted(ledger.read(table("trees", EPISODES)), 1))
    (beat,) = await heartbeats.beats()
    assert "paused" not in beat.about


async def _counted(read: Awaitable[Any], at_least: int) -> bool:
    return len(await read) >= at_least


async def _paused_in_beat(heartbeats: FilePresence) -> bool:
    return any(beat.about.get("paused") for beat in await heartbeats.beats())


async def a_start(ledger: Ledger, run: str, **said: JsonValue) -> None:
    fence = await ledger.take(scope(run))
    begun: dict[str, JsonValue] = {"from": None, "started": time.time(), **said}
    await ledger.append(table(run, STARTS), str(fence.number), begun, fence)


async def test_a_run_beating_is_resumed_in_place_if_paused_and_not_launched_again(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    await a_start(ledger, "train", directory=str(tmp_path / "run"), profile=str(tmp_path / "small.toml"))
    heartbeats, store = presence_of(ledger), desired_settings_of(ledger)
    assert heartbeats is not None and store is not None
    await heartbeats.beat("here/run", {"run": "train"})
    with pytest.raises(ValueError, match="is running"):
        await resume(ledger, "train")  # (beating, and not paused)
    await pause(ledger, "train")
    resumed = await resume(ledger, "train")
    assert resumed.how == IN_PLACE and resumed.launch is None
    found = await store.desired("train")
    assert found is not None and found.settings[PAUSED] is False
    launches = launches_of(ledger)
    assert launches is not None and await launches.all() == []  # (nothing launched)
    with pytest.raises(KeyError, match="no run"):
        await pause(ledger, "nothing")
    fence = await ledger.take(scope("train"))
    await ledger.append(table("train", STARTS), str(fence.number), {"started": time.time()}, fence)
    await ledger.append(table("train", ENDS), str(fence.number), {"how": FINISHED}, fence)
    with pytest.raises(ValueError, match="finished"):
        await resume(ledger, "train")  # (its process beat a moment ago, and said it finished)


RESUMABLE = """
directory = "{directory}"

[channels.policy]
model = "a-checkpoint"
renderer = "rollout_train.testing:plain_renderer"
engine = "rollout_train.testing:scripted_engine"

[trainer]
kind = "tests.rollout_train.test_pausing:Slow"
channel = "policy"
segment_tokens = 900
segments_per_step = 3
"""


class Slow(Steps):
    """Steps that take a while, so that a run can be stopped between two."""

    async def step(self, batch: Sequence[Item], *, seed: int, parent: Files | None, into: Path) -> Step:
        await asyncio.sleep(0.05)
        return await super().step(batch, seed=seed, parent=parent, into=into)


class Started:
    """A launched `rollout train`, played in this process: it ends when the run ends, or on an interrupt."""

    def __init__(self, work: asyncio.Task[None]) -> None:
        self.pid, self.work = 4343, work

    def send_signal(self, number: int) -> None:
        self.work.cancel()

    async def wait(self) -> int:
        try:
            await self.work
        except BaseException:
            return 1
        return 0


async def test_a_stopped_run_resumed_is_launched_again_into_itself_and_goes_on_from_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "resumable.toml").write_text(RESUMABLE.format(directory=tmp_path / "run"))
    ledger = FileLedger(tmp_path / "run" / "ledger")
    checkpoints = Checkpoints(ledger, FileBlobStore(tmp_path / "run" / "blobs"))
    first = asyncio.create_task(_train(profiles / "resumable.toml", None, ENVIRONMENT, groups=8, groups_per_step=1,
                                       seed=1))  # fmt: skip
    await until(lambda: _counted(checkpoints.all(), 2), seconds=30)
    first.cancel()  # (an interrupt: the run says it stopped)
    with contextlib.suppress(asyncio.CancelledError):
        await first
    (run,) = {each.run for each in await checkpoints.all()}
    assert run is not None
    ends: Any = await ledger.read(table(run, ENDS))
    assert [each["how"] for each in ends.values()] == [STOPPED]
    episodes, claims = await ledger.read(table(run, EPISODES)), await ledger.read(table(run, CLAIMS))
    made = {each.id for each in await checkpoints.all()}
    played = len(await ledger.read(table(run, RESULTS)))
    assert 0 < played < 8  # (stopped part of the way)
    with pytest.raises(KeyError, match="no launcher alive"):
        await resume(ledger, run)  # (it beat a moment ago, but said it stopped: no launcher offers its profile)

    spawned: list[list[str]] = []

    async def spawn(*command: str, **_: Any) -> Started:
        spawned.append(list(command))

        def after(flag: str) -> str:
            return command[command.index(flag) + 1]

        work = _train(
            Path(command[4]), Path(after("--directory")), command[5], int(after("--groups")),
            int(after("--groups-per-step")), int(after("--seed")), name=after("--name"),
        )  # fmt: skip
        return Started(asyncio.create_task(work))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    launches, heartbeats = launches_of(ledger), presence_of(ledger)
    assert launches is not None and heartbeats is not None
    launcher = Launcher("launcher/here", launches, heartbeats, profiles, [], tmp_path / "runs", every=0.01)
    serving = asyncio.create_task(launcher.serve())
    try:
        await until(lambda: _launcher_beats(heartbeats))
        resumed = await resume(ledger, run)
        launch = resumed.launch
        assert resumed.how == LAUNCHED and launch is not None
        asked = launch.asked
        assert (asked.resumes, asked.directory, asked.profile) == (run, str(tmp_path / "run"), "resumable")
        assert asked.groups == 8 - played  # (the groups it had left)
        with pytest.raises(ValueError, match="being launched"):
            await resume(ledger, run)
        await until(lambda: _state(launches, launch.id, ENDED), seconds=30)
    finally:
        serving.cancel()
    (command,) = spawned
    assert command[command.index("--directory") + 1] == str(tmp_path / "run")  # (its own directory: the same run)
    assert {each.run for each in await checkpoints.all()} == {run}
    ends = await ledger.read(table(run, ENDS))
    assert [ends[key]["how"] for key in sorted(ends, key=int)] == [STOPPED, FINISHED]
    assert sorted(await ledger.read(table(run, GROUPS)), key=int) == [str(number) for number in range(1, 9)]
    after = await ledger.read(table(run, EPISODES))
    assert all(after[key] == record for key, record in episodes.items())  # (no recorded episode played again)
    again: Any = await ledger.read(table(run, CLAIMS))
    attempts = {each.rsplit("/", 1)[0] for each in set(again) - set(claims)}  # (the episodes claimed since)
    assert not attempts & set(episodes)
    steps = Counter(each.step for each in await checkpoints.all())
    assert made <= {each.id for each in await checkpoints.all()} and set(steps.values()) == {1}  # (no step made twice)
    covered: Any = await ledger.read(table(run, STEPS))
    trained = [group for step in covered.values() for group in step["groups"]]
    assert len(trained) == len(set(trained))  # (no group trained on twice)


async def _launcher_beats(heartbeats: Any) -> bool:
    return any(beat.about.get("kind") == LAUNCHER for beat in await heartbeats.beats())


async def _state(launches: Any, id: str, state: str) -> bool:
    return any(each.id == id and each.state == state for each in await launches.all())


def test_the_commands_pause_and_resume_a_run_by_its_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rollout_train.cli import main

    ledger = FileLedger(tmp_path / "ledger")

    async def set_up() -> str:
        registry, heartbeats = registry_of(ledger), presence_of(ledger)
        assert registry is not None and heartbeats is not None
        entry = await registry.create("scout")
        await a_start(ledger, entry.id)
        await heartbeats.beat("here/scout", {"run": entry.id})
        return entry.id

    run = asyncio.run(set_up())

    def command(*arguments: str) -> str:
        monkeypatch.setattr("sys.argv", ["rollout", *arguments, "--ledger", str(tmp_path / "ledger")])
        main()
        return capsys.readouterr().out

    assert command("pause", "scout") == "scout is paused: what is playing plays out, and nothing new starts\n"
    assert asyncio.run(paused(ledger, run))
    assert command("resume", "scout") == "scout goes on\n" and not asyncio.run(paused(ledger, run))
    with pytest.raises(SystemExit, match="is running, and not paused"):
        command("resume", run)  # (by its id too)
    with pytest.raises(SystemExit, match="no run 'nobody'"):
        command("pause", "nobody")
