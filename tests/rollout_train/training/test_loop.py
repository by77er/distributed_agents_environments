"""The training loop with a runner playing the episodes it asks for in the ledger, under two runners: one in this
process, and a durable one. The loop's code is the same; so is what it does."""

import asyncio
import contextlib
import functools
import json
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import JsonValue

from rollout.harness import Runner
from rollout.harness.blobs import Blobs, FileBlobStore
from rollout.local import LocalRunner
from rollout_train import (
    Budget,
    Checkpoint,
    Colocated,
    FileLedger,
    Ledger,
    Step,
    StepFailed,
    Version,
    Versions,
    Weighted,
    results,
    train,
    trained,
)
from rollout_train import loop as loop_module
from rollout_train.record import STEPS, table
from rollout_train.recorder import Recorder
from rollout_train.rollouts import EpisodeRunner, Hooks, Record, episodes_of, loaded, playing
from rollout_train.rollouts.scheduler import EPISODES
from rollout_train.testing import ScriptedEngine, plain_channel
from rollout_train.trainer import STATE, WEIGHTS
from rollout_train.versions import Retention
from tests.rollout_train.rollouts.games import Words


@pytest.fixture(autouse=True)
def quickly(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loop looks for its groups' episodes in the ledger often (a run looks twice a second)."""
    monkeypatch.setattr(loop_module, "episodes_of", functools.partial(episodes_of, every=0.01))


class Counting:
    """A trainer that trains nothing: it writes down what it was given and leaves files as a trainer would."""

    budget = Budget(segments=3)

    def __init__(self, fails: int = 0) -> None:
        self.batches: list[list[Weighted]] = []
        self.parents: list[str | None] = []
        """What each step started from: the text of its parent's weights."""
        self.fails = fails
        """Steps that fail before one succeeds."""

    async def step(self, batch: Sequence[Weighted], *, seed: int, parent: Checkpoint | None, into: Path) -> Step:
        if self.fails:
            self.fails -= 1
            raise StepFailed("the trainer failed:\nout of memory")
        self.batches.append(list(batch))
        self.parents.append((parent.weights / "adapter.bin").read_text() if parent else None)
        (into / WEIGHTS).mkdir(parents=True)
        (into / WEIGHTS / "adapter.bin").write_text(f"weights after {len(self.batches)} steps")
        (into / STATE).mkdir()
        (into / STATE / "optimizer.bin").write_text(f"moments after {len(self.batches)} steps")
        return Step({"segments": float(len(batch))})


class Notes(Hooks):
    def __init__(self) -> None:
        self.kinds: list[str] = []

    def on_note(self, event: Mapping[str, JsonValue]) -> None:
        self.kinds.append(str(event["kind"]))


def versions_in(directory: Path) -> Versions:
    return Versions(FileLedger(directory / "ledger"), FileBlobStore(directory / "blobs"))


async def made_by(versions: Versions, run: str = "train") -> list[Version]:
    """The versions a run made, oldest first."""
    return sorted((v for v in await versions.all() if v.run == run), key=lambda version: version.depth)


def answering() -> Recorder:
    return Recorder({"policy": plain_channel(always=[("yes\n", "stop"), ("no\n", "stop")])})


@contextlib.asynccontextmanager
async def here(
    ledger: Ledger,
    recorder: Recorder,
    blobs: Blobs,
    *,
    runner: Runner | None = None,
    hooks: Sequence[Hooks] = (),
    places: int = 6,
    name: str = "here",
) -> AsyncGenerator[EpisodeRunner]:
    """A runner that plays what runs ask for in `ledger`, while the block runs."""
    played = runner if runner is not None else LocalRunner(recorder=recorder)
    episodes = EpisodeRunner(name, ledger, played, recorder, blobs, places, hooks=hooks, every=0.01)
    async with playing(episodes):
        yield episodes


@pytest.fixture(params=["in process", "durable runner"])
async def wiring(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[tuple[Runner, Recorder]]:
    recorder = answering()
    if request.param == "in process":
        yield LocalRunner(recorder=recorder), recorder
        return
    pytest.importorskip("dbos")
    from rollout_durable import DurableRunner

    runner = DurableRunner(tmp_path / "runs", recorder=recorder)
    await runner.launch()
    try:
        yield cast(Runner, runner), recorder
    finally:
        await runner.close()


async def test_the_loop_records_each_group_and_steps_on_what_it_played(
    wiring: tuple[Runner, Recorder], tmp_path: Path
) -> None:
    runner, recorder = wiring
    versions, trainer, notes = versions_in(tmp_path), Counting(), Notes()

    async def more(groups: int) -> None:
        async with here(versions.ledger, recorder, versions.blobs, runner=runner, hooks=[notes]):
            await train(
                Words(), trainer, versions, base="words-base", channel="policy", directory=tmp_path / "versions",
                publish=recorder.publish, groups=groups, groups_per_step=1, seed=1, hooks=[notes],
            )  # fmt: skip

    await more(3)
    lines = await results(versions.ledger)
    assert [line.group for line in lines] == [1, 2, 3]
    assert all(len(line.rewards) == 4 and line.failed == 0 for line in lines)
    # The policy says yes and no in turn, so in most groups some episodes said the word and some did not: those
    # groups train, and the others are skipped for having nothing to compare.
    played = [line for line in lines if line.segments]
    assert all(len(set(line.rewards)) == 1 and line.skipped for line in lines if not line.segments)
    assert len(played) == len(trainer.batches) >= 1

    # A step over each group with something to train on (one group a step, here) made a version, each from the one
    # before (the first from the base model), and the channel serves the newest, by its id and its depth.
    made = await made_by(versions)
    covered = await trained(versions.ledger)
    assert [version.depth for version in made] == list(range(1, len(played) + 1))
    assert [version.parents for version in made] == [(), *[(version.id,) for version in made[:-1]]]
    assert all(version.base == "words-base" and version.run == "train" for version in made)
    assert sorted(covered) == [line.group for line in played]
    assert sorted(str(outcome.version) for outcome in covered.values()) == sorted(version.id for version in made)
    assert sorted(version.step or 0 for version in made) == sorted(covered[line.group].step for line in played)
    channel = recorder.channels["policy"]
    assert (channel.adapter, channel.version) == (made[-1].id, len(made))
    assert trainer.parents == [None, *[f"weights after {n} steps" for n in range(1, len(played))]]
    for version, batch in zip(made, trainer.batches, strict=True):
        assert version.state is not None and list(version.state.files) == ["optimizer.bin"]
        assert version.batch is not None  # what it was trained on: each segment by its episode and its place there
        listed = json.loads(await versions.blobs.read(version.batch))
        assert listed == [[weighted.source, weighted.advantage] for weighted in batch]
        (line,) = [line for line in played if covered[line.group].version == version.id]
        assert line.segments_recorded == 4 and line.segments == len(batch) == 3  # the trainer's budget
        if sum(line.rewards) >= 2:  # those that said it were as fast as each other
            assert line.notes["speed_bonus"] == [1.0 if reward else 0.0 for reward in line.rewards]
        word = line.task.removeprefix("say-")
        for weighted in batch:  # whoever said the word is above the group's mean, and the others below it
            said = "".join(chr(token) for token in weighted.segment.tokens[weighted.segment.spans[0].start :]).strip()
            assert (weighted.advantage > 0) == (said == word)
    assert notes.kinds.count("result") == 3 and notes.kinds.count("step") == notes.kinds.count("published")
    assert notes.kinds.count("published") == len(played)
    assert notes.kinds.count("started") == notes.kinds.count("ended") == 12

    await more(2)  # started again: it goes on after the last group, from the newest version it made
    assert [line.group for line in await results(versions.ledger)] == [1, 2, 3, 4, 5]
    later = (await made_by(versions))[len(made) :]
    if later:  # (unless no group of the two had anything to train on)
        assert later[0].parents == (made[-1].id,) and later[0].depth == len(made) + 1
        assert trainer.parents[len(played)] == f"weights after {len(played)} steps"


async def test_a_step_waits_for_its_groups_and_takes_them_together(tmp_path: Path) -> None:
    recorder = answering()
    versions, trainer = versions_in(tmp_path), Counting()
    trainer.budget = Budget(segments=100)
    async with here(versions.ledger, recorder, versions.blobs):
        await train(
            Words(), trainer, versions, base="words-base", channel="policy", directory=tmp_path / "v",
            publish=recorder.publish, groups=6, groups_per_step=2, seed=1,
        )  # fmt: skip
    lines = await results(versions.ledger)
    played = sorted(line.group for line in lines if line.segments)
    steps = await versions.ledger.read(table("train", STEPS))
    covers = [[int(group) for group in step["groups"]] for step in steps.values()]  # type: ignore[index, union-attr]
    # Every group with something to train on is in one step; every step but the last waited for at least two of
    # them (and took every one queued by then, played while the step before ran).
    assert sorted(group for groups in covers for group in groups) == played
    assert all(len(groups) >= 2 for groups in covers[:-1]) and covers[-1]
    assert [len(batch) for batch in trainer.batches] == [4 * len(groups) for groups in covers]
    assert len(await made_by(versions)) == len(covers)


class Running(Hooks):
    """Counts the episodes running, and the groups they are of, as the runner tells of them."""

    def __init__(self) -> None:
        self.running: dict[str, int] = {}
        """Each episode running: its group."""
        self.most = 0
        self.groups_at_once = 0

    def on_note(self, event: Mapping[str, JsonValue]) -> None:
        if event["kind"] == "started":
            self.running[str(event["run_id"])] = cast(int, event["group"])
        elif event["kind"] == "ended":
            self.running.pop(str(event["run_id"]), None)
        self.most = max(self.most, len(self.running))
        self.groups_at_once = max(self.groups_at_once, len(set(self.running.values())))


async def test_episodes_of_any_groups_run_at_once_up_to_the_runners_places(tmp_path: Path) -> None:
    channel = plain_channel(always=[("yes\n", "stop"), ("no\n", "stop")])
    engine = cast(ScriptedEngine, channel.engines[0])
    answer = engine.generate

    async def slowly(*arguments: Any, **options: Any) -> Any:  # (long enough that the next group starts meanwhile)
        await asyncio.sleep(0.05)
        return await answer(*arguments, **options)

    engine.generate = slowly
    recorder = Recorder({"policy": channel})
    running = Running()
    versions = versions_in(tmp_path)
    async with here(versions.ledger, recorder, versions.blobs, hooks=[running], places=6):
        await train(
            Words(), Counting(), versions, base="words-base", channel="policy", directory=tmp_path / "v",
            publish=recorder.publish, groups=5, groups_per_step=2, episodes_at_once=6, seed=1,
        )  # fmt: skip
    # Four episodes a group: six at once are a group and half the next.
    assert running.most == 6 and running.groups_at_once >= 2


async def test_runners_share_a_runs_episodes_and_none_is_played_twice(tmp_path: Path) -> None:
    recorder = answering()
    versions, first, second = versions_in(tmp_path), Notes(), Notes()
    ledger = versions.ledger
    async with (
        here(ledger, recorder, versions.blobs, hooks=[first], places=2, name="one"),
        here(FileLedger(tmp_path / "ledger"), recorder, versions.blobs, hooks=[second], places=2, name="two"),
    ):
        await train(
            Words(), Counting(), versions, base="words-base", channel="policy", directory=tmp_path / "v",
            publish=recorder.publish, groups=4, groups_per_step=2, seed=1,
        )  # fmt: skip
    assert first.kinds.count("ended") and second.kinds.count("ended")  # each played some
    assert first.kinds.count("ended") + second.kinds.count("ended") == 16
    assert sorted(await ledger.read(table("train", EPISODES))) == sorted(
        f"{group}/{number}" for group in range(1, 5) for number in range(1, 5)
    )


async def test_a_trainer_that_shares_the_engines_gpu_puts_them_to_sleep_around_each_step(tmp_path: Path) -> None:
    recorder = answering()
    channel = recorder.channels["policy"]
    engine: Any = channel.engines[0]
    assert isinstance(engine, ScriptedEngine)
    guarded: list[str] = []
    trainer = Colocated(Counting(), [channel], guard=lambda: guarded.append(engine.told[-1]))
    versions = versions_in(tmp_path)
    async with here(versions.ledger, recorder, versions.blobs):
        await train(
            Words(), trainer, versions, base="words-base", channel="policy", directory=tmp_path / "v",
            publish=recorder.publish, groups=2,
        )  # fmt: skip
    made = await made_by(versions)
    assert made and engine.told[:4] == ["sleep", "wake", f"load {made[0].id}", "sleep"][: len(engine.told[:4])]
    assert guarded and set(guarded) == {"sleep"}  # the guard runs once the engines are asleep
    assert all("update_seconds" in version.metrics for version in made)


async def test_a_step_that_fails_leaves_the_weights_and_the_run_goes_on(tmp_path: Path) -> None:
    recorder = answering()
    versions, trainer = versions_in(tmp_path), Counting(fails=1)
    async with here(versions.ledger, recorder, versions.blobs):
        await train(
            Words(), trainer, versions, base="words-base", channel="policy", directory=tmp_path / "v",
            publish=recorder.publish, groups=3, groups_per_step=1,
        )  # fmt: skip
    covered = await trained(versions.ledger)
    first, *rest = [covered[group] for group in sorted(covered)]
    assert first.error == "out of memory" and first.version is None  # written down, and the weights as they were
    assert rest and all(outcome.version is not None and outcome.error is None for outcome in rest)
    assert [version.depth for version in await made_by(versions)] == list(range(1, len(rest) + 1))


async def test_the_ledger_keeps_every_episode_a_step_trained_on(tmp_path: Path) -> None:
    recorder = answering()
    versions, trainer = versions_in(tmp_path), Counting()
    async with here(versions.ledger, recorder, versions.blobs):
        await train(
            Words(), trainer, versions, base="words-base", channel="policy", directory=tmp_path / "v",
            publish=recorder.publish, groups=2,
        )  # fmt: skip
    ended = await versions.ledger.read(table("train", EPISODES))
    for version, batch in zip(await made_by(versions), trainer.batches, strict=True):
        assert version.batch is not None
        run, group, number, slot, index = json.loads(await versions.blobs.read(version.batch))[0][0].split("/")
        assert run == "train"
        record = Record.from_json(cast(Mapping[str, Any], ended[f"{group}/{number}"]))
        episode = await loaded(record, versions.blobs)
        assert episode.trajectories[slot].segments[int(index)] == batch[0].segment


async def test_a_run_started_from_another_runs_version_forks_there(tmp_path: Path) -> None:
    recorder = answering()
    versions, trainer = versions_in(tmp_path), Counting()
    keep_newest = Retention(recent=1, every=0)

    async def first_run(groups: int) -> None:
        await train(
            Words(), trainer, versions, base="words-base", channel="policy", directory=tmp_path / "v",
            publish=recorder.publish, run="first", groups=groups, groups_per_step=1, seed=1, retention=keep_newest,
        )  # fmt: skip

    async with here(versions.ledger, recorder, versions.blobs):
        await first_run(3)
        before = await made_by(versions, "first")
        assert before
        fork = before[-1]  # the first run's newest: another run starts from it
        forked = answering()
        async with here(versions.ledger, forked, versions.blobs, name="other"):
            await train(
                Words(), trainer, versions, start=fork.id, channel="policy", directory=tmp_path / "w",
                publish=forked.publish, run="second", groups=3, groups_per_step=1, seed=2,
            )  # fmt: skip
        trained_first = len(trainer.batches)
        await first_run(4)  # the first run goes on, and thins its own versions as it goes
    second = await made_by(versions, "second")
    assert second and second[0].parents == (fork.id,) and second[0].depth == fork.depth + 1
    assert all(version.base == "words-base" for version in second)  # its base, inherited from where it began
    assert forked.channels["policy"].adapter == second[-1].id
    assert trainer.parents[len(before)] == f"weights after {len(before)} steps"  # trained from the fork's weights
    after = await made_by(versions, "first")
    assert len(after) > len(before) and len(trainer.batches) > trained_first
    assert after[len(before)].parents == (fork.id,)  # the first run went on from its own newest, not the second's
    assert (await versions.version(fork.id)).weights is not None  # a run starts from it: its files are kept
    served = {after[-1].id, *after[-1].parents}  # (and what is served, with what it was trained from)
    assert all(version.weights is None for version in after if version.id not in {fork.id, *served})
    assert len(after) - len(served | {fork.id}) >= 1  # something was thinned


async def test_each_version_made_is_told_of_once_it_is_served(tmp_path: Path) -> None:
    recorder = answering()
    versions = versions_in(tmp_path)
    told: list[tuple[str, str | None]] = []

    async def made(version: Version) -> None:
        told.append((version.id, recorder.channels["policy"].adapter))

    async with here(versions.ledger, recorder, versions.blobs):
        await train(
            Words(), Counting(), versions, channel="policy", directory=tmp_path / "v", publish=recorder.publish,
            groups=3, groups_per_step=1, seed=1, made=made,
        )  # fmt: skip
    ids = [version.id for version in await made_by(versions)]
    assert ids and told == [(id, id) for id in ids]  # once each, and the channel already serves it
