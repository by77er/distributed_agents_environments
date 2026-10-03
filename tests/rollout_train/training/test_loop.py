"""The training loop under two wirings: everything in this process, and a durable runner with the rollout jobs
behind HTTP. The loop's code is the same; so is what it does."""

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import JsonValue

from rollout.harness import Runner
from rollout.harness.blobs import FileBlobStore
from rollout.local import LocalRunner
from rollout_train import (
    Budget,
    Checkpoint,
    Colocated,
    FileLedger,
    Policies,
    Step,
    StepFailed,
    Weighted,
    results,
    train,
    trained,
)
from rollout_train.record import STEPS, table
from rollout_train.recorder import Recorder
from rollout_train.rollouts import JobHooks, Jobs, RolloutJobs, loaded
from rollout_train.testing import ScriptedEngine, plain_channel
from rollout_train.trainer import STATE, WEIGHTS
from tests.rollout_train.rollouts.games import Words


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


class Notes(JobHooks):
    def __init__(self) -> None:
        self.kinds: list[str] = []

    def on_job(self, event: Mapping[str, JsonValue]) -> None:
        self.kinds.append(str(event["kind"]))


def policies_in(directory: Path) -> Policies:
    return Policies(FileLedger(directory / "ledger"), FileBlobStore(directory / "blobs"))


@pytest.fixture(params=["in process", "durable runner, jobs over HTTP"])
async def wiring(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[tuple[Jobs, Recorder, Notes]]:
    recorder = Recorder({"policy": plain_channel(always=[("yes\n", "stop"), ("no\n", "stop")])})
    notes = Notes()
    blobs = FileBlobStore(tmp_path / "blobs")
    if request.param == "in process":
        local = RolloutJobs(LocalRunner(recorder=recorder), recorder, log=tmp_path / "log", blobs=blobs, hooks=[notes])
        yield local, recorder, notes
        await local.close()
        return
    pytest.importorskip("dbos")
    pytest.importorskip("starlette")
    from rollout_durable import DurableRunner
    from rollout_train.rollouts.service import RolloutClient, create_app

    runner = DurableRunner(tmp_path / "runs", recorder=recorder)
    await runner.launch()
    served = RolloutJobs(cast_runner(runner), recorder, log=tmp_path / "log", blobs=blobs, hooks=[notes])
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(served)), base_url="http://rollouts")
    try:
        yield RolloutClient("http://rollouts", blobs, client=client), recorder, notes
    finally:
        await served.close()
        await runner.close()


def cast_runner(runner: Any) -> Runner:
    return runner


async def test_the_loop_records_each_group_and_steps_on_what_it_played(
    wiring: tuple[Jobs, Recorder, Notes], tmp_path: Path
) -> None:
    jobs, recorder, notes = wiring
    policies, trainer = policies_in(tmp_path), Counting()

    async def more(groups: int) -> None:
        await train(
            jobs, Words(), trainer, policies, policy="words", channel="policy", directory=tmp_path / "versions",
            groups=groups, groups_per_step=1, seed=1,
        )  # fmt: skip

    await more(3)
    lines = await results(policies.ledger)
    assert [line.group for line in lines] == [1, 2, 3]
    assert all(len(line.rewards) == 4 and line.failed == 0 for line in lines)
    # The policy says yes and no in turn, so in most groups some episodes said the word and some did not: those
    # groups train, and the others are skipped for having nothing to compare.
    played = [line for line in lines if line.segments]
    assert all(len(set(line.rewards)) == 1 and line.skipped for line in lines if not line.segments)
    assert len(played) == len(trainer.batches) >= 1

    # A step over each group with something to train on (one group a step, here) made a version of the policy,
    # each from the one before, and the channel serves the newest.
    versions = await policies.versions("words")
    covered = await trained(policies.ledger)
    assert [version.name for version in versions] == [f"words@{n}" for n in range(1, len(played) + 1)]
    assert [version.parent for version in versions] == [None, *[version.name for version in versions[:-1]]]
    assert sorted(covered) == [line.group for line in played]
    assert sorted(str(outcome.version) for outcome in covered.values()) == [version.name for version in versions]
    channel = recorder.channels["policy"]
    assert (channel.adapter, channel.version) == (versions[-1].name, len(versions))
    assert trainer.parents == [None, *[f"weights after {n} steps" for n in range(1, len(played))]]
    for version, batch in zip(versions, trainer.batches, strict=True):
        assert version.state is not None and list(version.state.files) == ["optimizer.bin"]
        assert version.batch is not None  # what it was trained on: each segment by its place in the job's log
        listed = json.loads(await policies.blobs.read(version.batch))
        assert listed == [[weighted.source, weighted.advantage] for weighted in batch]
        (line,) = [line for line in played if covered[line.group].version == version.name]
        assert line.segments_recorded == 4 and line.segments == len(batch) == 3  # the trainer's budget
        if sum(line.rewards) >= 2:  # those that said it were as fast as each other
            assert line.notes["speed_bonus"] == [1.0 if reward else 0.0 for reward in line.rewards]
        word = line.task.removeprefix("say-")
        for weighted in batch:  # whoever said the word is above the group's mean, and the others below it
            said = "".join(chr(token) for token in weighted.segment.tokens[weighted.segment.spans[0].start :]).strip()
            assert (weighted.advantage > 0) == (said == word)
    assert notes.kinds.count("result") == 3 and notes.kinds.count("step") == notes.kinds.count("published")
    assert notes.kinds.count("published") == len(played) and notes.kinds.count("episode") == 12

    await more(2)  # started again: it goes on after the last group, from the newest version
    assert [line.group for line in await results(policies.ledger)] == [1, 2, 3, 4, 5]
    assert trainer.parents[len(played)] in (f"weights after {len(played)} steps", None)


async def test_a_step_waits_for_its_groups_and_takes_them_together(tmp_path: Path) -> None:
    recorder = Recorder({"policy": plain_channel(always=[("yes\n", "stop"), ("no\n", "stop")])})
    rollouts = RolloutJobs(LocalRunner(recorder=recorder), recorder)
    policies, trainer = policies_in(tmp_path), Counting()
    trainer.budget = Budget(segments=100)
    await train(
        rollouts, Words(), trainer, policies, policy="words", channel="policy", directory=tmp_path / "v",
        groups=6, groups_per_step=2, seed=1,
    )  # fmt: skip
    await rollouts.close()
    lines = await results(policies.ledger)
    played = sorted(line.group for line in lines if line.segments)
    steps = await policies.ledger.read(table("train", STEPS))
    covers = [[int(group) for group in step["groups"]] for step in steps.values()]  # type: ignore[index, union-attr]
    # Every group with something to train on is in one step; every step but the last waited for at least two of
    # them (and took every one queued by then, played while the step before ran).
    assert sorted(group for groups in covers for group in groups) == played
    assert all(len(groups) >= 2 for groups in covers[:-1]) and covers[-1]
    assert [len(batch) for batch in trainer.batches] == [4 * len(groups) for groups in covers]
    assert len(await policies.versions("words")) == len(covers)


async def test_a_trainer_that_shares_the_engines_gpu_puts_them_to_sleep_around_each_step(tmp_path: Path) -> None:
    recorder = Recorder({"policy": plain_channel(always=[("yes\n", "stop"), ("no\n", "stop")])})
    rollouts = RolloutJobs(LocalRunner(recorder=recorder), recorder)
    channel = recorder.channels["policy"]
    engine: Any = channel.engines[0]
    assert isinstance(engine, ScriptedEngine)
    guarded: list[str] = []
    trainer = Colocated(Counting(), [channel], guard=lambda: guarded.append(engine.told[-1]))
    policies = policies_in(tmp_path)
    await train(
        rollouts, Words(), trainer, policies, policy="words", channel="policy", directory=tmp_path / "v", groups=2
    )
    await rollouts.close()
    versions = await policies.versions("words")
    assert versions and engine.told[:4] == ["sleep", "wake", "load words@1", "sleep"][: len(engine.told[:4])]
    assert guarded and set(guarded) == {"sleep"}  # the guard runs once the engines are asleep
    assert all("update_seconds" in version.metrics for version in versions)


async def test_a_step_that_fails_leaves_the_weights_and_the_run_goes_on(tmp_path: Path) -> None:
    recorder = Recorder({"policy": plain_channel(always=[("yes\n", "stop"), ("no\n", "stop")])})
    rollouts = RolloutJobs(LocalRunner(recorder=recorder), recorder)
    policies, trainer = policies_in(tmp_path), Counting(fails=1)
    await train(
        rollouts, Words(), trainer, policies, policy="words", channel="policy", directory=tmp_path / "v", groups=3,
        groups_per_step=1,
    )  # fmt: skip
    await rollouts.close()
    covered = await trained(policies.ledger)
    first, *rest = [covered[group] for group in sorted(covered)]
    assert first.error == "out of memory" and first.version is None  # written down, and the weights as they were
    assert rest and all(outcome.version is not None and outcome.error is None for outcome in rest)
    assert [version.number for version in await policies.versions("words")] == list(range(1, len(rest) + 1))


async def test_the_job_keeps_every_episode_a_step_trained_on(tmp_path: Path) -> None:
    recorder = Recorder({"policy": plain_channel(always=[("yes\n", "stop"), ("no\n", "stop")])})
    rollouts = RolloutJobs(LocalRunner(recorder=recorder), recorder, log=tmp_path / "log")
    assert rollouts.blobs is not None
    policies, trainer = Policies(FileLedger(tmp_path / "ledger"), rollouts.blobs), Counting()
    await train(
        rollouts, Words(), trainer, policies, policy="words", channel="policy", directory=tmp_path / "v", groups=2
    )
    job = rollouts.job("train")
    for version, batch in zip(await policies.versions("words"), trainer.batches, strict=True):
        assert version.batch is not None
        cursor, slot, index = json.loads(await policies.blobs.read(version.batch))[0][0].split("/")
        (record,) = [record for record in job.after(0) if record.episode.cursor == int(cursor)]
        episode = await loaded(record, rollouts.blobs)  # the log still has it, long after it was acknowledged
        assert episode.trajectories[slot].segments[int(index)] == batch[0].segment
    await rollouts.close()
