"""The training loop under two wirings: everything in this process, and a durable runner with the rollout jobs
behind HTTP. The loop's code is the same; so is what it does."""

import io
import json
import tarfile
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import JsonValue

from rollout.contracts import BlobReference
from rollout.harness import Runner
from rollout.local import LocalRunner
from rollout_train import Budget, Colocated, Directory, Step, StepFailed, Weighted, iterations, train
from rollout_train.recorder import Recorder
from rollout_train.rollouts import JobHooks, Jobs, RolloutJobs, loaded
from rollout_train.testing import ScriptedEngine, plain_channel
from tests.rollout_train.rollouts.games import Words


class Counting:
    """A trainer that trains nothing and keeps what it was given."""

    budget = Budget(sequences=3)
    latest: tuple[str, str] | None = None

    def __init__(self, fails: int = 0) -> None:
        self.batches: list[list[Weighted]] = []
        self.fails = fails
        """Steps that fail before one succeeds."""

    async def step(self, batch: Sequence[Weighted], *, seed: int) -> Step:
        if self.fails:
            self.fails -= 1
            raise StepFailed("the trainer failed:\nout of memory")
        self.batches.append(list(batch))
        name = f"step-{len(self.batches)}"
        self.latest = (name, f"/adapters/{name}")
        return Step(name, f"/adapters/{name}", {"sequences": float(len(batch))})


class Notes(JobHooks):
    def __init__(self) -> None:
        self.kinds: list[str] = []

    def on_job(self, event: Mapping[str, JsonValue]) -> None:
        self.kinds.append(str(event["kind"]))


@pytest.fixture(params=["in process", "durable runner, jobs over HTTP"])
async def wiring(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[tuple[Jobs, Recorder, Notes]]:
    recorder = Recorder({"policy": plain_channel(always=[("yes\n", "stop"), ("no\n", "stop")])})
    notes = Notes()
    if request.param == "in process":
        local = RolloutJobs(LocalRunner(recorder=recorder), recorder, hooks=[notes])
        yield local, recorder, notes
        await local.close()
        return
    pytest.importorskip("dbos")
    pytest.importorskip("starlette")
    from rollout_durable import DurableRunner
    from rollout_train.rollouts.service import RolloutClient, create_app

    runner = DurableRunner(tmp_path / "runs", recorder=recorder)
    await runner.launch()
    served = RolloutJobs(cast_runner(runner), recorder, log=tmp_path / "log", hooks=[notes])
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(served)), base_url="http://rollouts")
    try:
        assert served.blobs is not None  # (the store both sides share: here, files under the log)
        yield RolloutClient("http://rollouts", served.blobs, client=client), recorder, notes
    finally:
        await served.close()
        await runner.close()


def cast_runner(runner: Any) -> Runner:
    return runner


async def test_the_loop_trains_groups_as_they_finish_and_goes_on_where_it_stopped(
    wiring: tuple[Jobs, Recorder, Notes], tmp_path: Path
) -> None:
    jobs, recorder, notes = wiring
    store, trainer = Directory(tmp_path / "run"), Counting()
    await train(jobs, Words(), trainer, store, channel="policy", groups=3, seed=1)

    lines = iterations(store)  # in the order the groups ended, which is not always the order they were started in
    assert sorted(line.iteration for line in lines) == [1, 2, 3]
    assert all(len(line.rewards) == 4 and line.failed == 0 for line in lines)
    # The policy says yes and no in turn, so in most groups some episodes said the word and some did not: those
    # groups train, and the others are skipped for having nothing to compare.
    trained = [line for line in lines if line.update is not None]
    assert all(len(set(line.rewards)) == 1 and line.skipped for line in lines if line.update is None)
    assert len(trained) == len(trainer.batches) >= 1
    assert [line.version for line in trained] == list(range(1, len(trained) + 1))
    assert recorder.channels["policy"].version == len(trained)
    for line, batch in zip(trained, trainer.batches, strict=True):
        assert line.sequences_recorded == 4 and line.sequences_trained == len(batch) == 3  # the trainer's budget
        if sum(line.rewards) >= 2:  # those that said it were as fast as each other
            assert line.notes["speed_bonus"] == [1.0 if reward else 0.0 for reward in line.rewards]
        word = line.task.removeprefix("say-")
        for weighted in batch:  # whoever said the word is above the group's mean, and the others below it
            said = "".join(chr(token) for token in weighted.epoch.tokens[weighted.epoch.spans[0].start :]).strip()
            assert (weighted.advantage > 0) == (said == word)
    saved = json.loads(store.read("curriculum.json") or "{}")
    assert {entry["title"] for entry in saved.values()} <= {"say yes", "say no", "say maybe"}
    assert notes.kinds.count("iteration") == 3 and notes.kinds.count("published") == len(trained)
    assert notes.kinds.count("episode") == 12

    await train(jobs, Words(), trainer, store, channel="policy", groups=2, seed=1)  # started again: it goes on
    assert sorted(line.iteration for line in iterations(store)) == [1, 2, 3, 4, 5]


async def test_a_trainer_that_shares_the_engines_gpu_puts_them_to_sleep_around_each_step(tmp_path: Path) -> None:
    recorder = Recorder({"policy": plain_channel(always=[("yes\n", "stop"), ("no\n", "stop")])})
    rollouts = RolloutJobs(LocalRunner(recorder=recorder), recorder)
    channel = recorder.channels["policy"]
    engine: Any = channel.engines[0]
    assert isinstance(engine, ScriptedEngine)
    guarded: list[str] = []
    trainer = Colocated(Counting(), [channel], guard=lambda: guarded.append(engine.told[-1]))
    await train(rollouts, Words(), trainer, Directory(tmp_path / "run"), channel="policy", groups=2)
    await rollouts.close()
    steps = [line for line in iterations(Directory(tmp_path / "run")) if line.update is not None]
    assert steps and engine.told[:4] == ["sleep", "wake", "load step-1", "sleep"][: len(engine.told[:4])]
    assert guarded and set(guarded) == {"sleep"}  # the guard runs once the engines are asleep
    assert all("update_seconds" in (line.update or {}) for line in steps)


async def test_a_step_that_fails_leaves_the_weights_and_the_run_goes_on(tmp_path: Path) -> None:
    recorder = Recorder({"policy": plain_channel(always=[("yes\n", "stop"), ("no\n", "stop")])})
    rollouts = RolloutJobs(LocalRunner(recorder=recorder), recorder)
    store, trainer = Directory(tmp_path / "run"), Counting(fails=1)
    await train(rollouts, Words(), trainer, store, channel="policy", groups=3)
    await rollouts.close()
    first, *rest = [line for line in iterations(store) if line.sequences_trained]
    assert first.error == "out of memory" and first.update is None and first.version is None
    assert rest and all(line.update is not None for line in rest) and recorder.channels["policy"].version == len(rest)


async def test_with_a_blob_store_every_step_keeps_what_it_trained_on_and_what_it_left(tmp_path: Path) -> None:
    class Saving(Counting):
        """A trainer that leaves a directory of weights and a file of state behind each step."""

        async def step(self, batch: Sequence[Weighted], *, seed: int) -> Step:
            step = await super().step(batch, seed=seed)
            weights = tmp_path / "adapters" / step.adapter
            weights.mkdir(parents=True)
            (weights / "adapter.bin").write_bytes(step.adapter.encode())
            (tmp_path / "optimizer.bin").write_bytes(b"state after " + step.adapter.encode())
            artifacts = {"adapter": str(weights), "optimizer": str(tmp_path / "optimizer.bin")}
            return Step(step.adapter, str(weights), step.metrics, artifacts)

    recorder = Recorder({"policy": plain_channel(always=[("yes\n", "stop"), ("no\n", "stop")])})
    rollouts = RolloutJobs(LocalRunner(recorder=recorder), recorder, log=tmp_path / "log")
    assert rollouts.blobs is not None
    store, trainer = Directory(tmp_path / "run"), Saving()
    await train(rollouts, Words(), trainer, store, channel="policy", groups=2, blobs=rollouts.blobs)
    job = rollouts.job("train")
    for line, batch in zip([line for line in iterations(store) if line.update], trainer.batches, strict=True):
        # What it trained on: each sequence by its place in the job's log, with its advantage.
        assert line.batch is not None
        trained = json.loads(await rollouts.blobs.read(BlobReference.model_validate(line.batch)))
        assert trained == [[weighted.source, weighted.advantage] for weighted in batch]
        cursor, slot, index = trained[0][0].split("/")
        (record,) = [record for record in job.after(0) if record.episode.cursor == int(cursor)]
        episode = await loaded(record, rollouts.blobs)  # the log still has it, long after it was acknowledged
        assert episode.traces[slot].epochs[int(index)] == batch[0].epoch
        # What it left: the weights (a directory, kept as an archive) and the state it would go on from (a file).
        assert set(line.checkpoint) == {"adapter", "optimizer"}
        state = await rollouts.blobs.read(BlobReference.model_validate(line.checkpoint["optimizer"]))
        assert state == b"state after " + str(line.adapter).encode()
        packed = await rollouts.blobs.read(BlobReference.model_validate(line.checkpoint["adapter"]))
        with tarfile.open(fileobj=io.BytesIO(packed)) as archive:
            assert archive.getnames() == [str(line.adapter), f"{line.adapter}/adapter.bin"]
    await rollouts.close()
