"""The training loop is killed at every point where it writes or calls the trainer, and started again each time:
the run ends with every group played and recorded once, every group with something to train on in one step, every
step taken once, and nothing asked for twice."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import JsonValue

from rollout.contracts import BlobReference
from rollout.harness.blobs import FileBlobStore
from rollout.local import LocalRunner
from rollout_train import Checkpoint, Fence, Fenced, FileLedger, Policies, Step, Weighted, results, train, trained
from rollout_train.record import GROUPS, STEPS, table
from rollout_train.recorder import Recorder
from rollout_train.rollouts import RolloutJobs
from rollout_train.rollouts.jobs import INTERRUPTED
from rollout_train.testing import plain_channel
from tests.rollout_train.rollouts.games import Words
from tests.rollout_train.training.test_loop import Counting

TOTAL = 4


class Killed(Exception):
    """The process died here."""


class Fuse:
    """Dies at the `at`th point it passes, once."""

    def __init__(self, at: int | None) -> None:
        self.at, self.passed = at, 0

    def point(self) -> None:
        self.passed += 1
        if self.passed == self.at:
            raise Killed(f"at point {self.at}")


class DyingLedger(FileLedger):
    """A ledger whose process may die before a write, or just after it."""

    def __init__(self, directory: Path, fuse: Fuse) -> None:
        super().__init__(directory)
        self.fuse = fuse

    async def append(self, table: str, key: str, record: JsonValue, fence: Fence) -> bool:
        self.fuse.point()
        appended = await super().append(table, key, record, fence)
        self.fuse.point()
        return appended


class DyingBlobs(FileBlobStore):
    def __init__(self, directory: Path, fuse: Fuse) -> None:
        super().__init__(directory)
        self.fuse = fuse

    async def put(self, data: bytes, media_type: str) -> BlobReference:
        self.fuse.point()
        return await super().put(data, media_type)


class DyingTrainer(Counting):
    """A trainer whose step may die before it starts, half way (its files half written), or once it is done."""

    def __init__(self, fuse: Fuse, steps: list[str]) -> None:
        super().__init__()
        self.fuse, self.steps = fuse, steps

    async def step(self, batch: Sequence[Weighted], *, seed: int, parent: Checkpoint | None, into: Path) -> Step:
        self.fuse.point()
        (into / "weights").mkdir(parents=True)
        (into / "weights" / "half").write_text("a step that died left this")
        self.fuse.point()
        (into / "weights" / "half").unlink()
        (into / "weights").rmdir()
        step = await super().step(batch, seed=seed, parent=parent, into=into)
        self.steps.append(into.name)  # (a step that finished: what the run pays for)
        self.fuse.point()
        return step


async def attempt(directory: Path, at: int | None, steps: list[str], *, hard: bool) -> bool:
    """One life of the loop, in a process of its own (a runner, a recorder, jobs, a trainer, all new) over the
    same directory. It dies at point `at`; `hard` is a kill that leaves no trace, and otherwise the job is closed
    as an interrupt would close it. Returns whether it lived to the end."""
    fuse = Fuse(at)
    recorder = Recorder({"policy": plain_channel(always=[("yes\n", "stop"), ("no\n", "stop")])})
    # (The loop's own writes are the points it can die at. The job's are in the same store, and die with it.)
    rollouts = RolloutJobs(
        LocalRunner(recorder=recorder), recorder, log=directory / "log", blobs=FileBlobStore(directory / "blobs")
    )
    policies = Policies(DyingLedger(directory / "ledger", fuse), DyingBlobs(directory / "blobs", fuse))
    done = len(await results(policies.ledger))
    try:
        await train(
            rollouts, Words(), DyingTrainer(fuse, steps), policies, policy="words", channel="policy",
            directory=directory / "versions", groups=TOTAL - done, groups_per_step=2, seed=3,
        )  # fmt: skip
    except Killed:
        if hard:  # nothing the dying process did after this point reaches the disk
            for job in [rollouts.job("train")] if done or fuse.passed else []:
                job._log, job.blobs = None, None  # pyright: ignore[reportPrivateUsage]
        fuse.at = None
        await rollouts.close()
        return False
    head = await policies.head("words")
    channel = recorder.channels["policy"]
    assert head is None or (channel.adapter, channel.version) == (head.name, head.number)  # it serves the newest
    await rollouts.close()
    return True


@pytest.mark.parametrize("hard", [True, False], ids=["killed outright", "interrupted"])
async def test_a_loop_killed_at_any_point_and_started_again_finishes_the_run_once(tmp_path: Path, hard: bool) -> None:
    lives = 0
    for at in range(1, 400):  # die at the first point, then at the second, ... until a life passes every point
        directory = tmp_path / f"run-{at}"
        steps: list[str] = []
        if await attempt(directory, at, steps, hard=hard):
            break  # the loop passed fewer than `at` points: every point has been died at
        lives += 1
        assert await attempt(directory, None, steps, hard=hard), "started again, it runs to the end"
        await check(directory, steps)
    else:
        pytest.fail("the loop never finished")
    assert lives > 15  # (it was killed at that many different points)


async def check(directory: Path, steps: list[str]) -> None:
    policies = Policies(FileLedger(directory / "ledger"), FileBlobStore(directory / "blobs"))
    ledger = policies.ledger
    lines = await results(ledger)
    assert [line.group for line in lines] == list(range(1, TOTAL + 1))  # every group played and recorded, once
    assert sorted(map(int, await ledger.read(table("train", GROUPS)))) == list(range(1, TOTAL + 1))
    # Every group with something to train on is in one step, and every step made one version, each from the one
    # before; no version is an orphan.
    versions = await policies.versions("words")
    covered = await trained(ledger)
    assert sorted(covered) == [line.group for line in lines if line.segments]
    assert all(outcome.version is not None for outcome in covered.values())
    assert sorted({str(outcome.version) for outcome in covered.values()}) == [version.name for version in versions]
    assert [version.parent for version in versions] == [None, *[version.name for version in versions[:-1]]]
    intents = await ledger.read(table("train", STEPS))
    assert sorted(f"words@{intent['number']}" for intent in intents.values()) == [v.name for v in versions]  # type: ignore[index]
    # A step is taken again only if the loop died before its version was written down: never once it was.
    assert set(steps) == {version.name for version in versions}
    assert all(steps.count(version.name) <= 2 for version in versions)
    for version in versions:
        assert version.batch is not None and json.loads(await policies.blobs.read(version.batch))
        fetched = await policies.files(version.weights, directory / "fetched" / version.name)
        assert sorted(path.name for path in fetched.iterdir()) == ["adapter.bin"]  # whole: no half-written file
    # Each group was asked for once, and four of its runs ended (a run a dying job cut short does not count).
    asked = [json.loads(line) for line in (directory / "log" / "train" / "tickets.jsonl").read_text().splitlines()]
    assert sorted(ticket["id"] for ticket in asked) == [f"t_train-{number:04d}" for number in range(1, TOTAL + 1)]
    records = [json.loads(line) for line in (directory / "log" / "train" / "episodes.jsonl").read_text().splitlines()]
    for ticket in asked:
        ended = [r for r in records if r["episode"]["ticket"] == ticket["id"] and r["episode"]["detail"] != INTERRUPTED]
        assert len(ended) == 4, ticket["id"]
    # No episode was trained on in two groups.
    cursors: list[set[str]] = []
    for version in versions:
        assert version.batch is not None
        trained_on = json.loads(await policies.blobs.read(version.batch))
        cursors.append({source.split("/")[0] for source, _ in trained_on})
    assert sum(len(each) for each in cursors) == len(set[str]().union(*cursors))


async def test_a_loop_that_was_replaced_cannot_write(tmp_path: Path) -> None:
    recorder = Recorder({"policy": plain_channel(always=[("yes\n", "stop"), ("no\n", "stop")])})
    rollouts = RolloutJobs(LocalRunner(recorder=recorder), recorder, log=tmp_path / "log")
    assert rollouts.blobs is not None
    policies = Policies(FileLedger(tmp_path / "ledger"), rollouts.blobs)
    stale = await policies.ledger.take("runs/train")
    stale_writer = await policies.writer("words")
    await train(
        rollouts, Words(), Counting(), policies, policy="words", channel="policy", directory=tmp_path / "v", groups=1
    )  # (another loop takes the run and the policy)
    with pytest.raises(Fenced):
        await policies.ledger.append(table("train", GROUPS), "9", {}, stale)
    with pytest.raises(Fenced):
        await policies.add(stale_writer, "words", 9, weights=tmp_path / "ledger")
    await rollouts.close()
