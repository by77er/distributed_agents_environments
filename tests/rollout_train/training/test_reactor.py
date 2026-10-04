"""The training loop is killed at every point where it writes or calls the trainer, and started again each time,
with the runner that plays its episodes in the same process: the run ends with every group played and recorded once,
every group with something to train on in one step, every step taken once, and every episode recorded once (one a
runner was playing when it died is claimed again, and played again)."""

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import JsonValue

from rollout.contracts import BlobReference
from rollout.harness.blobs import FileBlobStore
from rollout_train import Checkpoints, Fence, Fenced, FileLedger, Files, Step, Weighted, results, train, trained
from rollout_train.checkpoints import new_id
from rollout_train.record import GROUPS, STEPS, table
from rollout_train.rollouts.scheduler import CLAIMS, EPISODES, INTERRUPTED, runner_scope
from rollout_train.testing import Policy, ScriptedEngine, plain_channel
from tests.rollout_train.rollouts.games import Words
from tests.rollout_train.training.test_loop import Counting, answering, here, made_by, quickly

__all__ = ["quickly"]  # (the loop looks for its episodes often here too)

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


class Gone(Exception):
    """The process this was in is dead: nothing it does reaches the disk."""


class DeadAfterwards(FileLedger):
    """The runner's view of the ledger: once its process is dead, it writes nothing more."""

    dead = False

    async def append(self, table: str, key: str, record: JsonValue, fence: Fence) -> bool:
        if self.dead:
            raise Gone(table)
        return await super().append(table, key, record, fence)


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

    async def step(self, batch: Sequence[Weighted], *, seed: int, parent: Files | None, into: Path) -> Step:
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
    """One life of the loop, in a process of its own (a runner, a recorder, a trainer, all new) over the same
    directory. It dies at point `at`; `hard` is a kill that leaves no trace (the runner writes nothing more, not
    even that it cut its episodes short), and otherwise the runner closes as an interrupt would close it. Returns
    whether it lived to the end."""
    fuse = Fuse(at)
    recorder = answering()
    # (The loop's own writes are the points it can die at. The runner's are in the same stores, and die with it.)
    runners = DeadAfterwards(directory / "ledger")
    checkpoints = Checkpoints(DyingLedger(directory / "ledger", fuse), DyingBlobs(directory / "blobs", fuse))
    done = len(await results(checkpoints.ledger))
    try:
        async with here(runners, recorder, FileBlobStore(directory / "blobs")):
            try:
                await train(
                    Words(), DyingTrainer(fuse, steps), checkpoints, base="words-base", channel="policy",
                    directory=directory / "checkpoints", publish=recorder.publish, groups=TOTAL - done,
                    groups_per_step=2, seed=3,
                )  # fmt: skip
            except Killed:
                runners.dead = hard  # (before the runner closes: nothing the dying process did reaches the disk)
                raise
    except Killed:
        return False
    head = await checkpoints.head("train")
    channel = recorder.channels["policy"]
    assert head is None or (channel.adapter, channel.version) == (head.id, head.depth)  # it serves the newest
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


async def test_episodes_a_runner_was_playing_when_it_died_are_claimed_again(tmp_path: Path) -> None:
    channel = plain_channel(always=[("yes\n", "stop"), ("no\n", "stop")])
    engine = cast(ScriptedEngine, channel.engines[0])
    answer = engine.generate

    async def slowly(*arguments: Any, **options: Any) -> Any:  # (long enough to die while episodes play)
        await asyncio.sleep(0.2)
        return await answer(*arguments, **options)

    engine.generate = slowly
    recorder = Policy(channel)
    runners = DeadAfterwards(tmp_path / "ledger")
    checkpoints = Checkpoints(FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs"))
    steps: list[str] = []
    async with here(runners, recorder, checkpoints.blobs):
        playing = train(
            Words(), DyingTrainer(Fuse(None), steps), checkpoints, base="words-base", channel="policy",
            directory=tmp_path / "checkpoints", publish=recorder.publish, groups=TOTAL, groups_per_step=2, seed=3,
        )  # fmt: skip
        loop = asyncio.create_task(playing)
        while not await checkpoints.ledger.read(table("train", CLAIMS)):  # noqa: ASYNC110 (the runner writes)
            await asyncio.sleep(0.01)
        runners.dead = True  # killed outright, with episodes playing: nothing more of it reaches the disk
        loop.cancel()
        await asyncio.gather(loop, return_exceptions=True)
    assert not await checkpoints.ledger.read(table("train", EPISODES))
    assert await attempt(tmp_path, None, steps, hard=True), "started again, it runs to the end"
    assert await check(tmp_path, steps)  # what it was playing, its runner started again claimed anew


async def check(directory: Path, steps: list[str]) -> int:
    """Checks a run that ended; returns how many of its episodes were claimed again (their first claim's runner
    died)."""
    store = Checkpoints(FileLedger(directory / "ledger"), FileBlobStore(directory / "blobs"))
    ledger = store.ledger
    lines = await results(ledger)
    assert [line.group for line in lines] == list(range(1, TOTAL + 1))  # every group played and recorded, once
    assert sorted(map(int, await ledger.read(table("train", GROUPS)))) == list(range(1, TOTAL + 1))
    # Every group with something to train on is in one step, and every step made one checkpoint, each from the one
    # before; no checkpoint is an orphan.
    checkpoints = await made_by(store)
    ids = sorted(checkpoint.id for checkpoint in checkpoints)
    covered = await trained(ledger)
    assert sorted(covered) == [line.group for line in lines if line.segments]
    assert all(outcome.checkpoint is not None for outcome in covered.values())
    assert sorted({str(outcome.checkpoint) for outcome in covered.values()}) == ids
    assert [checkpoint.parents for checkpoint in checkpoints] == [
        (),
        *[(checkpoint.id,) for checkpoint in checkpoints[:-1]],
    ]
    assert [checkpoint.depth for checkpoint in checkpoints] == list(range(1, len(checkpoints) + 1))
    intents = await ledger.read(table("train", STEPS))
    assert sorted(str(cast(dict[str, Any], intent)["makes"]) for intent in intents.values()) == ids
    # A step is taken again only if the loop died before its checkpoint was written down: never once it was.
    assert set(steps) == set(ids)
    assert all(steps.count(checkpoint.id) <= 2 for checkpoint in checkpoints)
    for checkpoint in checkpoints:
        assert checkpoint.batch is not None and json.loads(await store.blobs.read(checkpoint.batch))
        assert checkpoint.weights is not None
        fetched = await store.files(checkpoint.weights, directory / "fetched" / checkpoint.id)
        assert sorted(path.name for path in fetched.iterdir()) == ["adapter.bin"]  # whole: no half-written file
    # Each group's four episodes were recorded, once each. An episode was claimed again only when the claim
    # before was cut short (its runner closed) or its runner died (a runner started again holds a new fence).
    ended = await ledger.read(table("train", EPISODES))
    assert sorted(ended) == sorted(f"{group}/{number}" for group in range(1, TOTAL + 1) for number in range(1, 5))
    claims: dict[str, dict[int, dict[str, Any]]] = {}
    for key, claim in (await ledger.read(table("train", CLAIMS))).items():
        episode, attempt = key.rsplit("/", 1)
        claims.setdefault(episode, {})[int(attempt)] = cast(dict[str, Any], claim)
    cut = await ledger.read(table("train", INTERRUPTED))
    fence = (await ledger.fences())[runner_scope("here")]
    again = 0
    for episode, made in claims.items():
        assert sorted(made) == list(range(1, len(made) + 1)), episode
        for attempt in sorted(made)[:-1]:
            assert f"{episode}/{attempt}" in cut or made[attempt]["fence"] < fence, (episode, attempt)
            again += f"{episode}/{attempt}" not in cut
    # No episode was trained on in two groups.
    trained_on: list[set[str]] = []
    for checkpoint in checkpoints:
        assert checkpoint.batch is not None
        sources = json.loads(await store.blobs.read(checkpoint.batch))
        trained_on.append({"/".join(source.split("/")[:3]) for source, _ in sources})
    assert sum(len(each) for each in trained_on) == len(set[str]().union(*trained_on))
    return again


async def test_a_loop_that_was_replaced_cannot_write(tmp_path: Path) -> None:
    recorder = answering()
    checkpoints = Checkpoints(FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs"))
    stale = await checkpoints.ledger.take("runs/train")
    async with here(checkpoints.ledger, recorder, checkpoints.blobs):
        await train(
            Words(),
            Counting(),
            checkpoints,
            channel="policy",
            directory=tmp_path / "v",
            publish=recorder.publish,
            groups=1,
        )  # (another loop takes the run)
    with pytest.raises(Fenced):
        await checkpoints.ledger.append(table("train", GROUPS), "9", {}, stale)
    with pytest.raises(Fenced):  # its checkpoints are the run's, appended under its fence
        weights = tmp_path / "stale-weights"
        weights.mkdir()
        (weights / "adapter.bin").write_bytes(b"stale")
        await checkpoints.add(stale, new_id(), weights=weights, run="train")
