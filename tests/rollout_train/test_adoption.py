"""A claim's lapse and a runner started again, with sandboxes: a pool beside the ledger refuses a key whose claim has
lapsed; an episode runner over a durable runner adopts, when started again, the runs of its claims that held until it
stopped, and they play on in the same sandboxes; a run whose claim lapsed meanwhile is cut short, and its sandboxes
deleted."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from rollout.harness import LeaseRefused, SandboxPool
from rollout.harness.blobs import FileBlobStore
from rollout.testing import FakeSandboxes
from rollout_train.ledger import FileLedger
from rollout_train.presence import FilePresence
from rollout_train.record import table
from rollout_train.recorder import Recorder
from rollout_train.rollouts import EpisodeRunner, episodes_of
from rollout_train.rollouts.scheduler import ADOPTED, CLAIMS, INTERRUPTED, LAPSED, LOST, RESUMED
from rollout_train.sandboxes import admits, leases_of
from rollout_train.testing import plain_channel
from tests.rollout_train.test_sandboxes import BOX, GATES, ask

pytest.importorskip("rollout_durable")
from rollout_durable import DurableRunner


async def test_a_pool_beside_the_ledger_refuses_a_key_whose_claim_has_lapsed(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    beats = FilePresence(ledger.directory)
    sandboxes = FakeSandboxes()
    pool = SandboxPool(sandboxes, leases=leases_of(ledger), admits=admits(ledger, beats))
    await ask(ledger, {1: ({}, 1)})
    fence = await ledger.take("runners/elsewhere")
    await ledger.append(table("train", CLAIMS), "1/1/1", {"runner": "elsewhere", "fence": fence.number}, fence)
    await beats.beat("elsewhere", {})
    lease = await pool.acquire(BOX, "train/1/1/1/box")  # its claim holds
    by_hand = await pool.acquire(BOX, "by-hand/box")  # a key the ledger knows no claim of: admitted
    with pytest.raises(LeaseRefused):  # an attempt nobody claimed
        await pool.acquire(BOX, "train/1/1/2/box")
    again = await ledger.take("runners/elsewhere")  # its runner was started again, and did not adopt it
    with pytest.raises(LeaseRefused):
        await pool.acquire(BOX, lease.key)
    assert lease.handle in sandboxes.deleted and [each.key for each in await pool.held()] == [by_hand.key]
    with pytest.raises(LeaseRefused):  # nor is a sandbox made for it again, even for a moment
        await pool.acquire(BOX, lease.key)
    await ledger.append(table("train", ADOPTED), f"1/1/1/{again.number}", {}, again)  # adopted: it holds again
    assert await admits(ledger, beats)(lease.key)


class Restarts:
    """An episode runner over a durable runner, both started again over the same state; one pool throughout (its
    sandboxes outlive the runners, as a pool's on a machine of its own do)."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.ledger = FileLedger(tmp_path / "ledger")
        self.beats = FilePresence(self.ledger.directory)
        self.sandboxes = FakeSandboxes()
        self.pool = SandboxPool(self.sandboxes, leases=leases_of(self.ledger), admits=admits(self.ledger, self.beats))
        self.blobs = FileBlobStore(tmp_path / "blobs")
        self.durable: Any = None
        self.runner: EpisodeRunner | None = None
        self.serving: asyncio.Task[None] | None = None

    async def start(self, *, fresh_sandboxes: bool = False) -> EpisodeRunner:
        """With `fresh_sandboxes`, the pool starts again too, its sandboxes gone (as a pool's in the runner's process,
        whose sandboxes end with it), its leases still beside the ledger."""
        if fresh_sandboxes:
            await self.pool.close(release=False)
            self.sandboxes = FakeSandboxes()
            self.pool = SandboxPool(
                self.sandboxes, leases=leases_of(self.ledger), admits=admits(self.ledger, self.beats)
            )
        recorder = Recorder({"policy": plain_channel(always=[("yes\n", "stop")])})
        self.durable = DurableRunner(self.tmp_path / "runs", recorder=recorder, pools={"boxes": self.pool})
        self.runner = EpisodeRunner(
            "here", self.ledger, self.durable, recorder, self.blobs, places=4, pools={"boxes": self.pool},
            presence=self.beats, every=0.02,
        )  # fmt: skip
        await self.runner.prepare()  # (before the durable runner recovers its runs, as a profile does)
        await self.durable.launch()
        self.serving = asyncio.create_task(self.runner.serve())
        return self.runner

    async def stop(self) -> None:
        if self.serving is not None:
            self.serving.cancel()
            await asyncio.gather(self.serving, return_exceptions=True)
        if self.durable is not None:
            await self.durable.close()
        self.serving = self.durable = None


async def until(condition: Any, seconds: float = 15.0) -> None:
    async with asyncio.timeout(seconds):
        while not await condition():  # noqa: ASYNC110 (the runners write)
            await asyncio.sleep(0.02)


async def test_a_runner_started_again_adopts_its_durable_runs_and_they_play_on_in_their_sandboxes(
    tmp_path: Path,
) -> None:
    played = Restarts(tmp_path)
    await ask(played.ledger, {1: ({"gate": "adopted"}, 1)})
    try:
        await played.start()

        async def leased() -> bool:
            return len(played.sandboxes.sandboxes) == 1

        await until(leased)
        (handle,) = played.sandboxes.sandboxes
        await played.stop()  # its run is left to be resumed, and keeps its sandbox
        assert await played.ledger.read(table("train", INTERRUPTED)) == {} and handle in played.sandboxes.sandboxes
        await played.start()
        GATES.setdefault("adopted", asyncio.Event()).set()
        (episode,) = await episodes_of(played.ledger, played.blobs, "train", 1, 1, every=0.02)
    finally:
        await played.stop()
    assert sorted(await played.ledger.read(table("train", CLAIMS))) == ["1/1/1"]  # no new attempt
    assert sorted(await played.ledger.read(table("train", ADOPTED))) == ["1/1/1/2"]  # adopted under its new fence
    assert episode.info["key"] == "train/1/1/1/box" and episode.info["handle"] == handle  # the same sandbox
    assert episode.trainable is False and episode.excluded == RESUMED  # (what it sampled before is not recorded)
    assert played.sandboxes.made == [handle] and played.sandboxes.deleted == [handle]  # released when it ended


async def test_a_run_whose_claim_lapsed_while_its_runner_was_stopped_is_cut_short(tmp_path: Path) -> None:
    played = Restarts(tmp_path)
    await ask(played.ledger, {1: ({"gate": "lapsed"}, 1)})
    try:
        await played.start()

        async def leased() -> bool:
            return len(played.sandboxes.sandboxes) == 1

        await until(leased)
        (handle,) = played.sandboxes.sandboxes
        claims = await played.ledger.read(table("train", CLAIMS))
        run_id = str(claims["1/1/1"]["run_id"])  # type: ignore[index]
        await played.stop()
        fence = await played.ledger.take("runners/elsewhere")  # meanwhile another runner took the episode up
        claim: JsonValue = {"runner": "elsewhere", "fence": fence.number}
        await played.ledger.append(table("train", CLAIMS), "1/1/2", claim, fence)
        await played.beats.beat("elsewhere", {})
        await played.start()

        async def deleted() -> bool:
            return handle in played.sandboxes.deleted and played.durable.run(run_id).done

        await until(deleted)  # the recovered run was refused its sandbox, and ended
    finally:
        await played.stop()
    assert (await played.ledger.read(table("train", INTERRUPTED)))["1/1/1"]["why"] == LAPSED  # type: ignore[index]
    assert sorted(await played.ledger.read(table("train", CLAIMS))) == ["1/1/1", "1/1/2"]  # (its own claimed nothing)
    assert await played.ledger.read(table("train", ADOPTED)) == {} and played.sandboxes.sandboxes == {}


async def test_an_adopted_run_whose_sandboxes_did_not_outlive_its_runner_is_played_again(tmp_path: Path) -> None:
    played = Restarts(tmp_path)
    await ask(played.ledger, {1: ({"gate": "lost"}, 1)})
    try:
        await played.start()

        async def leased() -> bool:
            return len(played.sandboxes.sandboxes) == 1

        await until(leased)
        (handle,) = played.sandboxes.sandboxes
        await played.stop()
        GATES.setdefault("lost", asyncio.Event()).set()
        await played.start(fresh_sandboxes=True)  # its world ended with the process
        (episode,) = await episodes_of(played.ledger, played.blobs, "train", 1, 1, every=0.02)
    finally:
        await played.stop()
    assert (await played.ledger.read(table("train", INTERRUPTED)))["1/1/1"]["why"] == LOST  # type: ignore[index]
    assert episode.info["key"] == "train/1/1/2/box" and episode.info["handle"] != handle  # a new attempt, a new one
    assert episode.trainable and await played.pool.held() == []
