"""An episode's fence: the runner whose claim of an attempt was appended takes it, and the episode is recorded under
it, so a newer attempt shuts out the ones before; a runner started again takes it anew to adopt; a pool's keeper takes
it to end a lapsed claim, after reading the claim again. Claims written before episodes had fences hold as before."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from rollout.harness import LeaseRefused, SandboxPool
from rollout.testing import FakeSandboxes, until
from rollout_train.ledger import Fence, Fenced, FileLedger
from rollout_train.loop import newest
from rollout_train.presence import FilePresence
from rollout_train.record import table
from rollout_train.rollouts import episodes_of, playing
from rollout_train.rollouts.scheduler import (
    ADOPTED,
    CLAIMS,
    EPISODES,
    INTERRUPTED,
    RELEASED,
    SUPERSEDED,
    EpisodeRunner,
    episode_scope,
    holding,
    of_episode,
    runner_scope,
)
from rollout_train.sandboxes import admits, ending, keep, leases_of, pool_scope
from tests.rollout_train.support import BOX, BOX_GATES, ask_boxed, episode_runner


async def test_each_episode_is_recorded_under_a_fence_of_its_own_taken_when_it_was_claimed(tmp_path: Path) -> None:
    played = episode_runner(tmp_path, SandboxPool(FakeSandboxes()))
    await ask_boxed(played.ledger, {1: ({}, 2)})
    async with playing(played):
        await episodes_of(played.ledger, played.blobs, "train", 1, 2, every=0.01)
    fences = await played.ledger.fences()
    assert fences[episode_scope("train", "1/1")] == fences[episode_scope("train", "1/2")] == 1
    assert of_episode(episode_scope("train", "1/1")) and not of_episode(runner_scope("here"))
    lines = (tmp_path / "ledger" / "runs" / "train" / "episodes.jsonl").read_text().splitlines()
    assert sorted(json.loads(line)["fence"] for line in lines) == [1, 1]  # (each under its episode's fence)


async def test_an_attempt_whose_episode_was_claimed_again_meanwhile_is_not_recorded(tmp_path: Path) -> None:
    played = episode_runner(tmp_path, SandboxPool(FakeSandboxes()))
    ledger = played.ledger
    await ask_boxed(ledger, {1: ({"gate": "superseded"}, 1)})
    async with playing(played):

        async def claimed() -> bool:
            return "1/1/1" in await ledger.read(table("train", CLAIMS))

        await until(claimed)
        other = await ledger.take(runner_scope("elsewhere"))  # another runner claims the episode's next attempt
        claim: JsonValue = {"runner": "elsewhere", "fence": other.number}
        await ledger.append(table("train", CLAIMS), "1/1/2", claim, other)
        await ledger.take(episode_scope("train", "1/1"))
        BOX_GATES.setdefault("superseded", asyncio.Event()).set()

        async def cut() -> bool:
            return "1/1/1" in await ledger.read(table("train", INTERRUPTED))

        await until(cut)
    assert await ledger.read(table("train", EPISODES)) == {}  # its record was refused
    assert (await ledger.read(table("train", INTERRUPTED)))["1/1/1"]["why"] == SUPERSEDED  # type: ignore[index]


async def test_claims_written_before_episodes_had_fences_hold_as_their_latest_attempt(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    beats = FilePresence(ledger.directory)
    await ask_boxed(ledger, {1: ({}, 2)})
    first, second = await ledger.take(runner_scope("first")), await ledger.take(runner_scope("second"))
    for key, runner, fence in [("1/1/1", "first", first), ("1/2/1", "first", first), ("1/2/2", "second", second)]:
        await ledger.append(table("train", CLAIMS), key, {"runner": runner, "fence": fence.number}, fence)
    await beats.beat("first", {})
    await beats.beat("second", {})
    found = {beat.runner: beat for beat in await beats.beats()}
    assert await holding(ledger, "train", await ledger.fences(), found) == {"1/1/1", "1/2/2"}
    record: JsonValue = {"runner": "first", "fence": first.number}  # (an episode recorded as before)
    await ledger.append(table("train", EPISODES), "1/1", record, first)
    assert await holding(ledger, "train", await ledger.fences(), found) == {"1/2/2"}


class Going:
    """A durable runner's runs as `EpisodeRunner._adopt` finds them: every run it is asked for is going."""

    resumes = True

    def run(self, run_id: str) -> Any:
        class Handle:
            def __init__(self) -> None:
                self.run_id, self.done = run_id, False

        return Handle()


class Forgetting:
    def admit(self, run_id: str, attempt: Any) -> None:
        pass

    async def reaches(self, run: str, binding: Any) -> bool:
        return True

    async def sessions(self, run: str, run_id: str) -> dict[str, list[Any]]:
        return {}

    def forget(self, run_id: str) -> None:
        pass


async def test_a_runner_started_again_takes_each_adopted_episodes_fence_anew(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    await ask_boxed(ledger, {1: ({}, 2)})
    before = await ledger.take(runner_scope("here"))
    for key in ("1/1/1", "1/2/1"):  # claims from before episodes had fences, and one from after
        claim: JsonValue = {"runner": "here", "fence": before.number, "run_id": f"run-{key}", "at": time.time()}
        await ledger.append(table("train", CLAIMS), key, claim, before)
    taken = await ledger.take(episode_scope("train", "1/2"))
    runner: Any = Going()
    again = EpisodeRunner("here", ledger, runner, Forgetting(), None, places=2)  # type: ignore[arg-type]
    await again.prepare()
    fences = await ledger.fences()
    assert fences[episode_scope("train", "1/1")] == 1 and fences[episode_scope("train", "1/2")] == taken.number + 1
    assert sorted(await ledger.read(table("train", ADOPTED))) == ["1/1/1/2", "1/2/1/2"]
    with pytest.raises(Fenced):  # what its runner before would still record of it is refused
        await ledger.append(table("train", EPISODES), "1/2", {}, taken)


async def test_the_keeper_reads_a_lapsed_claim_again_and_ends_it_in_the_ledger_before_releasing(
    tmp_path: Path,
) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    beats = FilePresence(ledger.directory)
    await ask_boxed(ledger, {1: ({}, 1)})
    before = await ledger.take(runner_scope("here"))
    await ledger.append(table("train", CLAIMS), "1/1/1", {"runner": "here", "fence": before.number}, before)
    episode = await ledger.take(episode_scope("train", "1/1"))  # (as its runner takes it once its claim is in)
    await beats.beat("here", {})
    again = await ledger.take(runner_scope("here"))  # started again: the claim lapses until it is adopted
    assert await holding(ledger, "train", await ledger.fences(), None) == set()
    key = "train/1/1/1/box"
    adopted = await ledger.take(episode_scope("train", "1/1"))  # it adopts the claim, after the keeper's two looks
    await ledger.append(table("train", ADOPTED), f"1/1/1/{again.number}", {}, adopted)
    assert await ending([key], ledger, beats) == set()  # read again: it holds, and keeps its lease
    third = await ledger.take(runner_scope("here"))  # started again once more, and not adopted this time
    assert await ending([key], ledger, beats) == {key}
    assert (await ledger.read(table("train", INTERRUPTED)))["1/1/1"]["why"] == RELEASED  # type: ignore[index]
    for stale in (episode, adopted):  # nothing its runners would still write of the attempt is taken
        with pytest.raises(Fenced):
            await ledger.append(table("train", EPISODES), "1/1", {}, stale)
    with pytest.raises(Fenced):
        await ledger.append(table("train", ADOPTED), f"1/1/1/{third.number}", {}, adopted)


async def test_the_keeper_releases_nothing_whose_episode_another_fenced_meanwhile(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    await ask_boxed(ledger, {1: ({}, 1)})
    before = await ledger.take(runner_scope("here"))
    await ledger.append(table("train", CLAIMS), "1/1/1", {"runner": "here", "fence": before.number}, before)
    await ledger.take(runner_scope("here"))  # (lapsed: its runner started again)

    class Racing(FileLedger):
        async def take(self, scope: str) -> Fence:
            taken = await super().take(scope)
            if of_episode(scope):
                await super().take(scope)  # its runner, adopting it, takes the episode's fence just after
            return taken

    assert await ending(["train/1/1/1/box"], Racing(ledger.directory), None) == set()
    assert await ledger.read(table("train", INTERRUPTED)) == {}


async def test_a_keeper_stops_sweeping_once_another_process_keeps_its_pool(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    pool = SandboxPool(FakeSandboxes(), name="boxes@here/run", leases=leases_of(ledger))
    keeping = asyncio.create_task(keep(pool, ledger, None, every=0.01))
    try:

        async def taken() -> bool:
            return pool_scope(pool.name) in await ledger.fences()

        await until(taken)
        await asyncio.sleep(0.05)
        assert not keeping.done()
        await ledger.take(pool_scope(pool.name))  # the same run's directory, started again on this machine
        async with asyncio.timeout(5.0):
            await keeping  # (it returns: it no longer sweeps)
    finally:
        keeping.cancel()
        await asyncio.gather(keeping, return_exceptions=True)


async def test_a_loop_checks_its_fence_before_a_side_effect_outside_the_ledger(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    fence = await ledger.take("runs/train")
    await newest(ledger, fence)
    await ledger.take("runs/train")  # a loop that replaced it
    with pytest.raises(Fenced):
        await newest(ledger, fence)


async def test_a_lease_acquired_under_a_lapsed_attempt_is_refused_beside_a_newer_one(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    beats = FilePresence(ledger.directory)

    pool = SandboxPool(FakeSandboxes(), leases=leases_of(ledger), admits=admits(ledger, beats))
    await ask_boxed(ledger, {1: ({}, 1)})
    old, new = await ledger.take(runner_scope("old")), await ledger.take(runner_scope("new"))
    await ledger.append(table("train", CLAIMS), "1/1/1", {"runner": "old", "fence": old.number}, old)
    await ledger.append(table("train", CLAIMS), "1/1/2", {"runner": "new", "fence": new.number}, new)
    await beats.beat("old", {})
    await beats.beat("new", {})
    assert await admits(ledger, beats)("train/1/1/2/box") and not await admits(ledger, beats)("train/1/1/1/box")

    with pytest.raises(LeaseRefused):
        await pool.acquire(BOX, "train/1/1/1/box")
