"""A claim's lapse, with sandboxes: a pool beside the ledger refuses a key whose claim has lapsed, and admits it again
once the claim is adopted. An adopted run that an error outside it cuts short is noted, and its episode played again."""

import time
from collections.abc import AsyncIterator, Awaitable, Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from rollout.harness import LeaseRefused, SandboxPool
from rollout.testing import FakeSandboxes, until
from rollout_train.ledger import FileLedger
from rollout_train.presence import FilePresence
from rollout_train.record import table
from rollout_train.rollouts import EpisodeRunner, playing
from rollout_train.rollouts.scheduler import ADOPTED, CLAIMS, FAULT, INTERRUPTED, runner_scope
from rollout_train.sandboxes import admits, leases_of
from tests.rollout_train.support import BOX, ask, ask_boxed


async def test_a_pool_beside_the_ledger_refuses_a_key_whose_claim_has_lapsed(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    beats = FilePresence(ledger.directory)
    sandboxes = FakeSandboxes()
    pool = SandboxPool(sandboxes, leases=leases_of(ledger), admits=admits(ledger, beats))
    await ask_boxed(ledger, {1: ({}, 1)})
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


class Unreadable:
    """A runner whose runs survive it, as `EpisodeRunner._adopt` asks of one: every run it is asked for is going, and
    what each did cannot be read now (its store does not answer, say)."""

    resumes = True

    def __init__(self) -> None:
        self.cancelled: list[str] = []

    def run(self, run_id: str) -> Any:
        class Handle:
            def __init__(self) -> None:
                self.run_id, self.done = run_id, False

            async def events(self) -> AsyncIterator[Any]:
                raise ConnectionError("the runner's store could not be reached")
                yield  # (an iterator of its events)

        return Handle()

    async def cancel(self, run_id: str, *, reason: str) -> None:
        self.cancelled.append(run_id)


class Recording:
    def admit(self, run_id: str, attempt: Any) -> None:
        pass

    async def reaches(self, run: str, binding: Any) -> bool:
        return True

    async def sessions(self, run: str, run_id: str) -> dict[str, list[Any]]:
        return {}

    def forget(self, run_id: str) -> None:
        pass


async def test_an_adopted_run_an_error_outside_it_cut_short_is_noted_and_its_episode_is_open_again(
    tmp_path: Path,
) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    await ask(ledger, "train", {1: ({"word": "yes"}, 1)})
    before = await ledger.take(runner_scope("here"))  # the runner before it stopped: its claim, its run
    claim: JsonValue = {"runner": "here", "fence": before.number, "run_id": "run-1", "at": time.time()}
    await ledger.append(table("train", CLAIMS), "1/1/1", claim, before)

    def full() -> None:  # (it claims nothing: it only plays what it adopts)
        raise MemoryError("the machine has no memory left for more runs")

    runner = Unreadable()
    again = EpisodeRunner("here", ledger, runner, Recording(), None, places=1, guard=full, every=0.01)  # type: ignore[arg-type]
    await again.prepare()
    assert sorted(await ledger.read(table("train", ADOPTED))) == [f"1/1/1/{before.number + 1}"]
    assert await again.open() == []  # its own claim holds: it plays it once it serves
    async with playing(again):
        await until(lambda: _counted(ledger.read(table("train", INTERRUPTED)), 1))
    (cut,) = (await ledger.read(table("train", INTERRUPTED))).values()
    assert cut["why"] == FAULT and runner.cancelled == ["run-1"]  # type: ignore[index]
    (found,) = await again.open()
    assert (found.group, found.number, found.attempt) == (1, 1, 2)


async def _counted(read: Awaitable[Mapping[str, Any]], at_least: int) -> bool:
    return len(await read) >= at_least
