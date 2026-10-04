"""A claim's lapse, with sandboxes: a pool beside the ledger refuses a key whose claim has lapsed, and admits it again
once the claim is adopted."""

from pathlib import Path

import pytest

from rollout.harness import LeaseRefused, SandboxPool
from rollout.testing import FakeSandboxes
from rollout_train.ledger import FileLedger
from rollout_train.presence import FilePresence
from rollout_train.record import table
from rollout_train.rollouts.scheduler import ADOPTED, CLAIMS
from rollout_train.sandboxes import admits, leases_of
from tests.rollout_train.support import BOX, ask_boxed


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
