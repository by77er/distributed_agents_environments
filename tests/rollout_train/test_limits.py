"""A run's limits: it ends once it has run `limits.hours`, or spent `limits.spend` (its pods' hours among it), stopped
with the reason; and its RayJob is stopped by Kubernetes half an hour after its hours in any case."""

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rollout_train.database import DatabaseLedger
from rollout_train.jobs import HoursReached, SpendReached, _within_limits  # pyright: ignore[reportPrivateUsage]
from rollout_train.record import ENDS, STARTS, ending, newest_record, scope, start_header, table
from rollout_train.run_settings import RunSettings


def live(ledger: Any, **settings: Any) -> Any:
    """What `_within_limits` reads of a run."""
    said = RunSettings({"kind": "train", **settings})
    return SimpleNamespace(settings=said, gateway=None, pods=None, began=time.time(), runs={"run_1"}, ledger=ledger,
                           run=SimpleNamespace(id="run_1"))  # fmt: skip


async def forever() -> None:
    await asyncio.sleep(3600)


async def test_a_run_ends_stopped_once_it_has_run_its_hours(tmp_path: Path) -> None:
    ledger = DatabaseLedger(f"sqlite:///{tmp_path / 'ledger.db'}")
    fence = await ledger.take(scope("run_1"))
    await ledger.append(table("run_1", STARTS), str(fence.number), start_header(kind="train"), fence)
    started = time.monotonic()
    with pytest.raises(HoursReached, match=r"it ran 0\.0001 hours, which reaches limits\.hours"):
        async with ending(ledger, "run_1"):
            await _within_limits(live(ledger, **{"limits.hours": 0.0001}), forever())
    assert time.monotonic() - started < 5
    ended = newest_record(await ledger.read(table("run_1", ENDS)))
    assert ended["how"] == "stopped" and ended["detail"].startswith("HoursReached: it ran")


async def test_a_run_whose_pods_cost_its_limit_ends(tmp_path: Path) -> None:
    run = live(None, **{"limits.spend": 1.0})
    run.pods = SimpleNamespace(spent=1.25)  # (its renewals counted its pods' hours at their price)
    with pytest.raises(SpendReached, match=r"it spent \$1.25, which reaches limits.spend \$1"):
        await _within_limits(run, forever())


async def test_a_run_within_its_limits_does_its_work() -> None:
    async def work() -> str:
        return "done"

    assert await _within_limits(live(None, **{"limits.hours": 1.0, "limits.spend": 5.0}), work()) == "done"
    assert await _within_limits(live(None), work()) == "done"


def test_a_runs_rayjob_is_stopped_half_an_hour_after_its_hours() -> None:
    from rollout_train.launches import TRAIN, Asked, new_launch
    from rollout_train.submitting import rendered

    template: dict[str, Any] = {"apiVersion": "ray.io/v1", "kind": "RayJob", "metadata": {}, "spec": {}}
    bounded = new_launch(Asked(TRAIN, "gridworld", {"environment": "e:env", "limits.hours": 5}), "run_1")
    made = rendered(template, bounded, "python -m rollout_train.jobs L", {}, "rollout")
    assert made["spec"]["activeDeadlineSeconds"] == 5 * 3600 + 1800
    unbounded = new_launch(Asked(TRAIN, "gridworld", {"environment": "e:env"}), "run_2")
    assert (
        "activeDeadlineSeconds" not in rendered(template, unbounded, "python -m rollout_train.jobs L", {}, "x")["spec"]
    )
