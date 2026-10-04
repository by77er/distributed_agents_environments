"""How a run ended: each process says how its own start of a run ended, under the fence it started with."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from rollout_train.ledger import FileLedger
from rollout_train.record import ENDS, FAILED, FINISHED, PROCESS, STARTS, STOPPED, ending, scope, table


async def started(ledger: FileLedger, run: str, process: str = PROCESS) -> int:
    fence = await ledger.take(scope(run))
    record: JsonValue = {"process": process, "started": 1.0}
    await ledger.append(table(run, STARTS), str(fence.number), record, fence)
    return fence.number


async def how(ledger: FileLedger, run: str) -> dict[str, Any]:
    ends: Any = await ledger.read(table(run, ENDS))
    return {key: value["how"] for key, value in ends.items()}


async def test_a_run_says_it_finished_was_stopped_or_failed(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    number = await started(ledger, "done")
    async with ending(ledger, "done"):
        pass
    assert await how(ledger, "done") == {str(number): FINISHED}

    await started(ledger, "broken")
    with pytest.raises(RuntimeError):
        async with ending(ledger, "broken"):
            raise RuntimeError("the trainer ran out of memory")
    (said,) = (await ledger.read(table("broken", ENDS))).values()
    assert said["how"] == FAILED and "ran out of memory" in said["detail"]  # type: ignore[index]

    await started(ledger, "interrupted")

    async def work() -> None:
        async with ending(ledger, "interrupted"):
            await asyncio.sleep(60)

    task = asyncio.create_task(work())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert list((await how(ledger, "interrupted")).values()) == [STOPPED]


async def test_a_process_another_has_replaced_says_nothing_of_the_newer_start(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    await started(ledger, "run")
    newer = await started(ledger, "run", process="elsewhere/1/abcd")  # (another process took the run)
    async with ending(ledger, "run"):
        pass
    assert await how(ledger, "run") == {}  # (its own fence is no longer the run's: the append was refused)
    assert newer == 2


async def test_the_page_says_a_run_finished_or_was_lost(tmp_path: Path) -> None:
    from rollout_train.monitor.system import _how_it_ended  # pyright: ignore[reportPrivateUsage]

    starts = {"1": {"process": "a"}, "2": {"process": "b"}}
    assert _how_it_ended("ended", starts, {"2": {"how": FINISHED, "at": 5.0}}, beat=True)["state"] == FINISHED
    assert _how_it_ended("ended", starts, {"1": {"how": FAILED, "at": 5.0}}, beat=True) == {"state": "lost"}
    assert _how_it_ended("running", starts, {}, beat=True) == {}  # (still beating: as its beats say)
    assert _how_it_ended("ended", starts, {}, beat=False) == {}  # (from before runs said how they ended)
