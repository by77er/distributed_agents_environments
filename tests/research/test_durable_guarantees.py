"""The durable runner's event log under a replay (docs/research/ledger-guarantees.md).

A test marked `xfail(strict=True)` shows a guarantee that does not hold; its reason says what goes wrong.
"""

import asyncio
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from rollout.contracts import TERMINAL_EVENT_TYPES
from rollout.harness import (
    DirectModel,
    ModelBinding,
    PoolBinding,
    Program,
    ProgramReference,
    RunBinding,
    RunContext,
    RunSpecification,
    SandboxPool,
    SandboxSpec,
    register,
)
from rollout.testing import FakeSandboxes, ScriptedModelEndpoint

pytest.importorskip("rollout_durable")
from rollout_durable import DurableRunner


class Described(Program):
    """Asks its box who it is, and ends."""

    def sandboxes(self) -> Mapping[str, SandboxSpec]:
        return {"box": SandboxSpec(kind="fake")}

    async def main(self, run: RunContext) -> None:
        await run.emit("said", (await run.sandbox("box").call("describe")).structured)


class SlowToRelease(SandboxPool):
    """A pool whose first release waits until `released` is set: a run that ended is still releasing its sandbox."""

    def __init__(self, sandboxes: FakeSandboxes) -> None:
        super().__init__(sandboxes)
        self.released = asyncio.Event()
        self.releasing = 0

    async def release(self, key: str) -> None:
        self.releasing += 1
        if self.releasing == 1:
            await self.released.wait()
        await super().release(key)


async def until(condition: Callable[[], bool], seconds: float = 15.0) -> None:
    async with asyncio.timeout(seconds):
        while not condition():  # noqa: ASYNC110 (the runner writes)
            await asyncio.sleep(0.05)


def durable(directory: Path, pool: SandboxPool) -> DurableRunner:
    return DurableRunner(
        directory, providers={"scripted": lambda model: ScriptedModelEndpoint([])}, pools={"boxes": pool}
    )


@pytest.mark.xfail(
    strict=True,
    reason="since 42e324d a terminal event recorded while any event is stored takes MAX(seq) + 1: a run replayed "
    "after its terminal event was stored and before DBOS recorded its end (its runner closed or died while it "
    "released its sandboxes) appends a second terminal event",
)
async def test_a_run_replayed_after_it_ended_has_one_terminal_event(tmp_path: Path) -> None:
    pool = SlowToRelease(FakeSandboxes())
    runner = durable(tmp_path / "state", pool)
    await runner.launch()
    binding = RunBinding(
        models={"policy": ModelBinding(direct=DirectModel(provider="scripted", model="s"))},
        pools={"fake": PoolBinding(local="boxes")},
    )
    specification = RunSpecification(program=ProgramReference(program=register(Described)), binding=binding)
    try:
        handle = await runner.start(specification, lease="train/1/1/1")

        def ended() -> bool:
            return any(event.type in TERMINAL_EVENT_TYPES for event in runner.store.events(handle.run_id))

        await until(ended)  # its terminal event is stored; it is releasing its sandbox
        await runner.close()  # its runner stops (or dies) before the run's end is recorded
        pool.released.set()
        runner = durable(tmp_path / "state", pool)
        await runner.launch()  # the run is recovered, and replayed
        await runner.run(handle.run_id).result()
    finally:
        await runner.close()
    terminal = [event for event in runner.store.events(handle.run_id) if event.type in TERMINAL_EVENT_TYPES]
    assert len(terminal) == 1, [(event.seq, event.type.value) for event in terminal]
