"""A durable run acquires its sandboxes each time it is executed, under the same lease: resumed, it gets the same
sandboxes back; unloaded, it keeps them; ended, it releases them."""

import asyncio
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path

import pytest

from rollout.contracts import RunEventType, Text
from rollout.harness import (
    Address,
    DirectModel,
    Envelope,
    Lease,
    ModelBinding,
    PoolBinding,
    Program,
    ProgramReference,
    RunBinding,
    RunContext,
    RunSpecification,
    RunStatus,
    SandboxPool,
    SandboxSpec,
    WaitFor,
    register,
)
from rollout.testing import FakeSandboxes, ScriptedModelEndpoint, payload
from rollout_durable import DurableRunner


class Resumes(Program):
    """Asks its box who it is, waits for a message, and asks again."""

    def sandboxes(self) -> Mapping[str, SandboxSpec]:
        return {"box": SandboxSpec(kind="fake")}

    async def main(self, run: RunContext) -> None:
        before = await run.sandbox("box").call("describe")
        await run.emit("before", before.structured)
        await run.wait_for_message(WaitFor("go"))
        after = await run.sandbox("box").call("describe")
        await run.emit("after", after.structured)


class Counted(SandboxPool):
    """A pool that keeps every acquire asked of it."""

    def __init__(self, sandboxes: FakeSandboxes) -> None:
        super().__init__(sandboxes)
        self.asked: list[str] = []

    async def acquire(self, spec: SandboxSpec, key: str, environment: Mapping[str, str] | None = None) -> Lease:
        self.asked.append(key)
        return await super().acquire(spec, key, environment)


async def until(condition, seconds: float = 15) -> None:  # type: ignore[no-untyped-def]
    for _ in range(int(seconds * 20)):
        if condition():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("condition not reached")


def durable(directory: Path, pool: SandboxPool) -> DurableRunner:
    return DurableRunner(
        directory,
        providers={"scripted": lambda model: ScriptedModelEndpoint([])},
        pools={"boxes": pool},
        evict_after=timedelta(seconds=0.3),
        eviction_interval=0.1,
    )


@pytest.mark.parametrize("resumed", ["woken", "restarted"])
async def test_a_resumed_durable_run_gets_its_sandbox_back(resumed: str, tmp_path: Path) -> None:
    sandboxes = FakeSandboxes()
    pool = Counted(sandboxes)
    runner = durable(tmp_path / "state", pool)
    await runner.launch()
    try:
        binding = RunBinding(
            models={"policy": ModelBinding(direct=DirectModel(provider="scripted", model="s"))},
            pools={"fake": PoolBinding(local="boxes")},
        )
        specification = RunSpecification(program=ProgramReference(program=register(Resumes)), binding=binding)
        handle = await runner.start(specification, lease="train/1/1/1")
        await until(lambda: runner.store.is_evicted(handle.run_id))  # unloaded while it waits: it keeps its box
        if resumed == "restarted":  # its runner stops, and starts again: the run is recovered
            await runner.close()
            runner = durable(tmp_path / "state", pool)
            await runner.launch()
        assert len(sandboxes.sandboxes) == 1 and sandboxes.deleted == []
        await runner.send(Address(kind="run", value=handle.run_id), Envelope(kind="go", content=[Text(text="go")]))
        assert (await runner.run(handle.run_id).result()).status is RunStatus.COMPLETED
    finally:
        await runner.close()
    events = runner.store.events(handle.run_id)
    said = {str(payload(e)["kind"]): payload(e)["payload"] for e in events if e.type is RunEventType.OUTPUT_EMITTED}
    assert said["before"] == said["after"]  # resumed and replayed, it reached the same sandbox
    assert pool.asked == ["train/1/1/1/box", "train/1/1/1/box"]  # acquired again when it resumed, under its lease
    (handle_made,) = sandboxes.made
    assert sandboxes.deleted == [handle_made] and sandboxes.sandboxes == {}  # and released once it ended
