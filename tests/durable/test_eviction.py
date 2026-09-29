"""Idle runs are evicted from memory and woken by a message, by their deadline, or to be cancelled."""

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import pytest
from dbos import DBOS

from rollout.core.contracts import Message, RunEventType, Text
from rollout.core.harness import (
    Address,
    Deployment,
    DirectModel,
    Envelope,
    ModelBinding,
    RunBinding,
    RunContext,
    RunSpecification,
    RunStatus,
    Task,
    WaitFor,
    agent_program,
)
from rollout.core.testing import ScriptedModelEndpoint, payload
from rollout.durable import DurableRunner


class Chat(Task):
    wait_seconds = 60.0
    teardowns = 0

    async def start(self, run: RunContext) -> WaitFor:
        return WaitFor("message")

    async def respond(self, run: RunContext, reply: Message) -> WaitFor:
        await run.emit("reply", reply.text)
        return WaitFor("message", timeout=timedelta(seconds=self.wait_seconds))

    async def teardown(self, run: RunContext) -> None:
        Chat.teardowns += 1


class ShortWait(Chat):
    wait_seconds = 2.0


type Setup = tuple[DurableRunner, ScriptedModelEndpoint]


@pytest.fixture
async def evicting(tmp_path: Path) -> AsyncIterator[Setup]:
    endpoint = ScriptedModelEndpoint(["first", "second", "third"])
    runner = DurableRunner(
        tmp_path / "state",
        providers={"scripted": lambda model: endpoint},
        evict_after=timedelta(seconds=0.3),
        eviction_interval=0.1,
    )
    await runner.launch()
    binding = RunBinding(models={"policy": ModelBinding(direct=DirectModel(provider="scripted", model="s"))})
    for task in (Chat, ShortWait):
        spec = RunSpecification(program=agent_program(task), binding=binding)
        runner.deploy(Deployment(name=f"test/{task.__name__.lower()}", specification=spec))
    yield runner, endpoint
    await runner.close()


async def until(condition, seconds: float = 15) -> None:  # type: ignore[no-untyped-def]
    for _ in range(int(seconds * 20)):
        if condition():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("condition not reached")


def replies(runner: DurableRunner, run_id: str) -> list[str]:
    events = runner.store.events(run_id)
    return [str(payload(e)["payload"]) for e in events if e.type is RunEventType.OUTPUT_EMITTED]


async def test_an_idle_run_is_evicted_and_woken_by_a_message(evicting: Setup) -> None:
    runner, endpoint = evicting
    address = Address(kind="conversation", value="test/chat/c1")
    await runner.send(address, Envelope(content=[Text(text="hello")]))
    (handle,) = runner.conversation_runs("test/chat", "c1")
    await until(lambda: replies(runner, handle.run_id) == ["first"])
    teardowns = Chat.teardowns
    await until(lambda: runner.store.is_evicted(handle.run_id))
    assert (await DBOS.get_workflow_status_async(handle.run_id)).status == "CANCELLED"  # type: ignore[union-attr]
    await until(lambda: handle.run_id not in runner.workflow_tasks)  # unloaded from memory
    assert Chat.teardowns == teardowns  # but not ended: no teardown

    await runner.send(address, Envelope(content=[Text(text="again")]))
    await until(lambda: replies(runner, handle.run_id) == ["first", "second"])
    assert len(endpoint.requests) == 2  # the replay on waking made no model call
    events = runner.store.events(handle.run_id)
    assert [e.seq for e in events] == list(range(len(events)))
    assert sum(e.type is RunEventType.RUN_CREATED for e in events) == 1


async def test_an_evicted_run_wakes_when_its_wait_times_out(evicting: Setup) -> None:
    runner, _ = evicting
    await runner.send(Address(kind="conversation", value="test/shortwait/c2"), Envelope(content=[Text(text="hi")]))
    (handle,) = runner.conversation_runs("test/shortwait", "c2")
    await until(lambda: runner.store.is_evicted(handle.run_id))
    outcome = await asyncio.wait_for(handle.result(), 15)  # the 2-second wait times out; the episode ends
    assert outcome.status is RunStatus.COMPLETED
    assert replies(runner, handle.run_id) == ["first"]


async def test_cancelling_an_evicted_run_still_tears_it_down(evicting: Setup) -> None:
    runner, _ = evicting
    await runner.send(Address(kind="conversation", value="test/chat/c3"), Envelope(content=[Text(text="hi")]))
    (handle,) = runner.conversation_runs("test/chat", "c3")
    await until(lambda: runner.store.is_evicted(handle.run_id))
    teardowns = Chat.teardowns
    await asyncio.wait_for(runner.cancel(handle.run_id, reason="done"), 15)
    assert (await handle.result()).status is RunStatus.CANCELLED
    assert Chat.teardowns == teardowns + 1
