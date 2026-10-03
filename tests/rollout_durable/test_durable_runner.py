import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import pytest

from rollout.contracts import Message, RunEvent, RunEventType, SampleRequest, Text
from rollout.harness import (
    Address,
    Deployment,
    DirectModel,
    End,
    Envelope,
    ModelBinding,
    Observation,
    Priority,
    RunBinding,
    RunContext,
    RunNotLive,
    RunSpecification,
    RunStatus,
    Task,
    WaitFor,
    agent_program,
)
from rollout.testing import ScriptedModelEndpoint, ScriptedReply, payload
from rollout_durable import DurableRunner


class Echo(Task):
    async def start(self, run: RunContext) -> Observation:
        return Observation("Say something.")

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        await run.emit("reply", reply.text)
        return End(reward=1.0)


class Chat(Task):
    teardowns = 0

    async def start(self, run: RunContext) -> WaitFor:
        return WaitFor("message", timeout=timedelta(seconds=30))

    async def respond(self, run: RunContext, reply: Message) -> WaitFor:
        await run.emit("reply", reply.text)
        return WaitFor("message", timeout=timedelta(seconds=30))

    async def teardown(self, run: RunContext) -> None:
        Chat.teardowns += 1


def binding() -> RunBinding:
    return RunBinding(models={"policy": ModelBinding(direct=DirectModel(provider="scripted", model="script"))})


type Durable = tuple[DurableRunner, list[ScriptedModelEndpoint]]


@pytest.fixture
async def durable(tmp_path: Path, database: str | None) -> AsyncIterator[Durable]:
    endpoints: list[ScriptedModelEndpoint] = []
    runner = DurableRunner(tmp_path / "state", providers={"scripted": lambda model: endpoints[0]}, database=database)
    await runner.launch()
    yield runner, endpoints
    await runner.close()


def script(endpoints: list[ScriptedModelEndpoint], replies: list[ScriptedReply]) -> ScriptedModelEndpoint:
    endpoints.append(ScriptedModelEndpoint(replies))
    return endpoints[0]


def outputs(events: list[RunEvent]) -> list[str]:
    return [str(payload(event)["payload"]) for event in events if event.type is RunEventType.OUTPUT_EMITTED]


async def test_a_durable_run_completes_and_its_events_persist(durable: Durable) -> None:
    runner, endpoints = durable
    script(endpoints, ["hello"])
    handle = await runner.start(RunSpecification(program=agent_program(Echo), binding=binding()))
    streamed = [event async for event in handle.events()]
    assert (await handle.result()).status is RunStatus.COMPLETED
    assert [event.seq for event in streamed] == list(range(len(streamed)))
    assert streamed[0].type is RunEventType.RUN_CREATED and streamed[-1].type is RunEventType.RUN_COMPLETED
    assert outputs(streamed) == ["hello"]
    assert handle.recorded_events() == streamed


async def test_a_conversation_waits_durably_and_answers_each_message(durable: Durable) -> None:
    runner, endpoints = durable
    script(endpoints, ["hi!", "fine, thanks"])
    spec = RunSpecification(program=agent_program(Chat), binding=binding())
    runner.deploy(Deployment(name="acme/chat", specification=spec))
    conversation = Address(kind="conversation", value="acme/chat/user:7")
    await runner.send(conversation, Envelope(content=[Text(text="hello")]))
    await runner.send(conversation, Envelope(content=[Text(text="hello")]), idempotency_key="k1")
    await runner.send(conversation, Envelope(content=[Text(text="hello")]), idempotency_key="k1")  # a retry
    (handle,) = runner.conversation_runs("acme/chat", "user:7")
    async for _ in handle.events():
        if len(outputs(handle.recorded_events())) == 2:
            break
    assert outputs(handle.recorded_events()) == ["hi!", "fine, thanks"]
    teardowns = Chat.teardowns
    await runner.cancel(handle.run_id, reason="done")
    assert (await handle.result()).status is RunStatus.CANCELLED
    assert Chat.teardowns == teardowns + 1
    types = [event.type for event in handle.recorded_events()]
    assert types[-2:] == [RunEventType.RUN_CANCEL_REQUESTED, RunEventType.RUN_CANCELLED]


async def test_a_failed_send_can_be_retried_and_messages_to_a_run_are_deduplicated(durable: Durable) -> None:
    runner, endpoints = durable
    script(endpoints, ["hi!", "fine, thanks"])
    conversation = Address(kind="conversation", value="acme/chat/user:9")
    hello = Envelope(content=[Text(text="hello")])
    with pytest.raises(ValueError, match="does not name a conversation of a deployed agent"):
        await runner.send(conversation, hello, idempotency_key="k1")  # nothing is deployed yet
    runner.deploy(
        Deployment(name="acme/chat", specification=RunSpecification(program=agent_program(Chat), binding=binding()))
    )
    await runner.send(conversation, hello, idempotency_key="k1")  # the failed send claimed nothing
    (handle,) = runner.conversation_runs("acme/chat", "user:9")
    run = Address(kind="run", value=handle.run_id)
    assert await runner.send(run, hello, idempotency_key="k2") == "k2"
    assert await runner.send(run, hello, idempotency_key="k2") == "k2"  # a retry
    async for _ in handle.events():
        if len(outputs(handle.recorded_events())) == 2:
            break
    await runner.cancel(handle.run_id, reason="done")
    received = [event for event in handle.recorded_events() if event.type is RunEventType.MESSAGE_RECEIVED]
    assert [payload(event)["envelope"]["message_id"] for event in received] == ["k1", "k2"]  # type: ignore[index, call-overload]
    with pytest.raises(RunNotLive):
        await runner.send(run, hello)


async def test_a_high_priority_message_interrupts_the_reply(durable: Durable) -> None:
    runner, endpoints = durable
    started = asyncio.Event()

    async def slow(request: SampleRequest) -> Message:
        started.set()
        await asyncio.sleep(30)
        return Message.assistant("too late")

    endpoint = script(endpoints, [slow, "EUR"])
    spec = RunSpecification(program=agent_program(Chat), binding=binding())
    runner.deploy(Deployment(name="acme/chat", specification=spec))
    conversation = Address(kind="conversation", value="acme/chat/user:8")
    await runner.send(conversation, Envelope(content=[Text(text="Tour the whole repository.")]))
    await asyncio.wait_for(started.wait(), 10)
    await runner.send(conversation, Envelope(content=[Text(text="Just the currency.")]), priority=Priority.HIGH)
    (handle,) = runner.conversation_runs("acme/chat", "user:8")
    async for _ in handle.events():
        if outputs(handle.recorded_events()):
            break
    assert outputs(handle.recorded_events()) == ["EUR"]
    assert endpoint.cancelled == [endpoint.requests[0].effect_id]
    interrupted = [e for e in handle.recorded_events() if e.type is RunEventType.TURN_INTERRUPTED]
    assert payload(interrupted[0])["reply_effect_id"] == endpoint.requests[0].effect_id
    assert [m.text for m in endpoint.requests[1].context.append] == ["Tour the whole repository.", "Just the currency."]
    await runner.cancel(handle.run_id, reason="done")


async def test_a_run_whose_binding_cannot_be_served_does_not_start(durable: Durable) -> None:
    runner, _ = durable
    unserved = RunBinding(models={"policy": ModelBinding(direct=DirectModel(provider="nobody", model="m"))})
    with pytest.raises(ValueError, match="nobody"):  # refused at once, as `LocalRunner` refuses it
        await runner.start(RunSpecification(program=agent_program(Echo), binding=unserved))
    empty = RunBinding(models={})
    with pytest.raises(ValueError, match="policy"):
        await runner.start(RunSpecification(program=agent_program(Echo), binding=empty))
