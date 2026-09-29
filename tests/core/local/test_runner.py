import asyncio
from datetime import timedelta

import pytest

from rollout.core.contracts import Message, RunEventType, RunFailureClass, Text
from rollout.core.harness import (
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
    RunSpecification,
    RunStatus,
    Task,
    WaitFor,
    agent_program,
)
from rollout.core.local import LocalRunner, RunNotLive
from rollout.core.testing import ScriptedModelEndpoint, ScriptedReply, payload


def binding() -> RunBinding:
    return RunBinding(models={"policy": ModelBinding(direct=DirectModel(provider="scripted", model="script"))})


def runner_with(replies: list[ScriptedReply]) -> tuple[LocalRunner, ScriptedModelEndpoint]:
    endpoint = ScriptedModelEndpoint(replies)
    return LocalRunner(providers={"scripted": lambda model: endpoint}), endpoint


class Echo(Task):
    async def start(self, run: RunContext) -> Observation:
        return Observation(str(self.parameters))

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        await run.emit("reply", reply.text)
        return End(reward=1.0)


class Broken(Task):
    async def start(self, run: RunContext) -> Observation:
        raise RuntimeError("no data")


class Invalid(Task):
    async def start(self, run: RunContext) -> Observation:
        return Observation(Message.assistant("not allowed"))


class Chat(Task):
    """A conversation: answers each message and emits the reply."""

    torn_down: list[str] = []

    async def start(self, run: RunContext) -> WaitFor:
        return WaitFor("message", timeout=timedelta(seconds=5))

    async def respond(self, run: RunContext, reply: Message) -> WaitFor:
        await run.emit("reply", reply.text)
        return WaitFor("message", timeout=timedelta(milliseconds=50))

    async def teardown(self, run: RunContext) -> None:
        Chat.torn_down.append(run.run_id)


async def test_a_run_completes_and_streams_its_events() -> None:
    runner, _ = runner_with(["hello back"])
    handle = await runner.start(
        RunSpecification(program=agent_program(Echo, task_parameters="hello"), binding=binding())
    )
    events = [event async for event in handle.events()]
    outcome = await handle.result()
    assert outcome.status is RunStatus.COMPLETED
    types = [event.type for event in events]
    assert types[0] is RunEventType.RUN_CREATED
    assert types[-1] is RunEventType.RUN_COMPLETED
    assert RunEventType.OUTPUT_EMITTED in types
    (output,) = [event for event in events if event.type is RunEventType.OUTPUT_EMITTED]
    assert payload(output)["payload"] == "hello back"
    # A late subscriber replays from any position.
    assert [event.seq async for event in handle.events(from_seq=len(events) - 1)] == [len(events) - 1]


@pytest.mark.parametrize(
    ("task", "failure"),
    [(Broken, RunFailureClass.TASK_ERROR), (Invalid, RunFailureClass.INVALID_OBSERVATION)],
)
async def test_failures_are_classified(task: type[Task], failure: RunFailureClass) -> None:
    runner, _ = runner_with([])
    handle = await runner.start(RunSpecification(program=agent_program(task), binding=binding()))
    outcome = await handle.result()
    assert outcome.status is RunStatus.FAILED
    assert outcome.failure_class is failure
    assert handle.context.events[-1].type is RunEventType.RUN_FAILED


async def test_cancel_runs_teardown() -> None:
    runner, _ = runner_with([])
    handle = await runner.start(RunSpecification(program=agent_program(Chat), binding=binding()))
    await asyncio.sleep(0.01)  # the run is now waiting for a message
    await runner.cancel(handle.run_id, reason="test")
    assert (await handle.result()).status is RunStatus.CANCELLED
    assert handle.run_id in Chat.torn_down
    types = [event.type for event in handle.context.events]
    assert types[-2:] == [RunEventType.RUN_CANCEL_REQUESTED, RunEventType.RUN_CANCELLED]


def deployed(replies: list[ScriptedReply]) -> tuple[LocalRunner, ScriptedModelEndpoint]:
    runner, endpoint = runner_with(replies)
    runner.deploy(
        Deployment(name="acme/chat", specification=RunSpecification(program=agent_program(Chat), binding=binding()))
    )
    return runner, endpoint


CONVERSATION = Address(kind="conversation", value="acme/chat/user:42")


def text(value: str) -> Envelope:
    return Envelope(content=[Text(text=value)])


async def test_a_message_starts_the_conversation_and_the_next_one_reaches_the_live_run() -> None:
    runner, endpoint = deployed(["hi!", "fine, thanks"])
    await runner.send(CONVERSATION, text("hello"))
    await asyncio.sleep(0.01)
    await runner.send(CONVERSATION, text("how are you?"))
    (handle,) = runner.conversation_runs("acme/chat", "user:42")
    await handle.result()
    outputs = [payload(e)["payload"] for e in handle.context.events if e.type is RunEventType.OUTPUT_EMITTED]
    assert outputs == ["hi!", "fine, thanks"]
    assert handle.conversation is not None and handle.conversation.key == "user:42"
    assert len(endpoint.requests) == 2


async def test_messages_are_deduplicated_by_idempotency_key() -> None:
    runner, endpoint = deployed(["once"])
    await runner.send(CONVERSATION, text("hello"), idempotency_key="event-1")
    await runner.send(CONVERSATION, text("hello"), idempotency_key="event-1")
    (handle,) = runner.conversation_runs("acme/chat", "user:42")
    await handle.result()
    assert len(endpoint.requests) == 1


async def test_unconsumed_messages_start_the_next_run() -> None:
    class OneShot(Task):
        async def start(self, run: RunContext) -> WaitFor:
            return WaitFor("message")

        async def respond(self, run: RunContext, reply: Message) -> Observation:
            return End()  # ends after one message, leaving any others unconsumed

    runner, _ = runner_with(["first answer", "second answer"])
    spec = RunSpecification(program=agent_program(OneShot), binding=binding())
    runner.deploy(Deployment(name="acme/oneshot", specification=spec))
    address = Address(kind="conversation", value="acme/oneshot/k")
    await runner.send(address, text("first"), priority=Priority.LOW)
    await runner.send(address, text("second"), priority=Priority.LOW)
    await asyncio.sleep(0.05)
    runs = runner.conversation_runs("acme/oneshot", "k")
    assert len(runs) == 2
    for handle in runs:
        await handle.result()
    assert [len(handle.context.history.turns) for handle in runs] == [2, 2]


async def test_sending_to_an_ended_run_fails() -> None:
    runner, _ = runner_with(["ok"])
    handle = await runner.start(RunSpecification(program=agent_program(Echo, task_parameters="x"), binding=binding()))
    await handle.result()
    with pytest.raises(RunNotLive):
        await runner.send(Address(kind="run", value=handle.run_id), text("late"))


async def test_unknown_providers_and_conversations_are_rejected() -> None:
    runner = LocalRunner()
    with pytest.raises(ValueError, match="no endpoint factory"):
        await runner.start(RunSpecification(program=agent_program(Echo), binding=binding()))
    with pytest.raises(ValueError, match="does not name a conversation"):
        await runner.send(Address(kind="conversation", value="acme/unknown/k"), text("hi"))
