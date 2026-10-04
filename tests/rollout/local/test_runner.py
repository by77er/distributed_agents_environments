import asyncio

import pytest
from pydantic import ValidationError

from rollout.contracts import Message, RunEventType, RunFailureClass
from rollout.harness import (
    DirectModel,
    End,
    ModelBinding,
    Observation,
    RecordedModel,
    RunBinding,
    RunContext,
    RunSpecification,
    RunStatus,
    Task,
    agent_program,
)
from rollout.local import LocalRunner
from rollout.testing import ScriptedModelEndpoint, ScriptedReply, payload


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


class Waiting(Task):
    """Waits in `start` until the run is cancelled."""

    torn_down: list[str] = []

    async def start(self, run: RunContext) -> Observation:
        await asyncio.sleep(5)
        return Observation("late")

    async def teardown(self, run: RunContext) -> None:
        Waiting.torn_down.append(run.run_id)


async def test_cancel_runs_teardown() -> None:
    runner, _ = runner_with([])
    handle = await runner.start(RunSpecification(program=agent_program(Waiting), binding=binding()))
    await asyncio.sleep(0.01)  # the run is now waiting in start
    await runner.cancel(handle.run_id, reason="test")
    assert (await handle.result()).status is RunStatus.CANCELLED
    assert handle.run_id in Waiting.torn_down
    types = [event.type for event in handle.context.events]
    assert types[-2:] == [RunEventType.RUN_CANCEL_REQUESTED, RunEventType.RUN_CANCELLED]


def test_a_model_binding_is_exactly_one_kind() -> None:
    direct, recorded = DirectModel(provider="scripted", model="script"), RecordedModel(channel="policy")
    with pytest.raises(ValidationError, match="exactly one"):
        ModelBinding()
    with pytest.raises(ValidationError, match="exactly one"):
        ModelBinding(direct=direct, recorded=recorded)
    assert ModelBinding(recorded=recorded).direct is None


async def test_unknown_providers_are_rejected() -> None:
    runner = LocalRunner()
    with pytest.raises(ValueError, match="no endpoint factory"):
        await runner.start(RunSpecification(program=agent_program(Echo), binding=binding()))
