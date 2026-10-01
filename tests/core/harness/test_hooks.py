"""Hooks: a runner tells them of every event it records and every sample its models make, with the content."""

from rollout.core.contracts import (
    CapabilityContract,
    FinishReason,
    Message,
    RunEvent,
    RunEventType,
    SampleRequest,
    SampleResult,
    ToolCall,
    ToolSpecification,
    Usage,
)
from rollout.core.harness import (
    DirectModel,
    ModelBinding,
    ModelSample,
    ModelSlot,
    Program,
    ProgramReference,
    RunBinding,
    RunContext,
    RunHooks,
    RunSpecification,
    RunStatus,
    register,
)
from rollout.core.local import LocalRunner
from rollout.core.testing import tool_call_reply

MINE = ToolSpecification(
    name="mine", description="Mine.", input_schema={"type": "object", "properties": {"x": {"type": "integer"}}}
)


class Miner:
    """Always mines at x=3."""

    def describe(self, session_id: str) -> CapabilityContract:
        return CapabilityContract(context_limit=100_000, max_output_tokens=1_000)

    async def sample(self, request: SampleRequest) -> SampleResult:
        return SampleResult(
            message=tool_call_reply(ToolCall(call_id="c1", name="mine", arguments={"x": 3})),
            finish_reason=FinishReason.TOOL_USE,
            usage=Usage(context_used=1, context_limit=100_000),
        )

    async def cancel(self, effect_id: str) -> None:
        pass


class OneTurn(Program):
    def model_slots(self) -> dict[str, ModelSlot]:
        return {"ada": ModelSlot()}

    async def main(self, run: RunContext) -> None:
        await run.models["ada"].sample([Message.system("You mine."), Message.user("You see a wall.")], tools=[MINE])
        run.reward(2.0, slot="ada")


class Watching(RunHooks):
    def __init__(self) -> None:
        self.events: list[RunEvent] = []
        self.samples: list[ModelSample] = []

    def on_event(self, event: RunEvent) -> None:
        self.events.append(event)

    def on_sample(self, sample: ModelSample) -> None:
        self.samples.append(sample)


class Broken(RunHooks):
    def on_event(self, event: RunEvent) -> None:
        raise RuntimeError("a hook's own trouble")

    def on_sample(self, sample: ModelSample) -> None:
        raise RuntimeError("a hook's own trouble")


def specification() -> RunSpecification:
    binding = RunBinding(models={"ada": ModelBinding(direct=DirectModel(provider="scripted", model="miner"))})
    return RunSpecification(program=ProgramReference(program=register(OneTurn)), binding=binding)


async def test_hooks_see_every_event_and_every_sample_with_its_content() -> None:
    watching = Watching()
    runner = LocalRunner(providers={"scripted": lambda model: Miner()}, hooks=[Broken(), watching])
    handle = await runner.start(specification(), labels={"group": "g1"})
    outcome = await handle.result()
    assert outcome.status is RunStatus.COMPLETED  # a hook that raises does not fail the run
    assert watching.events == handle.recorded_events()  # each one, in order, as it was recorded
    assert (
        watching.events[0].type is RunEventType.RUN_CREATED and watching.events[-1].type is RunEventType.RUN_COMPLETED
    )
    (sample,) = watching.samples
    assert (sample.run_id, sample.slot) == (handle.run_id, "ada")
    assert [message.text for message in sample.request.context.append] == ["You mine.", "You see a wall."]
    assert [tool.name for tool in sample.request.tools] == ["mine"]
    assert sample.result.message.tool_calls[0].arguments == {"x": 3}
    assert sample.seconds >= 0


async def test_a_runner_without_hooks_is_unchanged() -> None:
    runner = LocalRunner(providers={"scripted": lambda model: Miner()})
    handle = await runner.start(specification())
    assert (await handle.result()).status is RunStatus.COMPLETED
