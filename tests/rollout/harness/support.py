"""Helpers the tests of hooks share."""

from rollout.contracts import (
    CapabilityContract,
    FinishReason,
    Message,
    SampleRequest,
    SampleResult,
    ToolCall,
    ToolSpecification,
    Usage,
)
from rollout.harness import (
    DirectModel,
    ModelBinding,
    ModelSlot,
    Program,
    ProgramReference,
    RunBinding,
    RunContext,
    RunSpecification,
    register,
)
from rollout.testing import tool_call_reply

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


def specification() -> RunSpecification:
    binding = RunBinding(models={"ada": ModelBinding(direct=DirectModel(provider="scripted", model="miner"))})
    return RunSpecification(program=ProgramReference(program=register(OneTurn)), binding=binding)
