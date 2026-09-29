from collections.abc import Mapping, Sequence

import pytest
from pydantic import JsonValue

from rollout.core.contracts import (
    EffectKind,
    RunEventType,
    Text,
    ToolCall,
    ToolResult,
    ToolResultBlock,
    ToolSpecification,
)
from rollout.core.harness import (
    DirectModel,
    ModelBinding,
    Observation,
    RunBinding,
    RunContext,
    RunSpecification,
    RunStatus,
    Task,
    ToolBinding,
    agent_program,
    tool,
)
from rollout.core.local import LocalRunner
from rollout.core.testing import ScriptedModelEndpoint, ScriptedReply, payload, tool_call_reply


class Counter:
    """An in-process tool set that remembers which effects it performed."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail = fail

    def specifications(self) -> Sequence[ToolSpecification]:
        return [ToolSpecification(name="increment", description="Add one to a counter.")]

    async def call(
        self, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
        if self.fail:
            raise ConnectionError("counter service is down")
        self.calls.append((effect_id, arguments_digest))
        return ToolResult(content=[Text(text=str(len(self.calls)))])


class Counting(Task):
    imports = ["counter"]

    async def start(self, run: RunContext) -> Observation:
        return Observation("Increment the counter.")

    @tool
    async def local_echo(self, text: str) -> str:
        """Echo text."""
        return text


def run_with(counter: Counter, replies: list[ScriptedReply], imports: dict[str, ToolBinding] | None = None):
    endpoint = ScriptedModelEndpoint(replies)
    runner = LocalRunner(providers={"scripted": lambda model: endpoint}, tool_sets={"counter-service": counter})
    binding = RunBinding(
        models={"policy": ModelBinding(direct=DirectModel(provider="scripted", model="script"))},
        imports=imports if imports is not None else {"counter": ToolBinding(local="counter-service")},
    )
    return runner, endpoint, RunSpecification(program=agent_program(Counting), binding=binding)


INCREMENT = ToolCall(call_id="c1", name="increment", arguments={})


async def test_imported_tools_are_offered_and_called_as_effects() -> None:
    counter = Counter()
    runner, endpoint, specification = run_with(counter, [tool_call_reply(INCREMENT), "done"])
    handle = await runner.start(specification)
    assert (await handle.result()).status is RunStatus.COMPLETED

    assert [tool.name for tool in endpoint.requests[0].tools] == ["local_echo", "increment"]
    (resolved,) = [e for e in handle.context.events if e.type is RunEventType.TOOLS_RESOLVED]
    assert [spec["name"] for spec in payload(resolved)["specifications"]] == ["local_echo", "increment"]  # type: ignore[index, union-attr]

    requested = [e for e in handle.context.events if e.type is RunEventType.EFFECT_REQUESTED]
    tool_effect = next(payload(e) for e in requested if payload(e)["kind"] == EffectKind.TOOL_CALL.value)
    assert counter.calls == [(tool_effect["effect_id"], tool_effect["arguments_digest"])]

    tool_turn = handle.context.history.turns[1].observation
    assert tool_turn is not None
    (block,) = tool_turn.messages[0].content
    assert isinstance(block, ToolResultBlock) and block.result.content == (Text(text="1"),)


async def test_a_failing_tool_set_answers_with_an_error_and_a_failed_effect() -> None:
    runner, _, specification = run_with(Counter(fail=True), [tool_call_reply(INCREMENT), "gave up"])
    handle = await runner.start(specification)
    assert (await handle.result()).status is RunStatus.COMPLETED
    completed = [payload(e) for e in handle.context.events if e.type is RunEventType.EFFECT_COMPLETED]
    assert {"status": "failed", "error_class": "ConnectionError"}.items() <= completed[1].items()
    tool_turn = handle.context.history.turns[1].observation
    assert tool_turn is not None
    (block,) = tool_turn.messages[0].content
    assert isinstance(block, ToolResultBlock) and block.result.is_error


async def test_an_unbound_import_is_rejected() -> None:
    runner, _, specification = run_with(Counter(), [], imports={})
    with pytest.raises(ValueError, match="does not say how to serve the import 'counter'"):
        await runner.start(specification)
