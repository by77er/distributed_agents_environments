from datetime import timedelta
from typing import Annotated, Any

import pytest
from pydantic import Field

from rollout.core.contracts import Message, RetryClass, Role, Text, ToolCall, ToolResult, ToolResultBlock
from rollout.core.harness import Observation, RunContext, Task, tool
from rollout.core.testing import local_run


class Calculator(Task):
    @tool
    async def add(self, a: int, b: Annotated[int, Field(description="The second addend.")] = 0) -> int:
        """Add two integers."""
        return a + b

    @tool(name="divide", retry_class=RetryClass.IDEMPOTENT)
    def divide_numbers(self, numerator: float, denominator: float) -> str:
        """Divide two numbers."""
        return str(numerator / denominator)

    @tool(timeout=timedelta(milliseconds=10))
    async def slow(self) -> str:
        """Never finishes in time."""
        import asyncio

        await asyncio.sleep(1)
        return "late"

    @tool
    async def picture(self) -> ToolResult:
        """Return a prepared result."""
        return ToolResult(content=[Text(text="a picture")], structured={"width": 3})

    async def start(self, run: RunContext) -> Observation:
        return Observation("compute")


def call(name: str, call_id: str = "c1", **arguments: Any) -> ToolCall:
    return ToolCall(call_id=call_id, name=name, arguments=arguments)


def results_of(observation: Observation) -> dict[str, ToolResult]:
    (message,) = observation.messages
    assert message.role is Role.TOOL
    return {block.call_id: block.result for block in message.content if isinstance(block, ToolResultBlock)}


def test_specifications_come_from_signatures_and_docstrings() -> None:
    specifications = {tool.name: tool for tool in Calculator().tools_for_turn(local_run(Calculator())[0])}
    assert list(specifications) == ["add", "divide", "slow", "picture"]
    add = specifications["add"]
    assert add.description == "Add two integers."
    assert add.input_schema == {
        "type": "object",
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"title": "B", "type": "integer", "default": 0, "description": "The second addend."},
        },
        "required": ["a"],
        "additionalProperties": False,
    }
    assert specifications["divide"].retry_class is RetryClass.IDEMPOTENT
    assert specifications["slow"].timeout_ms == 10


async def test_calls_run_concurrently_and_results_are_normalized() -> None:
    task = Calculator()
    run, _ = local_run(task)
    reply = Message(
        role=Role.ASSISTANT,
        content=[call("add", "c1", a=2, b=3), call("divide", "c2", numerator=1, denominator=4), call("picture", "c3")],
    )
    results = results_of(await task.run_tools(run, reply))
    assert results["c1"] == ToolResult(content=[Text(text="5")], structured=5)
    assert results["c2"].content == (Text(text="0.25"),)
    assert results["c3"].structured == {"width": 3}


@pytest.mark.parametrize(
    ("tool_call", "expected"),
    [
        (call("divide", numerator=1, denominator=0), "ZeroDivisionError: float division by zero"),
        (call("add", b=1), "invalid arguments for add"),
        (call("add", a=1, c=2), "invalid arguments for add"),
        (call("missing"), "unknown tool 'missing'"),
        (call("slow"), "slow timed out"),
    ],
)
async def test_failures_become_error_results(tool_call: ToolCall, expected: str) -> None:
    task = Calculator()
    run, _ = local_run(task)
    result = results_of(await task.run_tools(run, Message(role=Role.ASSISTANT, content=[tool_call])))["c1"]
    assert result.is_error
    assert isinstance(result.content[0], Text)
    assert expected in result.content[0].text


async def test_a_reply_without_tool_calls_ends_the_episode() -> None:
    task = Calculator()
    run, _ = local_run(task)
    observation = await task.run_tools(run, Message.assistant("done"))
    assert observation.end is not None


def test_duplicate_tool_names_are_rejected() -> None:
    with pytest.raises(TypeError):

        class Duplicate(Task):  # pyright: ignore[reportUnusedClass]
            @tool
            async def first(self) -> str:
                return ""

            @tool(name="first")
            async def second(self) -> str:
                return ""
