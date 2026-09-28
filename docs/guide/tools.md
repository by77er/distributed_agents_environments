# Tools

Tools are the task's action space: functions the model can call. Declare them as methods decorated with `@tool`.
The default `respond` executes the reply's tool calls and returns their results as the next observation.

## Declaring a tool

The tool's specification comes from the method: its name, its docstring as the description, and its parameters'
type hints as a JSON Schema (built by pydantic). Describe a parameter with `Annotated[..., Field(description=...)]`.

```python
from typing import Annotated, Literal

from pydantic import Field

from rollout.core.harness import Observation, RunContext, Task, tool


class Library(Task):
    async def start(self, run: RunContext) -> Observation:
        return Observation("Find a book about tide pools.")

    @tool
    async def search(
        self,
        query: Annotated[str, Field(description="Words to search for.")],
        limit: int = 3,
        sort: Literal["relevance", "year"] = "relevance",
    ) -> list[str]:
        """Search the catalog. Returns matching titles."""
        titles = ["Tide Pools of the Pacific", "The Rocky Shore", "Life Between the Tides"]
        return titles[:limit]


(search,) = Library.declared_tools.values()
specification = search.specification
assert specification.name == "search"
assert specification.description == "Search the catalog. Returns matching titles."
assert specification.input_schema == {
    "type": "object",
    "properties": {
        "query": {"type": "string", "title": "Query", "description": "Words to search for."},
        "limit": {"type": "integer", "title": "Limit", "default": 3},
        "sort": {"type": "string", "title": "Sort", "enum": ["relevance", "year"], "default": "relevance"},
    },
    "required": ["query"],
    "additionalProperties": False,
}
```

- Parameters can use any type pydantic can validate, including pydantic models. Arguments are validated before the
  body runs; unknown arguments are rejected.
- `*args` and `**kwargs` are not allowed: every argument must appear in the schema.
- Methods may be `async` or plain functions.
- `@tool` methods are collected when the class is defined (`Task.declared_tools`); subclasses inherit them.
  Two tools with the same name in one class hierarchy are an error.

## Options

```python fragment
@tool(name="search_catalog", retry_class=RetryClass.IDEMPOTENT, timeout=timedelta(seconds=10))
async def search(self, query: str) -> list[str]: ...
```

| Option | Effect |
|---|---|
| `name` | The name the model sees (default: the method name). Must match `^[a-zA-Z0-9_-]{1,64}$`. |
| `retry_class` | What a durable runner may do after a crash: `PURE` (default), `IDEMPOTENT`, `SIDE_EFFECTING`, `UNKNOWN`. |
| `timeout` | A time limit for the body. Exceeding it returns an error result to the model. |

## What a tool returns

The return value becomes a `ToolResult` in a TOOL message:

| The body returns | The model receives |
|---|---|
| `str` | one text block |
| a `Text` or `Media` block, or a list of them | those blocks |
| a `ToolResult` | exactly that result (use it to set `structured`, `is_error`, …) |
| anything else | its JSON as text, and the same value in `structured` |

## Errors are observations

Tool failures are shown to the model so it can react, instead of failing the run. Each of these produces a result
with `is_error=True` and a message:

- the body raises an exception (`"ZeroDivisionError: division by zero"`);
- the arguments do not validate against the schema;
- the model calls a tool that does not exist;
- the body exceeds its `timeout`.

## Tool calls in an episode

With the default `respond`, a reply that makes tool calls gets their results back, and a reply without tool calls
ends the episode. `score` can then judge the final answer.

```python
import asyncio

from rollout.core.contracts import Role, ToolCall, ToolResultBlock
from rollout.core.harness import Agent, rollout
from rollout.core.testing import local_run, tool_call_reply


class FindBook(Library):
    async def score(self, run: RunContext) -> float | None:
        final_reply = run.history.turns[-1].reply
        return 1.0 if final_reply is not None and "Tide Pools" in final_reply.text else 0.0


async def main() -> None:
    task = FindBook()
    call = ToolCall(call_id="call_1", name="search", arguments={"query": "tide pools", "limit": 1})
    run, endpoint = local_run(task, replies=[tool_call_reply(call), "Try 'Tide Pools of the Pacific'."])
    await rollout(task, Agent(), run)

    # Turn 1: the reply's tool call, answered by one TOOL message.
    tool_turn = run.history.turns[1]
    (tool_message,) = tool_turn.observation.messages
    assert tool_message.role is Role.TOOL
    (block,) = tool_message.content
    assert isinstance(block, ToolResultBlock) and block.call_id == "call_1"
    assert block.result.structured == ["Tide Pools of the Pacific"]

    # The model was offered the tool on every turn, and the episode scored 1.
    assert [tool.name for tool in endpoint.requests[0].tools] == ["search"]
    assert run.rewards[0].value == 1.0


asyncio.run(main())
```

## Combining tools with your own `respond`

Override `respond` to add environment logic, and call `run_tools` for replies that make tool calls:

```python fragment
async def respond(self, run: RunContext, reply: Message) -> Observation:
    if reply.tool_calls:
        return await self.run_tools(run, reply)  # executes all calls concurrently
    return End(reward=self.grade(reply.text))
```

## Offering a subset per turn

`tools_for_turn(run)` returns the specifications offered on each turn; the default offers every declared tool.
Override it to change the action space as the episode progresses:

```python fragment
def tools_for_turn(self, run: RunContext) -> list[ToolSpecification]:
    tools = super().tools_for_turn(run)
    return [tool for tool in tools if tool.name != "submit"] if run.turn < 2 else tools
```

## What tools are not

- **Tool bodies are task code, not effects.** Under the durable runner (M2), a tool body may run again during
  replay. Work that must happen exactly once belongs in effects: model samples, environment operations, imported
  tools.
- **Imported tools** (`imports = ["github"]`: MCP servers, HTTP services, other agents) are designed but not built
  yet. See [task](../core/harness/task.md#tools).
