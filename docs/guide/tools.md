# Tools

Tools are the task's action space: functions the model can call. Declare them as methods decorated with `@tool`.
The default `respond` executes the reply's tool calls and returns their results as the next observation.

## Declaring a tool

The tool's specification comes from the method: its name, its docstring as the description, and its parameters'
type hints as a JSON Schema (built by pydantic). Describe a parameter with `Annotated[..., Field(description=...)]`.

```python
from typing import Annotated, Literal

from pydantic import Field

from rollout.harness import Observation, RunContext, Task, tool


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
- A parameter named `run` receives the [run context](tasks.md#the-run-context) (for `run.now()`, `run.emit`,
  other model slots). It is not part of the schema, so the model never sees it.
- Methods may be `async` or plain functions.
- `@tool` methods are collected when the class is defined (`Task.declared_tools`); subclasses inherit them.
  Two tools with the same name in one class hierarchy are an error.

## Options

```py
@tool(name="search_catalog", retry_class=RetryClass.IDEMPOTENT, timeout=timedelta(seconds=10))
async def search(self, query: str) -> list[str]: ...
```

| Option | Effect |
|---|---|
| `name` | The name the model sees (default: the method name). Must match `^[a-zA-Z0-9_-]{1,64}$`. |
| `retry_class` | The specification's `retry_class`: `PURE` (default), `IDEMPOTENT`, `SIDE_EFFECTING` or `UNKNOWN`. Runners act on it for [imported tools](#imported-tools). |
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

from rollout.contracts import Role, ToolCall, ToolResultBlock
from rollout.harness import Agent, rollout
from rollout.testing import local_run, tool_call_reply


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

```py
async def respond(self, run: RunContext, reply: Message) -> Observation:
    if reply.tool_calls:
        return await self.run_tools(run, reply)  # executes all calls concurrently
    return End(reward=self.grade(reply.text))
```

## Offering a subset per turn

`tools_for_turn(run)` returns the specifications offered on each turn; the default offers every `@tool` method,
then every imported tool. Override it to change the action space as the episode progresses:

```py
def tools_for_turn(self, run: RunContext) -> list[ToolSpecification]:
    tools = super().tools_for_turn(run)
    return [tool for tool in tools if tool.name != "submit"] if run.turn < 2 else tools
```

## Imported tools

Tools that live outside the task are **imported**: a task declares an import by name, and the run's binding says
which tool set serves it. Anything that keeps state across runs, such as memory or notes, belongs here rather than
in a `@tool` body. Each call to an imported tool is a `tool.call` effect: the tool set receives the call's
`effect_id` and arguments digest, and can use them to perform each call at most once.

A tool set implements `ToolSet`: `specifications()` and `call(name, arguments, *, effect_id, arguments_digest)`.
The binding's `ToolBinding` says where it is served:

| `ToolBinding` | The tool set is |
|---|---|
| `ToolBinding(local="name")` | in the runner's process, registered as `tool_sets={"name": tool_set}` |
| `ToolBinding(url="http://host:8700")` | served over HTTP, wherever its own infrastructure runs ([below](#serving-a-tool-set-over-http)) |

Tool names are unique within a run: an imported tool that shares a name with another import or with a `@tool`
method raises `ValueError`.

```python
from collections.abc import Mapping, Sequence

from pydantic import JsonValue

from rollout.contracts import Text, ToolResult, ToolSpecification
from rollout.harness import DirectModel, ModelBinding, RunBinding, RunSpecification, ToolBinding, agent_program
from rollout.local import LocalRunner
from rollout.testing import ScriptedModelEndpoint


class Bookmarks:
    """An in-process tool set that remembers bookmarks across runs, once per effect."""

    def __init__(self) -> None:
        self.saved: dict[str, str] = {}      # effect_id → title

    def specifications(self) -> Sequence[ToolSpecification]:
        return [ToolSpecification(
            name="bookmark",
            description="Save a book title for later.",
            input_schema={"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]},
        )]

    async def call(self, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str,
                   arguments_digest: str) -> ToolResult:
        self.saved.setdefault(effect_id, str(arguments["title"]))     # a repeated effect_id is not saved twice
        return ToolResult(content=[Text(text=f"Saved. {len(self.saved)} bookmarks.")])


class Librarian(Library):
    imports = ["bookmarks"]


async def imported() -> None:
    bookmarks = Bookmarks()
    call = ToolCall(call_id="call_1", name="bookmark", arguments={"title": "The Rocky Shore"})
    endpoint = ScriptedModelEndpoint([tool_call_reply(call), "Saved it."])
    runner = LocalRunner(providers={"scripted": lambda model: endpoint}, tool_sets={"bookmark-store": bookmarks})
    binding = RunBinding(
        models={"policy": ModelBinding(direct=DirectModel(provider="scripted", model="script"))},
        imports={"bookmarks": ToolBinding(local="bookmark-store")},
    )
    handle = await runner.start(RunSpecification(program=agent_program(Librarian), binding=binding))
    await handle.result()

    assert [tool.name for tool in endpoint.requests[0].tools] == ["search", "bookmark"]   # @tool methods, then imports
    assert list(bookmarks.saved.values()) == ["The Rocky Shore"]


asyncio.run(imported())
```

If a tool set raises, the effect is recorded as failed and the model receives an error result, so every tool call
still gets an answer.

A tool set is shared by the runs it serves. What a run needs for itself alone, made when it starts and deleted when
it ends (a Minecraft world, a container), is a [sandbox](../libraries/rollout/sandboxes.md): the program declares it,
the runner acquires it before the program starts and releases it after, and the program reaches it as
`run.sandbox(name)`, whose operations are recorded effects like imported tools, with no id to pass.

### After a crash

Under the durable runner, a call that a crash interrupted is made again, with the same `effect_id`, when the tool's
`retry_class` is `PURE` or `IDEMPOTENT`, or when its tool set deduplicates: a `DeduplicatingToolSet` whose
`deduplicates` is true performs each `effect_id` at most once, and a tool set served over HTTP reports the attribute
of the one behind it. Otherwise (`SIDE_EFFECTING`, or `UNKNOWN`, which `ToolSpecification` defaults to) it is not
made again: the effect completes as `outcome_unknown`, and the model receives an error result saying the call may or
may not have taken effect ([effects](../libraries/rollout/contracts/effects.md#receivers-that-deduplicate)).

### Serving a tool set over HTTP

`rollout.harness.remote.serve(tool_set)` returns a Starlette application with two routes:

| Route | Body | Answer |
|---|---|---|
| `GET /specifications` | | `specifications`: the tool specifications; `deduplicates`: whether the tool set deduplicates |
| `POST /call` | `name`, `arguments`, `effect_id`, `arguments_digest` | the `ToolResult`; status 500 with `error` when the tool set raised |

`rollout tools FACTORY [--directory DIRECTORY] [--host 127.0.0.1] [--port 8700]` serves the tool set that
`FACTORY` (`module:function`, called with the directory) returns. A run reaches it with `ToolBinding(url=...)`;
task code calls `run.tools` the same way in both cases. A deployment profile names each tool set once
([deploying](deploying.md)).

## What tools are not

- **Tool bodies are task code, not effects.** The durable runner resumes a run by running its code again, so a
  tool body may run more than once. Work that must happen once belongs in effects: model samples, environment
  operations, imported tools.
