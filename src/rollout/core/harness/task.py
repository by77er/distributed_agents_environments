"""`Task`: the environment an agent acts in, in the reinforcement-learning sense (docs/core/harness/task.md)."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, ClassVar

from rollout.core.contracts import Message, Role, Text, ToolResult, ToolSpecification
from rollout.core.harness.context import RunContext
from rollout.core.harness.conversations import Envelope
from rollout.core.harness.history import ContextHints
from rollout.core.harness.observation import End, Observation, WaitFor
from rollout.core.harness.tools import DeclaredTool, collect_tools, error_result, execute_tool, tool_message


@dataclass(frozen=True)
class ModelSlot:
    """A model the task declares. The agent acts through `policy`; other slots (a simulated user, an opponent) are
    sampled by the task itself."""

    trainable: bool = True


class Task:
    models: ClassVar[dict[str, ModelSlot]] = {"policy": ModelSlot()}
    imports: ClassVar[list[str]] = []
    """External tool sets, bound per run."""
    max_turns: ClassVar[int | None] = None
    """The loop truncates the episode after this many model turns."""
    context_hints: ClassVar[ContextHints] = ContextHints()

    declared_tools: ClassVar[dict[str, DeclaredTool]] = {}
    """The `@tool` methods of this class, collected when the class is defined."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.declared_tools = collect_tools(cls)

    def __init__(self, parameters: Any = None) -> None:
        self.parameters = parameters

    # Lifecycle hooks

    async def setup(self, run: RunContext) -> None:
        """Once per run, before `start`."""

    async def start(self, run: RunContext) -> Observation | WaitFor:
        """Open the episode, or wait for the first message."""
        raise NotImplementedError(f"{type(self).__name__} must implement start")

    async def respond(self, run: RunContext, reply: Message) -> Observation | WaitFor:
        """The environment's step. Default: execute the reply's tool calls; end the episode when there are none."""
        return await self.run_tools(run, reply)

    async def resume(self, run: RunContext, envelope: Envelope) -> Observation | WaitFor:
        """Turn a message that satisfied a `WaitFor`, or interrupted a turn, into the next observation."""
        return Observation(_as_user_message(envelope))

    async def steer(self, run: RunContext, envelopes: list[Envelope], observation: Observation) -> Observation:
        """Merge messages delivered with mode `STEER` into the next observation. Default: append as USER content."""
        messages = observation.messages + tuple(_as_user_message(envelope) for envelope in envelopes)
        return Observation(messages, reward=observation.reward, end=observation.end, info=observation.info)

    async def score(self, run: RunContext) -> float | None:
        """Episode-level reward, attached to the end of the trajectory."""
        return None

    async def teardown(self, run: RunContext) -> None:
        """Always runs if `setup` began; must be idempotent."""

    # Tools

    def tools_for_turn(self, run: RunContext) -> list[ToolSpecification]:
        """The tools offered this turn. Default: every declared tool."""
        return [declared.specification for declared in self.declared_tools.values()]

    async def run_tools(self, run: RunContext, reply: Message) -> Observation:
        """Execute every tool call in `reply` concurrently; one TOOL message answers them all."""
        calls = reply.tool_calls
        if not calls:
            return End()
        results = await asyncio.gather(*(self._call(call.name, call.arguments) for call in calls))
        return Observation(tool_message(calls, results))

    async def _call(self, name: str, arguments: Any) -> ToolResult:
        declared = self.declared_tools.get(name)
        if declared is None:
            return error_result(f"unknown tool {name!r}")
        return await execute_tool(self, declared, arguments)


def _as_user_message(envelope: Envelope) -> Message:
    content = list(envelope.content)
    if not content and envelope.data is not None:
        content = [Text(text=json.dumps(envelope.data, ensure_ascii=False))]
    return Message(role=Role.USER, content=content)
