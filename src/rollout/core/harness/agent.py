"""`Agent`: the policy side of the loop (docs/core/harness/agent.md)."""

from typing import Any

from rollout.core.contracts import Message, ToolSpecification
from rollout.core.harness.context import RunContext
from rollout.core.harness.history import ContextHints, History


class Agent:
    system_prompt: str | None = None

    def __init__(self, configuration: Any = None) -> None:
        self.configuration = configuration

    def select_context(self, history: History, hints: ContextHints) -> list[Message]:
        """What the model sees this turn. Default: the system prompt and the history shaped by the task's hints."""
        system = [Message.system(self.system_prompt)] if self.system_prompt else []
        return system + history.messages(hints)

    async def act(self, run: RunContext, history: History, tools: list[ToolSpecification]) -> Message:
        """Produce one action. Default: one sample of the policy slot."""
        return await run.model.sample(self.select_context(history, run.context_hints), tools=tools)
