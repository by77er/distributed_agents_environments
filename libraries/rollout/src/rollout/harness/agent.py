"""`Agent`: the policy side of the loop (docs/guide/agents.md)."""

from typing import Any

from rollout.contracts import Message, ToolSpecification
from rollout.harness.context import RunContext
from rollout.harness.history import ContextHints, History


class Agent:
    """The policy side of the loop: what the model sees each turn, and how its output becomes one action.

    The default agent samples the policy slot once per turn with the whole history. Subclass it to change the
    context (compaction, windows) or the way it acts (plan-then-act, self-critique).
    """

    system_prompt: str | None = None
    """Sent first in every context when set."""

    def __init__(self, configuration: Any = None) -> None:
        """Must be deterministic: under a durable runner the agent is re-created on replay."""
        self.configuration = configuration

    def select_context(self, history: History, hints: ContextHints) -> list[Message]:
        """What the model sees this turn. Default: the system prompt and the history shaped by the task's hints."""
        system = [Message.system(self.system_prompt)] if self.system_prompt else []
        return system + history.messages(hints)

    async def act(self, run: RunContext, history: History, tools: list[ToolSpecification]) -> Message:
        """Produce one action. Default: one sample of the policy slot."""
        return await run.model.sample(self.select_context(history, run.context_hints), tools=tools)
