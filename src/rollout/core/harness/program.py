"""Programs: what a run executes (docs/core/harness/README.md#program)."""

from collections.abc import Mapping

from rollout.core.harness.agent import Agent
from rollout.core.harness.context import RunContext
from rollout.core.harness.history import ContextHints
from rollout.core.harness.loop import rollout
from rollout.core.harness.task import ModelSlot, Task


class Program:
    """What a run executes. `AgentProgram` is the task loop; plain durable workflows are other programs."""

    def model_slots(self) -> Mapping[str, ModelSlot]:
        """The model slots the program samples; a runner binds an endpoint to each."""
        return {"policy": ModelSlot()}

    def context_hints(self) -> ContextHints:
        return ContextHints()

    async def main(self, run: RunContext) -> None:
        raise NotImplementedError


class AgentProgram(Program):
    """The task loop: a task and an agent."""

    def __init__(self, task: Task, agent: Agent) -> None:
        self.task = task
        self.agent = agent

    def model_slots(self) -> Mapping[str, ModelSlot]:
        return self.task.models

    def context_hints(self) -> ContextHints:
        return self.task.context_hints

    async def main(self, run: RunContext) -> None:
        await rollout(self.task, self.agent, run)
