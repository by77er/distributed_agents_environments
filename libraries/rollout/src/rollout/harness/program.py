"""Programs: what a run executes (docs/libraries/rollout/README.md#program)."""

from collections.abc import Mapping

from rollout.contracts import ToolSpecification
from rollout.harness.agent import Agent
from rollout.harness.context import RunContext
from rollout.harness.history import ContextHints
from rollout.harness.loop import rollout
from rollout.harness.task import ModelSlot, Task


class Program:
    """What a run executes. `AgentProgram` is the task loop; plain durable workflows are other programs."""

    def model_slots(self) -> Mapping[str, ModelSlot]:
        """The model slots the program samples; a runner binds an endpoint to each."""
        return {"policy": ModelSlot()}

    def context_hints(self) -> ContextHints:
        return ContextHints()

    def imports(self) -> list[str]:
        """The imported tool sets the program needs; the run's binding says how each is served."""
        return []

    def tool_specifications(self) -> list[ToolSpecification]:
        """Tools the program itself defines (`@tool` methods), for the run's `tools.resolved` event."""
        return []

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

    def imports(self) -> list[str]:
        return list(self.task.imports)

    def tool_specifications(self) -> list[ToolSpecification]:
        return [declared.specification for declared in self.task.declared_tools.values()]

    async def main(self, run: RunContext) -> None:
        await rollout(self.task, self.agent, run)
