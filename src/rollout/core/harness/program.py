"""Programs: what a run executes (docs/core/harness/README.md#program)."""

from rollout.core.harness.agent import Agent
from rollout.core.harness.context import RunContext
from rollout.core.harness.loop import rollout
from rollout.core.harness.task import Task


class Program:
    """What a run executes. `AgentProgram` is the task loop; plain durable workflows are other programs."""

    async def main(self, run: RunContext) -> None:
        raise NotImplementedError


class AgentProgram(Program):
    """The task loop: a task and an agent."""

    def __init__(self, task: Task, agent: Agent) -> None:
        self.task = task
        self.agent = agent

    async def main(self, run: RunContext) -> None:
        await rollout(self.task, self.agent, run)
