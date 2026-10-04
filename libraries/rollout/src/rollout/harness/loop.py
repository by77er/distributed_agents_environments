"""The framework-owned rollout loop (docs/libraries/rollout/README.md#the-loop)."""

from rollout.contracts import Message, Role, ToolResultBlock
from rollout.harness.agent import Agent
from rollout.harness.context import RunContext
from rollout.harness.observation import End, InvalidObservation, Observation
from rollout.harness.task import Task


async def rollout(task: Task, agent: Agent, run: RunContext) -> None:
    """Run one episode of `task` with `agent`. Raises `InvalidObservation` or whatever a hook raised."""
    try:
        await task.setup(run)
        observation = _checked(await task.start(run), reply=None)
        run.record(observation)
        while True:
            if observation.end is not None:
                break
            if task.max_turns is not None and run.turn >= task.max_turns:
                observation = End(truncated=True)
                run.record(observation)
                break
            reply = await agent.act(run, run.history, task.tools_for_turn(run))
            if reply.role is not Role.ASSISTANT:
                raise TypeError(f"{type(agent).__name__}.act returned a {reply.role} message, not an ASSISTANT one")
            observation = _checked(await task.respond(run, reply), reply=reply)
            run.record(observation, reply=reply)
        episode_reward = await task.score(run)
        if episode_reward is not None:
            run.reward(episode_reward)
    finally:
        await task.teardown(run)


def _checked(observation: Observation, *, reply: Message | None) -> Observation:
    """Enforce the validation rules of docs/guide/tasks.md#validation; violations raise `InvalidObservation`."""
    calls = {call.call_id for call in reply.tool_calls} if reply is not None else set[str]()
    answered: set[str] = set()
    for message in observation.messages:
        if message.role not in (Role.USER, Role.TOOL):
            raise InvalidObservation(f"observations contain only USER and TOOL messages, not {message.role}")
        for block in message.content:
            if isinstance(block, ToolResultBlock):
                if block.call_id not in calls:
                    raise InvalidObservation(f"tool result {block.call_id!r} answers no tool call of the reply")
                answered.add(block.call_id)
    if missing := calls - answered:
        raise InvalidObservation(f"tool calls without a result: {sorted(missing)}")
    if observation.end is None and not observation.messages:
        raise InvalidObservation("a non-terminal observation needs at least one message")
    return observation
