"""The framework-owned rollout loop (docs/core/harness/README.md#the-loop-normative)."""

import asyncio

from rollout.core.contracts import Message, Role, ToolResultBlock
from rollout.core.harness.agent import Agent
from rollout.core.harness.context import Interrupted, RunContext
from rollout.core.harness.observation import End, InvalidObservation, Observation, WaitFor
from rollout.core.harness.task import Task

UNLOAD = "rollout: unload"
"""The message of a cancellation that unloads a waiting run from memory without ending it (no `teardown`)."""


async def rollout(task: Task, agent: Agent, run: RunContext) -> None:
    """Run one episode of `task` with `agent`. Raises `InvalidObservation` or whatever a hook raised."""
    unloading = False
    try:
        await task.setup(run)
        observation = _checked(await task.start(run), reply=None)
        run.record(observation)
        while True:
            if isinstance(observation, WaitFor):
                envelope = await run.wait_for_message(observation)
                if envelope is None:
                    observation = observation.on_timeout
                else:
                    observation = _checked(await task.resume(run, envelope), reply=None)
                run.record(observation)
                continue
            if observation.end is not None:
                break
            if task.max_turns is not None and run.turn >= task.max_turns:
                observation = End(truncated=True)
                run.record(observation)
                break
            try:
                reply = await run.interruptible(agent.act(run, run.history, task.tools_for_turn(run)))
            except Interrupted as interruption:
                observation = _checked(await task.resume(run, interruption.envelope), reply=None)
                run.record(observation)
                continue
            if reply.role is not Role.ASSISTANT:
                raise TypeError(f"{type(agent).__name__}.act returned a {reply.role} message, not an ASSISTANT one")
            observation = _checked(await task.respond(run, reply), reply=reply)
            if isinstance(observation, Observation) and observation.end is None:
                steering = await run.take_steering_messages()
                if steering:
                    observation = _checked(await task.steer(run, steering, observation), reply=reply)
            run.record(observation, reply=reply)
        episode_reward = await task.score(run)
        if episode_reward is not None:
            run.reward(episode_reward)
    except asyncio.CancelledError as cancelled:
        unloading = cancelled.args == (UNLOAD,)  # an evicted run has not ended: it resumes by replay
        raise
    finally:
        if not unloading:
            await task.teardown(run)


def _checked(observation: Observation | WaitFor, *, reply: Message | None) -> Observation | WaitFor:
    """Enforce the validation rules of docs/core/harness/task.md; violations raise `InvalidObservation`."""
    calls = {call.call_id for call in reply.tool_calls} if reply is not None else set[str]()
    if isinstance(observation, WaitFor):
        if calls:
            raise InvalidObservation("a reply with tool calls must be answered by an Observation, not a WaitFor")
        return observation
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
