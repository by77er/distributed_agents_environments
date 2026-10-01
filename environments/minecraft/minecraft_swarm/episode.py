"""The swarm episode: four agents, one shared reward, the world frozen while they think.

Each turn, every agent observes, thinks and calls one action tool, all at once while the world is frozen; then the
world runs one window while the actions happen. At the end, the team's diamonds (ground truth from the plugin) are
the reward of every agent: the swarm is rewarded equally.

Each agent is a model slot of its own (`ada`, `ben`, `cy`, `dee`): its own context and its own recorded session.
Bound to the same recorded channel, they are one policy. An agent sees its system prompt and its last few turns.
"""

from collections.abc import Mapping, Sequence
from typing import Any, cast

from pydantic import JsonValue

from minecraft_swarm.prompts import ACTIONS, TEAM, describe, system_prompt
from minecraft_swarm.tasks import Task, catalog
from rollout.core.contracts import Message, Role, Text, ToolCall, ToolResult, ToolResultBlock
from rollout.core.harness import ModelSlot, Program, RunContext

HISTORY_TURNS = 4
"""Earlier turns an agent sees besides the current one."""


class SwarmEpisode(Program):
    """Parameters: `task` (an id from the catalog), `world_seed`, `layout_seed`; optionally `turns`."""

    def __init__(self, parameters: Mapping[str, JsonValue] | None = None) -> None:
        parameters = parameters or {}
        tasks = {task.id: task for task in catalog()}
        self.task: Task = tasks[str(parameters.get("task", "t001"))]
        self.world_seed = int(cast(int, parameters.get("world_seed", 12345)))
        self.layout_seed = int(cast(int, parameters.get("layout_seed", 0)))
        self.turns = int(cast(int, parameters.get("turns", self.task.turns)))

    def model_slots(self) -> Mapping[str, ModelSlot]:
        return {name: ModelSlot() for name in TEAM}

    def imports(self) -> list[str]:
        return ["minecraft"]

    async def main(self, run: RunContext) -> None:
        begun = await self._call(
            run, "begin", {"task": self.task.id, "world_seed": self.world_seed, "layout_seed": self.layout_seed}
        )
        episode = str(begun["episode"])
        histories: dict[str, list[Message]] = {name: [] for name in TEAM}
        try:
            for turn in range(1, self.turns + 1):
                observations = await run.gather(
                    *(self._call(run, "observe", {"episode": episode, "agent": name}) for name in TEAM)
                )
                actions = await run.gather(
                    *(
                        self._think(run, name, turn, observation, histories[name])
                        for name, observation in zip(TEAM, observations, strict=True)
                    )
                )
                await run.gather(
                    *(
                        self._call(run, "act", {"episode": episode, "agent": name, "action": action})
                        for name, action in zip(TEAM, actions, strict=True)
                    )
                )
                await self._call(run, "window", {"episode": episode})
            score = await self._call(run, "score", {"episode": episode})
        finally:
            await self._call(run, "end", {"episode": episode})
        diamonds = float(cast(int, score["team_diamonds"]))
        for name in TEAM:  # the swarm is rewarded equally
            run.reward(diamonds, slot=name)
        await run.emit("result", {"task": self.task.id, "turns": self.turns, **score})

    async def _think(
        self, run: RunContext, name: str, turn: int, observation: Mapping[str, Any], history: list[Message]
    ) -> dict[str, JsonValue]:
        """One agent's turn: its context, one sample, and the action it chose (`wait` if it chose none)."""
        if history and history[-1].role is Role.ASSISTANT:  # answer the previous call with how it went
            previous = history[-1].tool_calls
            result = describe_result(observation.get("last_action"))
            if previous:
                history.append(
                    Message(
                        role=Role.TOOL,
                        content=[
                            ToolResultBlock(call_id=previous[0].call_id, result=ToolResult(content=[Text(text=result)]))
                        ],
                    )
                )
            else:
                history.append(Message.user(result))
        history.append(Message.user(describe(observation, turn, self.turns)))
        context = [Message.system(system_prompt(name, self.turns)), *recent(history, HISTORY_TURNS)]
        reply = await run.models[name].sample(context, tools=ACTIONS)
        history.append(reply)
        calls = reply.tool_calls
        if not calls:
            return {"name": "wait"}
        call: ToolCall = calls[0]  # one action per turn
        return {"name": call.name, **dict(call.arguments)}

    async def _call(self, run: RunContext, operation: str, arguments: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        result = await run.tools.call(operation, arguments)
        if result.is_error or not isinstance(result.structured, dict):
            raise RuntimeError(
                f"minecraft.{operation} failed: {''.join(p.text for p in result.content if isinstance(p, Text))}"
            )
        return result.structured


def describe_result(result: Any) -> str:
    if not isinstance(result, dict):
        return "No result yet."
    outcome = cast(dict[str, Any], result)
    if outcome.get("ok"):
        return json_brief({key: value for key, value in outcome.items() if key not in ("action", "ok")}) or "Done."
    if outcome.get("interrupted"):
        return "Cut off when the world froze; repeat it to continue."
    return f"Failed: {outcome.get('error', 'unknown error')}"


def json_brief(value: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}: {item}" for key, item in value.items())


def recent(history: Sequence[Message], turns: int) -> list[Message]:
    """The last `turns` turns (each starts at an observation, a user message that follows a tool result or nothing)."""
    starts = [
        index
        for index, message in enumerate(history)
        if message.role is Role.USER and (index == 0 or history[index - 1].role is not Role.USER)
    ]
    first = starts[-(turns + 1)] if len(starts) > turns else 0
    return list(history[first:])
