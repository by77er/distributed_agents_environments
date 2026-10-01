"""The swarm episode: four agents, one shared reward, the world frozen while they think.

Each turn, every agent observes, thinks and calls one action tool, all at once while the world is frozen; then the
world runs one window while the actions happen. The episode ends when its budget of game time is spent, or earlier
when nothing is left to earn. Its reward, the task's objective scored from the plugin's ground truth, goes to every
agent: the swarm is rewarded equally.

Each agent is a model slot of its own (`ada`, `ben`, `cy`, `dee`): its own context and its own recorded session.
Bound to the same recorded channel, they are one policy. An agent sees its system prompt and its last few turns;
what it wants to keep longer it writes in its notes (`note`) or on the team board (`post`), which every observation
shows. Those two tools take no game time: an agent that only writes waits that turn.
"""

from collections.abc import Mapping, Sequence
from typing import Any, cast

from pydantic import JsonValue

from minecraft_swarm.prompts import ACTIONS, MAX_BOARD_LINES, MAX_NOTES, MEMORY_ACTIONS, TEAM, describe, system_prompt
from minecraft_swarm.tasks import Task, catalog
from rollout.core.contracts import Message, Role, Text, ToolCall, ToolResult, ToolResultBlock
from rollout.core.harness import ModelSlot, Program, RunContext

HISTORY_TURNS = 4
"""Earlier turns an agent sees besides the current one."""
TICKS_PER_MINUTE = 1200


class SwarmEpisode(Program):
    """Parameters: `task` (an id from the catalog), `world_seed`, `layout_seed`; optionally `minutes` (a shorter
    budget of game time) and `turns` (a cap on turns)."""

    def __init__(self, parameters: Mapping[str, JsonValue] | None = None) -> None:
        parameters = parameters or {}
        tasks = {task.id: task for task in catalog()}
        self.task: Task = tasks[str(parameters.get("task", "t001"))]
        self.world_seed = int(cast(int, parameters.get("world_seed", 12345)))
        self.layout_seed = int(cast(int, parameters.get("layout_seed", 0)))
        self.minutes = min(self.task.minutes, float(cast(float, parameters.get("minutes", self.task.minutes))))
        turns = parameters.get("turns")
        self.max_turns = int(cast(int, turns)) if turns is not None else None
        self.notes: dict[str, str] = {name: "" for name in TEAM}
        self.board: list[str] = []
        self._wrote: set[str] = set()
        """Agents whose last turn was a memory tool (they waited in the world)."""

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
        budget = self.minutes * TICKS_PER_MINUTE
        spent = 0.0
        turn = 0
        try:
            while spent < budget and (self.max_turns is None or turn < self.max_turns):
                turn += 1
                left = (budget - spent) / TICKS_PER_MINUTE
                observations = await run.gather(
                    *(self._call(run, "observe", {"episode": episode, "agent": name}) for name in TEAM)
                )
                actions = await run.gather(
                    *(
                        self._think(run, name, left, observation, histories[name])
                        for name, observation in zip(TEAM, observations, strict=True)
                    )
                )
                await run.gather(
                    *(
                        self._call(run, "act", {"episode": episode, "agent": name, "action": action})
                        for name, action in zip(TEAM, actions, strict=True)
                    )
                )
                window = await self._call(run, "window", {"episode": episode})
                spent += float(cast(int, window["ticks"]))
                if window.get("done"):
                    break
            score = await self._call(run, "score", {"episode": episode})
        finally:
            await self._call(run, "end", {"episode": episode})
        reward = float(cast(float, score["reward"]))
        for name in TEAM:  # the swarm is rewarded equally
            run.reward(reward, slot=name)
        await run.emit(
            "result", {"task": self.task.id, "turns": turn, "game_minutes": spent / TICKS_PER_MINUTE, **score}
        )

    async def _think(
        self, run: RunContext, name: str, minutes_left: float, observation: Mapping[str, Any], history: list[Message]
    ) -> dict[str, JsonValue]:
        """One agent's turn: its context, one sample, and the action it chose (`wait` if it chose none)."""
        wrote = name in self._wrote
        self._wrote.discard(name)
        if wrote:  # the world saw a wait; the agent is told its writing was saved
            observation = {**observation, "last_action": None}
        if history and history[-1].role is Role.ASSISTANT:  # answer the previous call with how it went
            previous = history[-1].tool_calls
            result = "Saved." if wrote else describe_result(observation.get("last_action"))
            if previous:
                answered = ToolResultBlock(call_id=previous[0].call_id, result=ToolResult(content=[Text(text=result)]))
                history.append(Message(role=Role.TOOL, content=[answered]))
            else:
                history.append(Message.user(result))
        text = describe(observation, minutes_left=minutes_left, notes=self.notes[name], board=self.board)
        history.append(Message.user(text))
        context = [Message.system(system_prompt(name, self.task)), *recent(history, HISTORY_TURNS)]
        reply = await run.models[name].sample(context, tools=ACTIONS)
        history.append(reply)
        calls = reply.tool_calls
        if not calls:
            return {"name": "wait"}
        call: ToolCall = calls[0]  # one action per turn
        if call.name in MEMORY_ACTIONS:
            self._remember(name, call)
            self._wrote.add(name)
            return {"name": "wait"}
        return {"name": call.name, **dict(call.arguments)}

    def _remember(self, name: str, call: ToolCall) -> None:
        text = str(call.arguments.get("text", "")).strip()
        if call.name == "note":
            self.notes[name] = text[:MAX_NOTES]
        elif text:
            self.board.append(f"{name}: {text[:200]}")
            del self.board[:-MAX_BOARD_LINES]

    async def _call(self, run: RunContext, operation: str, arguments: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        result = await run.tools.call(operation, arguments)
        if result.is_error or not isinstance(result.structured, dict):
            detail = "".join(part.text for part in result.content if isinstance(part, Text))
            raise RuntimeError(f"minecraft.{operation} failed: {detail}")
        return result.structured


def describe_result(result: Any) -> str:
    if not isinstance(result, dict):
        return "No result."
    outcome = cast(dict[str, Any], result)
    if outcome.get("ok"):
        details = {key: value for key, value in outcome.items() if key not in ("action", "ok")}
        return ", ".join(f"{key}: {item}" for key, item in details.items()) or "Done."
    if outcome.get("interrupted"):
        return "Cut off when the world froze; repeat it to continue."
    return f"Failed: {outcome.get('error', 'unknown error')}"


def recent(history: Sequence[Message], turns: int) -> list[Message]:
    """The last `turns` turns and the current one (a turn starts at an observation: a user message that does not
    follow another user message)."""
    starts = [
        index
        for index, message in enumerate(history)
        if message.role is Role.USER and (index == 0 or history[index - 1].role is not Role.USER)
    ]
    first = starts[-(turns + 1)] if len(starts) > turns else 0
    return list(history[first:])
