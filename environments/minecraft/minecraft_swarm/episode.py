"""The swarm episode: four agents, one shared reward, the world frozen while they think.

Each turn, every agent observes, thinks and calls one action tool, all at once while the world is frozen; then the
world runs one window while the actions happen. The episode ends when its budget of game time or of turns is spent,
or earlier when nothing is left to earn. Its reward, the task's objective scored from the plugin's ground truth,
goes to every agent: the swarm is rewarded equally.

Each agent is a model slot of its own (`ada`, `ben`, `cy`, `dee`): its own context and its own memory. What plays
them is not this program's business: it sends each slot's model messages and tools and gets a message back. The team
talks through the game's chat: an observation shows the last few messages an agent has heard or said, each with
its age in turns.

What an agent remembers is a `rollout.harness.Memory`: the current observation in full, with the map; its recent
turns as `describe(recalled=True)` gives them (what was in sight, what it did, how that went, without the map); and,
of everything older, a summary it wrote itself when its memory was full. The team compacts in the same turn: a turn
waits for its slowest agent, so agents that write their summaries together hold the team up once, and agents that
write them in different turns once each.

The episode's result says how it went, in the game's terms: `solved`, `saturated` (nothing was left to earn),
`duration` (game minutes), `turns`, and the ground truth the reward was scored from.
"""

from collections.abc import Mapping
from typing import Any, cast

from pydantic import JsonValue

from minecraft_swarm.limits import LIMITS, TICKS_PER_SECOND
from minecraft_swarm.prompts import (
    ACTIONS,
    CHAT_LINES,
    COMPACT,
    NO_CALL,
    ONE_CALL,
    REMEMBERED,
    describe,
    describe_result,
    system_prompt,
)
from minecraft_swarm.tasks import TEAM, TURNS_PER_MINUTE, Task, catalog
from rollout.contracts import Message, Text, ToolCall
from rollout.harness import Memory, ModelSlot, Program, RunContext

TICKS_PER_MINUTE = 60 * TICKS_PER_SECOND
SPARE_TURNS = 4
"""An agent compacts along with a teammate whose memory is full if it remembers at least this many turns."""


class SwarmEpisode(Program):
    """Parameters: `task` (an id from the catalog), `world_seed`, `layout_seed`; optionally `minutes` (a shorter
    budget of game time) and `turns` (a lower cap on turns than the task's own)."""

    def __init__(self, parameters: Mapping[str, JsonValue] | None = None) -> None:
        parameters = parameters or {}
        tasks = {task.id: task for task in catalog()}
        self.task: Task = tasks[str(parameters.get("task", "t001"))]
        self.world_seed = int(cast(int, parameters.get("world_seed", 12345)))
        self.layout_seed = int(cast(int, parameters.get("layout_seed", 0)))
        self.minutes = min(self.task.minutes, float(cast(float, parameters.get("minutes", self.task.minutes))))
        turns = parameters.get("turns")
        budget = round(self.minutes * TURNS_PER_MINUTE)  # the task's budget of turns, for the game time it is given
        self.max_turns = budget if turns is None else min(budget, int(cast(int, turns)))
        self.system = Message.system(system_prompt(self.task))
        self.chat: dict[str, list[tuple[int, str, str]]] = {name: [] for name in TEAM}
        """What each agent has heard and said lately: (turn, speaker, message), oldest first."""

    def model_slots(self) -> Mapping[str, ModelSlot]:
        return {name: ModelSlot() for name in TEAM}

    def imports(self) -> list[str]:
        return ["minecraft"]

    async def main(self, run: RunContext) -> None:
        begun = await self._call(
            run, "begin", {"task": self.task.id, "world_seed": self.world_seed, "layout_seed": self.layout_seed}
        )
        episode = str(begun["episode"])
        memories = {name: Memory(prompt=COMPACT, remembered=REMEMBERED) for name in TEAM}
        budget = self.minutes * TICKS_PER_MINUTE
        spent = 0.0
        turn = 0
        try:
            saturated = False
            while spent < budget and turn < self.max_turns:
                turn += 1
                observations = await run.gather(
                    *(self._call(run, "observe", {"episode": episode, "agent": name}) for name in TEAM)
                )
                for name, observation in zip(TEAM, observations, strict=True):
                    answer(memories[name], observation)
                full = [name for name in TEAM if memories[name].crowded(run.models[name])]
                if full:
                    spare = [name for name in TEAM if name in full or len(memories[name].turns) >= SPARE_TURNS]
                    await run.gather(*(memories[name].compact(run.models[name], self.system) for name in spare))
                actions = await run.gather(
                    *(
                        self._think(run, name, turn, observation, memories[name])
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
                    saturated = True
                    break
            score = await self._call(run, "score", {"episode": episode})
        finally:
            await self._call(run, "end", {"episode": episode})
        reward = float(cast(float, score["reward"]))
        for name in TEAM:  # the swarm is rewarded equally
            run.reward(reward, slot=name)
        result = {
            **score,
            "task": self.task.id,
            "turns": turn,
            "duration": spent / TICKS_PER_MINUTE,
            "saturated": saturated,
            "ended": "nothing left to earn" if saturated else "game time" if spent >= budget else "turns",
            "compactions": max(memory.compactions for memory in memories.values()),
        }
        await run.emit("result", result)

    async def _think(
        self, run: RunContext, name: str, turn: int, observation: Mapping[str, Any], memory: Memory
    ) -> dict[str, JsonValue]:
        """One agent's turn: what it sees, one reply, and the action it chose (`idle` if it chose none)."""
        heard = self.chat[name]
        # (What a teammate said reached this agent after the turn it was said in: the turn before this one.)
        heard.extend((turn - 1, str(said["from"]), str(said["message"])) for said in observation.get("messages", []))
        del heard[:-CHAT_LINES]
        chat = [(turn - at, who, message) for at, who, message in heard]
        seen = Message.user(describe(observation, chat=chat))
        reply = await memory.sample(run.models[name], system=self.system, current=[seen], tools=ACTIONS)
        memory.remember(Message.user(describe(observation, recalled=True)), reply)
        calls = reply.tool_calls
        if not calls:
            return {"name": "idle"}
        call: ToolCall = calls[0]  # one action per turn
        said = " ".join(str(call.arguments.get("message", "")).split())[: LIMITS.chat_characters]
        if call.name == "chat" and said:  # an agent sees what it said among what it heard, as the harness says it
            heard.append((turn, name, said))
        return action(call)

    async def _call(self, run: RunContext, operation: str, arguments: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        result = await run.tools.call(operation, arguments)
        if result.is_error or not isinstance(result.structured, dict):
            detail = "".join(part.text for part in result.content if isinstance(part, Text))
            raise RuntimeError(f"minecraft.{operation} failed: {detail}")
        return result.structured


def action(call: ToolCall) -> dict[str, JsonValue]:
    """A tool call as the harness takes an action: its arguments, and its name (whatever an argument is called)."""
    return {**dict(call.arguments), "name": call.name}


def answer(memory: Memory, observation: Mapping[str, Any]) -> None:
    """Close the agent's last remembered turn with how its action went (the observation that follows says)."""
    if memory.turns:
        called = memory.turns[-1][-1].tool_calls
        memory.answer(describe_result(observation.get("last_action")) if called else NO_CALL, others=ONE_CALL)
