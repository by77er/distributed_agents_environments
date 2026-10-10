"""The horizon episode: one to four agents race a budget of game time for as much of an objective as they can get.

It plays as the team package's episode does (`minecraft_team.episode`): each turn every agent observes, thinks and
calls one action tool while the world is frozen, then the world runs one window while the actions happen; the team
talks in chat; each agent remembers its recent turns and a summary of older ones. What differs:

- every observation begins with what is left of the game (`prompts.clock`): game time and turns, both;
- nothing ends the game early, since there is always more to get: it lasts until its game time or its turns are
  spent, or for a speedrun until its goal is reached (`done`), whose time left is what it scores;
- the reward is the objective's amount scored at the end (`objectives.reward`: `log(1 + amount)`), the same for every
  agent of the team.

The result reports the amount, its unit and breakdown, the amount a minute of game time, what the team holds, and the
game time and turns spent (`duration` is turns). `solved` (what unlocks rows) is a speedrun's goal reached, or at least
one of any other objective.
"""

import random
from collections.abc import Mapping
from typing import Any, cast

from pydantic import JsonValue

from minecraft_horizons import worlds
from minecraft_horizons.objectives import Measure
from minecraft_horizons.prompts import clock, system_prompt
from minecraft_horizons.tasks import TASKS, Task
from minecraft_team.episode import TICKS_PER_MINUTE, action, answer, call
from minecraft_team.limits import LIMITS
from minecraft_team.prompts import (
    ACTIONS,
    CHAT_LINES,
    COMPACT,
    COMPACT_ALONE,
    REMEMBERED,
    describe,
)
from minecraft_team.tasks import NAMES, TEAM
from rollout.contracts import Message, ToolCall
from rollout.harness import Memory, ModelSlot, Program, RunContext, SandboxSpec

__all__ = ["HorizonEpisode"]

SPARE_TURNS = 4
"""An agent compacts along with a teammate whose memory is full if it remembers at least this many turns."""


class HorizonEpisode(Program):
    """Parameters: `task` (an id from the catalog), `world_seed`, `layout_seed`; optionally `names` (the names the
    players play under, all different: one to four of them, played by the first that many slots of `TEAM`) or
    `players` (how many play, under names drawn from `NAMES` with the layout seed)."""

    def __init__(self, parameters: Mapping[str, JsonValue] | None = None) -> None:
        parameters = parameters or {}
        self.task: Task = TASKS[str(parameters.get("task", next(iter(TASKS))))]
        self.world_seed = int(cast(int, parameters.get("world_seed", 12345)))
        self.layout_seed = int(cast(int, parameters.get("layout_seed", 0)))
        given = parameters.get("names")
        names = [str(name) for name in cast(list[object], given)] if isinstance(given, list) else []
        if not names:  # (a start that names nobody: names drawn from its layout)
            players = int(cast(int, parameters.get("players", len(TEAM))))
            names = random.Random(self.layout_seed).sample(NAMES, min(max(players, 1), len(TEAM)))
        if not 1 <= len(names) <= len(TEAM) or len(set(names)) != len(names):
            raise ValueError(f"from one to {len(TEAM)} players, all named differently: not {names}")
        self.team = TEAM[: len(names)]
        """The model slots that play: the first of `TEAM` (every slot is declared; the others take no turn)."""
        self.names = dict(zip(self.team, names, strict=True))
        """The name each slot plays under."""
        self.system = Message.system(system_prompt(self.task, names))
        self.tools = ACTIONS if len(names) > 1 else [tool for tool in ACTIONS if tool.name != "chat"]
        self.chat: dict[str, list[tuple[int, str, str]]] = {slot: [] for slot in self.team}
        """What each agent has heard and said lately: (turn, speaker, message), oldest first."""

    def model_slots(self) -> Mapping[str, ModelSlot]:
        return {name: ModelSlot() for name in TEAM}

    def sandboxes(self) -> Mapping[str, SandboxSpec]:
        names = [self.names[slot] for slot in self.team]
        return {"world": worlds.world(self.task.id, self.world_seed, self.layout_seed, names)}

    async def main(self, run: RunContext) -> None:
        world = run.sandbox("world")
        compact = COMPACT if len(self.team) > 1 else COMPACT_ALONE
        memories = {name: Memory(prompt=compact, remembered=REMEMBERED) for name in self.team}
        budget = self.task.minutes * TICKS_PER_MINUTE
        spent = 0.0
        turn = 0
        reached = False
        while spent < budget and turn < self.task.turns:
            turn += 1
            observations = await run.gather(
                *(call(world, "observe", {"agent": self.names[slot]}) for slot in self.team)
            )
            for name, observation in zip(self.team, observations, strict=True):
                answer(memories[name], observation)
            full = [name for name in self.team if memories[name].crowded(run.models[name])]
            if full:
                spare = [name for name in self.team if name in full or len(memories[name].turns) >= SPARE_TURNS]
                await run.gather(*(memories[name].compact(run.models[name], self.system) for name in spare))
            left = clock(spent / TICKS_PER_MINUTE, self.task.minutes, turn, self.task.turns)
            actions = await run.gather(
                *(
                    self._think(run, name, turn, left, observation, memories[name])
                    for name, observation in zip(self.team, observations, strict=True)
                )
            )
            await run.gather(
                *(
                    call(world, "act", {"agent": self.names[name], "action": action})
                    for name, action in zip(self.team, actions, strict=True)
                )
            )
            window = await call(world, "window")
            spent += float(cast(int, window["ticks"]))
            if window.get("done"):  # a speedrun's goal: the sooner, the better
                reached = True
                break
        score = await call(world, "score")
        reward = float(cast(float, score["reward"]))
        for name in self.team:  # the team is rewarded equally
            run.reward(reward, slot=name)
        minutes = spent / TICKS_PER_MINUTE
        amount = float(cast(float, score["amount"]))
        result: dict[str, JsonValue] = {
            **score,
            "task": self.task.id,
            "budget_minutes": self.task.minutes,
            "solved": reached or (amount >= 1.0 and self.task.objective.measure is not Measure.PROGRESS),
            "amount_per_minute": amount / minutes if minutes else 0.0,
            "turns": turn,
            "duration": turn,
            "game_minutes": minutes,
            "ended": "goal reached" if reached else "game time" if spent >= budget else "turns",
            "compactions": max(memory.compactions for memory in memories.values()),
            "team": [self.names[slot] for slot in self.team],
        }
        await run.emit("result", result)

    async def _think(
        self, run: RunContext, name: str, turn: int, left: str, observation: Mapping[str, Any], memory: Memory
    ) -> dict[str, JsonValue]:
        """One agent's turn: what is left of the game and what it sees, one reply, and the action it chose (`idle` if
        it chose none)."""
        heard = self.chat[name]
        # (What a teammate said reached this agent after the turn it was said in: the turn before this one.)
        heard.extend((turn - 1, str(said["from"]), str(said["message"])) for said in observation.get("messages", []))
        del heard[:-CHAT_LINES]
        chat = [(turn - at, who, message) for at, who, message in heard]
        seen = Message.user(f"{left}\n\n{describe(observation, chat=chat if len(self.team) > 1 else None)}")
        reply = await memory.sample(run.models[name], system=self.system, current=[seen], tools=self.tools)
        memory.remember(Message.user(f"{left}\n\n{describe(observation, recalled=True)}"), reply)
        calls = reply.tool_calls
        if not calls:
            return {"name": "idle"}
        chosen: ToolCall = calls[0]  # one action per turn
        said = " ".join(str(chosen.arguments.get("message", "")).split())[: LIMITS.chat_characters]
        if chosen.name == "chat" and said:  # an agent sees what it said among what it heard, as the harness says it
            heard.append((turn, self.names[name], said))
        return action(chosen)
