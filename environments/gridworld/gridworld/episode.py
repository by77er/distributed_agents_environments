"""The episode: two or more agents on one level, acting at once each turn, rewarded together.

Each agent is a model slot of its own (`agent-1` to `agent-4`, of which the first as many as the start has agents
play) and plays under a name the start's seed draws. Each turn every agent is shown the system prompt and its latest
observation, all agents sample at once (`run.gather`), and the game plays the first tool call of each reply. The
episode ends when every final plate is pressed at once (solved, reward 1 for every agent) or when the turn budget is
spent (reward 0).

The episode's result says `solved`, `duration` (turns played), how it ended, which doors opened and when, the actions
taken by kind (and the moves that failed), and the chat.
"""

from collections.abc import Mapping
from typing import cast

from pydantic import JsonValue

from gridworld.game import Game, names_for
from gridworld.level import generate
from gridworld.prompts import TOOLS, observe, parse, system_prompt
from rollout.contracts import Message
from rollout.harness import ModelSlot, Program, RunContext

__all__ = ["MAX_AGENTS", "TEAM", "TURNS", "GridEpisode"]

MAX_AGENTS = 4
TEAM = [f"agent-{number}" for number in range(1, MAX_AGENTS + 1)]
"""Every model slot the program declares; a start of `n` agents plays the first `n`."""
TURNS = 30
"""The budget of a start that names none."""


class GridEpisode(Program):
    """Parameters: `layout` (of `gridworld.level.LAYOUTS`), `agents` (two to `MAX_AGENTS`), `seed` (draws the level
    and the agents' names) and `turns` (the budget)."""

    def __init__(self, parameters: Mapping[str, JsonValue] | None = None) -> None:
        parameters = parameters or {}
        self.layout = str(parameters.get("layout", "open"))
        self.agents = int(cast(int, parameters.get("agents", 2)))
        if not 2 <= self.agents <= MAX_AGENTS:
            raise ValueError(f"from two to {MAX_AGENTS} agents, not {self.agents}")
        self.seed = int(cast(int, parameters.get("seed", 0)))
        self.turns = int(cast(int, parameters.get("turns", TURNS)))
        self.level = generate(self.layout, self.agents, self.seed)
        self.names = names_for(self.agents, self.seed)
        self.team = TEAM[: self.agents]

    def model_slots(self) -> Mapping[str, ModelSlot]:
        return {slot: ModelSlot() for slot in TEAM}

    def game(self) -> Game:
        return Game(self.level, self.names, self.turns, self.seed)

    async def main(self, run: RunContext) -> None:
        game = self.game()
        system = Message.system(system_prompt(self.level, self.names, self.turns))
        while not game.over:
            seen = [observe(game, agent) for agent in range(self.agents)]
            replies = await run.gather(
                *(
                    run.models[slot].sample([system, Message.user(text)], tools=TOOLS)
                    for slot, text in zip(self.team, seen, strict=True)
                )
            )
            game.step([parse(reply) for reply in replies])
        reward = 1.0 if game.solved else 0.0
        for slot in self.team:  # the team is rewarded together
            run.reward(reward, slot=slot)
        result: dict[str, JsonValue] = {
            "solved": game.solved,
            "duration": game.turn,
            "ended": "every final plate pressed" if game.solved else "turns",
            "layout": self.layout,
            "agents": self.agents,
            "seed": self.seed,
            "team": list(self.names),
            "opened": dict(game.opened),
            "actions": dict(game.counts),
            "chat": [[line.turn, self.names[line.speaker], line.message] for line in game.chat],
            "level": list(self.level.walls),
        }
        await run.emit("result", result)
