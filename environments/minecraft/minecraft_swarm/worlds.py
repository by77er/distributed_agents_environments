"""Temporary worlds for episodes, and the tool set episodes reach them through.

`MinecraftWorlds` starts a Paper server from a template, connects the team's bots, builds a task, runs windows of game
time while actions happen, and reports the score from the plugin's ground truth. Every episode of a GRPO group gets
its own server from the same world seed and layout seed, so they start identically.

`MinecraftTools` exposes it to programs as the imported tool set `minecraft`: every operation is a recorded effect.
"""

import asyncio
import json
import random
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from minecraft_swarm.control import Control
from minecraft_swarm.harness import Harness
from minecraft_swarm.paper import Installation, PaperServer
from minecraft_swarm.prompts import TEAM
from minecraft_swarm.tasks import Built, Start, Task, build, catalog, score, solved
from rollout.core.contracts import RetryClass, Text, ToolResult, ToolSpecification


@dataclass
class EpisodeWorld:
    id: str
    task: Task
    server: PaperServer
    control: Control
    harness: Harness
    built: Built
    team: list[str]
    windows: int = 0


@dataclass
class MinecraftWorlds:
    installation: Installation = field(default_factory=Installation)
    window_ticks: int = 100
    """At most this many game ticks per window (100 = five seconds)."""
    logs: Path | None = None
    tasks: dict[str, Task] = field(default_factory=lambda: {task.id: task for task in catalog()})
    _episodes: dict[str, EpisodeWorld] = field(default_factory=dict[str, EpisodeWorld])

    async def begin(
        self, task_id: str, world_seed: int, layout_seed: int, team: Sequence[str] = TEAM
    ) -> dict[str, Any]:
        task = self.tasks[task_id]
        episode = f"e-{uuid.uuid4().hex[:10]}"
        server = PaperServer(self.installation, seed=world_seed)
        await server.start()
        control = Control(server.control_url)
        harness: Harness | None = None
        try:
            harness = await Harness.start(log=self.logs / f"{episode}.harness.log" if self.logs else None)
            await harness.connect("127.0.0.1", server.port, list(team))
            await control.freeze()
            built = await build(task, control, list(team), random.Random(layout_seed))
            world = EpisodeWorld(episode, task, server, control, harness, built, list(team))
            self._episodes[episode] = world
            await self._run(world, ticks=20, settle=1.0)  # teleports and chunks reach the bots
            await control.baseline()  # advancements the kit granted are not the episode's
        except BaseException:
            if harness is not None:
                await harness.close()
            await control.close()
            await server.stop()
            raise
        return {
            "episode": episode,
            "task": task.model_dump(mode="json"),
            "available_diamonds": built.available_diamonds,
        }

    async def observe(self, episode: str, agent: str) -> dict[str, Any]:
        return await self._world(episode).harness.observe(agent)

    async def act(self, episode: str, agent: str, action: Mapping[str, JsonValue]) -> bool:
        world = self._world(episode)
        await world.harness.thaw()  # physics on while actions start; ticks still frozen
        return await world.harness.act(agent, dict(action))

    async def window(self, episode: str) -> dict[str, Any]:
        """Run game time until every action has finished or the window is over; then freeze. `done` says there is
        nothing left to earn: every staged diamond is held, or the dragon is dead."""
        world = self._world(episode)
        ran = await self._run(world, ticks=self.window_ticks, settle=0.3)
        world.windows += 1
        state = await world.control.state()
        staged = world.task.start in (Start.ITEMS, Start.CHESTS)  # the only starts whose diamonds are counted exactly
        done = bool(state.get("dragon_killed")) or (
            staged and int(state["team_diamonds"]) >= world.built.available_diamonds
        )
        return {
            "ticks": ran,
            "done": done,
            "team_diamonds": int(state["team_diamonds"]),  # for whoever watches; the reward is scored at the end
            "team_advancements": list(state.get("team_advancements", [])),
        }

    async def score(self, episode: str) -> dict[str, Any]:
        """Ground truth: the reward of the task's objective, the team's diamonds and advancements, and what
        happened."""
        world = self._world(episode)
        state = await world.control.state()
        events = await world.control.events()
        kinds = Counter(str(event["kind"]) for event in events)
        mined = Counter(str(event["block"]) for event in events if event["kind"] == "mined")
        return {
            "reward": score(world.task, state),
            "solved": solved(world.task, state),
            "objective": world.task.objective.value,
            "team_diamonds": int(state["team_diamonds"]),
            "team_advancements": list(state.get("team_advancements", [])),
            "dragon_killed": bool(state.get("dragon_killed", False)),
            "dragon_damage": float(state.get("dragon_damage", 0.0)),
            "players": {p["name"]: p["diamonds"] for p in state["players"] if p["name"] in world.team},
            "available_diamonds": world.built.available_diamonds,
            "events": dict(kinds),
            "mined": dict(mined),
        }

    async def end(self, episode: str) -> None:
        world = self._episodes.pop(episode, None)
        if world is None:
            return
        await world.harness.close()
        await world.control.close()
        await world.server.stop()

    async def close(self) -> None:
        await asyncio.gather(*(self.end(episode) for episode in list(self._episodes)), return_exceptions=True)

    async def _run(self, world: EpisodeWorld, *, ticks: int, settle: float) -> int:
        await world.harness.thaw()
        ran = 0
        while ran < ticks:
            await world.control.step(10)
            ran += 10
            if not await world.harness.busy():
                break
        await world.control.step(10)  # drops land and are picked up
        ran += 10
        await asyncio.sleep(settle)
        await world.harness.freeze()
        return ran

    def _world(self, episode: str) -> EpisodeWorld:
        world = self._episodes.get(episode)
        if world is None:
            raise KeyError(f"no episode {episode}")
        return world


def _object(properties: dict[str, JsonValue], required: list[str]) -> dict[str, JsonValue]:
    return {"type": "object", "properties": properties, "required": list[JsonValue](required)}


class MinecraftTools:
    """The `minecraft` tool set: the world operations an episode program performs, each a recorded effect."""

    def __init__(self, worlds: MinecraftWorlds) -> None:
        self.worlds = worlds

    def specifications(self) -> Sequence[ToolSpecification]:
        string: JsonValue = {"type": "string"}
        integer: JsonValue = {"type": "integer"}
        return [
            ToolSpecification(
                name="begin",
                description="Start a world for an episode.",
                input_schema=_object(
                    {"task": string, "world_seed": integer, "layout_seed": integer},
                    ["task", "world_seed", "layout_seed"],
                ),
                retry_class=RetryClass.SIDE_EFFECTING,
            ),
            ToolSpecification(
                name="observe",
                description="What an agent perceives.",
                input_schema=_object({"episode": string, "agent": string}, ["episode", "agent"]),
                retry_class=RetryClass.PURE,
            ),
            ToolSpecification(
                name="act",
                description="Start an agent's action.",
                input_schema=_object(
                    {"episode": string, "agent": string, "action": {"type": "object"}}, ["episode", "agent", "action"]
                ),
                retry_class=RetryClass.SIDE_EFFECTING,
            ),
            ToolSpecification(
                name="window",
                description="Run game time while actions happen.",
                input_schema=_object({"episode": string}, ["episode"]),
                retry_class=RetryClass.SIDE_EFFECTING,
            ),
            ToolSpecification(
                name="score",
                description="The team's diamonds (ground truth).",
                input_schema=_object({"episode": string}, ["episode"]),
                retry_class=RetryClass.PURE,
            ),
            ToolSpecification(
                name="end",
                description="Stop the episode's world.",
                input_schema=_object({"episode": string}, ["episode"]),
                retry_class=RetryClass.IDEMPOTENT,
            ),
        ]

    async def call(
        self, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
        value: JsonValue
        match name:
            case "begin":
                value = await self.worlds.begin(
                    str(arguments["task"]), _int(arguments["world_seed"]), _int(arguments["layout_seed"])
                )
            case "observe":
                value = await self.worlds.observe(str(arguments["episode"]), str(arguments["agent"]))
            case "act":
                action = arguments["action"]
                started = await self.worlds.act(
                    str(arguments["episode"]), str(arguments["agent"]), action if isinstance(action, dict) else {}
                )
                value = {"started": started}
            case "window":
                value = await self.worlds.window(str(arguments["episode"]))
            case "score":
                value = await self.worlds.score(str(arguments["episode"]))
            case "end":
                await self.worlds.end(str(arguments["episode"]))
                value = {"ended": True}
            case _:
                return ToolResult(content=[Text(text=f"unknown operation {name}")], is_error=True)
        return ToolResult(content=[Text(text=json.dumps(value))], structured=value)


def _int(value: JsonValue) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise TypeError(f"expected an integer, got {value!r}")
    return int(value)
