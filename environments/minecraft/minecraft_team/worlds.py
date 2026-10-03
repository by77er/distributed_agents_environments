"""Temporary worlds for episodes, and the tool set episodes reach them through.

`MinecraftWorlds` starts a Paper server from a template, connects the team's bots, builds a task, runs windows of game
time while actions happen, and reports the score from the plugin's ground truth. Every episode gets a server of its
own, and episodes given the same world seed and layout seed start identically.

`MinecraftTools` exposes it to programs as the imported tool set `minecraft`: every operation is a recorded effect.
It is served in the process that runs the episodes (`tools`), or from a machine of its own
(`rollout tools minecraft_team.worlds:tools`), and an episode cannot tell which.
"""

import asyncio
import json
import random
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from minecraft_team.control import Control
from minecraft_team.harness import Harness
from minecraft_team.limits import LIMITS, TICKS_PER_SECOND
from minecraft_team.paper import Installation, PaperServer, sweep
from minecraft_team.tasks import TEAM, Built, Task, build, catalog, saturated, score, solved
from rollout.contracts import RetryClass, Text, ToolResult, ToolSpecification

WINDOW_TICKS = LIMITS.window_seconds * TICKS_PER_SECOND
"""The most game ticks a window runs: it is over sooner when every action has finished."""


@dataclass
class EpisodeWorld:
    id: str
    task: Task
    server: PaperServer
    control: Control
    harness: Harness
    built: Built
    team: list[str]


@dataclass
class MinecraftWorlds:
    installation: Installation = field(default_factory=Installation)
    window_ticks: int = WINDOW_TICKS
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
            await harness.connect("127.0.0.1", server.port, list(team), version=self.installation.version)
            await control.freeze()
            built = await build(task, control, list(team), random.Random(layout_seed))
            world = EpisodeWorld(episode, task, server, control, harness, built, list(team))
            self._episodes[episode] = world
            await self._run(world, ticks=20, settle=1.0)  # teleports reach the bots
            await loaded(control, harness)  # and so does the world around them, before anyone looks at it
            await control.baseline()  # advancements the kit granted are not the episode's
        except BaseException:
            if harness is not None:
                await harness.close()
            await control.close()
            await server.stop()
            raise
        return {"episode": episode}

    async def observe(self, episode: str, agent: str) -> dict[str, Any]:
        return await self._world(episode).harness.observe(agent)

    async def act(self, episode: str, agent: str, action: Mapping[str, JsonValue]) -> bool:
        world = self._world(episode)
        await world.harness.thaw()  # physics on while actions start; ticks still frozen
        return await world.harness.act(agent, dict(action))

    async def window(self, episode: str) -> dict[str, Any]:
        """Run game time until every action has finished or the window is over; then freeze. `done` says there is
        nothing left to earn (`tasks.saturated`)."""
        world = self._world(episode)
        ran = await self._run(world, ticks=self.window_ticks, settle=0.3)
        state = await world.control.state()
        return {
            "ticks": ran,
            "done": saturated(world.task, state, world.built.available_diamonds),
            "team_diamonds": int(state["team_diamonds"]),  # for whoever watches; the reward is scored at the end
            "team_advancements": list(state.get("team_advancements", [])),
        }

    async def score(self, episode: str) -> dict[str, Any]:
        """Ground truth: the reward of the task's objective, the team's diamonds and advancements, and what
        happened."""
        world = self._world(episode)
        state = await world.control.state()
        team = {name.lower() for name in world.team}  # (someone watching is a player too, and not the episode's)
        events = [
            event
            for event in await world.control.events()
            if "player" not in event or str(event["player"]).lower() in team
        ]
        kinds = Counter(str(event["kind"]) for event in events)
        mined = Counter(str(event["block"]) for event in events if event["kind"] == "mined")
        return {
            "reward": score(world.task, state),
            "solved": solved(world.task, state, world.built.available_diamonds, len(world.team)),
            "objective": world.task.objective.value,
            "team_diamonds": int(state["team_diamonds"]),
            "team_advancements": list(state.get("team_advancements", [])),
            "team_obtained": dict(state.get("team_obtained", {})),
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
        try:
            await world.harness.close()
            await world.control.close()
        finally:
            await world.server.stop()

    async def close(self) -> None:
        await asyncio.gather(*(self.end(episode) for episode in list(self._episodes)), return_exceptions=True)

    async def _run(self, world: EpisodeWorld, *, ticks: int, settle: float) -> int:
        return await run_window(world.control, world.harness, ticks=ticks, settle=settle)

    def _world(self, episode: str) -> EpisodeWorld:
        world = self._episodes.get(episode)
        if world is None:
            raise KeyError(f"no episode {episode}")
        return world


TICK_SECONDS = 1 / TICKS_PER_SECOND
POLL_SECONDS = 0.1
DROP_TICKS = 10


async def run_window(control: Control, harness: Harness, *, ticks: int, settle: float) -> int:
    """Run game time while the bots act, then freeze; returns the ticks that ran. The window is over when every
    action has finished; when a bot whose own action has finished is hurt (it should not stand and take it while a
    teammate walks); or when `ticks` have run.

    The game runs at its own pace, twenty ticks a second, and is stopped when the bots are done: a bot's client moves
    and digs in real time, so a game stepped ten ticks at a time, with a pause to ask after each, gave the bots
    1.4 times the time the world had."""
    await harness.thaw()
    started = time.monotonic()
    await control.run(ticks)
    while time.monotonic() - started < ticks * TICK_SECONDS:
        await asyncio.sleep(POLL_SECONDS)
        acting, hurt = await harness.busy()
        if not acting or set(hurt) - set(acting):
            break
    ran = await control.stop()
    await control.step(DROP_TICKS)  # drops land and are picked up
    await harness.freeze()
    await asyncio.sleep(settle)  # the last of what the server sent reaches the bots
    return ran + DROP_TICKS


async def loaded(control: Control, harness: Harness, *, seconds: float = 30.0) -> None:
    """Wait until every bot holds the chunks around it (they are sent as the server ticks its connections)."""
    deadline = time.monotonic() + seconds
    while waiting := await harness.unloaded():
        if time.monotonic() > deadline:
            raise RuntimeError(f"the world around {', '.join(waiting)} did not load in {seconds:.0f} seconds")
        await asyncio.sleep(0.25)


def tools(directory: Path) -> "MinecraftTools":
    """The `minecraft` tool set for a deployment whose state is under `directory` (a profile's `[tools]` names this
    function): worlds on this machine, their logs kept there. Servers a stopped process left behind are removed."""
    worlds = MinecraftWorlds(logs=directory / "logs")
    (directory / "logs").mkdir(parents=True, exist_ok=True)
    sweep(worlds.installation)
    return MinecraftTools(worlds)


Arguments = Mapping[str, JsonValue]
STRING: JsonValue = {"type": "string"}
INTEGER: JsonValue = {"type": "integer"}


@dataclass(frozen=True)
class Operation:
    """An operation of the tool set: what it takes, what may be done with it after a crash, and what performs it."""

    description: str
    takes: Mapping[str, JsonValue]
    """Its arguments, each with its schema; every one is required."""
    retry_class: RetryClass
    perform: Callable[[MinecraftWorlds, Arguments], Awaitable[JsonValue]]


async def _begin(worlds: MinecraftWorlds, arguments: Arguments) -> JsonValue:
    names: Any = arguments["names"]
    team = [str(name) for name in names]
    return await worlds.begin(
        str(arguments["task"]), _int(arguments["world_seed"]), _int(arguments["layout_seed"]), team
    )


async def _observe(worlds: MinecraftWorlds, arguments: Arguments) -> JsonValue:
    return await worlds.observe(str(arguments["episode"]), str(arguments["agent"]))


async def _act(worlds: MinecraftWorlds, arguments: Arguments) -> JsonValue:
    action = arguments["action"]
    started = await worlds.act(
        str(arguments["episode"]), str(arguments["agent"]), action if isinstance(action, dict) else {}
    )
    return {"started": started}


async def _window(worlds: MinecraftWorlds, arguments: Arguments) -> JsonValue:
    return await worlds.window(str(arguments["episode"]))


async def _score(worlds: MinecraftWorlds, arguments: Arguments) -> JsonValue:
    return await worlds.score(str(arguments["episode"]))


async def _end(worlds: MinecraftWorlds, arguments: Arguments) -> JsonValue:
    await worlds.end(str(arguments["episode"]))
    return {"ended": True}


EPISODE: Mapping[str, JsonValue] = {"episode": STRING}
OPERATIONS: dict[str, Operation] = {
    "begin": Operation(
        "Start a world for an episode.",
        {"task": STRING, "world_seed": INTEGER, "layout_seed": INTEGER, "names": {"type": "array", "items": STRING}},
        RetryClass.SIDE_EFFECTING,
        _begin,
    ),
    # (Asked again before the game next runs, the harness answers the same: an observation uses nothing up.)
    "observe": Operation("What an agent perceives.", {**EPISODE, "agent": STRING}, RetryClass.PURE, _observe),
    "act": Operation(
        "Start an agent's action.",
        {**EPISODE, "agent": STRING, "action": {"type": "object"}},
        RetryClass.SIDE_EFFECTING,
        _act,
    ),
    "window": Operation("Run game time while actions happen.", EPISODE, RetryClass.SIDE_EFFECTING, _window),
    "score": Operation("The task's reward, and the ground truth it is scored from.", EPISODE, RetryClass.PURE, _score),
    "end": Operation("Stop the episode's world.", EPISODE, RetryClass.IDEMPOTENT, _end),
}
"""The operations of the tool set `minecraft`, by name: its specifications and what a call does both come from here."""


class MinecraftTools:
    """The `minecraft` tool set: the world operations an episode program performs, each a recorded effect."""

    deduplicates = False
    """An operation asked for twice is performed twice: nothing here remembers an effect's id."""

    def __init__(self, worlds: MinecraftWorlds) -> None:
        self.worlds = worlds

    async def close(self) -> None:
        await self.worlds.close()

    def specifications(self) -> Sequence[ToolSpecification]:
        return [
            ToolSpecification(
                name=name,
                description=operation.description,
                input_schema={
                    "type": "object",
                    "properties": dict(operation.takes),
                    "required": list[JsonValue](operation.takes),
                },
                retry_class=operation.retry_class,
            )
            for name, operation in OPERATIONS.items()
        ]

    async def call(
        self, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
        operation = OPERATIONS.get(name)
        if operation is None:
            return ToolResult(content=[Text(text=f"unknown operation {name}")], is_error=True)
        value = await operation.perform(self.worlds, arguments)
        return ToolResult(content=[Text(text=json.dumps(value))], structured=value)


def _int(value: JsonValue) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise TypeError(f"expected an integer, got {value!r}")
    return int(value)
