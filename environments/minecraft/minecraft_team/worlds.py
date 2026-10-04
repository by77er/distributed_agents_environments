"""Temporary worlds for episodes, as sandboxes of the kind `minecraft`, and the operations episodes perform on them.

`MinecraftWorlds` is a sandbox provider (`rollout.harness.sandboxes.Provider`): for each lease it starts a Paper server
from a template, connects the team's bots and builds the task. Its operations run windows of game time while actions
happen, and report the score from the plugin's ground truth; each one is a recorded effect of the episode that
performs it. Every episode gets a server of its own, and episodes given the same world seed and layout seed start
identically.

A pool over it is served in the process that runs the episodes (`worlds`, named in a profile's `[pools]`), or from a
machine of its own (`rollout pool minecraft_team.worlds:worlds`), and an episode cannot tell which.
"""

import asyncio
import json
import random
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import JsonValue

from minecraft_team.control import Control
from minecraft_team.harness import Harness
from minecraft_team.limits import LIMITS, TICKS_PER_SECOND
from minecraft_team.paper import Installation, PaperServer, sweep
from minecraft_team.tasks import TEAM, Built, Task, build, catalog, saturated, score, solved
from rollout.contracts import RetryClass, Text, ToolResult, ToolSpecification
from rollout.harness import Reach, SandboxSpec

WINDOW_TICKS = LIMITS.window_seconds * TICKS_PER_SECOND
"""The most game ticks a window runs: it is over sooner when every action has finished."""
KIND = "minecraft"
"""The kind of sandbox a world is."""


def world(task: str, world_seed: int, layout_seed: int, names: Sequence[str] = TEAM) -> SandboxSpec:
    """The sandbox an episode of `task` plays in: a world from `world_seed`, the task laid out with `layout_seed`,
    and a bot for each of `names`."""
    parameters: dict[str, JsonValue] = {
        "task": task,
        "world_seed": world_seed,
        "layout_seed": layout_seed,
        "names": list(names),
    }
    return SandboxSpec(kind=KIND, parameters=parameters)


@dataclass
class EpisodeWorld:
    handle: str
    task: Task
    server: PaperServer
    control: Control
    harness: Harness
    built: Built
    team: list[str]


@dataclass
class MinecraftWorlds:
    """At most `size` worlds at once, each a Paper server of its own (1 to 2 GB of memory)."""

    installation: Installation = field(default_factory=Installation)
    window_ticks: int = WINDOW_TICKS
    logs: Path | None = None
    tasks: dict[str, Task] = field(default_factory=lambda: {task.id: task for task in catalog()})
    size: int = 6
    deduplicates: ClassVar[bool] = False
    """An operation asked for twice is performed twice: nothing here remembers an effect's id."""
    _worlds: dict[str, EpisodeWorld] = field(default_factory=dict[str, EpisodeWorld])

    @property
    def kind(self) -> str:
        return KIND

    def operations(self) -> Sequence[ToolSpecification]:
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

    async def create(self, handle: str, spec: SandboxSpec, environment: Mapping[str, str]) -> Reach:
        """Start a world for an episode: `spec.parameters` are its task, world seed, layout seed and names."""
        if handle in self._worlds:
            return self._reach(self._worlds[handle])
        parameters: Any = spec.parameters
        task = self.tasks[str(parameters["task"])]
        team = [str(name) for name in parameters.get("names", TEAM)]
        server = PaperServer(self.installation, seed=int(parameters["world_seed"]))
        await server.start()
        control = Control(server.control_url)
        harness: Harness | None = None
        try:
            harness = await Harness.start(log=self.logs / f"{handle}.harness.log" if self.logs else None)
            await harness.connect("127.0.0.1", server.port, team, version=self.installation.version)
            await control.freeze()
            built = await build(task, control, team, random.Random(int(parameters["layout_seed"])))
            world = EpisodeWorld(handle, task, server, control, harness, built, team)
            await self._run(world, ticks=20, settle=1.0)  # teleports reach the bots
            await loaded(control, harness)  # and so does the world around them, before anyone looks at it
            await control.baseline()  # advancements the kit granted are not the episode's
        except BaseException:
            if harness is not None:
                await harness.close()
            await control.close()
            await server.stop()
            raise
        self._worlds[handle] = world
        return self._reach(world)

    async def delete(self, handle: str) -> None:
        world = self._worlds.pop(handle, None)
        if world is None:
            return
        try:
            await world.harness.close()
            await world.control.close()
        finally:
            await world.server.stop()

    async def held(self) -> Sequence[str]:
        return list(self._worlds)

    async def call(
        self, handle: str, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
        operation = OPERATIONS.get(name)
        if operation is None:
            return ToolResult(content=[Text(text=f"unknown operation {name}")], is_error=True)
        value = await operation.perform(self, self._world(handle), arguments)
        return ToolResult(content=[Text(text=json.dumps(value))], structured=value)

    async def close(self) -> None:
        await asyncio.gather(*(self.delete(handle) for handle in list(self._worlds)), return_exceptions=True)

    async def observe(self, world: EpisodeWorld, agent: str) -> dict[str, Any]:
        return await world.harness.observe(agent)

    async def act(self, world: EpisodeWorld, agent: str, action: Mapping[str, JsonValue]) -> bool:
        await world.harness.thaw()  # physics on while actions start; ticks still frozen
        return await world.harness.act(agent, dict(action))

    async def window(self, world: EpisodeWorld) -> dict[str, Any]:
        """Run game time until every action has finished or the window is over; then freeze. `done` says there is
        nothing left to earn (`tasks.saturated`)."""
        ran = await self._run(world, ticks=self.window_ticks, settle=0.3)
        state = await world.control.state()
        return {
            "ticks": ran,
            "done": saturated(world.task, state, world.built.available_diamonds),
            "team_diamonds": int(state["team_diamonds"]),  # for whoever watches; the reward is scored at the end
            "team_advancements": list(state.get("team_advancements", [])),
        }

    async def score(self, world: EpisodeWorld) -> dict[str, Any]:
        """Ground truth: the reward of the task's objective, the team's diamonds and advancements, and what
        happened."""
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

    async def _run(self, world: EpisodeWorld, *, ticks: int, settle: float) -> int:
        return await run_window(world.control, world.harness, ticks=ticks, settle=settle)

    def _world(self, handle: str) -> EpisodeWorld:
        world = self._worlds.get(handle)
        if world is None:
            raise KeyError(f"no world {handle}")
        return world

    @staticmethod
    def _reach(world: EpisodeWorld) -> Reach:
        """Where a player joins the world (to watch it), and its ground-truth plugin's control API."""
        return Reach(addresses={"game": f"127.0.0.1:{world.server.port}", "control": world.server.control_url})


TICK_SECONDS = 1 / TICKS_PER_SECOND
POLL_SECONDS = 0.1
DROP_TICKS = 10


async def run_window(control: Control, harness: Harness, *, ticks: int, settle: float) -> int:
    """Run game time while the bots act, then freeze; returns the ticks that ran. The window is over when every
    action has finished; when a bot whose own action has finished is hurt (it should not stand and take it while a
    teammate walks); when a bot becomes threatened (idle with a hostile mob closing in, or a creeper about to go off
    beside it: its agent should see it coming, as the first touch of a creeper is its explosion); or when `ticks`
    have run. A threat already there when the window began does not end it: its agent saw it and chose.

    The game runs at its own pace, twenty ticks a second, and is stopped when the bots are done: a bot's client moves
    and digs in real time, so a game stepped ten ticks at a time, with a pause to ask after each, gave the bots
    1.4 times the time the world had."""
    await harness.thaw()
    _, _, already = await harness.busy()
    started = time.monotonic()
    await control.run(ticks)
    while time.monotonic() - started < ticks * TICK_SECONDS:
        await asyncio.sleep(POLL_SECONDS)
        acting, hurt, threatened = await harness.busy()
        if not acting or set(hurt) - set(acting) or set(threatened) - set(already):
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


def worlds(directory: Path, size: int = 6) -> MinecraftWorlds:
    """The worlds of a deployment whose state is under `directory` (a profile's `[pools]` names this function): at
    most `size` at once on this machine, their logs kept there. Servers a stopped process left behind are removed."""
    made = MinecraftWorlds(logs=directory / "logs", size=size)
    (directory / "logs").mkdir(parents=True, exist_ok=True)
    sweep(made.installation)
    return made


Arguments = Mapping[str, JsonValue]
STRING: JsonValue = {"type": "string"}


@dataclass(frozen=True)
class Operation:
    """An operation on a world: what it takes, what may be done with it after a crash, and what performs it."""

    description: str
    takes: Mapping[str, JsonValue]
    """Its arguments, each with its schema; every one is required."""
    retry_class: RetryClass
    perform: Callable[[MinecraftWorlds, EpisodeWorld, Arguments], Awaitable[JsonValue]]


async def _observe(worlds: MinecraftWorlds, world: EpisodeWorld, arguments: Arguments) -> JsonValue:
    return await worlds.observe(world, str(arguments["agent"]))


async def _act(worlds: MinecraftWorlds, world: EpisodeWorld, arguments: Arguments) -> JsonValue:
    action = arguments["action"]
    started = await worlds.act(world, str(arguments["agent"]), action if isinstance(action, dict) else {})
    return {"started": started}


async def _window(worlds: MinecraftWorlds, world: EpisodeWorld, arguments: Arguments) -> JsonValue:
    return await worlds.window(world)


async def _score(worlds: MinecraftWorlds, world: EpisodeWorld, arguments: Arguments) -> JsonValue:
    return await worlds.score(world)


OPERATIONS: dict[str, Operation] = {
    # (Asked again before the game next runs, the harness answers the same: an observation uses nothing up.)
    "observe": Operation("What an agent perceives.", {"agent": STRING}, RetryClass.PURE, _observe),
    "act": Operation(
        "Start an agent's action.", {"agent": STRING, "action": {"type": "object"}}, RetryClass.SIDE_EFFECTING, _act
    ),
    "window": Operation("Run game time while actions happen.", {}, RetryClass.SIDE_EFFECTING, _window),
    "score": Operation("The task's reward, and the ground truth it is scored from.", {}, RetryClass.PURE, _score),
}
"""The operations on a world, by name: their specifications and what a call does both come from here."""
