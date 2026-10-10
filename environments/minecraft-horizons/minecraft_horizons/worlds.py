"""Temporary worlds for episodes, as sandboxes of the kind `minecraft-horizons`, and the operations episodes perform on
them.

`HorizonWorlds` is a sandbox provider (`rollout.harness.sandboxes.Provider`) built from the team package's parts: for
each lease a Paper server from a template, the team's bots, and the task's setting laid out by the team package's
builders (`minecraft_team.tasks.build`). Once the world is set up it notes what the team holds, so that what the team
began with does not count. Its operations are the team package's (observe, act, run a window of game time) and a
score: the objective's amount from the plugin's ground truth (`minecraft_horizons.objectives.measured`). Nothing ends
a game early: there is always more to get.

A pool over it is served in the process that runs the episodes, or from a machine of its own (`rollout pool
minecraft_horizons.worlds:worlds`), and an episode cannot tell which.
"""

import asyncio
import json
import random
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import JsonValue

from minecraft_horizons.objectives import holdings, measured, reward
from minecraft_horizons.tasks import TASKS, Task
from minecraft_team.control import Control
from minecraft_team.harness import Harness
from minecraft_team.paper import HEAP, Installation, PaperServer, sweep
from minecraft_team.tasks import TEAM, build
from minecraft_team.worlds import WINDOW_TICKS, loaded, run_window
from rollout.contracts import RetryClass, Text, ToolResult, ToolSpecification
from rollout.harness import Reach, SandboxSpec

__all__ = ["KIND", "OPERATIONS", "HorizonWorlds", "world", "worlds"]

KIND = "minecraft-horizons"
"""The kind of sandbox a world is."""
Arguments = Mapping[str, JsonValue]


def world(task: str, world_seed: int, layout_seed: int, names: Sequence[str] = TEAM) -> SandboxSpec:
    """The sandbox an episode of `task` plays in: a world from `world_seed`, the setting laid out with `layout_seed`,
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
    team: list[str]
    held_at_start: dict[str, float]
    """What the team held once the world was set up (its kit): not counted."""


@dataclass
class HorizonWorlds:
    """At most `size` worlds at once, each a Paper server of its own whose heap is at most `heap`, and the Node harness
    of its bots (1.1 to 2.4 GiB of memory a world: docs/research/minecraft-memory.md)."""

    installation: Installation = field(default_factory=Installation)
    logs: Path | None = None
    size: int = 6
    heap: str = HEAP
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
        task = TASKS[str(parameters["task"])]
        team = [str(name) for name in parameters.get("names", TEAM)]
        server = PaperServer(self.installation, seed=int(parameters["world_seed"]), heap=self.heap)
        await server.start()
        control = Control(server.control_url)
        harness: Harness | None = None
        try:
            log = self.logs / f"{handle}.harness.log" if self.logs else None
            harness = await Harness.start(log=log, installation=self.installation)
            await harness.connect("127.0.0.1", server.port, team, version=self.installation.version)
            await control.freeze()
            await build(task.laid_out(), control, team, random.Random(int(parameters["layout_seed"])))
            await run_window(control, harness, ticks=20, settle=1.0)  # teleports reach the bots
            await loaded(control, harness)  # and so does the world around them, before anyone looks at it
            await control.baseline()  # advancements the kit granted are not the episode's, nor containers placed
            held_at_start = holdings(await control.holdings())
            world = EpisodeWorld(handle, task, server, control, harness, team, held_at_start)
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

    # The operations (`OPERATIONS`), each given the episode's world and the call's arguments

    async def observe(self, world: EpisodeWorld, arguments: Arguments) -> JsonValue:
        return await world.harness.observe(str(arguments["agent"]))

    async def act(self, world: EpisodeWorld, arguments: Arguments) -> JsonValue:
        action = arguments["action"]
        await world.harness.thaw()  # physics on while actions start; ticks still frozen
        started = await world.harness.act(str(arguments["agent"]), dict(action) if isinstance(action, dict) else {})
        return {"started": started}

    async def window(self, world: EpisodeWorld, arguments: Arguments) -> JsonValue:
        """Run game time until every action has finished or the window is over; then freeze."""
        return {"ticks": await run_window(world.control, world.harness, ticks=WINDOW_TICKS, settle=0.3)}

    async def score(self, world: EpisodeWorld, arguments: Arguments) -> JsonValue:
        """Ground truth: the objective's amount (`objectives.measured`), its reward and what it is made of, everything
        the team holds, the advancements it earned, and what happened."""
        state = await world.control.state()
        found = await world.control.holdings()
        held = holdings(found)
        advancements = [str(name) for name in state.get("team_advancements", [])]
        amount = measured(world.task.objective, held, world.held_at_start, advancements)
        team = {name.lower() for name in world.team}  # (someone watching is a player too, and not the episode's)
        events = [
            event
            for event in await world.control.events()
            if "player" not in event or str(event["player"]).lower() in team
        ]
        scored: dict[str, JsonValue] = {
            "reward": reward(amount.amount),
            "amount": amount.amount,
            "amount_parts": dict[str, JsonValue](amount.parts),
            "objective": world.task.objective.id,
            "unit": world.task.objective.unit,
            "held": dict(found.get("held", {})),
            "stored": dict(found.get("stored", {})),
            "containers": int(found.get("containers", 0)),
            "held_at_start": dict[str, JsonValue](world.held_at_start),
            "team_advancements": list[JsonValue](advancements),
            "team_obtained": dict(state.get("team_obtained", {})),
            "events": dict[str, JsonValue](Counter(str(event["kind"]) for event in events)),
            "mined": dict[str, JsonValue](Counter(str(event["block"]) for event in events if event["kind"] == "mined")),
        }
        return scored

    def _world(self, handle: str) -> EpisodeWorld:
        world = self._worlds.get(handle)
        if world is None:
            raise KeyError(f"no world {handle}")
        return world

    @staticmethod
    def _reach(world: EpisodeWorld) -> Reach:
        """Where a player joins the world (to watch it), and its ground-truth plugin's control API."""
        return Reach(addresses={"game": f"127.0.0.1:{world.server.port}", "control": world.server.control_url})


def worlds(directory: Path, size: int = 6, heap: str = HEAP) -> HorizonWorlds:
    """The worlds of a deployment whose state is under `directory` (the cluster config's
    `[sandboxes.minecraft-horizons] provider` names this function, and its `size` and `heap` are this function's): at
    most `size` at once on this machine, each server's heap at most `heap`, their logs kept there. Servers a stopped
    process left behind are removed."""
    made = HorizonWorlds(logs=directory / "logs", size=size, heap=heap)
    (directory / "logs").mkdir(parents=True, exist_ok=True)
    sweep(made.installation)
    return made


STRING: JsonValue = {"type": "string"}


@dataclass(frozen=True)
class Operation:
    """An operation on a world: what it takes, what may be done with it after a crash, and what performs it."""

    description: str
    takes: Mapping[str, JsonValue]
    """Its arguments, each with its schema; every one is required."""
    retry_class: RetryClass
    perform: Callable[[HorizonWorlds, EpisodeWorld, Arguments], Awaitable[JsonValue]]


OPERATIONS: dict[str, Operation] = {
    "observe": Operation("What an agent perceives.", {"agent": STRING}, RetryClass.PURE, HorizonWorlds.observe),
    "act": Operation(
        "Start an agent's action.",
        {"agent": STRING, "action": {"type": "object"}},
        RetryClass.SIDE_EFFECTING,
        HorizonWorlds.act,
    ),
    "window": Operation("Run game time while actions happen.", {}, RetryClass.SIDE_EFFECTING, HorizonWorlds.window),
    "score": Operation(
        "The objective's amount, its reward, and the ground truth it is measured from.",
        {},
        RetryClass.PURE,
        HorizonWorlds.score,
    ),
}
"""The operations on a world, by name: their specifications and what a call does both come from here."""
