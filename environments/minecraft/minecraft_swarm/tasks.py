"""The task catalog: about a hundred tasks of increasing difficulty.

Every task has the same goal and reward: the diamonds the team holds when the episode ends. Tasks differ along four
axes:

- **Exploration** (where the diamonds are): loose items in sight, chests out of sight, natural ore in sight, ore
  hidden nearby, ore far away, diamond depth far below, or the surface.
- **Crafting** (how much of the tech tree is done for them): iron pickaxes given, down through ingots, raw iron, a
  stone pickaxe, a wooden pickaxe and logs, to nothing at all.
- **Coordination** (how gear is spread): everyone kitted, one kit for the whole team, or the kit's parts dealt
  across agents so no one can craft alone.
- **Hazards**: peaceful and lit, dark (mobs spawn), or hostile.

`catalog()` generates the tasks; `build()` turns one into a starting state in a live world, from server-side ground
truth (ore positions, safe places to stand) the agents never see. Difficulty here is an estimate for ordering; the
curriculum replaces it with measured success.
"""

import math
import random
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from minecraft_swarm.control import Control

DIAMOND_DEPTH = -58


class Exploration(IntEnum):
    ITEMS_IN_SIGHT = 0
    """Diamond items on the floor of a lit room around the team."""
    CHESTS_NEARBY = 1
    """Diamonds in chests in side rooms, around corners from where the team starts."""
    ORE_IN_SIGHT = 2
    """Natural diamond ore exposed in the wall of the pocket the team starts in."""
    ORE_NEARBY = 3
    """Natural ore hidden in stone 4 to 8 blocks from the start."""
    ORE_FAR = 4
    """The nearest ore 8 to 24 blocks away, in any direction."""
    ABOVE = 5
    """Underground, about 70 blocks above diamond depth."""
    SURFACE = 6
    """On the surface."""


class Crafting(IntEnum):
    NONE = 0
    """Diamonds need no tools (items and chests)."""
    IRON_PICKAXE = 1
    """Iron pickaxes given."""
    INGOTS = 2
    """Iron ingots, sticks and a crafting table: craft the pickaxe."""
    RAW_IRON = 3
    """Raw iron, coal, a furnace, sticks and a table: smelt, then craft."""
    STONE_PICKAXE = 4
    """A stone pickaxe, coal, a furnace, sticks and a table: find and mine iron first."""
    WOODEN_PICKAXE = 5
    """A wooden pickaxe, coal, sticks and a table: stone tools first."""
    LOGS = 6
    """Logs and coal: planks, sticks, a table and wooden tools first."""
    NOTHING = 7
    """Nothing at all (only on the surface, where there are trees)."""


class Coordination(StrEnum):
    KITTED = "kitted"
    """Everyone gets the kit."""
    ONE_KIT = "one_kit"
    """One agent gets the kit; the others start with food and torches only."""
    SPLIT = "split"
    """The kit's parts are dealt across agents: no one can craft alone."""


class Hazards(IntEnum):
    SAFE = 0
    """Peaceful and lit."""
    DARK = 1
    """Easy difficulty and unlit: mobs spawn in the dark."""
    HOSTILE = 2
    """Normal difficulty and unlit."""


class Task(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    exploration: Exploration
    crafting: Crafting
    coordination: Coordination = Coordination.KITTED
    hazards: Hazards = Hazards.SAFE
    apart: bool = False
    """The team starts in different places (only for items and chests)."""
    turns: int
    """Model turns per agent: each turn, every agent acts once and the world runs up to one window."""
    difficulty: float
    """Estimated; the curriculum measures it."""


SUPPLIES = [{"item": "torch", "count": 16}, {"item": "bread", "count": 6}]
"""What every agent gets whatever the task (except `NOTHING`, which gives nothing)."""

KITS: dict[Crafting, list[dict[str, Any]]] = {
    Crafting.NONE: [],
    Crafting.IRON_PICKAXE: [{"item": "iron_pickaxe"}],
    Crafting.INGOTS: [{"item": "iron_ingot", "count": 3}, {"item": "stick", "count": 2}, {"item": "crafting_table"}],
    Crafting.RAW_IRON: [
        {"item": "raw_iron", "count": 3}, {"item": "coal", "count": 2}, {"item": "furnace"},
        {"item": "stick", "count": 2}, {"item": "crafting_table"},
    ],
    Crafting.STONE_PICKAXE: [
        {"item": "stone_pickaxe"}, {"item": "coal", "count": 4}, {"item": "furnace"}, {"item": "stick", "count": 2},
        {"item": "crafting_table"},
    ],
    Crafting.WOODEN_PICKAXE: [
        {"item": "wooden_pickaxe"}, {"item": "coal", "count": 4}, {"item": "stick", "count": 6},
        {"item": "crafting_table"},
    ],
    Crafting.LOGS: [{"item": "oak_log", "count": 8}, {"item": "coal", "count": 4}],
    Crafting.NOTHING: [],
}  # fmt: skip
"""Per agent when `KITTED`; the tech tree's missing steps are the task."""

EXPLORATION_NAMES = {
    Exploration.ITEMS_IN_SIGHT: "diamonds lying in sight",
    Exploration.CHESTS_NEARBY: "diamonds in chests around corners",
    Exploration.ORE_IN_SIGHT: "diamond ore in the wall",
    Exploration.ORE_NEARBY: "diamond ore hidden nearby",
    Exploration.ORE_FAR: "diamond ore far away",
    Exploration.ABOVE: "far above diamond depth",
    Exploration.SURFACE: "from the surface",
}
CRAFTING_NAMES = {
    Crafting.NONE: "no tools needed",
    Crafting.IRON_PICKAXE: "with iron pickaxes",
    Crafting.INGOTS: "craft the pickaxe from ingots",
    Crafting.RAW_IRON: "smelt raw iron first",
    Crafting.STONE_PICKAXE: "find iron with stone pickaxes",
    Crafting.WOODEN_PICKAXE: "from wooden pickaxes",
    Crafting.LOGS: "from logs",
    Crafting.NOTHING: "from nothing",
}


def catalog() -> list[Task]:
    """Every task, ordered by estimated difficulty."""
    specifications: list[dict[str, Any]] = []
    for exploration in (Exploration.ITEMS_IN_SIGHT, Exploration.CHESTS_NEARBY):
        for hazards in (Hazards.SAFE, Hazards.DARK):
            for apart in (False, True):
                specifications.append(
                    {"exploration": exploration, "crafting": Crafting.NONE, "hazards": hazards, "apart": apart}
                )
    mining = (Exploration.ORE_IN_SIGHT, Exploration.ORE_NEARBY, Exploration.ORE_FAR, Exploration.ABOVE)
    tiers = [Crafting(level) for level in range(Crafting.IRON_PICKAXE, Crafting.LOGS + 1)]
    for exploration in mining:
        for crafting in tiers:
            for hazards in (Hazards.SAFE, Hazards.DARK):
                specifications.append({"exploration": exploration, "crafting": crafting, "hazards": hazards})
    for exploration in (Exploration.ORE_IN_SIGHT, Exploration.ORE_NEARBY):
        for crafting in tiers:
            specifications.append(
                {"exploration": exploration, "crafting": crafting, "coordination": Coordination.ONE_KIT}
            )
            if crafting >= Crafting.INGOTS:
                specifications.append(
                    {"exploration": exploration, "crafting": crafting, "coordination": Coordination.SPLIT}
                )
    for crafting in (Crafting.IRON_PICKAXE, Crafting.INGOTS, Crafting.RAW_IRON):
        specifications.append(
            {"exploration": Exploration.ORE_FAR, "crafting": crafting, "coordination": Coordination.ONE_KIT}
        )
    for exploration in (Exploration.ORE_NEARBY, Exploration.ORE_FAR):
        for crafting in (Crafting.IRON_PICKAXE, Crafting.RAW_IRON):
            specifications.append({"exploration": exploration, "crafting": crafting, "hazards": Hazards.HOSTILE})
    for crafting in (
        Crafting.IRON_PICKAXE,
        Crafting.STONE_PICKAXE,
        Crafting.WOODEN_PICKAXE,
        Crafting.LOGS,
        Crafting.NOTHING,
    ):
        for hazards in (Hazards.SAFE, Hazards.DARK):
            specifications.append({"exploration": Exploration.SURFACE, "crafting": crafting, "hazards": hazards})
    specifications.append(
        {"exploration": Exploration.SURFACE, "crafting": Crafting.NOTHING, "hazards": Hazards.HOSTILE}
    )

    tasks = [_task(**specification) for specification in specifications]
    tasks.sort(key=lambda task: (task.difficulty, task.exploration, task.crafting, task.coordination, task.hazards))
    return [task.model_copy(update={"id": f"t{index:03d}"}) for index, task in enumerate(tasks, start=1)]


def _task(
    *,
    exploration: Exploration,
    crafting: Crafting,
    coordination: Coordination = Coordination.KITTED,
    hazards: Hazards = Hazards.SAFE,
    apart: bool = False,
) -> Task:
    steps = max(0, crafting - Crafting.IRON_PICKAXE)
    difficulty = (
        exploration
        + steps
        + 0.75 * hazards
        + {Coordination.KITTED: 0.0, Coordination.ONE_KIT: 0.75, Coordination.SPLIT: 1.0}[coordination]
        + (0.5 if apart else 0.0)
    )
    turns = min(80, 10 + 4 * exploration + 6 * steps + (4 if coordination is not Coordination.KITTED else 0))
    parts = [EXPLORATION_NAMES[exploration], CRAFTING_NAMES[crafting]]
    if coordination is Coordination.ONE_KIT:
        parts.append("one kit for the team")
    elif coordination is Coordination.SPLIT:
        parts.append("parts split across the team")
    if hazards is not Hazards.SAFE:
        parts.append(hazards.name.lower())
    if apart:
        parts.append("starting apart")
    return Task(
        id="",
        title=", ".join(parts),
        exploration=exploration,
        crafting=crafting,
        coordination=coordination,
        hazards=hazards,
        apart=apart,
        turns=turns,
        difficulty=difficulty,
    )


def kits(task: Task, team: list[str], rng: random.Random) -> dict[str, list[dict[str, Any]]]:
    """Each agent's starting inventory."""
    supplies = [] if task.crafting is Crafting.NOTHING else SUPPLIES
    kit = KITS[task.crafting]
    if task.coordination is Coordination.KITTED:
        return {name: [*kit, *supplies] for name in team}
    if task.coordination is Coordination.ONE_KIT:
        holder = rng.choice(team)
        return {name: [*(kit if name == holder else []), *supplies] for name in team}
    order = team[:]
    rng.shuffle(order)
    dealt: dict[str, list[dict[str, Any]]] = {name: list(supplies) for name in team}
    for index, part in enumerate(kit):  # each part to the next agent: whoever crafts needs the others' parts
        dealt[order[index % len(order)]].append(part)
    return dealt


# Building a task in a live world


@dataclass
class Built:
    """A task built in a world: where everyone stands and what they start with."""

    task: Task
    placements: list[dict[str, Any]]
    available_diamonds: int
    """Diamonds placed or within 32 blocks as ore (each ore drops one): what a perfect team could reach soon."""
    anchor: tuple[int, int, int]


class BuildError(RuntimeError):
    """This world (or this spot in it) cannot host the task; try another seed or anchor."""


async def build(task: Task, control: Control, team: list[str], rng: random.Random) -> Built:
    """Prepare the world for `task` and place the team in it (they must be online)."""
    builders = {
        Exploration.ITEMS_IN_SIGHT: _items_in_sight,
        Exploration.CHESTS_NEARBY: _chests_nearby,
        Exploration.ORE_IN_SIGHT: _ore_in_sight,
        Exploration.ORE_NEARBY: _ore_nearby,
        Exploration.ORE_FAR: _ore_far,
        Exploration.ABOVE: _above,
        Exploration.SURFACE: _surface,
    }
    failures: list[str] = []
    for _ in range(8):  # anchors are random; some cannot host the task
        try:
            starts, available, anchor = await builders[task.exploration](task, control, rng)
            if len(starts) < len(team):
                raise BuildError(f"only {len(starts)} places to stand for {len(team)} agents")
            break
        except BuildError as error:
            failures.append(str(error))
    else:
        raise BuildError(f"no place in this world could host {task.id} ({task.title}): {failures}")
    inventories = kits(task, team, rng)
    placements = [
        {"name": name, "x": start[0] + 0.5, "y": start[1], "z": start[2] + 0.5, "kit": inventories[name]}
        for name, start in zip(team, starts, strict=False)
    ]
    setup = {
        "team": team,
        "spawn": {"x": anchor[0], "y": anchor[1], "z": anchor[2]},
        "placements": placements,
        "gamemode": "survival",
        "difficulty": {Hazards.SAFE: "peaceful", Hazards.DARK: "easy", Hazards.HOSTILE: "normal"}[task.hazards],
        "time": 18000 if task.hazards is not Hazards.SAFE and task.exploration is Exploration.SURFACE else 6000,
        "gamerules": {
            "do_daylight_cycle": False,
            "do_weather_cycle": False,
            "keep_inventory": task.hazards is Hazards.SAFE,
        },
    }
    result = await control.episode(setup)
    if result["missing"]:
        raise BuildError(f"not online: {result['missing']}")
    return Built(task, placements, available, anchor)


Start = tuple[int, int, int]
Builder = tuple[list[Start], int, tuple[int, int, int]]


def _anchor(rng: random.Random, y: int = DIAMOND_DEPTH, spread: int = 240) -> tuple[int, int, int]:
    return rng.randint(-spread, spread), y, rng.randint(-spread, spread)


async def _room_starts(control: Control, x: int, y: int, z: int, radius: int) -> list[Start]:
    spots = await control.standing_spots(x, y, z, radius=radius, limit=32)
    return [(spot["x"], spot["y"], spot["z"]) for spot in spots]


async def _items_in_sight(task: Task, control: Control, rng: random.Random) -> Builder:
    x, y, z = _anchor(rng, y=-30)
    await control.carve(x - 4, y, z - 4, width=9, height=3, depth=9, light=task.hazards is Hazards.SAFE)
    total = 0
    for _ in range(rng.randint(3, 5)):  # piles on the floor
        count = rng.randint(1, 4)
        total += await control.drop_items(
            x + rng.randint(-3, 3), y, z + rng.randint(-3, 3), [{"item": "diamond", "count": count}]
        )
    starts = await _room_starts(control, x, y, z, 4)
    if task.apart:  # the corners of the room
        corners = [(x - 3, y, z - 3), (x + 3, y, z - 3), (x - 3, y, z + 3), (x + 3, y, z + 3)]
        starts = [corner for corner in corners if corner in set(starts)] or starts
    return starts, total, (x, y, z)


async def _chests_nearby(task: Task, control: Control, rng: random.Random) -> Builder:
    x, y, z = _anchor(rng, y=-30)
    light = task.hazards is Hazards.SAFE
    await control.carve(x - 3, y, z - 3, width=7, height=3, depth=7, light=light)
    total = 0
    side_rooms: list[Start] = []
    for dx, dz, turn in ((1, 0, (0, 1)), (-1, 0, (0, -1)), (0, 1, (-1, 0)), (0, -1, (1, 0))):
        # a corridor out of the room, a bend, and a small room with a chest: out of sight from the start
        cx, cz = x + dx * 4, z + dz * 4
        await control.carve(
            min(cx, cx + dx * 4),
            y,
            min(cz, cz + dz * 4),
            width=abs(dx) * 5 or 1,
            height=2,
            depth=abs(dz) * 5 or 1,
            light=False,
        )
        bx, bz = cx + dx * 4, cz + dz * 4
        await control.carve(
            min(bx, bx + turn[0] * 4),
            y,
            min(bz, bz + turn[1] * 4),
            width=abs(turn[0]) * 5 or 1,
            height=2,
            depth=abs(turn[1]) * 5 or 1,
            light=False,
        )
        rx, rz = bx + turn[0] * 6, bz + turn[1] * 6
        await control.carve(rx - 2, y, rz - 2, width=5, height=3, depth=5, light=light)
        count = rng.randint(2, 6)
        await control.chest(rx, y, rz, [{"item": "diamond", "count": count}])
        total += count
        side_rooms.append((rx + 1, y, rz + 1))
    starts = side_rooms if task.apart else await _room_starts(control, x, y, z, 3)
    return starts, total, (x, y, z)


async def _ores_near(control: Control, x: int, y: int, z: int, radius: int) -> list[Start]:
    return [(ore["x"], ore["y"], ore["z"]) for ore in await control.ores(x, y, z, radius=radius)]


async def _pocket(control: Control, center: Start, task: Task) -> list[Start]:
    x, y, z = center
    await control.carve(
        x - 2, y, z - 2, width=5, height=3, depth=5, light=task.hazards is Hazards.SAFE, floor="deepslate"
    )
    return await _room_starts(control, x, y, z, 2)


async def _ore_in_sight(task: Task, control: Control, rng: random.Random) -> Builder:
    anchor = _anchor(rng)
    ores = await _ores_near(control, *anchor, radius=24)
    if not ores:
        raise BuildError("no diamond ore here")
    ox, oy, oz = rng.choice(ores)
    # A pocket whose west wall holds the ore at eye level: the ore is exposed and in sight from inside.
    await control.carve(
        ox + 1, oy - 1, oz - 2, width=5, height=3, depth=5, light=task.hazards is Hazards.SAFE, floor="deepslate"
    )
    starts = await _room_starts(control, ox + 3, oy - 1, oz, 2)
    return starts, len(await _ores_near(control, ox, oy, oz, 32)), (ox, oy, oz)


async def _ore_nearby(task: Task, control: Control, rng: random.Random) -> Builder:
    anchor = _anchor(rng)
    ores = await _ores_near(control, *anchor, radius=24)
    if not ores:
        raise BuildError("no diamond ore here")
    target = rng.choice(ores)
    angle = rng.uniform(0, 2 * math.pi)
    distance = rng.uniform(5, 8)
    center = (round(target[0] + distance * math.cos(angle)), target[1], round(target[2] + distance * math.sin(angle)))
    if min(_distance(center, ore) for ore in ores) < 4.5:  # the pocket must not expose any ore
        raise BuildError("an ore would be exposed")
    starts = await _pocket(control, center, task)
    return starts, len(await _ores_near(control, *center, radius=32)), center


async def _ore_far(task: Task, control: Control, rng: random.Random) -> Builder:
    anchor = _anchor(rng)
    ores = await _ores_near(control, *anchor, radius=40)
    for _ in range(40):
        center = (anchor[0] + rng.randint(-16, 16), DIAMOND_DEPTH + rng.randint(0, 6), anchor[2] + rng.randint(-16, 16))
        nearest = min((_distance(center, ore) for ore in ores), default=math.inf)
        if 8 <= nearest <= 24:
            starts = await _pocket(control, center, task)
            return starts, len(await _ores_near(control, *center, radius=32)), center
    raise BuildError("no spot with ore 8 to 24 blocks away")


async def _above(task: Task, control: Control, rng: random.Random) -> Builder:
    anchor = _anchor(rng)
    if not await _ores_near(control, *anchor, radius=24):
        raise BuildError("no diamond ore below")
    center = (anchor[0], DIAMOND_DEPTH + rng.randint(66, 74), anchor[2])
    starts = await _pocket(control, center, task)
    return starts, len(await _ores_near(control, anchor[0], DIAMOND_DEPTH, anchor[2], 32)), center


async def _surface(task: Task, control: Control, rng: random.Random) -> Builder:
    x, z = rng.randint(-160, 160), rng.randint(-160, 160)
    y = await control.surface(x, z) + 1
    if y < 63:
        raise BuildError("under water")
    starts = await _room_starts(control, x, y, z, 6)
    return starts, len(await _ores_near(control, x, DIAMOND_DEPTH, z, 32)), (x, y, z)


def _distance(a: Start, b: Start) -> float:
    return math.dist(a, b)
