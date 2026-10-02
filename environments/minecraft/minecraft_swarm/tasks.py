"""The task catalog: from staged skills to beating the game, in three tiers.

Every task is a starting state, a budget of game time and an objective scored from ground truth. The swarm shares the
reward. Three tiers:

- **Skills** (staged or safe): the plugin builds the situation from ground truth: diamonds lying in a lit room, chests
  around corners, natural ore exposed in a pocket's wall or hidden nearby. Kits remove steps of the tech tree.
  Objective: the diamonds the team holds at the end. Also here, the **crafting** tasks: among trees on a peaceful
  surface with nothing at all, make an item whose recipe is several steps deep, gathering every material (a crafting
  table; a stone pickaxe; a furnace; torches; an iron pickaxe). Objective: the steps of the chain the team got done.
- **Survival** (natural): a natural world, a real day and night, mobs and no kept inventory. Nothing is staged; only
  the start (a cave, the surface, the nether, beside a fortress or a stronghold, the end) and the kit decide where
  along the game the task begins. Objectives: diamonds held, or progress toward the dragon.
- **Game**: a bare spawn on the surface, nothing given. Objective: progress, ending with the dragon.

Progress is scored from Minecraft's own advancements, earned by any team member after the episode's start
(advancements a kit grants are not counted): each milestone along the path to the dragon has a weight.

`catalog()` generates the tasks; `build()` turns one into a starting state in a live world.
"""

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from minecraft_swarm.control import Control

DIAMOND_DEPTH = -58
TURNS_PER_MINUTE = 12
"""A task's budget of turns for each minute of its budget of game time: what the minute would take if every turn ran
its full five seconds. Turns are what cost real time (each is a round of thinking), and a turn whose actions end
quickly spends little game time: without this, four minutes of game time ran to over a hundred turns."""

MILESTONES: dict[str, float] = {
    "story/mine_stone": 1,
    "story/upgrade_tools": 1,
    "story/smelt_iron": 2,
    "story/iron_tools": 2,
    "story/mine_diamond": 4,
    "story/form_obsidian": 3,
    "story/enter_the_nether": 6,
    "nether/find_fortress": 6,
    "nether/obtain_blaze_rod": 8,
    "story/follow_ender_eye": 10,
    "story/enter_the_end": 12,
    "end/kill_dragon": 40,
}
"""Advancements along the path to the dragon and what each is worth."""
DRAGON_DAMAGE = 20.0
"""For a dragon left alive: this times the most it was hurt, as a share of its health."""


class Tier(StrEnum):
    SKILLS = "skills"
    SURVIVAL = "survival"
    GAME = "game"


class Objective(StrEnum):
    DIAMONDS = "diamonds"
    """The diamonds the team holds at the end (a diamond block counts 9)."""
    PROGRESS = "progress"
    """The weights of the milestones the team earned."""
    CRAFT = "craft"
    """The weights of the steps toward the task's item that the team got done: each thing gathered, crafted or
    smelted along the way, once."""


class Start(StrEnum):
    ITEMS = "items"
    """Staged: a lit room with diamond items on the floor."""
    CHESTS = "chests"
    """Staged: a room whose corridors bend toward chests of diamonds."""
    ORE_IN_SIGHT = "ore_in_sight"
    """Staged: a pocket at diamond depth whose wall exposes natural diamond ore."""
    ORE_NEARBY = "ore_nearby"
    """Staged: a pocket 4 to 8 blocks from hidden ore."""
    ORE_FAR = "ore_far"
    """Staged: a pocket 8 to 24 blocks from the nearest ore."""
    CAVE = "cave"
    """Natural: standing in a natural cave underground."""
    SURFACE = "surface"
    """Natural: on the surface."""
    WOODLAND = "woodland"
    """Natural: on the surface, with trees a few steps away."""
    NETHER = "nether"
    """Natural: in the nether, as if just through a portal."""
    FORTRESS = "fortress"
    """Natural: in the nether, beside a fortress."""
    STRONGHOLD_AREA = "stronghold_area"
    """Natural: on the surface, a few hundred blocks from a stronghold."""
    PORTAL_ROOM = "portal_room"
    """Natural: inside a stronghold, beside the end portal frame."""
    END = "end"
    """Natural: on the end's arrival platform, the dragon alive."""


class Kit(StrEnum):
    NONE = "none"
    IRON = "iron"
    """Iron pickaxe, sword and axe."""
    INGOTS = "ingots"
    """Iron ingots, sticks and a crafting table: craft the pickaxe."""
    RAW_IRON = "raw_iron"
    """Raw iron, coal, a furnace, sticks and a table: smelt, then craft."""
    STONE = "stone"
    """Stone pickaxe and sword, coal, a furnace, sticks and a table: find iron first."""
    WOODEN = "wooden"
    """Wooden pickaxe, sticks and a table."""
    NOTHING = "nothing"
    """Nothing at all, not even food."""
    NETHER_READY = "nether_ready"
    """Iron armor and tools, 14 obsidian and flint and steel: build and light a portal."""
    OBSIDIAN_MAKER = "obsidian_maker"
    """Iron armor, a diamond pickaxe, buckets and flint and steel: make the obsidian from water and lava."""
    FORTRESS_READY = "fortress_ready"
    """Iron armor, sword, bow and arrows, food and blocks: fight blazes."""
    EYES_READY = "eyes_ready"
    """Iron armor and tools and twelve eyes of ender: find the stronghold."""
    END_READY = "end_ready"
    """Diamond armor and sword, a bow, arrows, blocks, food and a water bucket: fight the dragon."""
    END_IRON = "end_iron"
    """The same with iron instead of diamond."""


class Coordination(StrEnum):
    KITTED = "kitted"
    """Everyone gets the kit."""
    ONE_KIT = "one_kit"
    """One agent gets the kit; the others start with food and torches only."""
    SPLIT = "split"
    """The kit's parts are dealt across agents: no one can finish alone."""


class Hazards(IntEnum):
    SAFE = 0
    """Peaceful, lit, inventory kept on death."""
    EASY = 1
    """Easy difficulty, unlit: mobs spawn in the dark."""
    NORMAL = 2
    HARD = 3


class Task(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    tier: Tier
    objective: Objective
    start: Start
    kit: Kit
    coordination: Coordination = Coordination.KITTED
    hazards: Hazards = Hazards.SAFE
    apart: bool = False
    """The team starts in different places (staged rooms only)."""
    minutes: float
    """Budget of game time (it passes only while actions happen). The task also ends after `turns` turns."""

    @property
    def turns(self) -> int:
        """Budget of turns."""
        return round(self.minutes * TURNS_PER_MINUTE)

    goal: str | None = None
    """What counts as solving the task: for a progress task the milestone it is about, for a crafting task the item
    to make."""
    difficulty: float
    """Estimated, for ordering; the curriculum measures it."""


def _stacks(*items: str | tuple[str, int]) -> list[dict[str, Any]]:
    return [{"item": item} if isinstance(item, str) else {"item": item[0], "count": item[1]} for item in items]


ARMOR_IRON = _stacks("iron_helmet", "iron_chestplate", "iron_leggings", "iron_boots")
ARMOR_DIAMOND = _stacks("diamond_helmet", "diamond_chestplate", "diamond_leggings", "diamond_boots")
SUPPLIES = _stacks(("torch", 32), ("bread", 8))
"""What every agent gets whatever the kit (except `NOTHING`)."""

KITS: dict[Kit, list[dict[str, Any]]] = {
    Kit.NONE: [],
    Kit.IRON: _stacks("iron_pickaxe", "iron_sword", "iron_axe"),
    Kit.INGOTS: _stacks(("iron_ingot", 3), ("stick", 2), "crafting_table"),
    Kit.RAW_IRON: _stacks(("raw_iron", 3), ("coal", 2), "furnace", ("stick", 2), "crafting_table"),
    Kit.STONE: _stacks("stone_pickaxe", "stone_sword", ("coal", 4), "furnace", ("stick", 4), "crafting_table"),
    Kit.WOODEN: _stacks("wooden_pickaxe", ("stick", 6), "crafting_table"),
    Kit.NOTHING: [],
    Kit.NETHER_READY: [*ARMOR_IRON, *_stacks("iron_pickaxe", "iron_sword", ("obsidian", 14), "flint_and_steel")],
    Kit.OBSIDIAN_MAKER: [
        *ARMOR_IRON,
        *_stacks("diamond_pickaxe", "iron_sword", ("bucket", 2), "water_bucket", "flint_and_steel"),
    ],
    Kit.FORTRESS_READY: [
        *ARMOR_IRON,
        *_stacks("iron_sword", "iron_pickaxe", "bow", ("arrow", 64), ("cobblestone", 64)),
    ],
    Kit.EYES_READY: [*ARMOR_IRON, *_stacks("iron_sword", "iron_pickaxe", ("ender_eye", 12), ("cobblestone", 64))],
    Kit.END_READY: [
        *ARMOR_DIAMOND,
        *_stacks("diamond_sword", "diamond_pickaxe", "bow", ("arrow", 128), ("cobblestone", 128), "water_bucket"),
    ],
    Kit.END_IRON: [
        *ARMOR_IRON,
        *_stacks("iron_sword", "iron_pickaxe", "bow", ("arrow", 128), ("cobblestone", 128), "water_bucket"),
    ],
}
"""Per agent when `KITTED`, besides `SUPPLIES`."""

START_DIFFICULTY = {
    Start.ITEMS: 0,
    Start.CHESTS: 1,
    Start.ORE_IN_SIGHT: 2,
    Start.ORE_NEARBY: 3,
    Start.ORE_FAR: 4,
    Start.CAVE: 6,
    Start.SURFACE: 7,
    Start.WOODLAND: 1,
    Start.FORTRESS: 10,
    Start.PORTAL_ROOM: 11,
    Start.NETHER: 12,
    Start.STRONGHOLD_AREA: 13,
    Start.END: 14,
}
KIT_STEPS = {
    Kit.NONE: 0,
    Kit.IRON: 0,
    Kit.INGOTS: 1,
    Kit.RAW_IRON: 2,
    Kit.STONE: 3,
    Kit.WOODEN: 4,
    Kit.NOTHING: 6,
    Kit.NETHER_READY: 0,
    Kit.OBSIDIAN_MAKER: 2,
    Kit.FORTRESS_READY: 0,
    Kit.EYES_READY: 0,
    Kit.END_READY: 0,
    Kit.END_IRON: 2,
}
"""Steps of the tech tree (or of preparation) a kit leaves to the team."""
GOALS = {
    Start.SURFACE: "story/enter_the_nether",
    Start.NETHER: "nether/find_fortress",
    Start.FORTRESS: "nether/obtain_blaze_rod",
    Start.STRONGHOLD_AREA: "story/follow_ender_eye",
    Start.PORTAL_ROOM: "story/enter_the_end",
    Start.END: "end/kill_dragon",
}
"""The milestone a survival task with the progress objective is about, by where it starts."""
TIER_ORDER = {Tier.SKILLS: 0, Tier.SURVIVAL: 1, Tier.GAME: 2}


Step = tuple[str, tuple[str, ...], float]
"""A step toward an item: its name, the items that show it was done (any of them; a trailing `*` matches any
beginning, as in `*_log`), and its weight."""

WOOD: list[Step] = [("logs", ("*_log",), 1), ("planks", ("*_planks",), 1)]
TABLE: list[Step] = [*WOOD, ("a crafting table", ("crafting_table",), 2)]
WOODEN_PICKAXE: list[Step] = [*TABLE, ("sticks", ("stick",), 1), ("a wooden pickaxe", ("wooden_pickaxe",), 3)]
STONE_PICKAXE: list[Step] = [
    *WOODEN_PICKAXE,
    ("cobblestone", ("cobblestone", "cobbled_deepslate"), 2),
    ("a stone pickaxe", ("stone_pickaxe",), 3),
]
FURNACE: list[Step] = [*STONE_PICKAXE, ("a furnace", ("furnace",), 3)]
IRON: list[Step] = [*FURNACE, ("raw iron", ("raw_iron",), 4), ("an iron ingot", ("iron_ingot",), 5)]
IRON_PICKAXE: list[Step] = [*IRON, ("an iron pickaxe", ("iron_pickaxe",), 6)]
CHAINS: dict[str, list[Step]] = {
    "crafting_table": TABLE,
    "wooden_pickaxe": WOODEN_PICKAXE,
    "stone_pickaxe": STONE_PICKAXE,
    "furnace": FURNACE,
    "torch": [*FURNACE, ("coal or charcoal", ("coal", "charcoal"), 3), ("torches", ("torch",), 3)],
    "iron_pickaxe": IRON_PICKAXE,
    "bucket": [*IRON, ("a bucket", ("bucket",), 6)],
    "shield": [*IRON, ("a shield", ("shield",), 6)],
    "diamond": [*IRON_PICKAXE, ("a diamond", ("diamond",), 8)],
    "diamond_pickaxe": [*IRON_PICKAXE, ("diamonds", ("diamond",), 8), ("a diamond pickaxe", ("diamond_pickaxe",), 10)],
}
"""Crafting tasks: for each item to make, the steps from nothing that lead to it. Everything must be gathered. The
longest run from wood to diamonds: the surface is far above diamond depth, so they include the way down."""
EARLY: list[Step] = [
    ("logs", ("*_log",), 0.5),
    ("planks", ("*_planks",), 0.5),
    ("a crafting table", ("crafting_table",), 1),
    ("a wooden pickaxe", ("wooden_pickaxe",), 1),
]
"""The first steps of the game, which have no advancement of their own: they count toward progress for a team that
starts with nothing."""
CRAFT_MINUTES = {
    "crafting_table": 6,
    "wooden_pickaxe": 8,
    "stone_pickaxe": 12,
    "furnace": 14,
    "torch": 20,
    "iron_pickaxe": 35,
    "bucket": 35,
    "shield": 35,
    "diamond": 60,
    "diamond_pickaxe": 70,
}


def done(steps: Sequence[Step], obtained: Mapping[str, Any]) -> list[str]:
    """The names of the steps that `obtained` (what the team got hold of, by item) shows were done."""

    def matches(pattern: str) -> bool:
        if pattern.startswith("*"):
            return any(item.endswith(pattern[1:]) for item in obtained)
        return pattern in obtained

    return [name for name, items, _ in steps if any(matches(item) for item in items)]


def catalog() -> list[Task]:
    """Every task, ordered by estimated difficulty."""
    specifications: list[dict[str, Any]] = []

    def add(tier: Tier, objective: Objective, start: Start, kit: Kit, minutes: float, **options: Any) -> None:
        entry = {"tier": tier, "objective": objective, "start": start, "kit": kit, "minutes": minutes, **options}
        specifications.append(entry)

    skills, survival, game = Tier.SKILLS, Tier.SURVIVAL, Tier.GAME
    diamonds, progress = Objective.DIAMONDS, Objective.PROGRESS

    # Skills: staged situations, gear given, diamonds held.
    for start in (Start.ITEMS, Start.CHESTS):
        add(skills, diamonds, start, Kit.NONE, 3)
        add(skills, diamonds, start, Kit.NONE, 4, hazards=Hazards.EASY)
        add(skills, diamonds, start, Kit.NONE, 5, apart=True)
    for kit in (Kit.IRON, Kit.INGOTS, Kit.RAW_IRON, Kit.STONE):
        add(skills, diamonds, Start.ORE_IN_SIGHT, kit, 4 + 2 * KIT_STEPS[kit])
        add(skills, diamonds, Start.ORE_NEARBY, kit, 6 + 2 * KIT_STEPS[kit])
    for kit in (Kit.IRON, Kit.INGOTS):
        add(skills, diamonds, Start.ORE_IN_SIGHT, kit, 6 + 2 * KIT_STEPS[kit], coordination=Coordination.ONE_KIT)
    for kit in (Kit.INGOTS, Kit.RAW_IRON):
        add(skills, diamonds, Start.ORE_IN_SIGHT, kit, 6 + 2 * KIT_STEPS[kit], coordination=Coordination.SPLIT)
    add(skills, diamonds, Start.ORE_NEARBY, Kit.IRON, 8, coordination=Coordination.ONE_KIT)
    for kit in (Kit.IRON, Kit.STONE):
        add(skills, diamonds, Start.ORE_FAR, kit, 10 + 2 * KIT_STEPS[kit])

    # Crafting: nothing given, a peaceful surface among trees; the item's whole chain is to be gathered and made.
    for item, minutes in CRAFT_MINUTES.items():
        add(skills, Objective.CRAFT, Start.WOODLAND, Kit.NOTHING, minutes, goal=item)

    # Survival: natural worlds; nothing is staged.
    for hazards in (Hazards.EASY, Hazards.NORMAL):
        for kit in (Kit.IRON, Kit.STONE, Kit.WOODEN):
            add(survival, diamonds, Start.CAVE, kit, 20 + 5 * KIT_STEPS[kit], hazards=hazards)
        for kit in (Kit.IRON, Kit.STONE, Kit.WOODEN, Kit.NOTHING):
            add(survival, diamonds, Start.SURFACE, kit, 30 + 6 * KIT_STEPS[kit], hazards=hazards)
    add(survival, diamonds, Start.CAVE, Kit.IRON, 25, hazards=Hazards.EASY, coordination=Coordination.ONE_KIT)
    add(survival, diamonds, Start.CAVE, Kit.RAW_IRON, 30, hazards=Hazards.EASY, coordination=Coordination.SPLIT)
    for hazards in (Hazards.EASY, Hazards.NORMAL):
        add(survival, progress, Start.SURFACE, Kit.NETHER_READY, 20, hazards=hazards)
    add(survival, progress, Start.SURFACE, Kit.OBSIDIAN_MAKER, 40, hazards=Hazards.NORMAL)
    add(survival, progress, Start.FORTRESS, Kit.FORTRESS_READY, 30, hazards=Hazards.NORMAL)
    add(survival, progress, Start.NETHER, Kit.FORTRESS_READY, 60, hazards=Hazards.NORMAL)
    add(survival, progress, Start.STRONGHOLD_AREA, Kit.EYES_READY, 40, hazards=Hazards.NORMAL)
    add(survival, progress, Start.PORTAL_ROOM, Kit.EYES_READY, 15, hazards=Hazards.NORMAL)
    add(survival, progress, Start.END, Kit.END_READY, 30, hazards=Hazards.NORMAL)
    add(survival, progress, Start.END, Kit.END_IRON, 40, hazards=Hazards.NORMAL)

    # The game.
    for hazards in (Hazards.EASY, Hazards.NORMAL, Hazards.HARD):
        add(game, progress, Start.SURFACE, Kit.NOTHING, 240, hazards=hazards)

    tasks = [_task(**specification) for specification in specifications]
    tasks.sort(key=lambda task: (TIER_ORDER[task.tier], task.difficulty, task.start, task.kit, task.coordination))
    return [task.model_copy(update={"id": f"t{index:03d}"}) for index, task in enumerate(tasks, start=1)]


def _task(
    *,
    tier: Tier,
    objective: Objective,
    start: Start,
    kit: Kit,
    minutes: float,
    coordination: Coordination = Coordination.KITTED,
    hazards: Hazards = Hazards.SAFE,
    apart: bool = False,
    goal: str | None = None,
) -> Task:
    steps = len(CHAINS[goal]) / 2 if objective is Objective.CRAFT and goal else KIT_STEPS[kit]
    difficulty = (
        START_DIFFICULTY[start]
        + steps
        + 0.75 * hazards
        + {Coordination.KITTED: 0.0, Coordination.ONE_KIT: 0.75, Coordination.SPLIT: 1.0}[coordination]
        + (1.0 if apart else 0.0)
    )
    if objective is Objective.PROGRESS:
        goal = "end/kill_dragon" if tier is Tier.GAME else GOALS[start]
    parts = [tier.value, objective.value, start.value.replace("_", " "), f"kit {kit.value.replace('_', ' ')}"]
    if objective is Objective.CRAFT and goal:
        parts = [tier.value, f"craft {goal.replace('_', ' ')}", "from nothing"]
    if coordination is not Coordination.KITTED:
        parts.append(coordination.value.replace("_", " "))
    if hazards is not Hazards.SAFE:
        parts.append(hazards.name.lower())
    if apart:
        parts.append("apart")
    return Task(
        id="",
        title=", ".join(parts),
        tier=tier,
        objective=objective,
        start=start,
        kit=kit,
        coordination=coordination,
        hazards=hazards,
        apart=apart,
        minutes=minutes,
        goal=goal,
        difficulty=difficulty,
    )


def kits(task: Task, team: Sequence[str], rng: random.Random) -> dict[str, list[dict[str, Any]]]:
    """Each agent's starting inventory."""
    supplies = [] if task.kit is Kit.NOTHING else SUPPLIES
    kit = KITS[task.kit]
    if task.coordination is Coordination.KITTED:
        return {name: [*kit, *supplies] for name in team}
    if task.coordination is Coordination.ONE_KIT:
        holder = rng.choice(list(team))
        return {name: [*(kit if name == holder else []), *supplies] for name in team}
    order = list(team)
    rng.shuffle(order)
    dealt: dict[str, list[dict[str, Any]]] = {name: list(supplies) for name in team}
    for index, part in enumerate(kit):  # each part to the next agent: whoever crafts needs the others' parts
        dealt[order[index % len(order)]].append(part)
    return dealt


def score(task: Task, state: Mapping[str, Any]) -> float:
    """The episode's reward from the plugin's ground truth (`Control.state()`)."""
    if task.objective is Objective.DIAMONDS:
        return float(state["team_diamonds"])
    if task.objective is Objective.CRAFT:
        steps = CHAINS[str(task.goal)]
        made = set(done(steps, state.get("team_obtained", {})))
        return float(sum(weight for name, _, weight in steps if name in made))
    earned = set(state.get("team_advancements", []))
    reward = float(sum(weight for key, weight in MILESTONES.items() if key in earned))
    if task.kit is Kit.NOTHING:  # from nothing, the steps before the first advancement count too
        made = set(done(EARLY, state.get("team_obtained", {})))
        reward += sum(weight for name, _, weight in EARLY if name in made)
    if "end/kill_dragon" not in earned:  # hurting the dragon counts for something
        reward += DRAGON_DAMAGE * float(state.get("dragon_damage", 0.0))
    return reward


def solved(task: Task, state: Mapping[str, Any]) -> bool:
    """Whether the team did what the task is about: holds a diamond, made the task's item, or earned its milestone."""
    if task.objective is Objective.DIAMONDS:
        return int(state["team_diamonds"]) > 0
    if task.objective is Objective.CRAFT:
        return str(task.goal) in state.get("team_obtained", {})
    return task.goal in set(state.get("team_advancements", []))


# Building a task in a live world

Point = tuple[int, int, int]


@dataclass
class Site:
    """Where a task starts: places to stand, and what there is to find."""

    starts: list[Point]
    available_diamonds: int
    """Diamonds placed, or natural diamond ore within 32 blocks (each ore drops one)."""
    anchor: Point
    world: str = "world"


@dataclass
class Built:
    """A task built in a world: where everyone stands and what they start with."""

    task: Task
    placements: list[dict[str, Any]]
    available_diamonds: int
    anchor: Point
    world: str


class BuildError(RuntimeError):
    """This world (or this spot in it) cannot host the task; try another seed or anchor."""


DIFFICULTY = {Hazards.SAFE: "peaceful", Hazards.EASY: "easy", Hazards.NORMAL: "normal", Hazards.HARD: "hard"}


async def build(task: Task, control: Control, team: list[str], rng: random.Random) -> Built:
    """Prepare the world for `task` and place the team in it (they must be online)."""
    builders = {
        Start.ITEMS: _items,
        Start.CHESTS: _chests,
        Start.ORE_IN_SIGHT: _ore_in_sight,
        Start.ORE_NEARBY: _ore_nearby,
        Start.ORE_FAR: _ore_far,
        Start.CAVE: _cave,
        Start.SURFACE: _surface,
        Start.WOODLAND: _woodland,
        Start.NETHER: _nether,
        Start.FORTRESS: _fortress,
        Start.STRONGHOLD_AREA: _stronghold_area,
        Start.PORTAL_ROOM: _portal_room,
        Start.END: _end,
    }
    failures: list[str] = []
    for _ in range(8):  # anchors are random; some cannot host the task
        try:
            site = await builders[task.start](task, control, rng)
            if len(site.starts) < len(team):
                raise BuildError(f"only {len(site.starts)} places to stand for {len(team)} agents")
            break
        except BuildError as error:
            failures.append(str(error))
    else:
        raise BuildError(f"no place in this world could host {task.id} ({task.title}): {failures}")
    inventories = kits(task, team, rng)
    placements = [
        {"name": name, "world": site.world, "x": x + 0.5, "y": y, "z": z + 0.5, "kit": inventories[name]}
        for name, (x, y, z) in zip(team, site.starts, strict=False)
    ]
    natural = task.tier is not Tier.SKILLS
    setup = {
        "team": team,  # each agent respawns where it is placed
        "spawn": {"world": site.world, "x": site.anchor[0], "y": site.anchor[1], "z": site.anchor[2]},
        "placements": placements,
        "gamemode": "survival",
        "difficulty": DIFFICULTY[task.hazards],
        "time": 1000,
        "gamerules": {
            "do_daylight_cycle": natural,
            "do_weather_cycle": natural,
            "keep_inventory": task.hazards is Hazards.SAFE,
        },
    }
    result = await control.episode(setup)
    if result["missing"]:
        raise BuildError(f"not online: {result['missing']}")
    return Built(task, placements, site.available_diamonds, site.anchor, site.world)


def _anchor(rng: random.Random, y: int = DIAMOND_DEPTH, spread: int = 240) -> Point:
    return rng.randint(-spread, spread), y, rng.randint(-spread, spread)


async def _spots(control: Control, point: Point, radius: int, world: str = "world", limit: int = 32) -> list[Point]:
    spots = await control.standing_spots(*point, radius=radius, limit=limit, world=world)
    return [(spot["x"], spot["y"], spot["z"]) for spot in spots]


def _spread(spots: Sequence[Point], count: int = 4) -> list[Point]:
    """Distinct spots a little apart (two agents cannot stand in one block)."""
    chosen: list[Point] = []
    for spot in spots:
        if all(math.dist(spot, other) >= 1.0 for other in chosen):
            chosen.append(spot)
        if len(chosen) == count:
            break
    return chosen


async def _ores_near(control: Control, point: Point, radius: int) -> list[Point]:
    return [(ore["x"], ore["y"], ore["z"]) for ore in await control.ores(*point, radius=radius)]


async def _ore_count(control: Control, point: Point) -> int:
    return len(await control.ores(point[0], DIAMOND_DEPTH, point[2], radius=32))


# Staged starts (skills)


async def _items(task: Task, control: Control, rng: random.Random) -> Site:
    x, y, z = _anchor(rng, y=-30)
    light = task.hazards is Hazards.SAFE
    if task.apart:  # diamonds in the middle room; each agent starts in its own side room, around a corner
        rooms = await _room_with_side_rooms(control, (x, y, z), light)
        total = await _drop_piles(control, (x, y, z), rng, spread=2)
        return Site([(rx + 1, y, rz + 1) for rx, rz in rooms], total, (x, y, z))
    await control.carve(x - 4, y, z - 4, width=9, height=3, depth=9, light=light)
    total = await _drop_piles(control, (x, y, z), rng, spread=3)
    return Site(_spread(await _spots(control, (x, y, z), 4)), total, (x, y, z))


async def _chests(task: Task, control: Control, rng: random.Random) -> Site:
    x, y, z = _anchor(rng, y=-30)
    rooms = await _room_with_side_rooms(control, (x, y, z), task.hazards is Hazards.SAFE)
    total = 0
    for rx, rz in rooms:
        count = rng.randint(2, 6)
        await control.chest(rx, y, rz, [{"item": "diamond", "count": count}])
        total += count
    # Apart: each agent in a side room of its own, beside a chest; the rest are found by exploring.
    starts = [(rx + 1, y, rz + 1) for rx, rz in rooms] if task.apart else _spread(await _spots(control, (x, y, z), 3))
    return Site(starts, total, (x, y, z))


async def _room_with_side_rooms(control: Control, center: Point, light: bool) -> list[tuple[int, int]]:
    """A 7 by 7 room; a corridor out of each side bends to a 5 by 5 room. Returns the side rooms' centres."""
    x, y, z = center
    await control.carve(x - 3, y, z - 3, width=7, height=3, depth=7, light=light)
    rooms: list[tuple[int, int]] = []
    for dx, dz, turn in ((1, 0, (0, 1)), (-1, 0, (0, -1)), (0, 1, (-1, 0)), (0, -1, (1, 0))):
        cx, cz = x + dx * 4, z + dz * 4  # out of the room
        await _corridor(control, (cx, y, cz), (dx, dz))
        bx, bz = cx + dx * 4, cz + dz * 4  # the bend
        await _corridor(control, (bx, y, bz), turn)
        rx, rz = bx + turn[0] * 6, bz + turn[1] * 6
        await control.carve(rx - 2, y, rz - 2, width=5, height=3, depth=5, light=light)
        rooms.append((rx, rz))
    return rooms


async def _corridor(control: Control, start: Point, direction: tuple[int, int]) -> None:
    """Five blocks long, one wide, two high, from `start` in `direction`."""
    x, y, z = start
    dx, dz = direction
    await control.carve(
        min(x, x + dx * 4), y, min(z, z + dz * 4), width=abs(dx) * 5 or 1, height=2, depth=abs(dz) * 5 or 1, light=False
    )


async def _drop_piles(control: Control, center: Point, rng: random.Random, *, spread: int) -> int:
    x, y, z = center
    total = 0
    for _ in range(rng.randint(3, 5)):
        pile = [{"item": "diamond", "count": rng.randint(1, 4)}]
        total += await control.drop_items(x + rng.randint(-spread, spread), y, z + rng.randint(-spread, spread), pile)
    return total


async def _pocket(control: Control, center: Point, task: Task) -> list[Point]:
    x, y, z = center
    light = task.hazards is Hazards.SAFE
    await control.carve(x - 2, y, z - 2, width=5, height=3, depth=5, light=light, floor="deepslate")
    return _spread(await _spots(control, center, 2))


async def _ore_in_sight(task: Task, control: Control, rng: random.Random) -> Site:
    ores = await _ores_near(control, _anchor(rng), 24)
    if not ores:
        raise BuildError("no diamond ore here")
    ox, oy, oz = rng.choice(ores)
    # A pocket whose west wall holds the ore at eye level: the ore is exposed and in sight from inside.
    light = task.hazards is Hazards.SAFE
    await control.carve(ox + 1, oy - 1, oz - 2, width=5, height=3, depth=5, light=light, floor="deepslate")
    starts = _spread(await _spots(control, (ox + 3, oy - 1, oz), 2))
    return Site(starts, await _ore_count(control, (ox, oy, oz)), (ox, oy, oz))


async def _ore_nearby(task: Task, control: Control, rng: random.Random) -> Site:
    ores = await _ores_near(control, _anchor(rng), 24)
    if not ores:
        raise BuildError("no diamond ore here")
    target = rng.choice(ores)
    angle, distance = rng.uniform(0, 2 * math.pi), rng.uniform(5, 8)
    center = (round(target[0] + distance * math.cos(angle)), target[1], round(target[2] + distance * math.sin(angle)))
    if min(math.dist(center, ore) for ore in ores) < 4.5:  # the pocket must not expose any ore
        raise BuildError("an ore would be exposed")
    return Site(await _pocket(control, center, task), await _ore_count(control, center), center)


async def _ore_far(task: Task, control: Control, rng: random.Random) -> Site:
    anchor = _anchor(rng)
    ores = await _ores_near(control, anchor, 40)
    for _ in range(40):
        center = (anchor[0] + rng.randint(-16, 16), DIAMOND_DEPTH + rng.randint(0, 6), anchor[2] + rng.randint(-16, 16))
        if 8 <= min((math.dist(center, ore) for ore in ores), default=math.inf) <= 24:
            return Site(await _pocket(control, center, task), await _ore_count(control, center), center)
    raise BuildError("no spot with ore 8 to 24 blocks away")


# Natural starts (survival and the game): nothing is built, only found.


async def _cave(task: Task, control: Control, rng: random.Random) -> Site:
    x, _, z = _anchor(rng)
    y = rng.randint(-50, -10)
    caves = [spot for spot in await _spots(control, (x, y, z), 12, limit=64) if abs(spot[1] - y) <= 12]
    if len(caves) < 4:
        raise BuildError("no natural cave here")
    first = caves[0]
    if await control.find_blocks("sculk_shrieker", *first, radius=24, limit=1):
        raise BuildError("an ancient city is here")  # its warden is no fair start
    together = [spot for spot in caves if math.dist(spot, first) <= 6]
    return Site(_spread(together), await _ore_count(control, first), first)


async def _surface(task: Task, control: Control, rng: random.Random) -> Site:
    x, z = rng.randint(-200, 200), rng.randint(-200, 200)
    y = await control.surface(x, z) + 1
    if y < 63:
        raise BuildError("under water")
    return Site(_spread(await _spots(control, (x, y, z), 6)), await _ore_count(control, (x, y, z)), (x, y, z))


async def _woodland(task: Task, control: Control, rng: random.Random) -> Site:
    site = await _surface(task, control, rng)
    logs = await control.find_blocks("#logs", *site.anchor, radius=10, limit=64)
    if not any(abs(log["y"] - site.anchor[1]) <= 2 for log in logs):  # a trunk within reach from the ground
        raise BuildError("no trees here")
    return site


async def _nether(task: Task, control: Control, rng: random.Random) -> Site:
    x, z = rng.randint(-100, 100), rng.randint(-100, 100)
    spots = await _spots(control, (x, 64, z), 24, world="world_nether", limit=64)
    if len(spots) < 4:
        raise BuildError("no ground here in the nether")
    return Site(_spread(spots), 0, spots[0], world="world_nether")


async def _fortress(task: Task, control: Control, rng: random.Random) -> Site:
    found = await control.locate("fortress", world="world_nether", x=rng.randint(-200, 200), z=rng.randint(-200, 200))
    if found is None:
        raise BuildError("no fortress near")
    spots = await _spots(control, (found["x"], 64, found["z"]), 32, world="world_nether", limit=64)
    if len(spots) < 4:
        raise BuildError("no ground beside the fortress")
    return Site(_spread(spots), 0, spots[0], world="world_nether")


async def _stronghold_area(task: Task, control: Control, rng: random.Random) -> Site:
    found = await control.locate("stronghold", world="world", x=rng.randint(-500, 500), z=rng.randint(-500, 500))
    if found is None:
        raise BuildError("no stronghold")
    angle, distance = rng.uniform(0, 2 * math.pi), rng.uniform(150, 300)
    x, z = round(found["x"] + distance * math.cos(angle)), round(found["z"] + distance * math.sin(angle))
    y = await control.surface(x, z) + 1
    if y < 63:
        raise BuildError("under water")
    return Site(_spread(await _spots(control, (x, y, z), 6)), 0, (x, y, z))


async def _portal_room(task: Task, control: Control, rng: random.Random) -> Site:
    found = await control.locate("stronghold", world="world", x=rng.randint(-500, 500), z=rng.randint(-500, 500))
    if found is None:
        raise BuildError("no stronghold")
    frames: list[dict[str, int]] = []
    for y in (found["y"], 30, 0, -20, 50):
        frames = await control.find_blocks("end_portal_frame", found["x"], y, found["z"], radius=64, limit=12)
        if frames:
            break
    if not frames:
        raise BuildError("no portal room found near the stronghold")
    frame = (frames[0]["x"], frames[0]["y"], frames[0]["z"])
    return Site(_spread(await _spots(control, frame, 6)), 0, frame)


async def _end(task: Task, control: Control, rng: random.Random) -> Site:
    spots = await _spots(control, (100, 49, 0), 8, world="world_the_end", limit=64)
    if len(spots) < 4:
        raise BuildError("no platform in the end")
    return Site(_spread(spots), 0, spots[0], world="world_the_end")
