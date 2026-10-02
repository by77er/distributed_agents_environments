"""The task catalog: three tiers from staged skills to the whole game, what each task rewards, and the gear each agent
gets."""

import random
from typing import Any

import pytest
from minecraft_swarm.prompts import goal
from minecraft_swarm.tasks import (
    CHAINS,
    KITS,
    MILESTONES,
    Coordination,
    Hazards,
    Kit,
    Objective,
    Start,
    Task,
    Tier,
    catalog,
    done,
    kits,
    score,
    solved,
)

TEAM = ["ada", "ben", "cy", "dee"]


def find(start: Start, kit: Kit, coordination: Coordination = Coordination.KITTED, tier: Tier | None = None) -> Task:
    return next(
        task
        for task in catalog()
        if task.start is start
        and task.kit is kit
        and task.coordination is coordination
        and (tier is None or task.tier is tier)
    )


def test_the_catalog_runs_from_staged_skills_to_the_whole_game() -> None:
    tasks = catalog()
    assert [task.id for task in tasks] == [f"t{index:03d}" for index in range(1, len(tasks) + 1)]
    assert len({task.title for task in tasks}) == len(tasks)  # every task is a different situation
    assert [task.tier for task in tasks] == sorted((task.tier for task in tasks), key=list(Tier).index)
    for tier in Tier:
        difficulties = [task.difficulty for task in tasks if task.tier is tier]
        assert difficulties == sorted(difficulties)
    assert {task.start for task in tasks} == set(Start)
    assert {task.kit for task in tasks} == set(Kit)
    assert {task.coordination for task in tasks} == set(Coordination)
    first, last = tasks[0], tasks[-1]
    assert first.start is Start.ITEMS and first.hazards is Hazards.SAFE and first.minutes <= 5
    assert last.tier is Tier.GAME and last.kit is Kit.NOTHING and last.hazards is Hazards.HARD and last.minutes >= 120
    # Skills reward diamonds or, starting from nothing among trees, the making of an item; nothing after them keeps
    # a peaceful world.
    assert {task.objective for task in tasks if task.tier is Tier.SKILLS} == {Objective.DIAMONDS, Objective.CRAFT}
    assert all(
        task.start is Start.WOODLAND and task.kit is Kit.NOTHING and task.goal in CHAINS
        for task in tasks
        if task.objective is Objective.CRAFT
    )
    assert all(task.hazards is not Hazards.SAFE for task in tasks if task.tier is not Tier.SKILLS)
    # A progress task is about one milestone; the game is about the dragon.
    assert all((task.goal in MILESTONES) == (task.objective is Objective.PROGRESS) for task in tasks)
    assert all(task.goal == "end/kill_dragon" for task in tasks if task.tier is Tier.GAME)


def test_diamond_tasks_reward_the_diamonds_the_team_holds() -> None:
    task = find(Start.ORE_IN_SIGHT, Kit.IRON)
    state: dict[str, Any] = {"team_diamonds": 5, "team_advancements": ["story/mine_stone", "story/mine_diamond"]}
    assert score(task, state) == 5.0 and solved(task, state)
    assert score(task, {"team_diamonds": 0}) == 0.0 and not solved(task, {"team_diamonds": 0})


def test_a_diamond_task_is_solved_by_most_of_what_was_laid_out_or_by_one_diamond_each() -> None:
    staged, ore = find(Start.CHESTS, Kit.NONE), find(Start.ORE_IN_SIGHT, Kit.IRON)
    assert not solved(staged, {"team_diamonds": 6}, available=12) and solved(staged, {"team_diamonds": 7}, available=12)
    assert not solved(ore, {"team_diamonds": 3}, available=274) and solved(ore, {"team_diamonds": 4}, available=274)


def test_progress_tasks_reward_milestones_once_and_are_solved_by_their_own() -> None:
    task = find(Start.FORTRESS, Kit.FORTRESS_READY)
    assert task.goal == "nether/obtain_blaze_rod"
    found: dict[str, Any] = {"team_diamonds": 3, "team_advancements": ["nether/find_fortress", "adventure/kill_a_mob"]}
    assert score(task, found) == MILESTONES["nether/find_fortress"] and not solved(task, found)
    rods: dict[str, Any] = {
        "team_diamonds": 0,
        "team_advancements": ["nether/find_fortress", "nether/obtain_blaze_rod"],
    }
    assert score(task, rods) == MILESTONES["nether/find_fortress"] + MILESTONES["nether/obtain_blaze_rod"]
    assert solved(task, rods)


def test_hurting_the_dragon_counts_and_killing_it_counts_most() -> None:
    task = find(Start.END, Kit.END_READY)
    hurt: dict[str, Any] = {"team_diamonds": 0, "team_advancements": [], "dragon_damage": 0.5}
    killed: dict[str, Any] = {"team_diamonds": 0, "team_advancements": ["end/kill_dragon"], "dragon_damage": 1.0}
    assert 0 < score(task, hurt) < score(task, killed) == MILESTONES["end/kill_dragon"]
    assert not solved(task, hurt) and solved(task, killed)
    # A dragon that died with no player credited for it (the advancement goes to a player) is as dead.
    uncredited: dict[str, Any] = {"team_advancements": [], "dragon_damage": 1.0, "dragon_killed": True}
    assert score(task, uncredited) == score(task, killed) and solved(task, uncredited)
    whole_game = catalog()[-1]
    assert score(whole_game, {"team_advancements": list(MILESTONES)}) == sum(MILESTONES.values())


def test_crafting_tasks_reward_each_step_of_the_chain_once_and_are_solved_by_the_item() -> None:
    task = next(t for t in catalog() if t.goal == "stone_pickaxe")
    assert [name for name, _, _ in CHAINS["stone_pickaxe"]][-2:] == ["cobblestone", "a stone pickaxe"]
    nothing: dict[str, Any] = {"team_diamonds": 0, "team_obtained": {}}
    assert score(task, nothing) == 0 and not solved(task, nothing)
    # Any log and any planks count; forty logs count as one step; things off the chain count for nothing.
    halfway: dict[str, Any] = {"team_obtained": {"birch_log": 40, "birch_planks": 8, "stick": 4, "dirt": 9}}
    assert done(CHAINS["stone_pickaxe"], halfway["team_obtained"]) == ["logs", "planks", "sticks"]
    assert score(task, halfway) == 3 and not solved(task, halfway)
    whole = {
        "team_obtained": dict.fromkeys(
            (
                "oak_log",
                "oak_planks",
                "crafting_table",
                "stick",
                "wooden_pickaxe",
                "cobbled_deepslate",
                "stone_pickaxe",
            ),
            1,
        )
    }
    assert score(task, whole) == sum(weight for _, _, weight in CHAINS["stone_pickaxe"]) == 13 and solved(task, whole)
    # The item made is the whole chain, however it was made: a furnace takes cobblestone, not a stone pickaxe.
    furnace = next(t for t in catalog() if t.goal == "furnace")
    obtained: dict[str, int] = dict.fromkeys(("oak_log", "oak_planks", "stick", "wooden_pickaxe", "cobblestone"), 1)
    direct = {"team_obtained": obtained}
    assert score(furnace, direct) == 8 and not solved(furnace, direct)  # no table, no stone pickaxe, no furnace
    obtained["furnace"] = 1
    assert score(furnace, direct) == sum(weight for _, _, weight in CHAINS["furnace"]) and solved(furnace, direct)
    text = goal(task)
    assert text.startswith("Goal: together, make a stone pickaxe. You start with nothing")
    assert text.endswith(
        "logs, planks, a crafting table, sticks, a wooden pickaxe, cobblestone, a stone pickaxe. "
        "The game is over when it is made."
    )
    assert next(t.goal for t in catalog() if t.objective is Objective.CRAFT) == "crafting_table"  # the shortest first


def test_the_crafting_ladder_runs_from_wood_to_diamonds() -> None:
    crafting = [task for task in catalog() if task.objective is Objective.CRAFT]
    assert [task.goal for task in crafting][-2:] == ["diamond", "diamond_pickaxe"]  # the longest chains last
    names = [name for name, _, _ in CHAINS["diamond_pickaxe"]]
    assert names[:2] == ["logs", "planks"] and names[-4:] == [
        "an iron ingot",
        "an iron pickaxe",
        "diamonds",
        "a diamond pickaxe",
    ]
    weights = [weight for _, _, weight in CHAINS["diamond_pickaxe"]]
    assert weights[-1] == max(weights) and sum(weights) == 49  # later steps are worth more


def test_from_nothing_the_first_steps_count_toward_progress() -> None:
    game = catalog()[-1]
    kitted = find(Start.FORTRESS, Kit.FORTRESS_READY)
    state: dict[str, Any] = {
        "team_advancements": ["story/mine_stone"],
        "team_obtained": {"oak_log": 3, "oak_planks": 12, "crafting_table": 1, "wooden_pickaxe": 1},
    }
    assert score(game, state) == MILESTONES["story/mine_stone"] + 3  # logs, planks, a table, a wooden pickaxe
    assert score(kitted, state) == MILESTONES["story/mine_stone"]  # a team given a kit earns nothing for wood
    assert "in order: logs, planks, a crafting table, a wooden pickaxe, mining stone" in goal(game)
    assert "in order: mining stone" in goal(kitted)


def test_a_task_has_a_budget_of_turns_as_well_as_of_game_time() -> None:
    tasks = {task.goal or task.id: task for task in catalog()}
    assert tasks["t001"].minutes == 3 and tasks["t001"].turns == 36  # twelve turns to the minute
    assert tasks["diamond_pickaxe"].turns == 840


def test_gear_is_dealt_by_coordination() -> None:
    every = kits(find(Start.ORE_IN_SIGHT, Kit.RAW_IRON), TEAM, random.Random(1))
    assert all({"item": "furnace"} in inventory for inventory in every.values())

    one = kits(find(Start.ORE_IN_SIGHT, Kit.IRON, Coordination.ONE_KIT), TEAM, random.Random(1))
    assert sum({"item": "iron_pickaxe"} in inventory for inventory in one.values()) == 1

    split = kits(find(Start.ORE_IN_SIGHT, Kit.RAW_IRON, Coordination.SPLIT), TEAM, random.Random(1))
    parts = KITS[Kit.RAW_IRON]
    holders = {str(part["item"]): name for name, inventory in split.items() for part in inventory if part in parts}
    assert len(holders) == len(parts)  # every part was dealt once
    assert len(set(holders.values())) == len(TEAM)  # five parts over four agents: everyone holds some
    assert all({"item": "torch", "count": 32} in inventory for inventory in split.values())

    nothing = kits(find(Start.SURFACE, Kit.NOTHING, tier=Tier.GAME), TEAM, random.Random(1))
    assert all(inventory == [] for inventory in nothing.values())


class Site:
    """Stands in for the plugin's setup API: remembers what a builder asked for."""

    def __init__(self) -> None:
        self.piles: list[tuple[int, int, int]] = []
        self.blocks: dict[tuple[int, int, int], str] = {}
        self.carved: list[tuple[int, int, int, int, int, int]] = []

    async def carve(self, x: int, y: int, z: int, *, width: int, height: int, depth: int, **_: Any) -> None:
        self.carved.append((x, y, z, width, height, depth))

    async def standing_spots(
        self, x: int, y: int, z: int, *, radius: int, limit: int, world: str
    ) -> list[dict[str, int]]:
        cells = [(x + dx, y, z + dz) for dx in range(-radius, radius + 1) for dz in range(-radius, radius + 1)]
        cells.sort(key=lambda cell: (cell[0] - x) ** 2 + (cell[2] - z) ** 2)
        return [{"x": cx, "y": cy, "z": cz} for cx, cy, cz in cells[:limit]]

    async def drop_items(self, x: int, y: int, z: int, items: list[dict[str, Any]]) -> int:
        self.piles.append((x, y, z))
        return sum(int(item["count"]) for item in items)

    async def set_block(self, x: int, y: int, z: int, block: str) -> None:
        self.blocks[(x, y, z)] = block

    async def ores(self, x: int, y: int, z: int, *, radius: int = 32, exposed: bool = False) -> list[dict[str, int]]:
        return [{"x": 10, "y": -55, "z": 10}] if radius >= 24 else []

    async def surface(self, x: int, z: int, world: str = "world") -> int:
        return 70

    async def find_blocks(self, block: str, x: int, y: int, z: int, **_: Any) -> list[dict[str, int]]:
        return [{"x": tx, "y": ty, "z": tz} for tx, tz in self.trees for ty in range(71, 76)]

    trees: tuple[tuple[int, int], ...] = ()


async def test_no_one_starts_within_reach_of_the_diamonds_on_the_floor() -> None:
    from minecraft_swarm.tasks import _items  # pyright: ignore[reportPrivateUsage]

    task = find(Start.ITEMS, Kit.NONE)
    for seed in range(40):
        control = Site()
        site = await _items(task, control, random.Random(seed))  # type: ignore[arg-type]
        assert len(site.starts) == 4 and len(set(site.starts)) == 4 and 3 <= len(control.piles) <= 5
        for px, _, pz in control.piles:
            assert all(max(abs(px - sx), abs(pz - sz)) > 2 for sx, _, sz in site.starts), (seed, site.starts)
        (x, _, z, width, _, depth) = control.carved[0]
        assert all(x <= px < x + width and z <= pz < z + depth for px, _, pz in control.piles)  # in the room


async def test_the_stone_kit_finds_iron_in_the_wall_of_its_pocket() -> None:
    from minecraft_swarm.tasks import _ore_in_sight  # pyright: ignore[reportPrivateUsage]

    control = Site()
    await _ore_in_sight(find(Start.ORE_IN_SIGHT, Kit.STONE), control, random.Random(1))  # type: ignore[arg-type]
    (x, y, z, width, _, depth) = control.carved[0]
    assert len(control.blocks) == 6 and set(control.blocks.values()) == {"deepslate_iron_ore"}
    assert all(bx == x + width and y <= by <= y + 1 and z < bz < z + depth - 1 for bx, by, bz in control.blocks)
    with_iron = Site()
    await _ore_in_sight(find(Start.ORE_IN_SIGHT, Kit.IRON), with_iron, random.Random(1))  # type: ignore[arg-type]
    assert not with_iron.blocks  # who has an iron pickaxe is given no iron


async def test_a_woodland_start_is_at_the_foot_of_the_nearest_tree_or_nowhere() -> None:
    from minecraft_swarm.tasks import BuildError, _woodland  # pyright: ignore[reportPrivateUsage]

    task = next(t for t in catalog() if t.goal == "wooden_pickaxe")
    control = Site()
    with pytest.raises(BuildError, match="no trees here"):  # (the driver then tries another world)
        await _woodland(task, control, random.Random(3))  # type: ignore[arg-type]
    control.trees = ((500, 500), (40, -30))
    site = await _woodland(task, control, random.Random(3))  # type: ignore[arg-type]
    assert site.anchor == (40, 71, -30)  # the nearer trunk's lowest log
    assert len(site.starts) == 4 and all(abs(x - 40) <= 4 and abs(z + 30) <= 4 and y == 71 for x, y, z in site.starts)
