"""The task catalog: three tiers from staged skills to the whole game, what each task rewards, and the gear each agent
gets."""

import random
from dataclasses import dataclass
from typing import Any

import pytest

from minecraft_team.environment import Teams
from minecraft_team.prompts import goal
from minecraft_team.tasks import (
    CHAINS,
    DRAGON_DAMAGE,
    KITS,
    MILESTONES,
    SOLVED_SHARE,
    TEAM,
    UNGUIDED,
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
    laddered,
    path_of,
    saturated,
    score,
    scored,
    solved,
)
from rollout.curriculum import Curriculum
from rollout.environment import Row


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
    guided = [task for task in tasks if task.guided]
    assert [task.id for task in guided] == [f"t{index:03d}" for index in range(1, len(guided) + 1)]
    # A task with a way to its goal has an unguided variant, the same situation a little harder.
    assert {task.id for task in tasks if not task.guided} == {f"{task.id}u" for task in guided if laddered(task)}
    for task in tasks:
        if not task.guided:
            original = next(each for each in guided if each.id == task.id.removesuffix("u"))
            assert (
                task.model_copy(
                    update={
                        "id": original.id,
                        "title": original.title,
                        "guided": True,
                        "difficulty": original.difficulty,
                    }
                )
                == original
            )
            assert task.difficulty == original.difficulty + UNGUIDED
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


def test_a_reward_runs_from_0_to_1_and_solving_the_task_outweighs_any_progress() -> None:
    for task in catalog():
        steps = path_of(task)
        unsolved: dict[str, Any] = {"team_diamonds": 0, "team_obtained": {}, "team_advancements": []}
        assert steps and score(task, unsolved, available=12) == 0.0
        # Every step done but the task itself: still less than any episode that solved it.
        almost = scored(task, nearly(task, steps), available=12)
        assert not almost.solved and almost.reward < SOLVED_SHARE <= score(task, solving(task), available=12)
        best = scored(task, solving(task, everything=True), available=12)
        assert best.reward == 1.0 and best.saturated and best.progress == 1.0


def nearly(task: Task, steps: list[tuple[str, tuple[str, ...], float]]) -> dict[str, Any]:
    """Every step of the path done, short of solving the task."""
    items = {item.replace("*", "oak"): 1 for _, shown, _ in steps for item in shown if item != str(task.goal)}
    advancements = [name for name, _, _ in steps if name in MILESTONES and name != task.goal]
    held = 6 if task.laid_out else 3  # of the 12 laid out, or one short of one each
    return {"team_diamonds": held, "team_obtained": items, "team_advancements": advancements, "dragon_damage": 0.9}


def solving(task: Task, everything: bool = False) -> dict[str, Any]:
    held = 12 if everything else 7
    goal = str(task.goal)
    return {
        "team_diamonds": held if task.objective is Objective.DIAMONDS else 0,
        "team_obtained": {goal: 1} if task.objective is Objective.CRAFT else {},
        "team_advancements": [goal] if task.objective is Objective.PROGRESS else [],
        "dragon_killed": goal == "end/kill_dragon",
    }


def test_diamonds_laid_out_count_as_a_share_of_those_laid_out() -> None:
    staged = find(Start.CHESTS, Kit.NONE)
    assert [name for name, _, _ in path_of(staged)] == ["diamonds"]
    assert score(staged, {"team_diamonds": 3}, available=12) == 0.125  # a quarter of them: half of the progress half
    assert score(staged, {"team_diamonds": 7}, available=12) == round(0.5 + 0.5 * 7 / 12, 6)  # more than half: solved
    assert score(staged, {"team_diamonds": 12}, available=12) == 1.0
    assert not solved(staged, {"team_diamonds": 6}, available=12) and solved(staged, {"team_diamonds": 7}, available=12)


def test_diamonds_from_ore_count_the_steps_the_kit_leaves_then_one_diamond_each() -> None:
    iron, ingots = find(Start.ORE_IN_SIGHT, Kit.IRON), find(Start.ORE_IN_SIGHT, Kit.INGOTS)
    assert [name for name, _, _ in path_of(iron)] == ["diamonds"]
    assert [(name, weight) for name, _, weight in path_of(ingots)] == [("an iron pickaxe", 6), ("diamonds", 8)]
    stone = find(Start.ORE_IN_SIGHT, Kit.STONE)
    assert [name for name, _, _ in path_of(stone)] == ["raw iron", "an iron ingot", "an iron pickaxe", "diamonds"]
    assert score(iron, {"team_diamonds": 2}, available=274, players=4) == 0.25  # half of one each
    assert score(iron, {"team_diamonds": 2}, available=274, players=2) == 1.0
    assert score(iron, {"team_diamonds": 20}, available=274) == score(iron, {"team_diamonds": 4}, available=274) == 1
    # From curriculum-9 (t013u, group 15): one team made the pickaxe and found no diamond, its teammates' teams four.
    pickaxe: dict[str, Any] = {
        "team_diamonds": 0,
        "team_obtained": {"cobbled_deepslate": 1, "crafting_table": 2, "iron_ingot": 6, "iron_pickaxe": 1, "stick": 4},
        "team_advancements": ["story/iron_tools", "story/mine_stone", "story/root", "story/smelt_iron"],
    }
    found = scored(ingots, pickaxe, available=313, players=4)
    assert not found.solved and found.progress == round(6 / 14, 6) and found.reward == round(0.5 * 6 / 14, 6)
    assert found.parts == {"solved": 0.0, "an iron pickaxe": round(0.5 * 6 / 14, 4), "diamonds": 0.0}
    # What the kit gave counts for nothing, picked up again or not (group 50: a table placed and taken back).
    assert score(ingots, {"team_diamonds": 0, "team_obtained": {"crafting_table": 1, "iron_ingot": 3}}) == 0.0
    four = {**pickaxe, "team_diamonds": 4, "team_obtained": {**pickaxe["team_obtained"], "diamond": 4}}
    assert scored(ingots, four, available=313, players=4).parts == {"solved": 0.5, "an iron pickaxe": 0.2143,
                                                                     "diamonds": 0.2857}  # fmt: skip


def test_an_episode_may_end_when_nothing_is_left_to_earn() -> None:
    staged, ore = find(Start.CHESTS, Kit.NONE), find(Start.ORE_IN_SIGHT, Kit.IRON)
    assert staged.laid_out and not ore.laid_out
    assert not saturated(staged, {"team_diamonds": 11}, 12) and saturated(staged, {"team_diamonds": 12}, 12)
    assert not saturated(ore, {"team_diamonds": 3}, 274) and saturated(ore, {"team_diamonds": 4}, 274)  # one each
    assert saturated(ore, {"team_diamonds": 1}, 274, players=1)
    furnace = next(t for t in catalog() if t.goal == "furnace")
    assert not saturated(furnace, {"team_diamonds": 0, "team_obtained": {"cobblestone": 8}}, 0)
    assert saturated(furnace, {"team_diamonds": 0, "team_obtained": {"furnace": 1}}, 0)
    dragon = find(Start.END, Kit.END_READY)
    assert not saturated(dragon, {"team_diamonds": 0, "dragon_damage": 0.9}, 0)
    assert saturated(dragon, {"team_diamonds": 0, "dragon_killed": True}, 0)


def test_progress_tasks_count_the_milestones_on_their_path_once_and_are_solved_by_their_own() -> None:
    task = find(Start.FORTRESS, Kit.FORTRESS_READY)
    assert task.goal == "nether/obtain_blaze_rod"
    assert [name for name, _, _ in path_of(task)] == ["nether/find_fortress", "nether/obtain_blaze_rod"]
    found: dict[str, Any] = {"team_diamonds": 3, "team_advancements": ["nether/find_fortress", "adventure/kill_a_mob"]}
    assert score(task, found) == round(0.5 * 6 / 14, 6) and not solved(task, found)
    rods: dict[str, Any] = {"team_diamonds": 0, "team_advancements": ["nether/obtain_blaze_rod"]}
    assert score(task, rods) == 1.0 and solved(task, rods)  # however it got there
    portal = find(Start.PORTAL_ROOM, Kit.EYES_READY)  # inside the stronghold: the eyes have been followed
    assert [name for name, _, _ in path_of(portal)] == ["story/enter_the_end"]


def test_hurting_the_dragon_counts_and_killing_it_counts_most() -> None:
    task = find(Start.END, Kit.END_READY)
    hurt: dict[str, Any] = {"team_diamonds": 0, "team_advancements": [], "dragon_damage": 0.5}
    killed: dict[str, Any] = {"team_diamonds": 0, "team_advancements": ["end/kill_dragon"], "dragon_damage": 1.0}
    assert score(task, hurt) == 0.5 * DRAGON_DAMAGE * 0.5 / MILESTONES["end/kill_dragon"] == 0.125
    assert score(task, killed) == 1.0 and not solved(task, hurt) and solved(task, killed)
    # A dragon that died with no player credited for it (the advancement goes to a player) is as dead.
    uncredited: dict[str, Any] = {"team_advancements": [], "dragon_damage": 1.0, "dragon_killed": True}
    assert score(task, uncredited) == 1.0 and solved(task, uncredited)
    whole_game = catalog()[-1]
    assert score(whole_game, {"team_advancements": list(MILESTONES)}) == 1.0
    almost = scored(whole_game, {"team_advancements": list(MILESTONES)[:-1], "dragon_damage": 0.5})
    assert almost.reward < 0.5 and almost.parts["end/kill_dragon"] == round(0.5 * 10 / 98, 4)


def test_crafting_tasks_count_each_step_of_the_chain_once_and_are_solved_by_the_item() -> None:
    task = next(t for t in catalog() if t.goal == "stone_pickaxe")
    assert [name for name, _, _ in CHAINS["stone_pickaxe"]][-2:] == ["cobblestone", "a stone pickaxe"]
    nothing: dict[str, Any] = {"team_diamonds": 0, "team_obtained": {}}
    assert score(task, nothing) == 0 and not solved(task, nothing)
    # Any log and any planks count; forty logs count as one step; things off the chain count for nothing.
    halfway: dict[str, Any] = {"team_obtained": {"birch_log": 40, "birch_planks": 8, "stick": 4, "dirt": 9}}
    assert done(CHAINS["stone_pickaxe"], halfway["team_obtained"]) == ["logs", "planks", "sticks"]
    assert score(task, halfway) == round(0.5 * 3 / 13, 6) and not solved(task, halfway)
    # From curriculum-9 (t019, group 46): two teams got as far as a wooden pickaxe, two as cobblestone.
    wooden = {"birch_log": 16, "birch_planks": 48, "crafting_table": 7, "stick": 48, "wooden_pickaxe": 1}
    assert score(task, {"team_obtained": wooden}) == round(0.5 * 8 / 13, 6)
    assert score(task, {"team_obtained": {**wooden, "cobblestone": 1}}) == round(0.5 * 10 / 13, 6)
    whole = {"team_obtained": {"stone_pickaxe": 1}}
    assert score(task, whole) == 1.0 and solved(task, whole)
    # The item made is the whole chain, however it was made: a furnace takes cobblestone, not a stone pickaxe.
    furnace = next(t for t in catalog() if t.goal == "furnace")
    obtained: dict[str, int] = dict.fromkeys(("oak_log", "oak_planks", "stick", "wooden_pickaxe", "cobblestone"), 1)
    direct = {"team_obtained": obtained}
    assert score(furnace, direct) == 0.25 and not solved(furnace, direct)  # 8 of 16: no table, stone pickaxe, furnace
    obtained["furnace"] = 1
    assert score(furnace, direct) == 1.0 and solved(furnace, direct)
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
    assert score(game, state) == round(0.5 * (MILESTONES["story/mine_stone"] + 3) / 98, 6)  # wood, a table, a pickaxe
    assert score(kitted, state) == 0.0  # mining stone is not on the path from a fortress
    assert "in order: logs, planks, a crafting table, a wooden pickaxe, mining stone" in goal(game)
    assert "in order: finding a fortress, getting a blaze rod." in goal(kitted)


def test_the_goal_says_what_counts() -> None:
    ingots, iron = find(Start.ORE_IN_SIGHT, Kit.INGOTS), find(Start.ORE_IN_SIGHT, Kit.IRON)
    assert goal(ingots).startswith("Goal: together, hold one diamond each, four in all.")
    assert "whoever does it: an iron pickaxe, diamonds. The game is over when you hold them." in goal(ingots)
    assert goal(iron, 1).startswith("Goal: hold one diamond.") and "step by step" not in goal(iron, 1)
    assert goal(find(Start.CHESTS, Kit.NONE)).startswith("Goal: together, hold as many diamonds as you can.")
    assert "What counts: killing the ender dragon. Hurting the dragon" in goal(find(Start.END, Kit.END_READY))


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

    async def ores(self, x: int, y: int, z: int, *, radius: int = 32) -> list[dict[str, int]]:
        return [{"x": 10, "y": -55, "z": 10}] if radius >= 24 else []

    async def surface(self, x: int, z: int, world: str = "world") -> int:
        return 70

    async def find_blocks(self, block: str, x: int, y: int, z: int, **_: Any) -> list[dict[str, int]]:
        return [{"x": tx, "y": ty, "z": tz} for tx, tz in self.trees for ty in range(71, 76)]

    trees: tuple[tuple[int, int], ...] = ()


async def test_no_one_starts_within_reach_of_the_diamonds_on_the_floor() -> None:
    from minecraft_team.tasks import _items  # pyright: ignore[reportPrivateUsage]

    task = find(Start.ITEMS, Kit.NONE)
    for seed in range(40):
        control = Site()
        site = await _items(task, control, random.Random(seed))  # type: ignore[arg-type]
        assert len(site.starts) == len(set(site.starts)) == len(TEAM) and 3 <= len(control.piles) <= 5
        for px, _, pz in control.piles:
            assert all(max(abs(px - sx), abs(pz - sz)) > 2 for sx, _, sz in site.starts), (seed, site.starts)
        (x, _, z, width, _, depth) = control.carved[0]
        assert all(x <= px < x + width and z <= pz < z + depth for px, _, pz in control.piles)  # in the room


async def test_the_stone_kit_finds_iron_in_the_wall_of_its_pocket() -> None:
    from minecraft_team.tasks import _ore_in_sight  # pyright: ignore[reportPrivateUsage]

    control = Site()
    await _ore_in_sight(find(Start.ORE_IN_SIGHT, Kit.STONE), control, random.Random(1))  # type: ignore[arg-type]
    (x, y, z, width, _, depth) = control.carved[0]
    assert len(control.blocks) == 6 and set(control.blocks.values()) == {"deepslate_iron_ore"}
    assert all(bx == x + width and y <= by <= y + 1 and z < bz < z + depth - 1 for bx, by, bz in control.blocks)
    with_iron = Site()
    await _ore_in_sight(find(Start.ORE_IN_SIGHT, Kit.IRON), with_iron, random.Random(1))  # type: ignore[arg-type]
    assert not with_iron.blocks  # who has an iron pickaxe is given no iron


async def test_a_woodland_start_is_at_the_foot_of_the_nearest_tree_or_nowhere() -> None:
    from minecraft_team.tasks import BuildError, _woodland  # pyright: ignore[reportPrivateUsage]

    task = next(t for t in catalog() if t.goal == "wooden_pickaxe")
    control = Site()
    with pytest.raises(BuildError, match="no trees here"):  # (the driver then tries another world)
        await _woodland(task, control, random.Random(3))  # type: ignore[arg-type]
    control.trees = ((500, 500), (40, -30))
    site = await _woodland(task, control, random.Random(3))  # type: ignore[arg-type]
    assert site.anchor == (40, 71, -30)  # the nearer trunk's lowest log
    assert len(site.starts) == len(TEAM) and all(
        abs(x - 40) <= 4 and abs(z + 30) <= 4 and y == 71 for x, y, z in site.starts
    )


@dataclass(frozen=True)
class Group:
    task: str
    title: str
    rewards: list[float]
    solved: list[bool]


def test_the_curriculum_unlocks_by_solved_and_weighs_rows_by_differing_rewards_on_this_scale() -> None:
    rows = list(Teams().rows())
    curriculum = Curriculum(rows, start=3, reach=4)
    assert len(curriculum.unlocked()) == 3
    first, second, third = rows[:3]

    def played(row: Row, rewards: list[float], solved: list[bool]) -> None:
        curriculum.recorded(Group(row.key, row.title, rewards, solved))

    played(first, [1.0, 1.0, 1.0, 1.0], [True] * 4)  # all saturated: solved, and nothing to compare
    assert len(curriculum.unlocked()) == 1 + 4 and curriculum.weight(first) == curriculum.floor
    played(second, [0.5 * 6 / 14, 0.0, 0.0, 0.0], [False] * 4)  # progress only: something to compare, nothing solved
    assert curriculum.weight(second) == 1.0 + curriculum.floor and curriculum.record(second).success == 0.0
    played(third, [0.75, 0.25, 0.75, 0.125], [True, False, True, False])  # a mixed group: half solved
    assert len(curriculum.unlocked()) == 3 + 4 and curriculum.weight(third) == 1.0 + curriculum.floor
