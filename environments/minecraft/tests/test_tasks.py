"""The task catalog: three tiers from staged skills to the whole game, what each task rewards, and the gear each agent
gets."""

import random
from typing import Any

from minecraft_swarm.tasks import (
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
    # Skills are staged and safe or easy and reward diamonds; nothing after them keeps a peaceful world.
    assert all(task.objective is Objective.DIAMONDS for task in tasks if task.tier is Tier.SKILLS)
    assert all(task.hazards is not Hazards.SAFE for task in tasks if task.tier is not Tier.SKILLS)
    # A progress task is about one milestone; the game is about the dragon.
    assert all((task.goal in MILESTONES) == (task.objective is Objective.PROGRESS) for task in tasks)
    assert all(task.goal == "end/kill_dragon" for task in tasks if task.tier is Tier.GAME)


def test_diamond_tasks_reward_the_diamonds_the_team_holds() -> None:
    task = find(Start.ORE_IN_SIGHT, Kit.IRON)
    state: dict[str, Any] = {"team_diamonds": 5, "team_advancements": ["story/mine_stone", "story/mine_diamond"]}
    assert score(task, state) == 5.0 and solved(task, state)
    assert score(task, {"team_diamonds": 0}) == 0.0 and not solved(task, {"team_diamonds": 0})


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
    whole_game = catalog()[-1]
    assert score(whole_game, {"team_advancements": list(MILESTONES)}) == sum(MILESTONES.values())


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
