"""The task catalog: a hundred-odd tasks of increasing difficulty with the same reward, and the gear each agent gets."""

import random

from minecraft_swarm.tasks import KITS, Coordination, Crafting, Exploration, Hazards, Task, catalog, kits

TEAM = ["ada", "ben", "cy", "dee"]


def test_the_catalog_covers_every_axis_in_order_of_difficulty() -> None:
    tasks = catalog()
    assert 90 <= len(tasks) <= 110
    assert [task.id for task in tasks] == [f"t{index:03d}" for index in range(1, len(tasks) + 1)]
    assert [task.difficulty for task in tasks] == sorted(task.difficulty for task in tasks)
    assert {task.exploration for task in tasks} == set(Exploration)
    assert {task.crafting for task in tasks} == set(Crafting)
    assert {task.coordination for task in tasks} == set(Coordination)
    assert len({task.title for task in tasks}) == len(tasks)  # every task is a different situation
    assert tasks[0].exploration is Exploration.ITEMS_IN_SIGHT
    assert tasks[-1].exploration is Exploration.SURFACE and tasks[-1].crafting is Crafting.NOTHING
    # No tools are needed only where diamonds lie around; with nothing at all, only the surface has trees.
    assert all((task.crafting is Crafting.NONE) == (task.exploration <= Exploration.CHESTS_NEARBY) for task in tasks)
    assert all(task.exploration is Exploration.SURFACE for task in tasks if task.crafting is Crafting.NOTHING)


def raw_iron_nearby(coordination: Coordination) -> Task:
    (task,) = [
        task
        for task in catalog()
        if task.exploration is Exploration.ORE_NEARBY
        and task.crafting is Crafting.RAW_IRON
        and task.coordination is coordination
        and task.hazards is Hazards.SAFE
    ]
    return task


def test_gear_is_dealt_by_coordination() -> None:
    every = kits(raw_iron_nearby(Coordination.KITTED), TEAM, random.Random(1))
    assert all({"item": "furnace"} in inventory for inventory in every.values())

    one = kits(raw_iron_nearby(Coordination.ONE_KIT), TEAM, random.Random(1))
    assert sum({"item": "furnace"} in inventory for inventory in one.values()) == 1

    split = kits(raw_iron_nearby(Coordination.SPLIT), TEAM, random.Random(1))
    parts = KITS[Crafting.RAW_IRON]
    holders = {str(part["item"]): name for name, inventory in split.items() for part in inventory if part in parts}
    assert len(holders) == len(parts)  # every part was dealt once
    assert len(set(holders.values())) == len(TEAM)  # five parts over four agents: everyone holds some
    assert all({"item": "torch", "count": 16} in inventory for inventory in split.values())
