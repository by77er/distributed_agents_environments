"""The curriculum unlocks harder tasks as easier ones are solved, and favors tasks solved about half the time."""

import random

from minecraft_swarm.curriculum import Curriculum
from minecraft_swarm.tasks import Tier, catalog


def test_tasks_unlock_in_order_as_easier_ones_are_solved() -> None:
    tasks = catalog()
    curriculum = Curriculum(tasks, random.Random(0))
    assert curriculum.unlocked() == tasks[:3]
    curriculum.update(tasks[2], [3, 2, 4, 1], [True, True, True, True])
    assert curriculum.unlocked() == tasks[:7]  # four past the hardest one solved
    curriculum.update(tasks[5], [0, 0, 0, 1], [False, False, False, True])  # tried, rarely solved: nothing unlocks
    assert curriculum.unlocked() == tasks[:7]


def test_every_task_can_be_reached() -> None:
    tasks = catalog()
    curriculum = Curriculum(tasks, random.Random(0))
    while len(curriculum.unlocked()) < len(tasks):
        curriculum.update(curriculum.unlocked()[-1], [1.0], [True])
    assert curriculum.unlocked()[-1].tier is Tier.GAME


def test_learning_progress_favors_tasks_solved_half_the_time() -> None:
    tasks = catalog()
    curriculum = Curriculum(tasks, random.Random(0))
    easy, middling, hopeless = tasks[0], tasks[1], tasks[2]
    curriculum.update(easy, [5, 5, 5, 5], [True] * 4)
    curriculum.update(middling, [0, 2, 0, 1], [False, True, False, True])
    curriculum.update(hopeless, [0, 0, 0, 0], [False] * 4)
    assert curriculum.weight(middling) > curriculum.weight(easy) > 0
    assert curriculum.weight(middling) > curriculum.weight(hopeless) > 0
    assert curriculum.weight(tasks[3]) == 1.0  # untried


def test_a_progress_task_counts_as_solved_only_by_its_own_milestone() -> None:
    # Rewards above zero (stone mined on the way) are not success: the curriculum is told what solved means.
    tasks = catalog()
    curriculum = Curriculum(tasks, random.Random(0))
    curriculum.update(tasks[0], [1.0, 2.0], [False, False])
    assert curriculum.record(tasks[0]).success == 0.0 and curriculum.record(tasks[0]).reward == 1.5
