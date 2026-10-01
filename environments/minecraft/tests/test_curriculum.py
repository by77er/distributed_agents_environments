"""The curriculum unlocks harder tasks as easier ones are solved, and favors tasks solved about half the time."""

import random

from minecraft_swarm.curriculum import Curriculum
from minecraft_swarm.tasks import catalog


def test_tasks_unlock_as_easier_ones_are_solved() -> None:
    tasks = catalog()
    curriculum = Curriculum(tasks, random.Random(0))
    first = curriculum.unlocked()
    assert first and all(task.difficulty <= 1.0 for task in first)
    for task in first:
        curriculum.update(task, [3, 2, 4, 1])  # solved every time
    harder = curriculum.unlocked()
    assert len(harder) > len(first) and max(task.difficulty for task in harder) <= 3.0


def test_learning_progress_favors_tasks_solved_half_the_time() -> None:
    tasks = catalog()
    curriculum = Curriculum(tasks, random.Random(0))
    easy, middling, hopeless = tasks[0], tasks[1], tasks[2]
    curriculum.update(easy, [5, 5, 5, 5])
    curriculum.update(middling, [0, 2, 0, 1])
    curriculum.update(hopeless, [0, 0, 0, 0])
    assert curriculum.weight(middling) > curriculum.weight(easy) > 0
    assert curriculum.weight(middling) > curriculum.weight(hopeless) > 0
    assert curriculum.weight(tasks[3]) == 1.0  # untried
