"""The curriculum unlocks harder tasks as easier ones are solved, and favors tasks whose groups differ."""

import random
from pathlib import Path

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


def test_tasks_whose_episodes_differ_are_trained_on_most() -> None:
    tasks = catalog()
    curriculum = Curriculum(tasks, random.Random(0))
    saturated, uneven, hopeless, unbuildable = tasks[0], tasks[1], tasks[2], tasks[3]
    curriculum.update(saturated, [5, 5, 5, 5], [True] * 4)
    curriculum.update(uneven, [9, 9, 20, 9], [True] * 4)  # all "solved", and yet a group with much to teach
    curriculum.update(hopeless, [0, 0, 0, 0], [False] * 4)
    curriculum.failed(unbuildable)
    assert curriculum.weight(uneven) == 1.05
    assert curriculum.weight(saturated) == curriculum.weight(hopeless) == curriculum.weight(unbuildable) == 0.05
    assert curriculum.weight(tasks[4]) == 1.0  # untried
    curriculum.update(uneven, [9, 9, 9, 9], [True] * 4)  # a moving average: one even group halves it
    assert curriculum.weight(uneven) == 0.55


def test_a_task_whose_group_is_still_running_is_not_chosen_again() -> None:
    tasks = catalog()
    curriculum = Curriculum(tasks, random.Random(0))
    for _ in range(50):
        assert curriculum.sample(pending=[tasks[0].id, tasks[2].id]) is tasks[1]
    assert curriculum.sample(pending=[task.id for task in tasks[:3]]) in tasks[:3]  # unless nothing else is unlocked


def test_records_are_kept_by_title_so_that_a_changed_catalog_does_not_move_them(tmp_path: Path) -> None:
    tasks = catalog()
    curriculum = Curriculum(tasks, random.Random(0))
    curriculum.update(tasks[4], [1, 2], [True, False])
    curriculum.save(tmp_path / "curriculum.json")
    shifted = [task.model_copy(update={"id": f"x{index}"}) for index, task in enumerate(tasks[2:])]  # two tasks gone
    again = Curriculum(shifted, random.Random(0))
    again.load(tmp_path / "curriculum.json")
    assert again.record(shifted[2]).attempts == 1 and again.record(shifted[2]).reward == 1.5
    assert [task_id for task_id, record in again.records.items() if record.attempts] == ["x2"]


def test_a_progress_task_counts_as_solved_only_by_its_own_milestone() -> None:
    # Rewards above zero (stone mined on the way) are not success: the curriculum is told what solved means.
    tasks = catalog()
    curriculum = Curriculum(tasks, random.Random(0))
    curriculum.update(tasks[0], [1.0, 2.0], [False, False])
    assert curriculum.record(tasks[0]).success == 0.0 and curriculum.record(tasks[0]).reward == 1.5
