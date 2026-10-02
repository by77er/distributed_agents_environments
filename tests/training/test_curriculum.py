"""The curriculum unlocks harder rows as easier ones are solved, and favors rows whose groups differ."""

import json
import random

from rollout.rollouts import Row
from rollout.training import Curriculum

ROWS = [Row(f"r{number:02d}", f"row {number}") for number in range(1, 21)]


def test_rows_unlock_in_order_as_easier_ones_are_solved() -> None:
    curriculum = Curriculum(ROWS, random.Random(0))
    assert curriculum.unlocked() == ROWS[:3]
    curriculum.update(ROWS[2], [3, 2, 4, 1], [True, True, True, True])
    assert curriculum.unlocked() == ROWS[:7]  # four past the hardest one solved
    curriculum.update(ROWS[5], [0, 0, 0, 1], [False, False, False, True])  # tried, rarely solved: nothing unlocks
    assert curriculum.unlocked() == ROWS[:7]
    while len(curriculum.unlocked()) < len(ROWS):  # every row can be reached
        curriculum.update(curriculum.unlocked()[-1], [1.0], [True])
    assert curriculum.unlocked()[-1] is ROWS[-1]


def test_rows_whose_episodes_differ_are_trained_on_most() -> None:
    curriculum = Curriculum(ROWS, random.Random(0))
    saturated, uneven, hopeless = ROWS[0], ROWS[1], ROWS[2]
    curriculum.update(saturated, [5, 5, 5, 5], [True] * 4)
    curriculum.update(uneven, [9, 9, 20, 9], [True] * 4)  # all "solved", and yet a group with much to teach
    curriculum.update(hopeless, [0, 0, 0, 0], [False] * 4)
    assert curriculum.weight(uneven) == 1.05
    assert curriculum.weight(saturated) == curriculum.weight(hopeless) == 0.05
    assert curriculum.weight(ROWS[4]) == 1.0  # untried
    curriculum.update(uneven, [9, 9, 9, 9], [True] * 4)  # a moving average: one even group halves it
    assert curriculum.weight(uneven) == 0.55
    # Solved is what the row's own result says, not a reward above zero.
    curriculum.update(ROWS[5], [1.0, 2.0], [False, False])
    assert curriculum.record(ROWS[5]).success == 0.0 and curriculum.record(ROWS[5]).reward == 1.5


def test_a_row_that_could_not_be_set_up_is_tried_again_before_it_counts_as_tried() -> None:
    curriculum = Curriculum(ROWS, random.Random(0))
    stubborn = ROWS[1]
    curriculum.failed(stubborn)
    curriculum.failed(stubborn)
    assert curriculum.weight(stubborn) == 1.0  # the next start may be one it can be set up from
    curriculum.failed(stubborn)
    assert curriculum.weight(stubborn) == 0.05 and curriculum.record(stubborn).attempts == 1


def test_a_row_whose_group_is_still_running_is_not_chosen_again() -> None:
    curriculum = Curriculum(ROWS, random.Random(0))
    for _ in range(50):
        assert curriculum.sample(pending=[ROWS[0].key, ROWS[2].key]) is ROWS[1]
    assert curriculum.sample(pending=[row.key for row in ROWS[:3]]) in ROWS[:3]  # unless nothing else is unlocked


def test_records_are_kept_by_title_so_that_a_changed_catalog_does_not_move_them() -> None:
    curriculum = Curriculum(ROWS, random.Random(0))
    curriculum.update(ROWS[4], [1, 2], [True, False])
    saved = json.loads(json.dumps(curriculum.saved()))
    shifted = [Row(f"x{index}", row.title) for index, row in enumerate(ROWS[2:])]  # two rows gone: other keys
    again = Curriculum(shifted, random.Random(0))
    again.restore(saved)
    assert again.record(shifted[2]).attempts == 1 and again.record(shifted[2]).reward == 1.5
    assert [key for key, record in again.records.items() if record.attempts] == ["x2"]
    old = Curriculum(ROWS)  # records saved without titles go by key, and fields a record no longer has are dropped
    old.restore({"r05": {"attempts": 2, "success": 1.0, "reward": 3.0, "gone": 1}})
    assert old.record(ROWS[4]).attempts == 2
