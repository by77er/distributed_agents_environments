"""The curriculum unlocks harder rows as easier ones are solved, and favors rows whose groups differ; an environment's
own can gate rows on evals."""

import random
from collections.abc import Sequence
from dataclasses import dataclass

from rollout.curriculum import Curriculum, GroupResult, curriculum_of, solved_share
from rollout.environment import Row
from rollout_train import Result
from tests.rollout_train.rollouts.games import Words

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


def test_a_curriculum_is_the_fold_of_a_runs_results_by_title_whatever_the_environment_has_become() -> None:
    lines = [
        Result(1, 0.0, ROWS[4].key, ROWS[4].title, rewards=[1.0, 2.0], solved=[True, False]),
        Result(2, 0.0, ROWS[0].key, ROWS[0].title),  # none of its episodes completed
        Result(3, 0.0, "gone", "a row the environment no longer has", rewards=[1.0], solved=[True]),
    ]
    shifted = [Row(f"x{index}", row.title) for index, row in enumerate(ROWS[2:])]  # two rows gone: other keys
    curriculum = Curriculum(shifted, random.Random(0))
    for line in lines:
        curriculum.recorded(line)
    assert curriculum.record(shifted[2]).attempts == 1 and curriculum.record(shifted[2]).reward == 1.5
    assert [key for key, record in curriculum.records.items() if record.attempts] == ["x2"]
    same = Curriculum(ROWS)
    for line in lines:
        same.recorded(line)
    assert same.record(ROWS[4]).attempts == 1 and same.record(ROWS[0]).failures == 1
    assert same.sample(rng=random.Random(5)) == same.sample(rng=random.Random(5))  # a choice can be made again


def test_a_group_counts_for_the_rows_its_row_counts_for_as_well() -> None:
    guided, alone = Row("t1", "a task"), Row("t1u", "a task, unguided", counts_for=("t1", "gone"))
    curriculum = Curriculum([guided, alone], random.Random(0))
    curriculum.recorded(Result(1, 0.0, "t1u", alone.title, rewards=[2.0, 2.0], solved=[True, True]))
    assert curriculum.record(alone).success == curriculum.record(guided).success == 1.0
    assert curriculum.record(alone).signal == curriculum.record(guided).signal == 0.0  # all alike: nothing to teach
    curriculum.recorded(Result(2, 0.0, "t1", guided.title, rewards=[1.0, 0.0], solved=[True, False]))
    assert curriculum.record(guided).attempts == 2 and curriculum.record(alone).attempts == 1  # not the other way


@dataclass
class Gated(Curriculum):
    """Rows past the third wait until the suite `held-out`'s entry of its own environment is solved at least 60% of
    the time by a checkpoint."""

    opened_by: str | None = None

    def evaluated(
        self, suite: str, checkpoint: str | None, results: Sequence[GroupResult], entry: str | None = None
    ) -> None:
        super().evaluated(suite, checkpoint, results, entry)
        if (suite, entry) == ("held-out", "words:words") and solved_share(results) >= 0.6 and self.opened_by is None:
            self.opened_by = checkpoint

    def unlocked(self) -> list[Row]:
        return super().unlocked() if self.opened_by else super().unlocked()[:3]


class Staged(Words):
    def curriculum(self) -> Curriculum:
        return Gated(ROWS)


def test_a_curriculum_can_gate_rows_on_an_evals_results() -> None:
    curriculum = curriculum_of(Staged())
    assert isinstance(curriculum, Gated) and isinstance(curriculum_of(Words()), Curriculum)
    curriculum.update(ROWS[2], [1.0], [True])  # solved: the generic curriculum would unlock four more
    assert curriculum.unlocked() == ROWS[:3]
    curriculum.evaluated("other", "c1", [Result(1, 0.0, "r01", rewards=[1.0], solved=[True])], "words:words")
    curriculum.evaluated(
        "held-out", "c1", [Result(1, 0.0, "r01", rewards=[1.0, 0.0], solved=[True, False])], "words:words"
    )
    assert curriculum.unlocked() == ROWS[:3]  # half solved is not enough
    played = [
        Result(1, 0.0, "r01", rewards=[1.0, 1.0], solved=[True, True]),
        Result(2, 0.0, "r02", solved=[True, False]),
    ]
    curriculum.evaluated("held-out", "c2", played, "games:other")  # (another environment's entry of the suite)
    assert curriculum.opened_by is None
    curriculum.evaluated("held-out", "c2", played, "words:words")
    assert curriculum.opened_by == "c2" and curriculum.unlocked() == ROWS[:7]
    assert curriculum.evaluations[("held-out", "words:words")] == ("c2", played)
    assert curriculum.evaluations[("other", "words:words")][0] == "c1"
    generic = Curriculum(ROWS)
    generic.evaluated("held-out", None, played, "words:words")  # (the newest of each suite's entry, deciding nothing)
    assert generic.evaluations == {("held-out", "words:words"): (None, played)} and generic.unlocked() == ROWS[:3]
