"""The gridworld's curriculum: the open rooms first, a row unlocking what follows once its success or its progress
passes a gate over its recent groups, rows with mixed outcomes drawn most and settled ones still drawn now and then,
and the same curriculum rebuilt from results as the ledger keeps them."""

import random
from collections import Counter
from dataclasses import dataclass, field

from gridworld.curriculum import GridCurriculum
from gridworld.environment import ROWS, environment
from rollout.curriculum import curriculum_of
from rollout_train.record import Result

KEYS = [row.key for row in ROWS]
ROW = {row.key: row for row in ROWS}


@dataclass
class Line:
    task: str
    rewards: list[float]
    solved: list[bool] = field(default_factory=list[bool])

    @property
    def title(self) -> str:
        return ROW[self.task].title

    def __post_init__(self) -> None:
        self.solved = self.solved or [reward == 1.0 for reward in self.rewards]


def unlocked(curriculum: GridCurriculum) -> list[str]:
    return [row.key for row in curriculum.unlocked()]


def test_the_rows_go_from_the_open_room_to_the_vault_each_with_two_to_four_agents() -> None:
    assert [f"{layout}-{agents}" for layout in ("open", "door", "gate", "vault") for agents in (2, 3, 4)] == KEYS
    assert isinstance(curriculum_of(environment), GridCurriculum)


def test_the_open_rooms_are_unlocked_first_and_a_row_unlocks_what_follows_once_learned() -> None:
    curriculum = GridCurriculum(ROWS)
    assert unlocked(curriculum) == ["open-2", "open-3", "open-4"]
    curriculum.recorded(Line("open-4", [1.0, 1.0, 0.25, 1.0]))
    assert unlocked(curriculum) == KEYS[:3], "one group is not enough to call a row learned"
    curriculum.recorded(Line("open-4", [1.0, 0.25, 0.25, 1.0]))
    assert unlocked(curriculum) == KEYS[:5], "solved over half the time: two rows past it"


def test_progress_unlocks_a_row_that_is_not_yet_solved() -> None:
    curriculum = GridCurriculum(ROWS)
    door = [0.375, 0.375, 0.375, 0.25]  # (the door open and one of two final plates pressed, mostly: none solved)
    for _ in range(2):
        curriculum.recorded(Line("door-2", door))
    assert curriculum.progress["door-2"] < 0.7 and unlocked(curriculum) == KEYS[:3]
    for _ in range(2):
        curriculum.recorded(Line("door-2", [0.375] * 4))
    assert curriculum.progress["door-2"] >= 0.7
    assert unlocked(curriculum) == KEYS[:6], "progress past 0.7 on door-2: two rows past it"


def test_rows_with_mixed_outcomes_are_drawn_most_and_settled_ones_still_now_and_then() -> None:
    curriculum = GridCurriculum(ROWS)
    for _ in range(3):
        curriculum.recorded(Line("open-2", [1.0] * 4))  # settled: every episode solved
        curriculum.recorded(Line("open-3", [1.0, 0.333, 1.0, 0.333]))  # solved half the time
        curriculum.recorded(Line("open-4", [0.25, 0.125, 0.125, 0.0]))  # progress with spread, none solved
    weights = {row.key: curriculum.weight(row) for row in curriculum.unlocked()}
    assert weights["open-3"] > weights["open-4"] > weights["open-2"] > 0
    assert weights["open-2"] == curriculum.floor
    rng = random.Random(0)
    drawn = Counter(curriculum.sample(rng=rng).key for _ in range(3000))
    assert drawn["open-3"] > drawn["open-4"] > drawn["open-2"] > 0


def test_a_row_not_yet_tried_has_full_weight() -> None:
    curriculum = GridCurriculum(ROWS)
    assert curriculum.weight(ROW["open-2"]) == 1.0


def test_the_curriculum_is_rebuilt_from_results_as_the_ledger_keeps_them() -> None:
    lines = [
        Line("open-2", [1.0] * 4),
        Line("open-3", [1.0, 0.333, 1.0, 1.0]),
        Line("open-3", [1.0, 1.0, 0.333, 1.0]),
        Line("door-2", [0.375, 0.25, 0.25, 0.0]),
        Line("open-4", [0.25, 0.0, 0.0, 0.0]),
    ]
    live = curriculum_of(environment)
    kept: dict[str, dict[str, object]] = {}
    groups: dict[str, dict[str, object]] = {}
    for number, line in enumerate(lines, start=1):
        result = Result(group=number, time=float(number), task=line.task, title=line.title, rewards=line.rewards,
                        solved=line.solved)  # fmt: skip
        live.recorded(result)
        kept[str(number)] = result.to_json()
        groups[str(number)] = {"task": line.task, "title": line.title}
    rebuilt = curriculum_of(environment)
    for number in sorted(kept, key=int):  # (as `rollout_train.loop.train` does when a run is started again)
        rebuilt.recorded(Result.from_json(kept[number], int(number), groups[number]))
    assert isinstance(live, GridCurriculum) and isinstance(rebuilt, GridCurriculum)
    assert rebuilt.records == live.records and rebuilt.progress == live.progress
    assert unlocked(rebuilt) == unlocked(live) == KEYS[:4]
    assert [rebuilt.weight(row) for row in ROWS] == [live.weight(row) for row in ROWS]
