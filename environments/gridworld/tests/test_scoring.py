"""The reward and its parts on games played by hand: solved, partly done, idle, and the ways to farm progress that do
not pay; the stages of each layout; and that any solved episode scores more than any unsolved one."""

import pytest

from gridworld.environment import ROWS
from gridworld.game import Action, Game, names_for
from gridworld.level import Door, DoorKind, Level, Plate, generate
from gridworld.scoring import FINAL, progress_of, scored, stages
from rollout.environment import Row

WAIT = Action("wait")


def move(direction: str) -> Action:
    return Action("move", direction=direction)


def door_level() -> Level:
    """Two rooms of three by three, x 1 to 3 and 5 to 7, joined by door a at (4, 2): plates 1 and 2 in the first open
    it, final plates 3 and 4 are in the second. The agents start at (2, 1) and (2, 3)."""
    walls = ("#########", "#...#...#", "#.......#", "#...#...#", "#########")
    plates = (Plate(1, (1, 1), "a"), Plate(2, (1, 3), "a"), Plate(3, (6, 1)), Plate(4, (6, 3)))
    return Level(walls, ((2, 1), (2, 3)), plates, (Door("a", (4, 2), DoorKind.PLATES),))


def game(level: Level, turns: int = 30) -> Game:
    return Game(level, names_for(len(level.starts), 0), turns, 0)


def play(played: Game, *turns: tuple[Action, Action]) -> Game:
    for actions in turns:
        played.step(list(actions))
    return played


TO_THREE = [move(each) for each in ("east", "south", "east", "east", "east", "north", "east")]
"""From plate 1, through door a, to final plate 3."""
TO_FOUR = [move(each) for each in ("east", "north", "east", "east", "east", "south", "east")]
"""From plate 2, through door a, to final plate 4."""


def open_door(played: Game) -> Game:
    """Both agents step west onto plates 1 and 2 at once: door a opens."""
    return play(played, (move("west"), move("west")))


def test_an_idle_team_scores_nothing() -> None:
    played = play(game(door_level()), *[(WAIT, WAIT)] * 30)
    score = scored(played)
    assert played.over and not played.solved
    assert (score.reward, score.progress) == (0.0, 0.0)
    assert score.parts == {"solved": 0.0, "door a": 0.0, FINAL: 0.0}


def test_wandering_that_opens_nothing_and_presses_no_final_plate_scores_nothing() -> None:
    played = play(game(door_level()), *[(move("east"), move("east")), (move("west"), move("west"))] * 5)
    assert scored(played).reward == 0.0


def test_opening_a_door_and_pressing_final_plates_are_graded_stages() -> None:
    played = open_door(game(door_level()))
    assert played.opened == {"a": 1}
    score = scored(played)
    assert score.parts == {"solved": 0.0, "door a": 0.25, FINAL: 0.0}
    assert (score.reward, score.progress) == (0.25, 0.5)
    play(played, *[(step, WAIT) for step in TO_THREE])
    assert played.positions[0] == (6, 1)
    score = scored(played)
    assert score.parts == {"solved": 0.0, "door a": 0.25, FINAL: 0.125}
    assert (score.reward, score.progress) == (0.375, 0.75)


def test_solving_earns_every_part() -> None:
    played = open_door(game(door_level()))
    play(played, *zip([*TO_THREE, WAIT, WAIT], [WAIT, WAIT, *TO_FOUR], strict=True))
    assert played.solved
    score = scored(played)
    assert score.parts == {"solved": 0.5, "door a": 0.25, FINAL: 0.25}
    assert (score.reward, score.progress, score.solved) == (1.0, 1.0, True)


def test_touring_the_final_plates_alone_or_stepping_on_and_off_earns_no_more_than_standing_on_one() -> None:
    standing = play(open_door(game(door_level())), *[(step, WAIT) for step in TO_THREE])
    touring = play(open_door(game(door_level())), *[(step, WAIT) for step in TO_THREE])
    tour = [move("south"), move("south"), move("north"), move("north"), move("south"), move("south")]
    play(touring, *[(step, WAIT) for step in tour])  # (onto plate 4, back onto plate 3, onto plate 4 again)
    assert touring.positions[0] == (6, 3)
    assert touring.most_pressed == standing.most_pressed == 1
    assert scored(touring).reward == scored(standing).reward == 0.375


def test_door_plates_pressed_one_at_a_time_open_nothing_and_earn_nothing() -> None:
    played = play(game(door_level()), (move("west"), WAIT), (move("east"), move("west")), (WAIT, move("east")))
    assert played.opened == {}
    assert scored(played).reward == 0.0


@pytest.mark.parametrize(
    ("layout", "expected"),
    [
        ("open", [FINAL]),
        ("door", ["door a", FINAL]),
        ("gate", ["door b", FINAL]),  # (the gate is held open, not opened for good: the lever behind it is the step)
        ("vault", ["door a", "door c", FINAL]),
    ],
)
def test_each_layout_has_its_doors_that_open_for_good_and_its_final_plates_for_stages(
    layout: str, expected: list[str]
) -> None:
    assert stages(generate(layout, 3, 0)) == expected


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row.key)
def test_the_most_an_unsolved_episode_can_score_is_below_any_solved_one(row: Row) -> None:
    level = generate(str(row.parameters["layout"]), int(str(row.parameters["agents"])), 0)
    played = game(level)
    played.opened = {door.label: 1 for door in level.doors if door.kind is not DoorKind.HELD}
    played.most_pressed = len(level.final) - 1  # (every door open, every final plate but one pressed at once)
    best = scored(played)
    assert not best.solved and 0.25 <= best.reward < 0.5
    assert progress_of(best.reward, False) == pytest.approx(best.progress, abs=1e-5)
    assert progress_of(1.0, True) == 1.0
