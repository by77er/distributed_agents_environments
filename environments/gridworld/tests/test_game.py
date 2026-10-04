"""The game, a turn at a time: moving and walls, squares no two share, doors, gates and levers, chat, winning only
with every final plate pressed at once, and the turn budget."""

import pytest

from gridworld.game import Action, Game, names_for
from gridworld.level import Door, DoorKind, Level, Lever, Plate

WAIT = Action("wait")


def move(direction: str, message: str | None = None) -> Action:
    return Action("move", direction=direction, message=message)


def room(starts: tuple[tuple[int, int], ...], final: tuple[tuple[int, int], ...] = ()) -> Level:
    """An open room of five by three: x from 1 to 5, y from 1 to 3."""
    walls = ("#######", "#.....#", "#.....#", "#.....#", "#######")
    return Level(walls, starts, tuple(Plate(index + 1, at) for index, at in enumerate(final)))


def game(level: Level, *, turns: int = 10, seed: int = 0) -> Game:
    return Game(level, names_for(len(level.starts), seed), turns, seed)


def test_an_agent_moves_one_square_and_a_wall_stops_it() -> None:
    played = game(room(((1, 1), (5, 3))))
    played.step([move("east"), move("south")])
    assert played.positions == [(2, 1), (5, 3)]
    assert played.outcomes[0] == "You moved east."
    assert played.outcomes[1] == "You could not move south: a wall is in the way."
    played.step([move("north"), move("west")])
    assert played.positions == [(2, 1), (4, 3)]
    assert played.counts["move"] == 4 and played.counts["blocked"] == 2


def test_two_agents_stepping_onto_one_square_one_gets_there_by_the_seed() -> None:
    winners: set[int] = set()
    for seed in range(20):
        played = game(room(((1, 2), (3, 2))), seed=seed)
        played.step([move("east"), move("west")])
        again = game(room(((1, 2), (3, 2))), seed=seed)
        again.step([move("east"), move("west")])
        assert played.positions == again.positions, "the same seed breaks the tie the same way"
        (winner,) = [index for index, at in enumerate(played.positions) if at == (2, 2)]
        loser = 1 - winner
        assert played.positions[loser] == ((1, 2), (3, 2))[loser]
        assert played.outcomes[loser].endswith(f"{played.names[winner]} got there first.")
        winners.add(winner)
    assert winners == {0, 1}, "either can win, as the seed has it"


def test_no_one_steps_onto_a_square_whose_occupant_stays_or_through_one_coming_the_other_way() -> None:
    played = game(room(((1, 2), (2, 2), (3, 2))))
    played.step([move("east"), move("east"), WAIT])  # (a chain stopped at its head stops whole)
    assert played.positions == [(1, 2), (2, 2), (3, 2)]
    assert played.outcomes[:2] == [
        f"You could not move east: {played.names[1]} is in the way.",
        f"You could not move east: {played.names[2]} is in the way.",
    ]
    swapped = game(room(((1, 2), (2, 2))))
    swapped.step([move("east"), move("west")])
    assert swapped.positions == [(1, 2), (2, 2)]
    assert swapped.outcomes == [
        f"You could not move east: you and {swapped.names[1]} bumped.",
        f"You could not move west: you and {swapped.names[0]} bumped.",
    ]


def test_an_agent_follows_into_a_square_being_left() -> None:
    played = game(room(((1, 2), (2, 2), (3, 2))))
    played.step([move("east"), move("east"), move("east")])
    assert played.positions == [(2, 2), (3, 2), (4, 2)]


def test_the_team_wins_only_with_every_final_plate_pressed_in_the_same_turn() -> None:
    played = game(room(((1, 1), (5, 3)), final=((2, 1), (4, 3))))
    played.step([move("east"), WAIT])
    assert not played.solved and not played.over
    played.step([move("west"), move("west")])  # (one steps off as the other steps on)
    assert not played.solved
    played.step([move("east"), WAIT])
    assert played.solved and played.over and played.turn == 3
    with pytest.raises(RuntimeError, match="over"):
        played.step([WAIT, WAIT])


def test_the_game_ends_unsolved_when_the_turns_are_spent() -> None:
    played = game(room(((1, 1), (5, 3)), final=((2, 1), (4, 3))), turns=3)
    for _ in range(3):
        played.step([WAIT, WAIT])
    assert played.over and not played.solved and played.turn == 3


def test_what_is_said_reaches_everyone_and_moving_or_waiting_can_carry_it() -> None:
    played = game(room(((1, 1), (5, 3), (3, 2))))
    played.step([move("east", "I take plate 1"), Action("say", message="ok"), Action("wait", message="me too")])
    assert [(line.turn, line.speaker, line.message) for line in played.chat] == [
        (1, 0, "I take plate 1"),
        (1, 1, "ok"),
        (1, 2, "me too"),
    ]
    assert played.positions[0] == (2, 1), "talking took no turn"
    assert played.outcomes[1] == "You spoke and stayed where you were."


def test_a_call_that_is_no_action_does_nothing_and_says_why() -> None:
    played = game(room(((1, 1), (5, 3))))
    played.step([Action("none"), Action("invalid", problem="there is no tool jump", extra=True)])
    assert played.positions == [(1, 1), (5, 3)]
    assert played.outcomes == [
        "You called no tool, so you did nothing.",
        "Nothing happened: there is no tool jump. Only your first call counted.",
    ]


TWO_ROOMS = ("#######", "#..#..#", "#.....#", "#..#..#", "#######")
"""Two rooms of two by three, and a doorway between them at (3, 2)."""


def test_a_door_with_plates_opens_for_good_when_they_are_pressed_at_once() -> None:
    plates = (Plate(1, (1, 1), "a"), Plate(2, (1, 3), "a"), Plate(3, (5, 1)), Plate(4, (5, 3)))
    level = Level(TWO_ROOMS, ((2, 1), (2, 3)), plates, (Door("a", (3, 2), DoorKind.PLATES),))
    played = game(level)
    played.step([move("west"), WAIT])
    assert played.events == [] and not played.is_open(level.doors[0])
    played.step([move("east"), move("west")])  # (one steps off as the other steps on: not at once)
    assert played.events == []
    played.step([move("west"), WAIT])
    assert played.events == ["Plates 1 and 2 were pressed at once: door a is open for good."]
    assert played.opened == {"a": 3}
    played.step([move("east"), move("east")])
    assert played.is_open(level.doors[0]) and played.events == []


def test_a_closed_door_is_in_the_way() -> None:
    level = Level(
        TWO_ROOMS, ((2, 2), (1, 1)), (Plate(1, (5, 1)), Plate(2, (5, 3))), (Door("a", (3, 2), DoorKind.LEVER),)
    )
    played = game(level)
    played.step([move("east"), WAIT])
    assert played.positions[0] == (2, 2)
    assert played.outcomes[0] == "You could not move east: the closed door a is in the way."


def test_a_gate_is_open_only_while_its_plate_is_pressed_or_someone_stands_in_it() -> None:
    plates = (Plate(1, (2, 1), "a"), Plate(2, (5, 1)), Plate(3, (5, 3)))
    level = Level(TWO_ROOMS, ((1, 1), (2, 2)), plates, (Door("a", (3, 2), DoorKind.HELD),))
    gate = level.doors[0]
    played = game(level)
    played.step([WAIT, move("east")])
    assert played.positions[1] == (2, 2), "closed"
    played.step([move("east"), WAIT])
    assert played.is_open(gate) and played.events == ["Gate a opened."]
    played.step([WAIT, move("east")])
    assert played.positions[1] == (3, 2)
    played.step([move("west"), WAIT])  # (the plate let go while someone stands in the gate)
    assert played.is_open(gate) and played.events == []
    played.step([WAIT, move("east")])
    assert not played.is_open(gate) and played.events == ["Gate a closed."]
    played.step([WAIT, move("west")])
    assert played.positions[1] == (4, 2), "shut behind"


def test_a_lever_stepped_on_opens_its_door_for_good() -> None:
    walls = ("#######", "#.....#", "#######")
    level = Level(
        walls,
        ((1, 1), (2, 1)),
        (Plate(1, (5, 1)), Plate(2, (4, 1))),
        (Door("a", (3, 1), DoorKind.LEVER),),
        (Lever((1, 1), "a"),),
    )
    played = game(level)
    played.step([WAIT, WAIT])
    assert played.opened == {"a": 1}, "standing on it from the start pulls it on the first turn"
    assert played.events == [f"{played.names[0]} pulled the lever: door a is open for good."]
    played.step([WAIT, move("east")])
    assert played.positions[1] == (3, 1) and played.events == []
