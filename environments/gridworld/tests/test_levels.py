"""Levels: the same seed draws the same level, every level drawn can be solved, the steps before the final plates
grow from layout to layout, and the soundness check catches levels that cannot be solved."""

import pytest

from gridworld.environment import ROWS
from gridworld.level import STEPS, Door, DoorKind, Level, Lever, Plate, generate, opened, problems, reachable

SEEDS = range(100)
SHAPES = sorted({(str(row.parameters["layout"]), int(str(row.parameters["agents"]))) for row in ROWS})


@pytest.mark.parametrize(("layout", "agents"), SHAPES)
def test_the_same_seed_draws_the_same_level_and_others_differ(layout: str, agents: int) -> None:
    assert generate(layout, agents, 7) == generate(layout, agents, 7)
    assert len({generate(layout, agents, seed) for seed in range(20)}) == 20


@pytest.mark.parametrize(("layout", "agents"), SHAPES)
def test_every_level_drawn_is_sound(layout: str, agents: int) -> None:
    for seed in SEEDS:
        level = generate(layout, agents, seed)
        assert problems(level, agents) == [], (seed, level)
        assert len(level.starts) == agents and len(level.final) == agents
        doorways = {door.at for door in level.doors}
        beside = {(x + dx, y + dy) for x, y in doorways for dx, dy in STEPS}
        standing = {plate.at for plate in level.plates} | {lever.at for lever in level.levers} | set(level.starts)
        assert not beside & standing, "nothing stands by a doorway"
        assert opened(level, agents) == {door.label for door in level.doors}, "every door can be opened"


@pytest.mark.parametrize(("layout", "agents"), SHAPES)
def test_each_step_waits_behind_the_one_before(layout: str, agents: int) -> None:
    """With every door shut, no final plate can be reached; the lever only through its gate; the gate's plate only
    through the door before it."""
    for seed in range(20):
        level = generate(layout, agents, seed)
        every = {door.label for door in level.doors}
        shut = reachable(level, set(level.starts), set())
        if every:
            assert not any(plate.at in shut for plate in level.final), seed
        for gate in (door for door in level.doors if door.kind is DoorKind.HELD):
            without = reachable(level, set(level.starts), every - {gate.label})
            assert not any(lever.at in without for lever in level.levers), seed
            before = {door.label for door in level.doors if door.kind is DoorKind.PLATES}
            for plate in level.plates_of(gate.label):
                assert before or plate.at in shut, "without a door before it, the gate's plate is in the first room"
                for label in before:
                    assert plate.at not in reachable(level, set(level.starts), every - {label}), seed


def test_the_steps_grow_from_layout_to_layout() -> None:
    kinds = {layout: sorted(door.kind for door in generate(layout, agents, 0).doors) for layout, agents in SHAPES}
    assert kinds == {
        "open": [],
        "door": [DoorKind.PLATES],
        "gate": [DoorKind.HELD, DoorKind.LEVER],
        "vault": [DoorKind.HELD, DoorKind.LEVER, DoorKind.PLATES],
    }
    assert [len(kinds[str(row.parameters["layout"])]) for row in ROWS] == sorted(
        len(kinds[str(row.parameters["layout"])]) for row in ROWS
    )


def test_no_layout_or_too_many_agents_is_refused() -> None:
    with pytest.raises(ValueError, match="no layout"):
        generate("maze", 2, 0)
    with pytest.raises(ValueError, match="could be drawn"):
        generate("open", 40, 0)


ROOMS = (
    "#########",
    "#...#...#",
    "#...#...#",
    "#...#...#",
    "#########",
)
"""Two rooms of three by three, with a door to make between them at (4, 2)."""


def two_rooms(*, final: tuple[tuple[int, int], ...] = ((6, 1), (6, 3)), pair: int = 2) -> Level:
    walls: list[str] = list(ROOMS)
    walls[2] = "#.......#"
    pressing = [Plate(1, (1, 1), "a"), Plate(2, (1, 3), "a"), Plate(3, (2, 1), "a")][:pair]
    finals = [Plate(len(pressing) + index + 1, at) for index, at in enumerate(final)]
    return Level(tuple(walls), ((3, 1), (3, 3)), (*pressing, *finals), (Door("a", (4, 2), DoorKind.PLATES),))


def test_a_sound_level_has_no_problems() -> None:
    assert problems(two_rooms(), 2) == []


def test_a_door_needing_more_plates_than_agents_cannot_be_solved() -> None:
    found = problems(two_rooms(pair=3), 2)
    assert any("door a needs 3 plates" in problem for problem in found)
    assert any("final plates [4, 5] cannot be reached" in problem for problem in found)


def test_a_final_plate_for_each_agent() -> None:
    assert any("2 final plates, not 1" in problem for problem in problems(two_rooms(final=((6, 1),)), 2))


def test_plates_that_cut_the_floor_in_two_are_found() -> None:
    walls = ("########", "#......#", "########")
    level = Level(walls, ((1, 1), (6, 1)), (Plate(1, (3, 1)), Plate(2, (4, 1))))
    assert "plates and starts cut the floor in two" in problems(level, 2)


def test_a_gate_needs_someone_to_hold_it_while_another_passes() -> None:
    walls = ("#####", "#.#.#", "#...#", "#####")
    gate = Door("a", (1, 2), DoorKind.HELD)
    level = Level(walls, ((3, 2),), (Plate(1, (3, 1), "a"), Plate(2, (1, 1))), (gate,))
    assert any("gate a needs a plate, and an agent to hold it" in problem for problem in problems(level, 1))


def test_a_lever_behind_a_gate_opens_its_door() -> None:
    level = generate("gate", 3, 0)
    assert opened(level, 3) == {"a", "b"}
    no_lever = Level(level.walls, level.starts, level.plates, level.doors)
    assert any("door b has no lever" in problem for problem in problems(no_lever, 3))
    assert all(isinstance(lever, Lever) for lever in level.levers)
