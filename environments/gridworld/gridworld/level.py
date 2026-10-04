"""Levels: rooms of floor walled off from each other, the doors between them, and the plates and levers that open
the doors.

A level is drawn from its layout, its number of agents and a seed (`generate`): the same three give the same level.
Every agent starts in the first room. The final plates, one for each agent, are in the last room; the team wins when
all of them are pressed at once. The rooms between are steps on the way:

- `open`: one room, with the starts and the final plates.
- `door`: two rooms. A pair of plates in the first opens the door to the second for good, when both are pressed at
  once.
- `gate`: a side room above the first, behind a gate that is open only while a plate in the first room is pressed,
  holds a lever; pulling it (stepping on it) opens the door to the last room for good. One agent holds the gate while
  another goes in, pulls the lever and comes back out.
- `vault`: the two together. A pair of plates opens the door from the first room to a hall; the hall's gate leads to
  the lever, and the lever opens the door from the hall to the last room.

Generation checks that what it drew can be solved (`problems`), and draws again until it can.
"""

import random
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "LAYOUTS",
    "Door",
    "DoorKind",
    "Level",
    "Lever",
    "Plate",
    "Point",
    "generate",
    "problems",
]

type Point = tuple[int, int]
"""(x, y): x grows to the east, y to the south."""

WALL = "#"
FLOOR = "."
STEPS: tuple[Point, ...] = ((0, -1), (0, 1), (1, 0), (-1, 0))
DRAWS = 200
"""Levels drawn before `generate` gives up on a layout."""


class DoorKind(StrEnum):
    PLATES = "plates"
    """Opens for good when all its plates are pressed at once."""
    HELD = "held"
    """A gate: open only while one of its plates is pressed, or someone stands in it."""
    LEVER = "lever"
    """Opens for good when its lever is pulled."""


@dataclass(frozen=True)
class Plate:
    number: int
    at: Point
    door: str | None = None
    """The label of the door it works; None for a final plate."""


@dataclass(frozen=True)
class Door:
    label: str
    at: Point
    kind: DoorKind


@dataclass(frozen=True)
class Lever:
    at: Point
    door: str


@dataclass(frozen=True)
class Level:
    walls: tuple[str, ...]
    """Rows of wall (`#`) and floor (`.`), north first; a door's square is floor."""
    starts: tuple[Point, ...]
    """Where each agent starts, in the order of the agents."""
    plates: tuple[Plate, ...]
    doors: tuple[Door, ...] = ()
    levers: tuple[Lever, ...] = ()

    @property
    def width(self) -> int:
        return len(self.walls[0])

    @property
    def height(self) -> int:
        return len(self.walls)

    def wall(self, at: Point) -> bool:
        x, y = at
        return not (0 <= y < self.height and 0 <= x < self.width) or self.walls[y][x] == WALL

    def floor(self) -> list[Point]:
        return [(x, y) for y in range(self.height) for x in range(self.width) if not self.wall((x, y))]

    @property
    def final(self) -> list[Plate]:
        """The plates the team wins on."""
        return [plate for plate in self.plates if plate.door is None]

    def plates_of(self, label: str) -> list[Plate]:
        return [plate for plate in self.plates if plate.door == label]

    def door_at(self, at: Point) -> Door | None:
        return next((door for door in self.doors if door.at == at), None)

    def lever_of(self, label: str) -> Lever | None:
        return next((lever for lever in self.levers if lever.door == label), None)


@dataclass(frozen=True)
class Room:
    x: int
    y: int
    width: int
    height: int

    def tiles(self) -> list[Point]:
        return [(x, y) for y in range(self.y, self.y + self.height) for x in range(self.x, self.x + self.width)]


class Plan:
    """A level being drawn: rooms in a row along the south (the band), perhaps one room above one of them, and the
    openings between them."""

    def __init__(self, rng: random.Random, widths: Sequence[int], height: int, above: tuple[int, int, int] | None):
        """`above`: the index of the band's room it is over, its width and its height."""
        self.rng = rng
        top = 1 + (above[2] + 1 if above else 0)
        self.band: list[Room] = []
        x = 1
        for width in widths:
            self.band.append(Room(x, top, width, height))
            x += width + 1
        self.above: Room | None = None
        if above:
            under, width, tall = above
            room = self.band[under]
            self.above = Room(room.x + rng.randint(0, room.width - width), 1, width, tall)
        self.grid = [[WALL] * x for _ in range(top + height + 1)]
        for room in [*self.band, *([self.above] if self.above else [])]:
            for tx, ty in room.tiles():
                self.grid[ty][tx] = FLOOR
        self.openings: list[Point] = []
        self.taken: set[Point] = set()

    def between(self, index: int) -> Point:
        """An opening in the wall between the band's room `index` and the next."""
        room = self.band[index]
        at = (room.x + room.width, self.rng.randint(room.y, room.y + room.height - 1))
        return self._open(at)

    def up(self) -> Point:
        """An opening in the wall between the room above and the room under it."""
        assert self.above is not None
        room = self.above
        return self._open((self.rng.randint(room.x, room.x + room.width - 1), room.y + room.height))

    def _open(self, at: Point) -> Point:
        self.grid[at[1]][at[0]] = FLOOR
        self.openings.append(at)
        return at

    def place(self, room: Room, count: int) -> list[Point]:
        """`count` squares of `room`, none taken and none next to an opening (where someone standing would be in the
        way). Raises `Redraw` when the room has too few."""
        near = {(x + dx, y + dy) for x, y in self.openings for dx, dy in STEPS}
        free = [tile for tile in room.tiles() if tile not in self.taken and tile not in near]
        if len(free) < count:
            raise Redraw
        chosen = self.rng.sample(free, count)
        self.taken.update(chosen)
        return chosen

    def walls(self) -> tuple[str, ...]:
        return tuple("".join(row) for row in self.grid)


class Redraw(Exception):
    """What was drawn does not fit: draw again."""


def numbered(groups: Iterable[tuple[Sequence[Point], str | None]]) -> tuple[Plate, ...]:
    """Plates numbered from 1 in the order given: the steps' plates first, the final plates last."""
    plates: list[Plate] = []
    for points, door in groups:
        first = len(plates) + 1
        plates += [Plate(first + index, at, door) for index, at in enumerate(points)]
    return tuple(plates)


def _open(agents: int, rng: random.Random) -> Level:
    plan = Plan(rng, [rng.randint(agents + 2, agents + 4)], rng.randint(3, 5), None)
    (room,) = plan.band
    starts = plan.place(room, agents)
    return Level(plan.walls(), tuple(starts), numbered([(plan.place(room, agents), None)]))


def _door(agents: int, rng: random.Random) -> Level:
    plan = Plan(rng, [rng.randint(3, 5), rng.randint(3, 4)], rng.randint(3, 5), None)
    first, last = plan.band
    door = Door("a", plan.between(0), DoorKind.PLATES)
    starts = plan.place(first, agents)
    plates = numbered([(plan.place(first, 2), "a"), (plan.place(last, agents), None)])
    return Level(plan.walls(), tuple(starts), plates, (door,))


def _gate(agents: int, rng: random.Random) -> Level:
    first = rng.randint(4, 6)
    plan = Plan(
        rng, [first, rng.randint(3, 4)], rng.randint(3, 4), (0, rng.randint(2, min(4, first)), rng.randint(1, 2))
    )
    start, last = plan.band
    assert plan.above is not None
    gate = Door("a", plan.up(), DoorKind.HELD)
    door = Door("b", plan.between(0), DoorKind.LEVER)
    starts = plan.place(start, agents)
    plates = numbered([(plan.place(start, 1), "a"), (plan.place(last, agents), None)])
    (lever,) = plan.place(plan.above, 1)
    return Level(plan.walls(), tuple(starts), plates, (gate, door), (Lever(lever, "b"),))


def _vault(agents: int, rng: random.Random) -> Level:
    hall = rng.randint(3, 5)
    widths = [rng.randint(4, 5), hall, rng.randint(3, 4)]
    plan = Plan(rng, widths, rng.randint(3, 4), (1, rng.randint(2, min(3, hall)), rng.randint(1, 2)))
    start, middle, last = plan.band
    assert plan.above is not None
    door = Door("a", plan.between(0), DoorKind.PLATES)
    gate = Door("b", plan.up(), DoorKind.HELD)
    final = Door("c", plan.between(1), DoorKind.LEVER)
    starts = plan.place(start, agents)
    plates = numbered([(plan.place(start, 2), "a"), (plan.place(middle, 1), "b"), (plan.place(last, agents), None)])
    (lever,) = plan.place(plan.above, 1)
    return Level(plan.walls(), tuple(starts), plates, (door, gate, final), (Lever(lever, "c"),))


LAYOUTS: dict[str, Callable[[int, random.Random], Level]] = {
    "open": _open,
    "door": _door,
    "gate": _gate,
    "vault": _vault,
}


def generate(layout: str, agents: int, seed: int) -> Level:
    """The level of `layout` for `agents` agents that `seed` draws. Raises `ValueError` for a layout there is not, or
    one that cannot be drawn for that many agents."""
    if layout not in LAYOUTS:
        raise ValueError(f"there is no layout {layout!r}: {', '.join(LAYOUTS)}")
    rng = random.Random(f"gridworld-{layout}-{agents}-{seed}")
    for _ in range(DRAWS):
        try:
            level = LAYOUTS[layout](agents, rng)
        except Redraw:
            continue
        if not problems(level, agents):
            return level
    raise ValueError(f"no level of {layout} for {agents} agents could be drawn")


def problems(level: Level, agents: int) -> list[str]:
    """Why `level` cannot be solved by `agents` agents, or why someone in it could cut others off; empty when it is
    sound:

    - the agents start on different squares of floor, and there is a final plate for each of them;
    - every door that opens with plates has no more plates than there are agents, and a gate has a plate and at
      least two agents (one to hold it, one to pass);
    - opening the doors in the order they can be opened (a door when its plates or lever can be reached, a gate
      when its plate can) reaches every final plate;
    - with every door open, the floor that is neither a plate nor a start is connected and touches every plate and
      start: someone standing on a plate, or still at their start, never cuts anyone off.
    """
    found: list[str] = []
    starts = list(level.starts)
    if len(starts) != agents or len(set(starts)) != agents:
        found.append(f"{agents} agents need {agents} different starts, not {starts}")
    if any(level.wall(at) for at in starts):
        found.append("an agent starts in a wall")
    marked = [plate.at for plate in level.plates] + [lever.at for lever in level.levers]
    if len(set(marked)) != len(marked) or set(marked) & set(starts):
        found.append("two things share a square")
    if any(level.door_at(at) for at in [*marked, *starts]):
        found.append("something is in a doorway")
    if len(level.final) != agents:
        found.append(f"{agents} agents need {agents} final plates, not {len(level.final)}")
    for door in level.doors:
        pressing = level.plates_of(door.label)
        if door.kind is DoorKind.PLATES and not 1 <= len(pressing) <= agents:
            found.append(f"door {door.label} needs {len(pressing)} plates pressed by {agents} agents")
        if door.kind is DoorKind.HELD and (not pressing or agents < 2):
            found.append(f"gate {door.label} needs a plate, and an agent to hold it while another passes")
        if door.kind is DoorKind.LEVER and level.lever_of(door.label) is None:
            found.append(f"door {door.label} has no lever")
    reached = reachable(level, set(starts), opened(level, agents))
    if missing := [plate.number for plate in level.final if plate.at not in reached]:
        found.append(f"final plates {missing} cannot be reached")
    standing = {plate.at for plate in level.plates} | set(starts)
    free = [at for at in level.floor() if at not in standing]
    connected: set[Point] = flood(level, {free[0]}, lambda at: at not in standing) if free else set()
    if len(connected) != len(free):
        found.append("plates and starts cut the floor in two")
    if any(not any((x + dx, y + dy) in connected for dx, dy in STEPS) for x, y in standing):
        found.append("a plate or a start is walled in by others")
    return found


def opened(level: Level, agents: int) -> set[str]:
    """The doors `agents` agents can open, opening each as soon as it can (a gate counts as open once its plate can
    be reached: one of them holds it while the others pass)."""
    done: set[str] = set()
    while True:
        reached = reachable(level, set(level.starts), done)
        more = {
            door.label for door in level.doors if door.label not in done and _can_open(level, door, reached, agents)
        }
        if not more:
            return done
        done |= more


def _can_open(level: Level, door: Door, reached: set[Point], agents: int) -> bool:
    pressing = [plate.at in reached for plate in level.plates_of(door.label)]
    match door.kind:
        case DoorKind.PLATES:
            return 1 <= len(pressing) <= agents and all(pressing)
        case DoorKind.HELD:
            return agents >= 2 and any(pressing)
        case DoorKind.LEVER:
            lever = level.lever_of(door.label)
            return lever is not None and lever.at in reached


def reachable(level: Level, starts: set[Point], open_doors: set[str]) -> set[Point]:
    """The floor that can be walked to from `starts`, through the doors in `open_doors` and no other."""

    def passable(at: Point) -> bool:
        door = level.door_at(at)
        return door is None or door.label in open_doors

    return flood(level, starts, passable)


def flood(level: Level, starts: set[Point], passable: Callable[[Point], bool]) -> set[Point]:
    seen = set(starts)
    queue = deque(starts)
    while queue:
        x, y = queue.popleft()
        for dx, dy in STEPS:
            step = (x + dx, y + dy)
            if step not in seen and not level.wall(step) and passable(step):
                seen.add(step)
                queue.append(step)
    return seen
