"""The game: where everyone stands, which doors are open, what was said, and one turn at a time.

Every agent acts at once each turn. Talking takes no turn: what anyone says is heard by everyone from the next turn
on. Then the moves resolve: a step into a wall or a closed door fails; when several agents step onto one square,
one of them gets there, chosen by the level's seed and the turn, and the others stay put; an agent cannot step onto a
square whose occupant stays put, and two agents cannot pass through each other. Then the world answers what the
agents are standing on: a lever stepped on is pulled and opens its door for good, a door's plates all pressed at once
open it for good, and a gate is open while one of its plates is pressed or someone stands in it. The team wins when
every final plate is pressed at the end of a turn; the most final plates pressed at once at the end of a turn is kept
(`most_pressed`), for the reward's progress (`gridworld.scoring`).
"""

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from gridworld.level import Door, DoorKind, Level, Plate, Point

__all__ = ["DIRECTIONS", "NAMES", "Action", "Game", "Line", "names_for"]

DIRECTIONS: dict[str, Point] = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
NAMES = [
    "Ada", "Ben", "Cy", "Dee", "Eve", "Finn", "Gus", "Hal", "Ivy", "Jo", "Kit", "Lou",
    "Max", "Ned", "Oona", "Pia", "Rex", "Sal", "Tess", "Uma", "Vic", "Wes", "Yan", "Zed",
]  # fmt: skip
"""Names agents play under, with different initials: the map shows each teammate by theirs."""

type Kind = Literal["move", "wait", "say", "none", "invalid"]


def names_for(agents: int, seed: int) -> list[str]:
    """The names a start's agents play under, drawn with its seed: so that the policy learns no name's part."""
    return random.Random(f"names-{seed}").sample(NAMES, agents)


@dataclass(frozen=True)
class Action:
    kind: Kind
    direction: str | None = None
    message: str | None = None
    """What the agent says this turn, with whatever it does."""
    problem: str | None = None
    """Why the call was no action (`invalid`)."""
    extra: bool = False
    """It made more calls than the first, which count for nothing."""


@dataclass(frozen=True)
class Line:
    turn: int
    """The turn it was said in."""
    speaker: int
    message: str


@dataclass
class Game:
    level: Level
    names: Sequence[str]
    turns: int
    """The budget: the game ends unsolved after this many turns."""
    seed: int
    """Breaks ties for a square."""
    positions: list[Point] = field(default_factory=list[Point])
    turn: int = 0
    """Turns played."""
    opened: dict[str, int] = field(default_factory=dict[str, int])
    """The doors open for good, and the turn each opened in."""
    pulled: set[Point] = field(default_factory=set[Point])
    chat: list[Line] = field(default_factory=list[Line])
    outcomes: list[str] = field(default_factory=list[str])
    """What each agent's last action did."""
    events: list[str] = field(default_factory=list[str])
    """What changed in the world last turn."""
    counts: Counter[str] = field(default_factory=Counter[str])
    """Actions taken, by kind, and moves that failed (`blocked`)."""
    solved: bool = False
    most_pressed: int = 0
    """The most final plates pressed at once at the end of a turn."""

    def __post_init__(self) -> None:
        if len(self.names) != len(self.level.starts):
            raise ValueError(f"{len(self.level.starts)} agents need as many names, not {list(self.names)}")
        self.positions = self.positions or list(self.level.starts)
        self.outcomes = self.outcomes or [""] * len(self.names)

    @property
    def over(self) -> bool:
        return self.solved or self.turn >= self.turns

    def occupant(self, at: Point) -> int | None:
        return next((index for index, position in enumerate(self.positions) if position == at), None)

    def pressed(self, plate: Plate) -> bool:
        return plate.at in self.positions

    def is_open(self, door: Door) -> bool:
        if door.kind is DoorKind.HELD:
            return any(map(self.pressed, self.level.plates_of(door.label))) or door.at in self.positions
        return door.label in self.opened

    def blocked(self, at: Point) -> str | None:
        """What keeps anyone from stepping onto `at`: a wall, a closed door, or nothing."""
        if self.level.wall(at):
            return "a wall"
        door = self.level.door_at(at)
        if door is not None and not self.is_open(door):
            return f"the closed {'gate' if door.kind is DoorKind.HELD else 'door'} {door.label}"
        return None

    def step(self, actions: Sequence[Action]) -> None:
        """Play one turn: every agent's action at once, then the world."""
        if self.over:
            raise RuntimeError("the game is over")
        if len(actions) != len(self.names):
            raise ValueError(f"{len(self.names)} agents act each turn, not {len(actions)}")
        self.turn += 1
        gates = {door.label: self.is_open(door) for door in self.level.doors if door.kind is DoorKind.HELD}
        order = list(range(len(actions)))
        random.Random(f"{self.seed}-{self.turn}").shuffle(order)  # (who gets a square first)
        outcomes = [""] * len(actions)
        movers: dict[int, Point] = {}
        for index in order:
            action = actions[index]
            self.counts[action.kind] += 1
            match action.kind:
                case "move":
                    assert action.direction is not None
                    dx, dy = DIRECTIONS[action.direction]
                    x, y = self.positions[index]
                    target = (x + dx, y + dy)
                    if (reason := self.blocked(target)) is not None:
                        outcomes[index] = f"You could not move {action.direction}: {reason} is in the way."
                    else:
                        movers[index] = target
                case "wait":
                    outcomes[index] = "You waited."
                case "say":
                    outcomes[index] = "You spoke and stayed where you were."
                case "none":
                    outcomes[index] = "You called no tool, so you did nothing."
                case "invalid":
                    outcomes[index] = f"Nothing happened: {action.problem}."
        for target in set(movers.values()):
            contending = [index for index in order if movers.get(index) == target]
            for loser in contending[1:]:
                del movers[loser]
                direction = actions[loser].direction
                outcomes[loser] = f"You could not move {direction}: {self.names[contending[0]]} got there first."
        swapping = [
            index
            for index, target in movers.items()
            if (other := self.occupant(target)) is not None and movers.get(other) == self.positions[index]
        ]
        for index in swapping:  # (two cannot pass through each other)
            other = self.occupant(movers[index])
            assert other is not None
            outcomes[index] = f"You could not move {actions[index].direction}: you and {self.names[other]} bumped."
        for index in swapping:
            del movers[index]
        changed = True
        while changed:  # (until no one left moving would step onto someone who stays put)
            changed = False
            for index, target in movers.items():
                other = self.occupant(target)
                if other is None or other in movers:
                    continue
                outcomes[index] = f"You could not move {actions[index].direction}: {self.names[other]} is in the way."
                del movers[index]
                changed = True
                break
        for index, target in movers.items():
            self.positions[index] = target
            outcomes[index] = f"You moved {actions[index].direction}."
        self.counts["blocked"] += sum(action.kind == "move" for action in actions) - len(movers)
        for index, action in enumerate(actions):
            if action.message:
                self.chat.append(Line(self.turn, index, action.message))
            if action.extra:
                outcomes[index] += " Only your first call counted."
        self.outcomes = outcomes
        self.events = self._world(gates)
        pressed = sum(map(self.pressed, self.level.final))
        self.most_pressed = max(self.most_pressed, pressed)
        self.solved = bool(self.level.final) and pressed == len(self.level.final)

    def _world(self, gates: dict[str, bool]) -> list[str]:
        """Pull the levers stepped on and open the doors whose plates are all pressed; what changed."""
        events: list[str] = []
        for lever in self.level.levers:
            if lever.at in self.positions and lever.at not in self.pulled:
                self.pulled.add(lever.at)
                self.opened.setdefault(lever.door, self.turn)
                puller = self.names[self.positions.index(lever.at)]
                events.append(f"{puller} pulled the lever: door {lever.door} is open for good.")
        for door in self.level.doors:
            pressing = self.level.plates_of(door.label)
            if door.kind is DoorKind.PLATES and door.label not in self.opened and all(map(self.pressed, pressing)):
                self.opened[door.label] = self.turn
                numbers = " and ".join(str(plate.number) for plate in pressing)
                said = f"Plates {numbers} were pressed at once" if len(pressing) > 1 else f"Plate {numbers} was pressed"
                events.append(f"{said}: door {door.label} is open for good.")
            if door.kind is DoorKind.HELD and self.is_open(door) != gates[door.label]:
                events.append(f"Gate {door.label} {'opened' if self.is_open(door) else 'closed'}.")
        return events
