"""What agents read and call: the system prompt, an observation as text, the actions as tools, and a reply read as
an action.

An observation is everything an agent needs to act, so an agent is shown only the latest one: the turn, a map of the
whole level (space-separated, north up), where each teammate and each plate, door and lever is from it (`2N 1E`:
two squares north and one east), what each of those does and its state, the recent chat with each line's age in
turns, and what its last action did and what changed in the world.
"""

from collections.abc import Mapping, Sequence

from pydantic import JsonValue

from gridworld.game import DIRECTIONS, Action, Game
from gridworld.level import Door, DoorKind, Level, Plate, Point
from rollout.contracts import Message, ToolSpecification

__all__ = ["CHAT_LINES", "MESSAGE_CHARACTERS", "TOOLS", "observe", "offset", "parse", "system_prompt"]

CHAT_LINES = 10
"""The most recent lines of chat an observation shows."""
MESSAGE_CHARACTERS = 200
"""What is said is cut to this length."""

SAY: dict[str, JsonValue] = {"type": "string", "description": "Something to say to the team at the same time (free)."}
TOOLS = [
    ToolSpecification(
        name="move",
        description="Step one square north, south, east or west.",
        input_schema={
            "type": "object",
            "properties": {"direction": {"type": "string", "enum": list(DIRECTIONS)}, "say": SAY},
            "required": ["direction"],
        },
    ),
    ToolSpecification(
        name="wait",
        description="Stay where you are. Standing on a plate keeps it pressed.",
        input_schema={"type": "object", "properties": {"say": SAY}, "required": []},
    ),
    ToolSpecification(
        name="say",
        description="Say something to the whole team and stay where you are.",
        input_schema={"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
    ),
]

SYSTEM = """You are playing a cooperative game on a grid, in a team of {count}: {team}. The team wins when every final \
plate is pressed at the same time, each by a different one of you standing on it. You have {turns} turns.

Each turn, everyone acts at once: call exactly one tool (only your first call counts).
- move: step one square north, south, east or west.
- wait: stay where you are. Standing on a plate keeps it pressed.
- say: speak to the team and stay where you are.
Talking is free: move and wait take a `say` argument too. Everyone hears what is said from the next turn on.

Two of you cannot stand on the same square. When several step onto one square at once, one of them gets there and \
the others stay put. {obstacles} in the way.{mechanics}

Decide together who goes to which plate: one of you on each."""

COUNTS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
MECHANICS = {
    DoorKind.PLATES: "A door with plates opens for good when all its plates are pressed at once.",
    DoorKind.HELD: "A gate is open only while its plate is pressed: one of you holds it while others pass.",
    DoorKind.LEVER: "A lever opens its door for good when someone steps on it.",
}


def system_prompt(level: Level, names: Sequence[str], turns: int) -> str:
    """The same for every agent of a game (who each one is, its observations say)."""
    kinds = [kind for kind in DoorKind if any(door.kind is kind for door in level.doors)]
    mechanics = " ".join(MECHANICS[kind] for kind in kinds)
    mechanics = f" {mechanics} Your observations say what each plate, door and lever does." if mechanics else ""
    obstacles = "Walls and closed doors are" if level.doors else "Walls are"
    count = COUNTS.get(len(names), str(len(names)))
    team = ", ".join(names)
    return SYSTEM.format(count=count, team=team, turns=turns, obstacles=obstacles, mechanics=mechanics)


def offset(origin: Point, at: Point) -> str:
    """Where `at` is from `origin`: `2N 1E`, `3W`, or `here`."""
    dx, dy = at[0] - origin[0], at[1] - origin[1]
    parts = ([f"{-dy}N"] if dy < 0 else [f"{dy}S"] if dy > 0 else []) + (
        [f"{dx}E"] if dx > 0 else [f"{-dx}W"] if dx < 0 else []
    )
    return " ".join(parts) or "here"


def observe(game: Game, agent: int) -> str:
    """What `agent` sees at the start of the game's next turn."""
    level, me = game.level, game.positions[agent]
    lines = [f"Turn {game.turn + 1} of {game.turns}. You are {game.names[agent]}.", "", "Map (north is up):"]
    lines += [" ".join(row) for row in _map(game, agent)]
    lines += [_legend(game, agent), "", "Teammates (where they are from you):"]
    lines += [
        f"- {name}: {offset(me, game.positions[index])}" for index, name in enumerate(game.names) if index != agent
    ]
    lines += ["", "Plates, doors and levers (where they are from you):"]
    lines += [f"- plate {plate.number}: {offset(me, plate.at)}. {_plate(game, plate, agent)}" for plate in level.plates]
    lines += [f"- {_name(door)}: {offset(me, door.at)}. {_door(game, door)}" for door in level.doors]
    lines += [
        f"- lever: {offset(me, lever.at)}. Step on it to pull it: it opens door {lever.door} for good. "
        + ("Pulled." if lever.at in game.pulled else "Not pulled.")
        for lever in level.levers
    ]
    lines += ["", "Chat, oldest first:"]
    heard = game.chat[-CHAT_LINES:]
    lines += [
        f"- {'you' if line.speaker == agent else game.names[line.speaker]}, {_ago(game.turn + 1 - line.turn)}: "
        f"{line.message}"
        for line in heard
    ] or ["(Nothing has been said yet.)"]
    lines.append("")
    if game.turn == 0:
        lines.append("This is the first turn.")
    else:
        lines.append(" ".join(["Last turn:", game.outcomes[agent], *game.events]))
    return "\n".join(lines)


def _map(game: Game, agent: int) -> list[list[str]]:
    level = game.level
    grid = [list(row) for row in level.walls]
    for plate in level.plates:
        grid[plate.at[1]][plate.at[0]] = str(plate.number)
    for lever in level.levers:
        grid[lever.at[1]][lever.at[0]] = "="
    for door in level.doors:
        grid[door.at[1]][door.at[0]] = "/" if game.is_open(door) else "+"
    for index, (x, y) in enumerate(game.positions):
        grid[y][x] = "@" if index == agent else game.names[index][0]
    return grid


def _legend(game: Game, agent: int) -> str:
    level = game.level
    others = [f"{name[0]} {name}" for index, name in enumerate(game.names) if index != agent]
    numbers = [plate.number for plate in level.plates]
    parts = ["@ you", *others, f"{numbers[0]}-{numbers[-1]} plates" if len(numbers) > 1 else "1 a plate"]
    if level.doors:
        parts += ["+ a closed door", "/ an open door"]
    if level.levers:
        parts.append("= a lever")
    return ", ".join([*parts, "# wall", ". floor"]) + "."


def _name(door: Door) -> str:
    return f"{'gate' if door.kind is DoorKind.HELD else 'door'} {door.label}"


def _plate(game: Game, plate: Plate, agent: int) -> str:
    if plate.door is None:
        does = "A final plate."
    else:
        door = next(door for door in game.level.doors if door.label == plate.door)
        if door.kind is DoorKind.HELD:
            does = f"Holds gate {door.label} open while pressed."
        else:
            others = [str(each.number) for each in game.level.plates_of(door.label) if each != plate]
            together = f"With plate {' and '.join(others)}, it" if others else "It"
            does = f"{together} opens door {door.label} for good."
    standing = game.occupant(plate.at)
    if standing is None:
        return f"{does} Not pressed."
    return f"{does} Pressed by {'you' if standing == agent else game.names[standing]}."


def _door(game: Game, door: Door) -> str:
    numbers = " and ".join(str(plate.number) for plate in game.level.plates_of(door.label))
    match door.kind:
        case DoorKind.PLATES:
            does = f"Opens for good when plates {numbers} are pressed at once."
        case DoorKind.HELD:
            does = f"Open only while plate {numbers} is pressed, or someone stands in it."
        case DoorKind.LEVER:
            does = "Opens for good when the lever is pulled."
    return f"{does} {'Open' if game.is_open(door) else 'Closed'}."


def _ago(turns: int) -> str:
    return f"{turns} turn{'s' * (turns != 1)} ago"


def parse(reply: Message) -> Action:
    """The action a reply chose: its first tool call, with what it says (`none` when it called no tool)."""
    calls = reply.tool_calls
    if not calls:
        return Action("none")
    call, extra = calls[0], len(calls) > 1
    arguments: Mapping[str, object] = call.arguments
    said = _said(arguments.get("message" if call.name == "say" else "say"))
    match call.name:
        case "move":
            direction = str(arguments.get("direction", "")).strip().lower()
            if direction not in DIRECTIONS:
                return Action(
                    "invalid", message=said, problem="move needs a direction: north, south, east or west", extra=extra
                )
            return Action("move", direction=direction, message=said, extra=extra)
        case "wait":
            return Action("wait", message=said, extra=extra)
        case "say":
            if said is None:
                return Action("invalid", problem="say needs a message", extra=extra)
            return Action("say", message=said, extra=extra)
        case _:
            return Action("invalid", message=said, problem=f"there is no tool {call.name}", extra=extra)


def _said(value: object) -> str | None:
    said = " ".join(str(value).split())[:MESSAGE_CHARACTERS] if isinstance(value, str) else ""
    return said or None
