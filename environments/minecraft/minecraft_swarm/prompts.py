"""What agents read and call: the system prompt, observations as text, and the actions as tools."""

from collections.abc import Mapping, Sequence
from typing import Any, cast

from pydantic import JsonValue

from minecraft_swarm.tasks import Objective, Task
from rollout.core.contracts import ToolSpecification

TEAM = ["ada", "ben", "cy", "dee"]

GOALS = {
    Objective.DIAMONDS: (
        "Goal: when the game ends, the four of you together should hold as many diamonds as possible. Only diamonds "
        "in your inventories count (a diamond block counts as 9); diamond ore does not count until it is mined and "
        "picked up."
    ),
    Objective.PROGRESS: (
        "Goal: get as far toward beating the game as you can, together. What counts, in order: mining stone, a stone "
        "pickaxe, smelting iron, an iron pickaxe, mining diamonds, forming obsidian, entering the nether, finding a "
        "fortress, getting a blaze rod, following eyes of ender into a stronghold, entering the end, and killing the "
        "ender dragon, which counts most. A step counts once, whoever does it."
    ),
}

SYSTEM = """You are one of four players in Minecraft: {team}. You play together. Each observation says which one \
you are.

{goal}

How the game runs: the world is frozen while you think. Each turn every player chooses exactly one action by calling \
one tool; then the world runs for up to five seconds while the actions happen, and freezes again. Long actions \
(walking far, digging through rock, fighting) may be cut off; repeat them to continue. Writing a note or posting to \
the board costs no game time.

What you know: only what you have seen with your own eyes. Each turn shows a map of what you have seen close around \
you, and lists notable things in sight farther off. Coordinates are absolute, (x, y, z): +x is east, +y is up, +z is \
south. Walking digs through what is in the way and picks up items it passes over. If you die you start again where \
the game began. You see your earlier turns only in brief, so keep what matters in your notes (the note tool): they \
are shown to you every turn. The team board (the post tool) is shown to all four of you every turn; chat reaches \
teammates at their next turn. Use them to split up the work and to share what you find. Think briefly, then act."""


def system_prompt(task: Task) -> str:
    """The same for all four agents, so that their prompts share it (and the tools) as a prefix the engine caches.

    It says nothing of how long the game lasts, and neither do observations: an episode's length is a limit of
    training, not of the game, and a policy told the clock learns to play the clock. Doing more before the episode is
    cut off is rewarded all the same."""
    return SYSTEM.format(team=", ".join(TEAM), goal=GOALS[task.objective])


def _integer(description: str) -> dict[str, JsonValue]:
    return {"type": "integer", "description": description}


def _schema(properties: Mapping[str, JsonValue], required: Sequence[str]) -> dict[str, JsonValue]:
    return {"type": "object", "properties": dict(properties), "required": list(required)}


def _action(
    name: str, description: str, properties: Mapping[str, JsonValue], required: Sequence[str]
) -> ToolSpecification:
    return ToolSpecification(name=name, description=description, input_schema=_schema(properties, required))


STRING: dict[str, JsonValue] = {"type": "string"}
COUNT = _integer("how many")
DIRECTION: dict[str, JsonValue] = {"type": "string", "enum": ["north", "south", "east", "west", "up", "down"]}
SLOT: dict[str, JsonValue] = {"type": "string", "enum": ["hand", "off-hand", "head", "torso", "legs", "feet"]}
INTEGER: dict[str, JsonValue] = {"type": "integer"}
XYZ: dict[str, JsonValue] = {"x": INTEGER, "y": INTEGER, "z": INTEGER}
AT = ["x", "y", "z"]

ACTIONS = [
    _action("move_to", "Walk to a place you have seen, digging through what is in the way.", XYZ, AT),
    _action(
        "move",
        "Walk up to 32 blocks in a direction, digging through what is in the way.",
        {"direction": DIRECTION, "blocks": COUNT},
        ["direction"],
    ),
    _action("mine", "Mine a block you can see within reach (4.5 blocks) and pick up what drops.", XYZ, AT),
    _action(
        "place_at",
        "Place a block from your inventory at a free position you see within reach; it needs a block next to it.",
        {"item": STRING, **XYZ},
        ["item", *AT],
    ),
    _action(
        "use",
        "Use an item (the one you hold, or `item`). Without a position: eat food, or throw an eye of ender. On a block "
        "you see within reach: flint and steel on obsidian, a bucket on water or lava, an eye of ender on a portal "
        "frame; on a chest it shows what is inside; on a bed you sleep.",
        {"item": STRING, **XYZ},
        [],
    ),
    _action(
        "craft",
        "Craft an item from your inventory (3x3 recipes need a crafting table within reach).",
        {"item": STRING, "count": COUNT},
        ["item"],
    ),
    _action(
        "smelt",
        "Put items and fuel into a furnace within reach; each item takes 10 seconds.",
        {"item": STRING, "fuel": STRING, "count": COUNT},
        ["item"],
    ),
    _action("take_smelted", "Take what a furnace within reach has finished.", {}, []),
    _action("take", "Take items from a chest within reach.", {**XYZ, "item": STRING, "count": COUNT}, [*AT, "item"]),
    _action("store", "Put items into a chest within reach.", {**XYZ, "item": STRING, "count": COUNT}, [*AT, "item"]),
    _action(
        "toss",
        "Throw items from your inventory, toward a position if given: a teammate standing there picks them up (you "
        "cannot pick them back up for five seconds).",
        {"item": STRING, "count": COUNT, **XYZ},
        ["item"],
    ),
    _action(
        "equip",
        "Hold an item, or wear armor (slot: head, torso, legs, feet).",
        {"item": STRING, "slot": SLOT},
        ["item"],
    ),
    _action(
        "attack",
        "Attack a creature you can see (by its id from the observation) until it dies or time runs out.",
        {"target": _integer("the creature's id")},
        ["target"],
    ),
    _action(
        "shoot",
        "Shoot arrows at a creature or an end crystal you can see (by its id); needs a bow and arrows.",
        {"target": _integer("the creature's id")},
        ["target"],
    ),
    _action("chat", "Say something to your teammates.", {"message": STRING}, ["message"]),
    _action(
        "note",
        "Replace your private notes (shown to you every turn): where things are, your plan, what is done.",
        {"text": STRING},
        ["text"],
    ),
    _action("post", "Add a line to the team board (shown to all four of you every turn).", {"text": STRING}, ["text"]),
    _action("wait", "Do nothing while the world runs (up to five seconds of game time pass).", {}, []),
]
"""The actions, as tools the model calls (one per turn): motor control, not strategy. `note` and `post` are the
agents' memory; the rest act in the world."""

MEMORY_ACTIONS = ("note", "post")
"""Handled by the episode itself; they take no game time."""

MAX_NOTES = 1200
MAX_BOARD_LINES = 12


def describe(
    observation: Mapping[str, Any],
    *,
    notes: str = "",
    board: Sequence[str] = (),
    brief: bool = False,
) -> str:
    """An observation as text: who and where you are, the map of what you have seen, what is in sight, what happened,
    what you heard and remember. `brief` keeps only where you were and what you held (how earlier turns are kept)."""
    me = observation["self"]
    position = me["position"]
    world: Mapping[str, Any] = observation.get("world") or {}
    where = f"({position['x']}, {position['y']}, {position['z']})"
    place = [str(me.get("dimension", "overworld"))]
    if world.get("biome"):
        place.append(str(world["biome"]))
    if place[0] == "overworld":  # the nether and the end have no day and no sky
        time: Mapping[str, Any] = world.get("time") or {}
        if time:
            place.append(str(time.get("phase")))
        if world.get("sky") is False:
            place.append("no sky above")
    holding = f" Holding {me['holding']}." if me.get("holding") else ""
    wearing = f" Wearing {', '.join(me['wearing'].values())}." if me.get("wearing") else ""
    lines = [
        f"You are {me['name']}, at {where} ({', '.join(place)}); "
        f"health {me['health']}/20, "
        f"food {me['food']}/20, light {world.get('light', '?')}.",
        f"Inventory: {_items(me['inventory'])}.{holding}{wearing}",
    ]
    if brief:
        return "\n".join(lines)
    if observation.get("died"):
        lines.append("You died since your last turn and respawned.")
    result = observation.get("last_action")
    if result:
        lines.append(f"Your last action: {_result(result)}")
    if observation.get("messages"):
        lines.append("Teammates said: " + " | ".join(f"{m['from']}: {m['message']}" for m in observation["messages"]))
    if observation.get("map"):
        lines.append(render_map(observation))
    if observation["notable"]:
        lines.append("Notable in sight: " + "; ".join(_notable(kind) for kind in observation["notable"]) + ".")
    if observation["items"]:
        dropped = "; ".join(
            f"{i['count']} {i['item']} at ({i['x']}, {i['y']}, {i['z']})" for i in observation["items"][:8]
        )
        lines.append(f"Dropped items: {dropped}.")
    if observation["teammates"]:
        mates = "; ".join(f"{t['name']} at ({t['x']}, {t['y']}, {t['z']})" for t in observation["teammates"])
        lines.append(f"Teammates in sight: {mates}.")
    if observation.get("mobs"):
        hostile = "; ".join(f"{m['mob']} (id {m['id']}) {m['distance']} away" for m in observation["mobs"][:6])
        lines.append(f"Hostile: {hostile}.")
    if observation.get("animals"):
        animals = "; ".join(f"{a['animal']} (id {a['id']}) {a['distance']} away" for a in observation["animals"][:6])
        lines.append(f"Animals: {animals}.")
    lines.append(f"Your notes: {notes or '(empty)'}")
    lines.append("Team board: " + (" | ".join(board) if board else "(empty)"))
    return "\n".join(lines)


# The map

UNSEEN, EMPTY, SOLID, PASSABLE, SELF = "?", ".", "#", ",", "@"
SYMBOLS = {
    "water": "~",
    "lava": "%",
    "fire": "^",
    "chest": "h",
    "trapped_chest": "h",
    "barrel": "h",
    "crafting_table": "t",
    "furnace": "f",
    "blast_furnace": "f",
    "obsidian": "o",
    "crying_obsidian": "o",
    "nether_portal": "p",
    "end_portal": "p",
    "end_portal_frame": "m",
    "spawner": "s",
    "bedrock": "k",
    "torch": "j",
    "wall_torch": "j",
    "gravel": "z",
    "sand": "z",
    "ancient_debris": "a",
}
"""Blocks with a symbol of their own on the map. Everything else is `#` if solid and `,` if it can be walked through."""
ORES = {
    "diamond": "d",
    "iron": "i",
    "coal": "c",
    "gold": "g",
    "redstone": "r",
    "lapis": "l",
    "emerald": "e",
    "copper": "u",
    "quartz": "q",
}
SUFFIXES = {"_log": "w", "_leaves": "v", "_bed": "b"}
ITEM, HOSTILE, ANIMAL = "*", "!", "&"
HEIGHTS = {2: "above your head", 1: "your head", 0: "your feet", -1: "the floor under you", -2: "below the floor"}


def symbol(name: str, solid: bool) -> str:
    """The map's character for a block."""
    if name == "air":
        return EMPTY
    if name in SYMBOLS:
        return SYMBOLS[name]
    if name.endswith("_ore"):
        ore = name.removesuffix("_ore").removeprefix("deepslate_").removeprefix("nether_")
        return ORES.get(ore, SOLID)
    for suffix, character in SUFFIXES.items():
        if name.endswith(suffix):
            return character
    return SOLID if solid else PASSABLE


def render_map(observation: Mapping[str, Any]) -> str:
    """The map as text: one grid per height, highest first; one character per block, north up and east right. Only
    what the agent has seen is on it. Teammates, creatures and dropped items in sight are drawn where they stand."""
    local: Mapping[str, Any] = observation["map"]
    center, radius = local["center"], int(local["radius"])
    side = 2 * radius + 1
    palette = [(str(entry["name"]), bool(entry["solid"])) for entry in local["palette"]]
    characters = [symbol(name, solid) for name, solid in palette]
    legend: dict[str, set[str]] = {}
    for (name, _), character in zip(palette, characters, strict=True):
        if character not in (EMPTY, SOLID):
            legend.setdefault(character, set()).add(name)

    marks: dict[tuple[int, int, int], str] = {(int(center["x"]), int(center["y"]), int(center["z"])): SELF}
    for mate in observation.get("teammates", []):
        marks[(mate["x"], mate["y"], mate["z"])] = str(mate["name"])[:1].upper()
        legend.setdefault(str(mate["name"])[:1].upper(), set()).add(str(mate["name"]))
    for kind, character, meaning in (
        ("items", ITEM, "a dropped item"),
        ("mobs", HOSTILE, "a hostile creature"),
        ("animals", ANIMAL, "an animal"),
    ):
        for entity in observation.get(kind, []):
            marks.setdefault((entity["x"], entity["y"], entity["z"]), character)
            legend.setdefault(character, set()).add(meaning)

    west, north = int(center["x"]) - radius, int(center["z"]) - radius
    lines = [
        f"Map of what you have seen within {radius} blocks, one grid per height, highest first. North is up, east is "
        f"right: columns are x={west} to x={west + side - 1}, and each row starts with its z. "
        f"{UNSEEN} not seen, {EMPTY} empty, {SOLID} solid, {PASSABLE} something you can walk through, {SELF} you."
    ]
    shown: set[str] = set()
    for layer in local["layers"]:
        y, title = int(layer["y"]), HEIGHTS.get(int(layer["dy"]), "")
        rows: list[list[str]] = []
        for row in range(side):
            indices = [int(index) for index in layer["cells"][row * side : (row + 1) * side]]
            cells: list[str] = [UNSEEN if index < 0 else characters[index] for index in indices]
            for column in range(side):
                mark = marks.get((west + column, y, north + row))
                if mark is not None:
                    cells[column] = mark
            rows.append(cells)
        kinds = {cell for cells in rows for cell in cells}
        shown |= kinds
        heading = f"y={y} ({title})" if title else f"y={y}"
        if kinds == {UNSEEN}:
            lines.append(f"{heading}: not seen")
        elif len(kinds - {UNSEEN}) == 1:  # one kind of thing wherever it was seen: a line says as much as a grid
            lines.append(f"{heading}: all {next(iter(kinds - {UNSEEN}))} where seen")
        else:
            lines.append(f"{heading}:")
            lines.extend(f"{north + row} {' '.join(cells)}" for row, cells in enumerate(rows))
    meanings = [
        f"{character} {'/'.join(sorted(names))}" for character, names in sorted(legend.items()) if character in shown
    ]
    if meanings:
        lines.append("On the map: " + "; ".join(meanings) + ".")
    return "\n".join(lines)


def _position(position: Any) -> str:
    if not isinstance(position, Mapping):
        return "where you are now"
    place = cast(Mapping[str, Any], position)
    return f"({place.get('x')}, {place.get('y')}, {place.get('z')})"


def _notable(kind: Mapping[str, Any]) -> str:
    """One kind of block in sight: the nearest, how many, and where the next few are."""
    count = int(kind.get("count", 1))
    text = f"{kind['block']} at ({kind['x']}, {kind['y']}, {kind['z']}), {kind['distance']} away"
    also = ", ".join(f"({other['x']}, {other['y']}, {other['z']})" for other in kind.get("also", []))
    if count > 1:
        text += f" ({count} in sight" + (f"; also {also}" if also else "") + ")"
    return text


def _items(inventory: Mapping[str, int]) -> str:
    return ", ".join(f"{count} {name}" for name, count in sorted(inventory.items())) or "empty"


def _result(result: Mapping[str, Any]) -> str:
    action: Mapping[str, Any] = result.get("action") or {}
    name = str(action.get("name", "?"))
    if result.get("ok"):
        details = {key: value for key, value in result.items() if key not in ("action", "ok")}
        return f"{name} succeeded: {details}" if details else f"{name} succeeded."
    if result.get("interrupted"):
        return f"{name} was cut off when the world froze; you got to {_position(result.get('now_at'))}."
    return f"{name} failed: {result.get('error', 'unknown error')}"
