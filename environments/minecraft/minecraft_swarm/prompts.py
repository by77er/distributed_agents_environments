"""What agents read and call: the system prompt, observations as text, and the actions as tools."""

from collections.abc import Mapping, Sequence
from typing import Any

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

SYSTEM = """You are {name}, one of four players in Minecraft: {team}. You play together.

{goal}

How the game runs: the world is frozen while you think. Each turn every player chooses exactly one action by calling \
one tool; then the world runs for up to five seconds while the actions happen, and freezes again. Long actions \
(walking far, tunnelling, fighting) may be cut off; repeat them to continue. You have about {minutes:g} minutes of \
game time; time passes only while actions happen.

What you know: you see only what is in your line of sight. Coordinates are (x, y, z): +x is east, +y is up, +z is \
south. If you die you start again where the game began. You see only your last few turns, so keep what matters in \
your notes (the note tool): they are shown to you every turn. The team board (the post tool) is shown to all four \
of you every turn; chat reaches teammates at their next turn. Use them to split up the work and to share what you \
find. Think briefly, then act."""


def system_prompt(name: str, task: Task) -> str:
    return SYSTEM.format(name=name, team=", ".join(TEAM), goal=GOALS[task.objective], minutes=task.minutes)


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
WAY: dict[str, JsonValue] = {"type": "string", "enum": ["down", "up"]}
SLOT: dict[str, JsonValue] = {"type": "string", "enum": ["hand", "off-hand", "head", "torso", "legs", "feet"]}
XYZ: dict[str, JsonValue] = {"x": _integer("x"), "y": _integer("y"), "z": _integer("z")}
AT = ["x", "y", "z"]

ACTIONS = [
    _action("move_to", "Walk to a place near something you have seen (digging through if needed).", XYZ, AT),
    _action("move", "Walk up to 32 blocks in a direction.", {"direction": DIRECTION, "blocks": COUNT}, ["direction"]),
    _action("mine", "Mine a block you can see within reach (4.5 blocks) and pick up what drops.", XYZ, AT),
    _action(
        "tunnel",
        "Dig a 1x2 tunnel north, south, east or west, up to 16 blocks; stops if lava comes into sight.",
        {"direction": DIRECTION, "length": _integer("blocks")},
        ["direction"],
    ),
    _action(
        "stairs",
        "Dig a staircase down (or up) in a direction (north, south, east or west), one block forward per step.",
        {"direction": DIRECTION, "steps": _integer("steps"), "way": WAY},
        ["direction"],
    ),
    _action("collect", "Pick up dropped items you can see nearby.", {}, []),
    _action(
        "craft",
        "Craft an item from your inventory (3x3 recipes need a crafting table within reach).",
        {"item": STRING, "count": COUNT},
        ["item"],
    ),
    _action(
        "place", "Place a block from your inventory next to you (e.g. a crafting table).", {"item": STRING}, ["item"]
    ),
    _action(
        "place_at",
        "Place a block from your inventory at a free position you see within reach; it needs a block next to it.",
        {"item": STRING, **XYZ},
        ["item", *AT],
    ),
    _action(
        "use",
        "Use an item: on a block you see within reach (flint and steel on obsidian, a bucket on water or lava, an eye "
        "of ender on a portal frame), or, without a position, in the air (an eye of ender flies toward a stronghold).",
        {"item": STRING, **XYZ},
        [],
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
    _action(
        "smelt",
        "Put items and fuel into a furnace within reach; each item takes 10 seconds.",
        {"item": STRING, "fuel": STRING, "count": COUNT},
        ["item"],
    ),
    _action("take_smelted", "Take what a furnace within reach has finished.", {}, []),
    _action("open_chest", "Look into a chest you can see within reach.", XYZ, AT),
    _action("take", "Take items from a chest within reach.", {**XYZ, "item": STRING, "count": COUNT}, [*AT, "item"]),
    _action("store", "Put items into a chest within reach.", {**XYZ, "item": STRING, "count": COUNT}, [*AT, "item"]),
    _action(
        "give",
        "Walk to a teammate you can see and toss them items.",
        {"to": STRING, "item": STRING, "count": COUNT},
        ["to", "item"],
    ),
    _action(
        "equip",
        "Hold an item, or wear armor (slot: head, torso, legs, feet).",
        {"item": STRING, "slot": SLOT},
        ["item"],
    ),
    _action("eat", "Eat food from your inventory.", {"item": STRING}, ["item"]),
    _action("sleep", "Sleep in a bed you can see within reach (at night).", XYZ, AT),
    _action("chat", "Say something to your teammates.", {"message": STRING}, ["message"]),
    _action(
        "note",
        "Replace your private notes (shown to you every turn): where things are, your plan, what is done.",
        {"text": STRING},
        ["text"],
    ),
    _action("post", "Add a line to the team board (shown to all four of you every turn).", {"text": STRING}, ["text"]),
    _action("wait", "Do nothing this turn.", {}, []),
]
"""The actions, as tools the model calls (one per turn). `note` and `post` are the agents' memory; the rest act in
the world."""

MEMORY_ACTIONS = ("note", "post")
"""Handled by the episode itself; they take no game time."""

MAX_NOTES = 1200
MAX_BOARD_LINES = 12


def describe(observation: Mapping[str, Any], *, minutes_left: float, notes: str = "", board: Sequence[str] = ()) -> str:
    """An observation as text: who and where you are, what you see, what happened, what you heard and remember."""
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
        f"{minutes_left:.1f} minutes left. You are at {where} ({', '.join(place)}); health {me['health']}/20, "
        f"food {me['food']}/20, light {world.get('light', '?')}.",
        f"Inventory: {_items(me['inventory'])}.{holding}{wearing}",
    ]
    if observation.get("died"):
        lines.append("You died since your last turn and respawned.")
    result = observation.get("last_action")
    if result:
        lines.append(f"Your last action: {_result(result)}")
    if observation.get("messages"):
        lines.append("Teammates said: " + " | ".join(f"{m['from']}: {m['message']}" for m in observation["messages"]))
    surroundings = observation["surroundings"]
    openings = ", ".join(
        f"{direction} {value['open']}" + (f" then {value['then']}" if "then" in value else "")
        for direction, value in surroundings.items()
    )
    lines.append(f"Open space: {openings}.")
    seen = ", ".join(f"{b['block']} ({b['count']})" for b in observation["visible_blocks"][:6])
    lines.append(f"You see mostly: {seen}.")
    if observation["notable"]:
        lines.append("Notable: " + "; ".join(_notable(kind) for kind in observation["notable"]) + ".")
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
    if me.get("near"):
        lines.append("Within reach: " + ", ".join(s["station"] for s in me["near"]) + ".")
    lines.append(f"Your notes: {notes or '(empty)'}")
    lines.append("Team board: " + (" | ".join(board) if board else "(empty)"))
    return "\n".join(lines)


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
        return f"{name} was cut off when the world froze; repeat it to continue."
    return f"{name} failed: {result.get('error', 'unknown error')}"
