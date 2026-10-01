"""What agents read and call: the system prompt, observations as text, and the actions as tools."""

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import JsonValue

from rollout.core.contracts import ToolSpecification

SYSTEM = """You are {name}, one of four players in Minecraft: {team}. You play together.

Goal: when the game ends, the four of you together should hold as many diamonds as possible. Only diamonds in your \
inventories count (a diamond block counts as 9); diamond ore does not count until it is mined and picked up.

How the game runs: the world is frozen while you think. Each turn every player chooses exactly one action by calling \
one tool; then the world runs for up to five seconds while the actions happen, and freezes again. Long actions \
(walking far, tunnelling) may be cut off; repeat them to continue. You have {turns} turns in all.

What you know: you see only what is in your line of sight. Coordinates are (x, y, z): +x is east, +y is up, +z \
is south. Diamond ore is deepest, around y = -58. Iron pickaxes or better mine diamond ore; stone \
pickaxes mine iron ore; iron ore must be smelted into ingots in a furnace before crafting. Your teammates hear what \
you say with the chat tool; use it to split up the work and to share what you find. Think briefly, then act."""

TEAM = ["ada", "ben", "cy", "dee"]


def _integer(description: str) -> dict[str, JsonValue]:
    return {"type": "integer", "description": description}


def _schema(properties: Mapping[str, JsonValue], required: Sequence[str]) -> dict[str, JsonValue]:
    return {"type": "object", "properties": dict(properties), "required": list(required)}


DIRECTION: dict[str, JsonValue] = {"type": "string", "enum": ["north", "south", "east", "west", "up", "down"]}
XYZ: dict[str, JsonValue] = {"x": _integer("x"), "y": _integer("y"), "z": _integer("z")}

STRING: dict[str, JsonValue] = {"type": "string"}
WAY: dict[str, JsonValue] = {"type": "string", "enum": ["down", "up"]}


def _action(
    name: str, description: str, properties: Mapping[str, JsonValue], required: Sequence[str]
) -> ToolSpecification:
    return ToolSpecification(name=name, description=description, input_schema=_schema(properties, required))


ACTIONS = [
    _action(
        "move_to",
        "Walk to a place near something you have seen (digging through if needed).",
        XYZ,
        ["x", "y", "z"],
    ),
    _action(
        "move",
        "Walk up to 32 blocks in a direction.",
        {"direction": DIRECTION, "blocks": _integer("how far")},
        ["direction"],
    ),
    _action(
        "mine",
        "Mine a block you can see within reach (4.5 blocks) and pick up what drops.",
        XYZ,
        ["x", "y", "z"],
    ),
    _action(
        "tunnel",
        "Dig a 1x2 tunnel north, south, east or west, up to 16 blocks; stops if lava comes into sight.",
        {"direction": DIRECTION, "length": _integer("blocks")},
        ["direction"],
    ),
    _action(
        "stairs",
        "Dig a staircase down (or up) in a direction, one block forward and one down per step.",
        {"direction": DIRECTION, "steps": _integer("steps"), "way": WAY},
        ["direction"],
    ),
    _action(
        "collect",
        "Pick up dropped items you can see nearby.",
        {},
        [],
    ),
    _action(
        "craft",
        "Craft an item from your inventory (3x3 recipes need a crafting table within reach).",
        {"item": STRING, "count": _integer("how many")},
        ["item"],
    ),
    _action(
        "place",
        "Place a block from your inventory next to you (e.g. crafting_table, furnace, torch).",
        {"item": STRING},
        ["item"],
    ),
    _action(
        "smelt",
        "Put items and fuel into a furnace within reach; each item takes 10 seconds.",
        {"item": STRING, "fuel": STRING, "count": _integer("how many")},
        ["item"],
    ),
    _action(
        "take_smelted",
        "Take what a furnace within reach has finished.",
        {},
        [],
    ),
    _action(
        "open_chest",
        "Look into a chest you can see within reach.",
        XYZ,
        ["x", "y", "z"],
    ),
    _action(
        "take",
        "Take items from a chest you can see within reach.",
        {**XYZ, "item": STRING, "count": _integer("how many")},
        ["x", "y", "z", "item"],
    ),
    _action(
        "give",
        "Walk to a teammate you can see and toss them items.",
        {"to": STRING, "item": STRING, "count": _integer("how many")},
        ["to", "item"],
    ),
    _action(
        "equip",
        "Hold an item.",
        {"item": STRING},
        ["item"],
    ),
    _action(
        "eat",
        "Eat food from your inventory.",
        {"item": STRING},
        ["item"],
    ),
    _action(
        "chat",
        "Say something to your teammates.",
        {"message": STRING},
        ["message"],
    ),
    _action(
        "wait",
        "Do nothing this turn.",
        {},
        [],
    ),
]
"""The harness's actions, as tools the model calls (one per turn)."""


def system_prompt(name: str, turns: int) -> str:
    return SYSTEM.format(name=name, team=", ".join(TEAM), turns=turns)


def describe(observation: Mapping[str, Any], turn: int, turns: int) -> str:
    """An observation as text: who and where you are, what you see, what happened, what you heard."""
    me = observation["self"]
    position = me["position"]
    where = f"({position['x']}, {position['y']}, {position['z']})"
    holding = f" Holding {me['holding']}." if me.get("holding") else ""
    lines = [
        f"Turn {turn} of {turns}. You are at {where}; health {me['health']}/20, food {me['food']}/20.",
        f"Inventory: {_items(me['inventory'])}.{holding}",
    ]
    if observation.get("died"):
        lines.append("You died since your last turn and respawned.")
    result = observation.get("last_action")
    if result:
        lines.append(f"Your last action: {_result(result)}")
    if observation.get("messages"):
        lines.append("Teammates said: " + " | ".join(f"{m['from']}: {m['message']}" for m in observation["messages"]))
    surroundings = observation["surroundings"]
    lines.append(
        "Open space: "
        + ", ".join(
            f"{direction} {value['open']}" + (f" then {value['then']}" if "then" in value else "")
            for direction, value in surroundings.items()
        )
        + "."
    )
    lines.append(
        "You see mostly: " + ", ".join(f"{b['block']} ({b['count']})" for b in observation["visible_blocks"][:6]) + "."
    )
    if observation["notable"]:
        lines.append(
            "Notable: "
            + "; ".join(
                f"{n['block']} at ({n['x']}, {n['y']}, {n['z']}), {n['distance']} away"
                for n in observation["notable"][:12]
            )
            + "."
        )
    if observation["items"]:
        lines.append(
            "Dropped items: "
            + "; ".join(f"{i['count']} {i['item']} at ({i['x']}, {i['y']}, {i['z']})" for i in observation["items"][:8])
            + "."
        )
    if observation["teammates"]:
        lines.append(
            "Teammates in sight: "
            + "; ".join(f"{t['name']} at ({t['x']}, {t['y']}, {t['z']})" for t in observation["teammates"])
            + "."
        )
    if observation["mobs"]:
        lines.append("Hostile: " + "; ".join(f"{m['mob']} {m['distance']} away" for m in observation["mobs"][:6]) + ".")
    if me.get("near"):
        lines.append("Within reach: " + ", ".join(s["station"] for s in me["near"]) + ".")
    return "\n".join(lines)


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
