"""What agents read and call: the system prompt, observations as text, and the actions as tools."""

from collections.abc import Mapping, Sequence
from typing import Any, cast

from pydantic import JsonValue

from minecraft_team.limits import LIMITS
from minecraft_team.tasks import CHAINS, EARLY, KITS, TEAM, Coordination, Kit, Objective, Start, Task
from rollout.contracts import ToolSpecification

NUMBERS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
]  # fmt: skip


def spelled(number: int) -> str:
    """A small whole number as a sentence has it ("five seconds")."""
    return NUMBERS[number] if 0 <= number < len(NUMBERS) else str(number)


def listed(names: Sequence[str], last: str) -> str:
    """Names as a sentence lists them: "a, b or c"."""
    return f"{', '.join(names[:-1])} {last} {names[-1]}" if len(names) > 1 else "".join(names)


DIAMONDS_GOAL = (
    "Goal: together, hold as many diamonds as you can. Only diamonds in your inventories count (a diamond block "
    "counts as 9). Diamonds may be lying on the ground, stored in chests, or still in ore; ore counts only once "
    "it is mined and picked up. What is around you differs from game to game."
)
PROGRESS_GOAL = (
    "Goal: get as far toward beating the game as you can, together. What counts, in order: {early}mining stone, a "
    "stone pickaxe, smelting iron, an iron pickaxe, mining diamonds, getting obsidian, entering the nether, finding a "
    "fortress, getting a blaze rod, following eyes of ender into a stronghold, entering the end, and killing the "
    "ender dragon, which counts most; hurting the dragon without killing it counts for a little. A step counts "
    "once, whoever does it, and only if it is done in this game: what you start with does not count."
)
"""`early` names the steps that count before mining stone, each followed by a comma: none, or those of `EARLY`."""
CRAFT_GOAL = (
    "Goal: together, make {item}. You start with nothing: everything it takes must be gathered and crafted. Getting "
    "there counts step by step, each step once, whoever does it: {steps}. The game is over when it is made."
)
"""What each objective asks, in the words agents read: what is scored. How to get there is said apart (`way`), from
what the team starts with."""

TABLE_NEEDED = "recipes on a 3x3 grid need a crafting table within reach"
MAKING = {
    "logs": "Get logs: mine the trunk of a tree (any *_log); it needs no tool.",
    "planks": "Craft planks from a log: one log makes four, named after its wood (oak_log makes oak_planks).",
    "a crafting table": f"Craft a crafting_table from four planks and place it (place_at): {TABLE_NEEDED}.",
    "sticks": "Craft sticks from two planks (they make four).",
    "a wooden pickaxe": "Within reach of the table, craft a wooden_pickaxe: three planks and two sticks.",
    "cobblestone": "Mine stone with any pickaxe: it drops cobblestone (deepslate drops cobbled_deepslate, which serves "
    "as well). Without a pickaxe, stone drops nothing.",
    "a stone pickaxe": "Within reach of the table, craft a stone_pickaxe: three cobblestone and two sticks.",
    "a furnace": "Within reach of the table, craft a furnace from eight cobblestone, and place it (place_at).",
    "raw iron": "Find iron ore (iron_ore, or deepslate_iron_ore deep down) and mine it with a stone pickaxe or better: "
    "it drops raw_iron (to a wooden pickaxe it drops nothing). An iron pickaxe takes three.",
    "an iron ingot": "Within reach of the placed furnace, smelt raw_iron with the count you want (one goes in "
    f"otherwise), with fuel you carry ({listed(LIMITS.fuels, 'or')}). Each takes {LIMITS.smelt_seconds} seconds of "
    "game time, which passes while actions happen; then take the iron_ingot out with take_smelted.",
    "an iron pickaxe": "Within reach of the table, craft an iron_pickaxe: three iron_ingot and two sticks.",
    "coal or charcoal": "Get coal: mine coal ore with any pickaxe (it drops coal), or smelt a log in the furnace for "
    "charcoal.",
    "torches": "Craft torches from coal (or charcoal) and a stick.",
    "a bucket": "Within reach of the table, craft a bucket: three iron_ingot.",
    "a shield": "Within reach of the table, craft a shield: six planks and one iron_ingot.",
    "diamonds": "Find diamond ore: it occurs only deep underground, below y = 16 (as deepslate_diamond_ore below "
    "y = 0), and only an iron pickaxe or better gets a diamond out of it.",
    "a diamond pickaxe": "Within reach of the table, craft a diamond_pickaxe: three diamonds and two sticks.",
}
"""How to make each step of a crafting chain (`CHAINS`), in the words agents read. Mining takes the best tool an
agent carries by itself, so no step says to hold one."""
MAKING["a diamond"] = MAKING["diamonds"]

PLACE_TABLE = f"Place the crafting_table (place_at): {TABLE_NEEDED}."
PLACE_FURNACE = "Place the furnace (place_at)."
FUEL = "Fuel for the furnace: mine coal ore with any pickaxe (it drops coal), or burn planks or logs."
MINE_DIAMONDS = (
    "Mine the diamond ore within reach: the diamond drops and you pick it up (mining takes the best tool you carry by "
    "itself)."
)
DIAMOND_WAYS: dict[Kit, list[str]] = {
    Kit.IRON: [MAKING["diamonds"], MINE_DIAMONDS],
    Kit.INGOTS: [PLACE_TABLE, MAKING["an iron pickaxe"], MAKING["diamonds"], MINE_DIAMONDS],
    Kit.RAW_IRON: [PLACE_FURNACE, MAKING["an iron ingot"], PLACE_TABLE, MAKING["an iron pickaxe"], MAKING["diamonds"],
                   MINE_DIAMONDS],
    Kit.STONE: [MAKING["raw iron"], PLACE_FURNACE, MAKING["an iron ingot"], PLACE_TABLE, MAKING["an iron pickaxe"],
                MAKING["diamonds"], MINE_DIAMONDS],
    Kit.WOODEN: [PLACE_TABLE, MAKING["cobblestone"], MAKING["a stone pickaxe"], MAKING["raw iron"], MAKING["a furnace"],
                 FUEL, MAKING["an iron ingot"], MAKING["an iron pickaxe"], MAKING["diamonds"], MINE_DIAMONDS],
    Kit.NOTHING: [*(MAKING[name] for name, _, _ in CHAINS["iron_pickaxe"]), FUEL, MAKING["diamonds"], MINE_DIAMONDS],
}  # fmt: skip
"""The way to diamonds from each kit, step by step."""
NO_DIGGING = "You have no pickaxe, and nothing here needs digging: the rock around you drops nothing when mined."
TAKE = "take the diamonds out (take, at a chest within reach)"


def laid_out(task: Task, players: int) -> str:
    """Where the diamonds of a staged start are and how to get to them: the rooms `tasks.build` carves for it (a
    middle room, and a corridor out of each of its walls bending to a side room)."""
    you = "You each start" if players > 1 else "You start"
    match task.start, task.apart:
        case Start.ITEMS, False:
            return "The diamonds here lie on the floor of the room you start in: walk over them (move) to pick them up."
        case Start.ITEMS, True:
            return (
                f"{you} in a side room; the diamonds lie on the floor of the middle room. A corridor leads out of your "
                f"room and bends to it: walk there, and over the diamonds to pick them up (move). {NO_DIGGING}"
            )
        case Start.CHESTS, False:
            return (
                "The diamonds here are in four chests, one in each of four side rooms. A corridor leads out of each "
                f"wall of the room you start in and bends to a side room: walk there (move), then {TAKE}. {NO_DIGGING}"
            )
        case Start.CHESTS, True:
            return (
                f"{you} in a side room, beside a chest: {TAKE}. Each of the four side rooms has a chest. A corridor "
                f"leads out of your room and bends to a middle room, and from it to each of the others: walk there "
                f"(move) for the rest. {NO_DIGGING}"
            )
        case _:
            return ""


SYSTEM = """{opening}

{goal}{way}

How the game runs: the world is frozen while you think. {turns} An action that takes more than {window} seconds (a \
long walk, digging far through rock, a long fight) is cut off there: the result says where you got to.

What you know: only what you have seen with your own eyes. Each turn shows a map of what you have seen close around \
you, names the blocks that touch you, and lists notable things in sight farther off. Coordinates are absolute, \
(x, y, z): +x is east, +y is up, +z is south. Walking picks up items it passes over. If you die you start again \
where the game began, {death}. Your memory is limited: you see your recent turns without their maps, and anything \
older only as a summary that you write yourself when asked.{chat}{teamwork}

When an action fails, or you are getting no closer to the goal, stop and think: what does the goal need that you do \
not have yet? Plan how to get those things, one step at a time, and act on the first. Do not try what already failed \
again unless something has changed. Otherwise, think briefly, then act."""

TEAM_OPENING = (
    "You are one of {count} players in Minecraft: {team}. You play together. Each observation says which one \
you are."
)
ALONE_OPENING = "You are {name}, playing Minecraft on your own."
TEAM_TURNS = "Each turn every player chooses exactly one action by calling one tool (only your first call counts); \
then the world runs until all {count} actions have finished, and freezes again."
ALONE_TURNS = "Each turn you choose exactly one action by calling one tool (only your first call counts); then the \
world runs until it has finished, and freezes again."
CHAT = " Chat reaches teammates at their next turn; every observation shows the team's last {chat_lines} messages and \
how old each is."
TEAMWORK = (
    "Work as a team: together you get further than apart. Talk in chat: early on, say what you carry and what you "
    "will do; split the work so that no two of you do the same thing; hand teammates what they need (toss); say what "
    "you find, and where; and answer when you are asked."
)
"""What a team is told about playing as one: guidance (`guidance`), as the way to the goal is."""

COMPACT = """The turns above are about to leave your memory. Write what you need to remember from them, and from \
your earlier summary if there is one, to keep playing well: what you have learned about the world (places and things, \
with their coordinates), what you and your teammates have done and agreed, and what you intend to do next. Be \
specific and brief, and leave out what no longer matters. Reply with the summary only."""
COMPACT_ALONE = COMPACT.replace("what you and your teammates have done and agreed", "what you have done")
"""What an agent is asked when its oldest turns are compacted into a summary."""

REMEMBERED = "What you remember from earlier in this game (your own summary):\n{summary}"
"""How its summary is shown to it afterwards."""

NO_CALL = "You called no tool, so you did nothing that turn."
"""What a turn in which an agent called no tool is answered with."""
ONE_CALL = "Not done: only your first call of a turn counts."
"""What every call of a turn after the first is answered with."""


def system_prompt(task: Task, team: Sequence[str]) -> str:
    """The same for every agent of the team: nothing in it says which of them reads it (each observation does). The
    team plays under the names in `team`; a player on their own is told so, and nothing of chat or teammates.

    It says nothing of how long the game lasts, and neither do observations: how long an episode lasts is no rule of
    the game, and an agent told the clock plays the clock. Doing more before the episode is cut off is rewarded all
    the same."""
    death = "with what you carried" if task.keeps_inventory else "and what you carried lies where you died"
    players = len(team)
    told = guidance(task, players)
    return SYSTEM.format(
        opening=TEAM_OPENING.format(count=spelled(players), team=", ".join(team)) if players > 1
        else ALONE_OPENING.format(name=team[0]),
        goal=goal(task, players),
        way="".join(f"\n\n{told['way']}" for _ in [0] if "way" in told),
        turns=TEAM_TURNS.format(count=spelled(players)) if players > 1 else ALONE_TURNS,
        window=spelled(LIMITS.window_seconds),
        chat=CHAT.format(chat_lines=CHAT_LINES) if players > 1 else "",
        teamwork=f" {told['teamwork']}" if "teamwork" in told else "",
        death=death,
    )  # fmt: skip


def guidance(task: Task, players: int = len(TEAM)) -> dict[str, str]:
    """The guidance in a task's system prompt, by kind, word for word: the way to the goal (`way`), and for a team
    how to play as one (`teamwork`). An episode reports it, so that a learner can take it out of the prompts again."""
    told = {"way": way(task, players), "teamwork": TEAMWORK if players > 1 else ""}
    return {kind: text for kind, text in told.items() if text}


def goal(task: Task, players: int = len(TEAM)) -> str:
    """What the task asks, as agents read it (for one player, as one player reads it)."""
    if task.objective is Objective.DIAMONDS:
        text = DIAMONDS_GOAL
    elif task.objective is Objective.PROGRESS:
        early = [name for name, _, _ in EARLY] if task.counts_early_steps else []
        text = PROGRESS_GOAL.format(early="".join(f"{name}, " for name in early))
    else:
        steps = [name for name, _, _ in CHAINS[str(task.goal)]]
        text = CRAFT_GOAL.format(item=steps[-1], steps=", ".join(steps))
    if players > 1:
        return text
    for team, alone in ALONE_GOAL:
        text = text.replace(team, alone)
    return text


ALONE_GOAL = [
    ("Goal: together, ", "Goal: "), ("in your inventories", "in your inventory"), (", whoever does it", ""),
    ("A step counts once, whoever does it,", "A step counts once,"), (", together.", "."),
]  # fmt: skip
"""What the goals say of a team, and what they say instead to a player on their own."""


def way(task: Task, players: int = len(TEAM)) -> str:
    """How to get to the goal, step by step, from what the team starts with: who carries what, every recipe and rule
    on the way, in order. Empty for a task whose way is not one ladder (the progress tasks), and for an unguided
    variant of one that is."""
    if not task.guided:
        return ""
    if task.objective is Objective.CRAFT:
        steps = [MAKING[name] for name, _, _ in CHAINS[str(task.goal)]]
    elif task.objective is Objective.DIAMONDS and task.kit is Kit.NONE:
        return laid_out(task, players)
    elif task.objective is Objective.DIAMONDS:
        steps = DIAMOND_WAYS.get(task.kit, [])
    else:
        return ""
    if not steps:
        return ""
    kit = listed([_stack(stack) for stack in KITS[task.kit]], "and")
    supplies = "" if task.kit is Kit.NOTHING else ", besides food and torches"
    coordination = task.coordination if players > 1 else Coordination.KITTED  # (one player holds the whole kit)
    you = "Each of you starts" if players > 1 else "You start"
    carried = {
        Coordination.KITTED: f"{you} with {kit}{supplies}." if kit else "You start with nothing.",
        Coordination.ONE_KIT: f"One of you starts with {kit} (your inventory shows whether it is you); the others "
        "start with food and torches. Whoever holds the parts does the crafting, or tosses them to a teammate "
        "(toss: to a player standing within three blocks).",
        Coordination.SPLIT: f"These are dealt among you, so no one can finish alone: {kit}. Bring the parts to one "
        "player (toss: to a player standing within three blocks), who does the crafting.",
    }[coordination]
    numbered = "\n".join(f"{index}. {step}" for index, step in enumerate(steps, 1))
    return f"How to get there. {carried}\n{numbered}"


def _stack(stack: Mapping[str, Any]) -> str:
    """A stack as an inventory shows it: "3 iron_ingot"."""
    return f"{stack.get('count', 1)} {stack['item']}"


def describe_result(result: Any) -> str:
    """How an action went (the harness's result of it), as the agent reads it in answer to its call."""
    if not isinstance(result, dict):
        return "No result."
    outcome = cast(dict[str, Any], result)
    if outcome.get("ok"):
        details = {key: value for key, value in outcome.items() if key not in ("action", "ok")}
        return ", ".join(f"{key}: {item}" for key, item in details.items()) or "Done."
    if outcome.get("interrupted"):
        at = cast(dict[str, Any], outcome.get("now_at") or {})
        return f"Cut off when the world froze; you got to ({at.get('x')}, {at.get('y')}, {at.get('z')})."
    return f"Failed: {outcome.get('error', 'unknown error')}"


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

WALKING = (
    f"Walking digs through what is in the way if your tools break it within {spelled(LIMITS.walk_dig_seconds)} seconds "
    f"a block, and bridges or pillars with {listed(LIMITS.scaffolding, 'or')} you carry where there is no other way. "
    "Where it cannot get through, it stops and says what is in the way."
)

ACTIONS = [
    _action("move_to", f"Walk to a place you have seen. {WALKING}", XYZ, AT),
    _action(
        "move",
        f"Walk up to {LIMITS.move_blocks} blocks in a direction. {WALKING}",
        {"direction": DIRECTION, "blocks": COUNT},
        ["direction"],
    ),
    _action(
        "mine",
        f"Mine a block you can see within reach ({LIMITS.reach_blocks:g} blocks) and pick up what drops.",
        XYZ,
        AT,
    ),
    _action(
        "place_at",
        "Place a block from your inventory at a free position you see within reach; it needs a block next to it.",
        {"item": STRING, **XYZ},
        ["item", *AT],
    ),
    _action(
        "use",
        "Use the item you hold (or `item`). With no position: eat it, or throw an eye of ender. On a block within "
        "reach: apply it (flint and steel, a bucket, an eye of ender on a frame), look into a chest, sleep in a bed.",
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
        "Put items into a furnace within reach, with the fuel for them (the best you carry, unless you name one: "
        f"{', '.join(LIMITS.fuels)}); each item takes {LIMITS.smelt_seconds} seconds.",
        {"item": STRING, "fuel": STRING, "count": COUNT},
        ["item"],
    ),
    _action("take_smelted", "Take what a furnace within reach has finished.", {}, []),
    _action("take", "Take items from a chest within reach.", {**XYZ, "item": STRING, "count": COUNT}, [*AT, "item"]),
    _action("store", "Put items into a chest within reach.", {**XYZ, "item": STRING, "count": COUNT}, [*AT, "item"]),
    _action(
        "toss",
        "Throw items (one, unless you give a count) toward a position within three blocks: whoever stands there "
        "picks them up.",
        {"item": STRING, "count": COUNT, **XYZ},
        ["item"],
    ),
    _action(
        "equip",
        "Hold an item, or put armor on (it goes where it is worn; `slot` is for the off-hand).",
        {"item": STRING, "slot": SLOT},
        ["item"],
    ),
    _action(
        "attack",
        "Attack a creature you can see, by its id, until it dies or time runs out.",
        {"target": INTEGER},
        ["target"],
    ),
    _action(
        "shoot",
        "Shoot arrows at a creature or end crystal you can see, by its id (needs a bow and arrows).",
        {"target": INTEGER},
        ["target"],
    ),
    _action(
        "chat",
        "Say something to your teammates.",
        {"message": STRING},
        ["message"],
    ),
    _action("wait", f"Do nothing for {spelled(LIMITS.wait_seconds)} seconds.", {}, []),
]
"""The actions, as tools the model calls (one per turn): motor control; strategy is the agents'. The harness takes
each by its name, with these arguments (harness/lib/actions.js)."""

CHAT_LINES = 6
"""Messages of the team's chat an observation shows: the newest, each with its age in turns."""


def describe(
    observation: Mapping[str, Any],
    *,
    chat: Sequence[tuple[int, str, str]] | None = (),
    recalled: bool = False,
) -> str:
    """An observation as text: who and where you are, the map of what you have seen, the blocks that touch you,
    what is in sight (every list as long as the harness made it), and what the team has said lately. `chat` holds
    (age in turns, speaker, message), oldest first.
    `recalled` is the form in which a turn stays in memory: everything but the map, the blocks beside the agent and
    the chat. (How the last action went is not in it: the action's own result says that.)"""
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
        f"You are {me['name']}, at {where} ({', '.join(place)}); health {me['health']}/20, food {me['food']}/20.",
        f"Inventory: {_items(me['inventory'])}.{holding}{wearing}",
    ]
    if observation.get("died"):
        lines.append("You died since your last turn and respawned.")
    if observation.get("map") and not recalled:
        lines.append(render_map(observation))
        lines.append(beside(observation))
    if observation["notable"]:
        lines.append("Notable in sight: " + "; ".join(_notable(kind) for kind in observation["notable"]) + ".")
    if observation["items"]:
        dropped = "; ".join(f"{i['count']} {i['item']} at ({i['x']}, {i['y']}, {i['z']})" for i in observation["items"])
        lines.append(f"Dropped items: {dropped}.")
    if observation["teammates"]:
        mates = "; ".join(f"{t['name']} at ({t['x']}, {t['y']}, {t['z']})" for t in observation["teammates"])
        lines.append(f"Teammates in sight: {mates}.")
    if observation.get("mobs"):
        hostile = "; ".join(f"{m['mob']} (id {m['id']}) {m['distance']} away" for m in observation["mobs"])
        lines.append(f"Hostile: {hostile}.")
    if observation.get("animals"):
        animals = "; ".join(f"{a['animal']} (id {a['id']}) {a['distance']} away" for a in observation["animals"])
        lines.append(f"Animals: {animals}.")
    if recalled:
        return "\n".join(lines)
    if chat is None:  # (a player on their own)
        return "\n".join(lines)
    if chat:
        lines.append("Team chat, oldest first:")
        lines.extend(f"- {'you' if who == me['name'] else who}, {_age(age)}: {message}" for age, who, message in chat)
    else:
        lines.append("Team chat: (nothing yet)")
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
ITEM, HOSTILE, ANIMAL, TEAMMATE = "*", "!", "&", "+"
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
    what the agent has seen is on it. Teammates, creatures and dropped items in sight are drawn where they stand:
    those the observation lists, and those it holds as `unlisted` (in sight, past the end of a list)."""
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
    for mate in observation.get("teammates", []):  # (not their initials: a teammate's D was read as diamonds)
        marks[(mate["x"], mate["y"], mate["z"])] = TEAMMATE
        legend.setdefault(TEAMMATE, set()).add("a teammate")
    unlisted: Mapping[str, Any] = observation.get("unlisted") or {}
    for kind, character, meaning in (
        ("items", ITEM, "a dropped item"),
        ("mobs", HOSTILE, "a hostile creature"),
        ("animals", ANIMAL, "an animal"),
    ):
        for entity in (*observation.get(kind, []), *unlisted.get(kind, [])):
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


SIDES = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}


def beside(observation: Mapping[str, Any]) -> str:
    """The ten blocks that touch the agent, each with its coordinates and what the map knows of it: on every side at
    head and at foot height, over the head and under the feet. They are on the map too; here nothing has to be
    counted out along a row. (Agents shown the map alone aimed more than half their `mine` calls at empty blocks
    next to them.)"""
    local: Mapping[str, Any] = observation["map"]
    center, radius = local["center"], int(local["radius"])
    side = 2 * radius + 1
    x0, y0, z0 = int(center["x"]), int(center["y"]), int(center["z"])
    layers = {int(layer["y"]): layer["cells"] for layer in local["layers"]}
    standing = {
        (int(mate["x"]), int(mate["y"]) + up, int(mate["z"])): str(mate["name"])
        for mate in observation.get("teammates", [])
        for up in (0, 1)
    }

    def at(x: int, y: int, z: int) -> str:
        where = f"({x}, {y}, {z})"
        if (x, y, z) in standing:
            return f"{where} {standing[(x, y, z)]}"
        cells = layers.get(y)
        index = -1 if cells is None else int(cells[(z - z0 + radius) * side + (x - x0 + radius)])
        if index < 0:
            return f"{where} not seen"
        name = str(local["palette"][index]["name"])
        return f"{where} {'empty' if name == 'air' else name}"

    lines = ["Next to you, at head height and at foot height:"]
    lines.extend(
        f"- {name}: {at(x0 + dx, y0 + 1, z0 + dz)}, {at(x0 + dx, y0, z0 + dz)}" for name, (dx, dz) in SIDES.items()
    )
    lines.append(f"- over your head: {at(x0, y0 + 2, z0)}; under your feet: {at(x0, y0 - 1, z0)}")
    return "\n".join(lines)


def _age(turns: int) -> str:
    return "this turn" if turns <= 0 else "1 turn ago" if turns == 1 else f"{turns} turns ago"


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
