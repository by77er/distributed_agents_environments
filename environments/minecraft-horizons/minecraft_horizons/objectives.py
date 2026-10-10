"""Objectives with no ceiling: an amount the team ends a game with, measured from the plugin's ground truth.

An objective is something the team holds (`Measure.HELD`: items, each worth so much of the objective's unit, in
members' inventories and in the containers they placed), the advancements it earned (`Measure.ADVANCEMENTS`), or a
milestone on the way to the dragon reached as fast as it can be (`Measure.PROGRESS`: a speedrun, whole or a segment of
one). What the team held when the game began does not count: the amount is what it holds at the end beyond that,
never below nothing. Nothing caps it, so a team can always do better in the time it has; how it gets there (mining
as it goes, better tools first, a farm) is its own affair. A speedrun's amount is its progress along the milestones
to its goal (from 0 to 1) and, once the goal is reached, which ends the game, the share of its time left
(`progressed`): from 0 to 2, and higher the sooner the goal falls.

The reward is `log(1 + amount)` (`reward`): each doubling is worth as much at any scale, so a team that builds a farm
and ends with ten times its group's iron is ahead by a clear margin without its group's numbers swamping every other
group's in an update. Results report the amount itself.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "OBJECTIVES", "Measure", "Measured", "Objective", "holdings", "measured", "progressed", "reward", "value",
]  # fmt: skip


class Measure(StrEnum):
    HELD = "held"
    """Items the team holds at the end, each worth `Objective.values` of the objective's unit."""
    ADVANCEMENTS = "advancements"
    """Advancements the team earned (each once, whoever earned it; recipes are not advancements)."""
    PROGRESS = "progress"
    """A milestone toward the dragon (`Objective.goal`), as fast as it can be reached."""


@dataclass(frozen=True)
class Objective:
    id: str
    title: str
    measure: Measure
    unit: str
    """What the amount counts, as agents read it ("iron", "food points")."""
    counted: str
    """What counts toward it, as agents read it: the items and what each is worth."""
    values: Mapping[str, float] = field(default_factory=dict[str, float])
    """For `Measure.HELD`: what one of each item (by its id) is worth."""
    goal: str | None = None
    """For `Measure.PROGRESS`: the milestone (an advancement, of `minecraft_team.tasks.MILESTONES`) that ends the
    game."""


def _equivalents(material: str, ore: bool = True) -> dict[str, float]:
    """A metal or gem as raw item, ingot or gem, nugget, block and ore: one each, a block nine, a nugget a ninth."""
    values = {f"raw_{material}": 1.0, f"{material}_ingot": 1.0, f"{material}_nugget": 1 / 9, f"{material}_block": 9.0}
    values[f"raw_{material}_block"] = 9.0
    if ore:
        values |= {f"{material}_ore": 1.0, f"deepslate_{material}_ore": 1.0}
    return values


WOODS = ("oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "mangrove", "cherry", "pale_oak")
STEMS = ("crimson", "warped")
"""The overworld's trees and the nether's fungi, whose logs (stems) are wood."""
WOOD: dict[str, float] = {
    **{f"{prefix}{kind}_{part}": 1.0 for kind in WOODS for part in ("log", "wood") for prefix in ("", "stripped_")},
    **{f"{prefix}{kind}_{part}": 1.0 for kind in STEMS for part in ("stem", "hyphae") for prefix in ("", "stripped_")},
    **{f"{kind}_planks": 0.25 for kind in (*WOODS, *STEMS)},
    "bamboo_block": 1.0,
    "stripped_bamboo_block": 1.0,
}
"""A log counts one; four planks are made of a log, so a plank a quarter."""

FOOD: dict[str, float] = {
    "apple": 4, "baked_potato": 5, "beef": 3, "beetroot": 1, "beetroot_soup": 6, "bread": 5, "cake": 14, "carrot": 3,
    "chicken": 2, "chorus_fruit": 4, "cod": 2, "cooked_beef": 8, "cooked_chicken": 6, "cooked_cod": 5,
    "cooked_mutton": 6, "cooked_porkchop": 8, "cooked_rabbit": 5, "cooked_salmon": 6, "cookie": 2, "dried_kelp": 1,
    "glow_berries": 2, "golden_apple": 4, "golden_carrot": 6, "honey_bottle": 6, "melon_slice": 2,
    "mushroom_stew": 6, "mutton": 2, "porkchop": 3, "potato": 1, "pumpkin_pie": 8, "rabbit": 3, "rabbit_stew": 10,
    "salmon": 2, "sweet_berries": 2, "tropical_fish": 1, "enchanted_golden_apple": 4,
}  # fmt: skip
"""Food by the hunger it restores (half shanks), as the game has it: cooking raises it, so a furnace pays. Food that
hurts whoever eats it (rotten flesh, spider eyes, poisonous potatoes, pufferfish) is not food here."""

OBJECTIVES: dict[str, Objective] = {
    objective.id: objective
    for objective in [
        Objective(
            "wood", "Wood", Measure.HELD, "wood",
            "logs, wood and stems of any tree, one each (stripped too); planks a quarter each", WOOD,
        ),
        Objective(
            "food", "Food", Measure.HELD, "food points",
            "food, by the hunger it restores: bread 5, a cooked steak or porkchop 8, a raw one 3, an apple 4, a carrot "
            "3, and so on (cooked food is worth more than raw)", FOOD,
        ),
        Objective(
            "coal", "Coal", Measure.HELD, "coal",
            "coal and coal ore one each, a block of coal nine (charcoal is not coal)",
            {"coal": 1.0, "coal_ore": 1.0, "deepslate_coal_ore": 1.0, "coal_block": 9.0},
        ),
        Objective(
            "iron", "Iron", Measure.HELD, "iron",
            "raw iron, iron ingots and iron ore one each, a block of iron or of raw iron nine, a nugget a ninth (iron "
            "made into tools or armor does not count)", _equivalents("iron"),
        ),
        Objective(
            "gold", "Gold", Measure.HELD, "gold",
            "raw gold, gold ingots and gold ore one each, a block of gold or of raw gold nine, a nugget a ninth (gold "
            "made into anything else does not count)", _equivalents("gold"),
        ),
        Objective(
            "diamonds", "Diamonds", Measure.HELD, "diamonds",
            "diamonds and diamond ore one each, a block of diamond nine (diamonds made into tools or armor do not "
            "count)", {"diamond": 1.0, "diamond_ore": 1.0, "deepslate_diamond_ore": 1.0, "diamond_block": 9.0},
        ),
        Objective(
            "advancements", "Advancements", Measure.ADVANCEMENTS, "advancements",
            "every advancement the team earns, once, whoever earns it (recipes are not advancements)",
        ),
        *(
            Objective(key, title, Measure.PROGRESS, "progress", "", goal=goal)
            for key, title, goal in [
                ("iron-tools", "An iron pickaxe", "story/iron_tools"),
                ("nether", "Into the nether", "story/enter_the_nether"),
                ("blaze-rods", "A blaze rod", "nether/obtain_blaze_rod"),
                ("stronghold", "Into a stronghold", "story/follow_ender_eye"),
                ("end", "Into the end", "story/enter_the_end"),
                ("dragon", "The ender dragon", "end/kill_dragon"),
            ]
        ),
    ]
}  # fmt: skip
"""Every objective, by its id, from the quickest to get some of to the slowest."""


@dataclass(frozen=True)
class Measured:
    amount: float
    """What the team ended with beyond what it began with, in the objective's unit (never below nothing)."""
    parts: dict[str, float]
    """What each item (or each advancement) added to it: the amount's breakdown, for results."""


def value(objective: Objective, items: Mapping[str, float]) -> tuple[float, dict[str, float]]:
    """What items are worth to a held objective, and what each kind of them is worth."""
    parts = {item: count * objective.values[item] for item, count in items.items() if item in objective.values}
    return sum(parts.values()), {item: worth for item, worth in parts.items() if worth}


def holdings(found: Mapping[str, object]) -> dict[str, float]:
    """The plugin's holdings (`held` and `stored`) as one count of each item."""
    total: dict[str, float] = {}
    for part in ("held", "stored"):
        counts = found.get(part)
        if isinstance(counts, Mapping):
            for item, count in counts.items():  # pyright: ignore[reportUnknownVariableType]
                if isinstance(count, int | float):
                    total[str(item)] = total.get(str(item), 0.0) + float(count)  # pyright: ignore[reportUnknownArgumentType]
    return total


def measured(
    objective: Objective,
    held: Mapping[str, float],
    held_at_start: Mapping[str, float],
    advancements: list[str],
) -> Measured:
    """The amount of a held or advancements `objective` a team ended with: what it holds (`held`) beyond what it held
    at the start, or the advancements it earned since. (A speedrun's is `progressed`.)"""
    if objective.measure is Measure.ADVANCEMENTS:
        earned = sorted({name for name in advancements if not name.startswith("recipes/")})
        return Measured(float(len(earned)), dict.fromkeys(earned, 1.0))
    end, parts = value(objective, held)
    start, _ = value(objective, held_at_start)
    return Measured(max(end - start, 0.0), parts)


def progressed(progress: float, reached: bool, minutes_spent: float, minutes: float) -> Measured:
    """A speedrun's amount: its `progress` along the milestones to its goal (from 0 to 1) and, if the goal was
    `reached`, the share of its `minutes` of game time left then."""
    left = max(1.0 - minutes_spent / minutes, 0.0) if reached and minutes else 0.0
    return Measured(progress + left, {"progress": progress, "time_left": left})


def reward(amount: float) -> float:
    """What an amount is worth as a reward: `log(1 + amount)`."""
    return math.log1p(max(amount, 0.0))
