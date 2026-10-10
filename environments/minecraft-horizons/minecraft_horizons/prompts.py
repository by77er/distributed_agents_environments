"""What agents read: the system prompt, which says what counts and how long the game lasts, and the clock at the top of
every observation.

Unlike the team package's prompts, these say how long the game lasts, and every observation shows what is left of it
(`clock`): an objective with no ceiling is a race against a budget, and what pays (the nearest ore, or better tools
first, or a farm) depends on how much of it there is. The rest of the system prompt (how the game runs, what an agent
knows, chat and teamwork) and how an observation reads are the team package's own (`minecraft_team.prompts`).
"""

from collections.abc import Sequence

from minecraft_horizons.objectives import Measure
from minecraft_horizons.tasks import Task
from minecraft_team.limits import LIMITS
from minecraft_team.prompts import (
    ALONE_OPENING,
    ALONE_TURNS,
    CHAT,
    CHAT_LINES,
    DRAGON,
    DRAGON_ONLY,
    MILESTONE_WORDS,
    SYSTEM,
    TEAM_OPENING,
    TEAM_TURNS,
    TEAMWORK,
    spelled,
)
from minecraft_team.tasks import KILL, path_of

__all__ = ["clock", "goal", "system_prompt"]

GOAL = """Goal: {together}end the game with as much {unit} as you can. What counts is what {holder} when the game \
ends, beyond what {began}: {counted}. The game lasts {minutes} minutes of game time{turns}; every observation shows \
how much is left. Plan for the time you have: with little of it, use what is at hand; with more, what takes time to \
set up (better tools, enchantments, a farm) can pay off. Nothing caps the count: there is always more to get."""

ADVANCEMENTS_GOAL = """Goal: {together}earn as many advancements as you can: {counted}. The game lasts {minutes} \
minutes of game time{turns}; every observation shows how much is left. Plan for the time you have: with little of it, \
earn what is close at hand; with more, what takes time to set up opens up many more. Nothing caps the count: there is \
always another to earn."""


SPEEDRUN_GOAL = """Goal: a speedrun{together}. The finish line is {goal}: get there as fast as you can. \
Getting there counts step by step, each step once, whoever does it, and only if it is done in this game: \
{steps}.{dragon} Reaching the finish ends the game, and the more of the time is left then, the higher the score: \
finishing sooner is always better. The game lasts at most {minutes} minutes of game time{turns}; every observation \
shows how much is left."""


def goal(task: Task, players: int) -> str:
    """What the task asks, and how long the game lasts, as agents read it."""
    together = "together, " if players > 1 else ""
    turns = f" or {task.turns} turns, whichever runs out first"
    if task.objective.measure is Measure.PROGRESS:
        path = path_of(task.laid_out())
        names = [MILESTONE_WORDS.get(name, name) for name, _, _ in path]
        dragon = "" if task.objective.goal != KILL else DRAGON if len(path) > 1 else DRAGON_ONLY
        return SPEEDRUN_GOAL.format(
            together=" as a team" if players > 1 else "", goal=MILESTONE_WORDS[str(task.objective.goal)],
            steps=", ".join(names), dragon=dragon, minutes=task.minutes, turns=turns,
        )  # fmt: skip
    if task.objective.measure is Measure.ADVANCEMENTS:
        return ADVANCEMENTS_GOAL.format(
            together=together, counted=task.objective.counted, minutes=task.minutes, turns=turns
        )
    holder = (
        "the team holds in your inventories and in chests you placed"
        if players > 1
        else ("you hold in your inventory and in chests you placed")
    )
    began = "the team began with" if players > 1 else "you began with"
    return GOAL.format(
        together=together, unit=task.objective.unit, holder=holder, began=began, counted=task.objective.counted,
        minutes=task.minutes, turns=turns,
    )  # fmt: skip


def system_prompt(task: Task, team: Sequence[str]) -> str:
    """The same for every agent of the team: nothing in it says which of them reads it (each observation does)."""
    players = len(team)
    death = "with what you carried" if task.setting.keeps_inventory else "and what you carried lies where you died"
    return SYSTEM.format(
        opening=TEAM_OPENING.format(count=spelled(players), team=", ".join(team)) if players > 1
        else ALONE_OPENING.format(name=team[0]),
        goal=goal(task, players),
        way="",
        turns=TEAM_TURNS.format(count=spelled(players)) if players > 1 else ALONE_TURNS,
        window=spelled(LIMITS.window_seconds),
        chat=CHAT.format(chat_lines=CHAT_LINES) if players > 1 else "",
        teamwork=f" {TEAMWORK}" if players > 1 else "",
        death=death,
    )  # fmt: skip


def clock(minutes_spent: float, minutes: float, turn: int, turns: int) -> str:
    """What is left of the game, as the first line of an observation: the game time left (to the nearest tenth of a
    minute) and the turns left, this one among them."""
    left = max(minutes - minutes_spent, 0.0)
    return f"Time left: {left:.1f} of {minutes:g} minutes of game time; {turns - turn + 1} of {turns} turns."
