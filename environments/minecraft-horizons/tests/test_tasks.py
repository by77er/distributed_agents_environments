"""The catalog, the rows and starts, and what agents read: every objective at every budget, told the clock."""

import itertools
import random
from typing import cast

from minecraft_horizons.environment import Horizons
from minecraft_horizons.objectives import OBJECTIVES
from minecraft_horizons.prompts import clock, goal, system_prompt
from minecraft_horizons.tasks import LADDER, SETTINGS, TASKS, catalog
from minecraft_team.tasks import TURNS_PER_MINUTE, Hazards, Kit, Start, Tier


def test_every_objective_comes_from_each_of_its_settings_at_every_budget_of_the_ladder() -> None:
    tasks = catalog()
    assert len(tasks) == len(TASKS) == sum(len(SETTINGS[name]) for name in OBJECTIVES) * len(LADDER)
    for name in OBJECTIVES:
        for setting in SETTINGS[name]:
            assert sorted(t.minutes for t in tasks if t.objective.id == name and t.setting is setting) == list(LADDER)
    assert all(later == 2 * earlier for earlier, later in itertools.pairwise(LADDER))  # (it doubles)


def test_shorter_budgets_come_first_and_turns_follow_game_time() -> None:
    minutes = [task.minutes for task in catalog()]
    assert minutes == sorted(minutes)
    assert all(task.turns == round(task.minutes * TURNS_PER_MINUTE) for task in catalog())


def test_a_setting_is_laid_out_by_the_team_packages_builders_as_a_natural_world() -> None:
    task = TASKS["iron-underground-20m"]
    laid = task.laid_out()
    assert laid.start is Start.CAVE and laid.kit is Kit.STONE and laid.hazards is Hazards.EASY
    assert laid.tier is Tier.SURVIVAL and not laid.keeps_inventory  # (a real day and night; death drops what you carry)
    assert TASKS["wood-fresh-5m"].laid_out().kit is Kit.NOTHING


def test_a_start_draws_training_worlds_and_evals_play_in_worlds_training_never_draws() -> None:
    environment = Horizons()
    rows = environment.rows()
    assert [row.key for row in rows] == list(TASKS)
    trained = {environment.start(row, random.Random(seed))["world_seed"] for row in rows for seed in range(40)}  # type: ignore[index]
    starts = environment.evals()["horizons-held-out"]
    assert len(starts) == len(rows)
    held_out = {start.parameters["world_seed"] for start in starts}  # type: ignore[index]
    assert held_out and not held_out & trained
    for start in starts:
        names = cast(list[str], start.parameters["names"])  # type: ignore[index]
        assert 1 <= len(names) <= 4 and len(set(names)) == len(names)


def test_the_goal_says_what_counts_how_long_the_game_lasts_and_that_there_is_no_ceiling() -> None:
    task = TASKS["iron-fresh-40m"]
    team = goal(task, 3)
    assert "40 minutes of game time" in team and f"{task.turns} turns" in team
    assert "raw iron" in team and "chests you placed" in team and "beyond what the team began with" in team
    assert "Nothing caps the count" in team and team.startswith("Goal: together, ")
    alone = goal(task, 1)
    assert alone.startswith("Goal: end the game") and "your inventory" in alone and "you began with" in alone
    assert "advancements" in goal(TASKS["advancements-fresh-10m"], 2)
    prompt = system_prompt(task, ["ada", "ben"])
    assert "ada, ben" in prompt and "40 minutes of game time" in prompt and "how much is left" in prompt


def test_the_clock_shows_game_time_and_turns_left() -> None:
    assert clock(0.0, 5, 1, 60) == "Time left: 5.0 of 5 minutes of game time; 60 of 60 turns."
    assert clock(3.26, 5, 41, 60) == "Time left: 1.7 of 5 minutes of game time; 20 of 60 turns."
    assert clock(6.0, 5, 60, 60).startswith("Time left: 0.0 of 5")  # (a last window may run over)
