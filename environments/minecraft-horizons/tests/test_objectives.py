"""Objectives with no ceiling: what each counts, that what the team began with does not, and the reward."""

import math

import pytest

from minecraft_horizons.objectives import FOOD, OBJECTIVES, WOOD, Measure, holdings, measured, progressed, reward
from minecraft_team.tasks import MILESTONES


def test_metals_count_raw_ingot_and_ore_one_each_a_block_nine_and_a_nugget_a_ninth() -> None:
    iron = OBJECTIVES["iron"]
    held = {"raw_iron": 3, "iron_ingot": 2, "iron_ore": 1, "iron_block": 1, "raw_iron_block": 1, "iron_nugget": 9}
    found = measured(iron, held, {}, [])
    assert found.amount == pytest.approx(3 + 2 + 1 + 9 + 9 + 1)
    assert found.parts["iron_nugget"] == pytest.approx(1.0) and found.parts["iron_block"] == 9.0
    assert measured(iron, {"iron_pickaxe": 1, "iron_chestplate": 1}, {}, []).amount == 0.0  # (made into gear)


def test_what_the_team_began_with_does_not_count_and_the_amount_is_never_below_nothing() -> None:
    food = OBJECTIVES["food"]
    began = {"bread": 8}  # (a kit's supplies)
    assert measured(food, {"bread": 8, "cooked_beef": 2}, began, []).amount == pytest.approx(16.0)
    assert measured(food, {"bread": 3}, began, []).amount == 0.0  # (eaten: less than at the start, not less than 0)


def test_cooked_food_is_worth_more_than_raw_and_harmful_food_is_not_food() -> None:
    assert FOOD["cooked_beef"] > FOOD["beef"] and FOOD["cooked_porkchop"] > FOOD["porkchop"]
    assert "rotten_flesh" not in FOOD and "spider_eye" not in FOOD and "poisonous_potato" not in FOOD


def test_wood_counts_every_trees_logs_and_planks_as_a_quarter() -> None:
    assert WOOD["oak_log"] == WOOD["stripped_cherry_wood"] == WOOD["crimson_stem"] == 1.0
    assert WOOD["spruce_planks"] == 0.25
    assert measured(OBJECTIVES["wood"], {"birch_log": 3, "oak_planks": 8}, {}, []).amount == pytest.approx(5.0)


def test_advancements_count_each_one_once_and_not_recipes() -> None:
    found = measured(
        OBJECTIVES["advancements"],
        {},
        {},
        ["story/mine_stone", "story/mine_stone", "recipes/misc/torch", "adventure/root"],
    )
    assert found.amount == 2.0 and set(found.parts) == {"story/mine_stone", "adventure/root"}


def test_held_and_stored_are_one_count() -> None:
    assert holdings({"held": {"coal": 3}, "stored": {"coal": 5, "stick": 2}, "containers": 1}) == {
        "coal": 8.0,
        "stick": 2.0,
    }


def test_the_reward_is_log_one_plus_the_amount() -> None:
    assert reward(0.0) == 0.0 and reward(-3.0) == 0.0
    assert reward(9.0) == pytest.approx(math.log(10))
    assert reward(99.0) - reward(9.0) == pytest.approx(math.log(10))  # (each tenfold, the same step)


def test_every_held_objective_names_what_it_counts_and_every_speedrun_its_goal() -> None:
    for objective in OBJECTIVES.values():
        assert objective.unit
        assert (objective.measure is Measure.HELD) == bool(objective.values)
        assert (objective.measure is Measure.PROGRESS) == (objective.goal in MILESTONES)
        assert bool(objective.counted) == (objective.measure is not Measure.PROGRESS)


def test_a_speedrun_scores_its_progress_and_once_it_reaches_its_goal_the_share_of_its_time_left() -> None:
    assert progressed(0.4, False, 20.0, 20.0).amount == pytest.approx(0.4)  # (not there: progress alone)
    assert progressed(1.0, True, 15.0, 20.0).amount == pytest.approx(1.25)
    assert progressed(1.0, True, 5.0, 20.0).amount > progressed(1.0, True, 10.0, 20.0).amount  # (sooner is better)
    assert progressed(1.0, True, 21.0, 20.0).amount == pytest.approx(1.0)  # (a last window run over: nothing left)
    assert progressed(1.0, True, 5.0, 20.0).parts == {"progress": 1.0, "time_left": 0.75}
