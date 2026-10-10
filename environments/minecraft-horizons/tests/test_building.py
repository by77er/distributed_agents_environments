"""Building: the hut's blueprint, what counts as a building material, and what agents are told of the sites."""

from minecraft_horizons.building import HUT, SITES, Site, counts, describe, positions
from minecraft_horizons.prompts import goal
from minecraft_horizons.tasks import TASKS


def test_a_hut_is_walls_three_high_with_a_doorway_and_a_roof_seventy_one_blocks() -> None:
    assert len(HUT) == 71 and len(set(HUT)) == 71
    assert sum(1 for _, dy, _ in HUT if dy == 3) == 25  # (the roof)
    assert (2, 0, 0) not in HUT and (2, 1, 0) not in HUT and (2, 2, 0) in HUT  # (the doorway, and the wall above it)
    assert (2, 1, 2) not in HUT  # (inside is empty)
    assert SITES == 16


def test_building_materials_count_and_other_blocks_do_not() -> None:
    assert all(counts(block) for block in ("oak_planks", "cherry_log", "stripped_birch_wood", "cobblestone", "bricks"))
    assert not any(counts(block) for block in ("air", "dirt", "smooth_stone", "glass", "torch", "oak_leaves"))


def test_every_site_s_blueprint_is_placed_from_its_corner() -> None:
    sites = [Site(10, 64, 20), Site(18, 65, 20)]
    absolute = positions(sites)
    assert len(absolute) == 2 * len(HUT) and absolute[0] == (10, 64, 20) and absolute[len(HUT)] == (18, 65, 20)


def test_the_goal_names_every_site_and_the_blueprint() -> None:
    sites = [Site(10, 64, 20), Site(18, 65, 20)]
    told = goal(TASKS["huts-fresh-40m"], 2, sites)
    assert told.startswith("Goal: together, build as much of as many huts as you can")
    assert "1 at (10, 64, 20); 2 at (18, 65, 20)" in describe(sites) and describe(sites) in told
    assert "71 blocks a hut" in told and "40 minutes of game time" in told and "half a hut is worth half" in told
