"""The map an agent is shown: built by the harness from what the bot has seen (tested in Node on a made-up world, no
server), and drawn as text for the model."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from minecraft_swarm.prompts import describe, render_map, symbol

HARNESS = Path(__file__).resolve().parents[1] / "harness"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not (HARNESS / "node_modules").exists(), reason="Node and the harness's packages"
)


def test_the_harness_builds_the_map_from_line_of_sight_only() -> None:
    ran = subprocess.run(
        ["node", "--test", "test/map.test.js"], cwd=HARNESS, capture_output=True, text=True, check=False
    )
    assert ran.returncode == 0, ran.stdout + ran.stderr


def observation() -> dict[str, Any]:
    """What a bot observes in the made-up world: a room with a corridor east, ore in the west wall, water, a teammate
    and dropped diamonds; and a second room with a chest behind the south wall, which it cannot see."""
    ran = subprocess.run(["node", "test/scene.js"], cwd=HARNESS, capture_output=True, text=True, check=True)
    return json.loads(ran.stdout)


def test_the_map_is_drawn_one_grid_per_height_with_absolute_coordinates() -> None:
    seen = observation()
    lines = render_map(seen).splitlines()
    assert "columns are x=-6 to x=6, and each row starts with its z" in lines[0]
    feet = lines.index("y=64 (your feet):")
    grid = lines[feet + 1 : feet + 14]
    assert grid[6] == "0 ? ? # . . . @ . . . . . ."  # the corridor runs east from the agent's row
    assert grid[4] == "-2 ? ? # . . . . . B . # ? ?"  # ben
    assert grid[8] == "2 ? ? # * . . . . ~ . # ? ?"  # dropped diamonds, and water
    assert grid[10] == "4 ? ? ? # # # # # # # ? ? ?"  # the south wall; behind it, nothing is known
    assert grid[12] == "6 ? ? ? ? ? ? ? ? ? ? ? ? ?"
    head = lines.index("y=65 (your head):")
    assert lines[head + 7] == "0 ? ? d . . . . . . . . . ."  # ore in the west wall
    assert "y=63 (the floor under you): all # where seen" in lines
    assert "y=62 (below the floor): not seen" in lines
    assert lines[-1] == "On the map: * a dropped item; B ben; d deepslate_diamond_ore; ~ water."
    assert "h" not in "".join(grid)  # no chest


def test_an_observation_carries_the_map_and_exact_coordinates_and_earlier_turns_are_brief() -> None:
    seen = observation()
    text = describe(seen, notes="chest room south?", board=["ben: I go east"])
    assert text.startswith("You are ada, at (0, 64, 0) (overworld, plains, day, no sky above); health 20/20")
    assert "minute" not in text  # agents are not told the clock
    assert "Map of what you have seen within 6 blocks" in text
    assert "Notable in sight: deepslate_diamond_ore at (-4, 65, 0), 4.3 away." in text
    assert "Dropped items: 3 diamond at (-3, 64, 2)." in text
    assert "Your notes: chest room south?" in text and "Team board: ben: I go east" in text
    assert "Open space" not in text and "You see mostly" not in text  # the map shows it
    brief = describe(seen, brief=True)
    assert brief.splitlines() == [text.splitlines()[0], "Inventory: empty."]


def test_blocks_have_their_own_symbols_or_fall_back_to_solid_and_passable() -> None:
    assert symbol("air", False) == "."
    assert symbol("deepslate_diamond_ore", True) == symbol("diamond_ore", True) == "d"
    assert symbol("nether_quartz_ore", True) == "q"
    assert symbol("oak_log", True) == symbol("spruce_log", True) == "w"
    assert symbol("lava", False) == "%" and symbol("chest", True) == "h"
    assert symbol("deepslate", True) == "#" and symbol("poppy", False) == ","
