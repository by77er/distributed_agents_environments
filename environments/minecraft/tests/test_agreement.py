"""What is written in two places says the same in both: the actions and their limits (the prompts, and the Node
harness that performs them), the control API (the Python client, and the Java plugin that serves it), the tool set's
operations, the game's version, and how far off large things are seen (the harness, and the server's configuration).

No server is started: the files are read, and Node prints what the harness takes (a script that connects to nothing).
"""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from minecraft_team import control
from minecraft_team.limits import LIMITS
from minecraft_team.paper import CONFIG, PAPER_VERSION, PLUGIN_SOURCES, server_properties
from minecraft_team.prompts import ACTIONS, DIRECTION, SLOT, SYMBOLS, symbol, system_prompt
from minecraft_team.tasks import NAMES, catalog
from minecraft_team.worlds import OPERATIONS, MinecraftTools, MinecraftWorlds
from rollout.contracts import RetryClass

HARNESS = Path(__file__).resolve().parents[1] / "harness"
CHUNK = 16

needs_node = pytest.mark.skipif(
    shutil.which("node") is None or not (HARNESS / "node_modules").exists(), reason="Node and the harness's packages"
)


def vocabulary() -> dict[str, Any]:
    """What the harness takes and keeps, as it says itself."""
    ran = subprocess.run(["node", "test/vocabulary.js"], cwd=HARNESS, capture_output=True, text=True, check=True)
    return json.loads(ran.stdout)


@needs_node
def test_the_prompts_offer_the_actions_the_harness_handles_with_its_slots_and_directions() -> None:
    harness = vocabulary()
    offered = [tool.name for tool in ACTIONS]
    # The harness also handles `idle`, which no agent is offered: it is what calling no tool comes to.
    assert set(harness["actions"]) == {*offered, "idle"} and len(offered) == len(set(offered)) == 16
    assert harness["slots"] == SLOT["enum"]
    assert harness["directions"] == DIRECTION["enum"]
    assert harness["limits"] == LIMITS.model_dump(mode="json")


@needs_node
def test_the_map_draws_as_chests_and_furnaces_what_the_harness_opens_as_such() -> None:
    harness = vocabulary()
    assert {name for name, character in SYMBOLS.items() if character == symbol("chest", True)} == set(
        harness["containers"]
    )
    assert {name for name, character in SYMBOLS.items() if character == symbol("furnace", True)} == set(
        harness["furnaces"]
    )


def test_the_prompts_state_the_limits_in_these_words() -> None:
    described = {tool.name: tool.description for tool in ACTIONS}
    assert (LIMITS.reach_blocks, LIMITS.move_blocks, LIMITS.walk_dig_seconds) == (4.5, 32, 5)
    assert described["mine"] == "Mine a block you can see within reach (4.5 blocks) and pick up what drops."
    assert described["move"].startswith(
        "Walk up to 32 blocks in a direction. Walking digs through what is in the way if your tools break it within "
        "five seconds a block, and bridges or pillars with dirt, cobblestone, cobbled_deepslate or netherrack you carry"
    )
    assert described["move_to"].endswith(described["move"].removeprefix("Walk up to 32 blocks in a direction. "))
    assert (LIMITS.wait_seconds, LIMITS.smelt_seconds, LIMITS.window_seconds) == (5, 10, 20)
    assert described["wait"] == "Do nothing for five seconds."
    assert described["smelt"].endswith(
        "unless you name one: coal, charcoal, planks, logs, sticks); each item takes 10 seconds."
    )
    system = system_prompt(catalog()[0], ["ada", "ben", "cy", "dee"])
    assert system.startswith("You are one of four players in Minecraft: ada, ben, cy, dee. You play together.")
    assert "then the world runs until all four actions have finished, and freezes again." in system
    assert "An action that takes more than twenty seconds (a long walk" in system


def test_the_client_asks_only_for_what_the_plugin_serves_and_for_all_of_it() -> None:
    plugin = (PLUGIN_SOURCES / "src/dev/rollout/groundtruth/GroundTruthPlugin.java").read_text()
    served = re.findall(r'^\s*route\("([^"]+)"', plugin, flags=re.MULTILINE)
    client = Path(control.__file__).read_text()
    asked = re.findall(r'self\._request\(\s*"(?:GET|POST)",\s*"([^"]+)"', client)
    assert len(asked) == client.count("self._request(")  # every request names its path where it is made
    assert len(served) == len(set(served)) and set(asked) == set(served)


def test_the_tool_set_specifies_the_operations_it_performs() -> None:
    specifications = MinecraftTools(MinecraftWorlds()).specifications()
    assert [specification.name for specification in specifications] == list(OPERATIONS)
    for specification in specifications:
        takes = OPERATIONS[specification.name].takes
        assert specification.input_schema == {"type": "object", "properties": dict(takes), "required": list(takes)}
        assert specification.retry_class is OPERATIONS[specification.name].retry_class
    # An observation uses nothing up (harness/lib/news.js), so it may be asked for again; an action may not.
    assert OPERATIONS["observe"].retry_class is RetryClass.PURE
    assert OPERATIONS["act"].retry_class is OPERATIONS["window"].retry_class is RetryClass.SIDE_EFFECTING


async def test_an_operation_the_tool_set_does_not_have_is_an_error() -> None:
    result = await MinecraftTools(MinecraftWorlds()).call("teleport", {}, effect_id="e", arguments_digest="d")
    assert result.is_error


def test_the_plugin_is_built_for_the_version_the_servers_run() -> None:
    plugin: dict[str, Any] = yaml.safe_load((PLUGIN_SOURCES / "resources" / "plugin.yml").read_text())
    assert PAPER_VERSION.startswith(str(plugin["api-version"]))


@needs_node
def test_large_things_are_tracked_and_sent_as_far_off_as_the_harness_shows_them() -> None:
    far = vocabulary()["far_range"]  # the dragon, end crystals and ghasts
    spigot: dict[str, Any] = yaml.safe_load((CONFIG / "spigot.yml").read_text())
    assert spigot["world-settings"]["default"]["entity-tracking-range"]["other"] == far
    properties = server_properties()
    assert int(properties["view-distance"]) * CHUNK >= far and int(properties["simulation-distance"]) * CHUNK >= far


def test_every_task_with_a_ladder_says_its_way_step_by_step_in_actions_the_agents_have() -> None:
    from minecraft_team.prompts import way
    from minecraft_team.tasks import Kit, Objective

    actions = {tool.name for tool in ACTIONS}
    for task in catalog():
        said = way(task)
        if task.objective is Objective.PROGRESS or not task.guided:
            assert said == ""
            continue
        assert said and said in system_prompt(task, ["ada", "ben"])
        named = set(re.findall(r"\((\w+)(?=\)|:|, at)", said))  # "(place_at)", "(toss: ...)", "(take, at ...)"
        assert named <= actions, (task.id, named - actions)
        if task.kit not in (Kit.NONE, Kit.IRON):
            assert "craft an iron_pickaxe: three iron_ingot and two sticks" in said or task.objective is Objective.CRAFT
    ingots = next(task for task in catalog() if task.kit is Kit.INGOTS and task.coordination.value == "kitted")
    assert way(ingots).startswith(
        "How to get there. Each of you starts with 3 iron_ingot, 2 stick and 1 crafting_table, besides food and "
        "torches."
    )


def test_a_team_of_any_size_is_told_the_game_as_it_is_for_that_many() -> None:
    from minecraft_team.catalog import catalog as teams
    from minecraft_team.prompts import guidance, way

    task = next(task for task in catalog() if task.kit.value == "ingots" and task.coordination.value == "one_kit")
    alone = system_prompt(task, ["ada"])
    assert alone.startswith("You are ada, playing Minecraft on your own.")
    for team_word in ("together", "teammate", "chat", "Each of you", "One of you", "players"):
        assert team_word not in alone, team_word
    assert way(task, 1).startswith("How to get there. You start with 3 iron_ingot")  # one player holds the whole kit
    assert "teamwork" not in guidance(task, 1) and set(guidance(task, 3)) == {"way", "teamwork"}
    three = system_prompt(task, ["ada", "ben", "cy"])
    assert three.startswith("You are one of three players in Minecraft: ada, ben, cy.")
    assert "until all three actions have finished" in three and guidance(task, 3)["teamwork"] in three
    # A start names who plays, all differently: one to four, and at least two where the kit is dealt in parts.
    import random

    rng = random.Random(0)
    for row in teams.rows():
        starts: list[Any] = [teams.start(row, rng) for _ in range(40)]
        players = {len(start["names"]) for start in starts}
        for start in starts:
            assert len(set(start["names"])) == len(start["names"]) and set(start["names"]) <= set(NAMES)
        split = next(each for each in catalog() if each.id == row.key).coordination.value == "split"
        assert players == ({2, 3, 4} if split else {1, 2, 3, 4}), (row.key, players)


def test_an_unguided_row_counts_for_its_guided_twin() -> None:
    from minecraft_team.catalog import catalog as teams

    rows = {row.key: row for row in teams.rows()}
    for row in rows.values():
        if row.key.endswith("u"):
            assert (
                row.counts_for == (row.key.removesuffix("u"),)
                and row.title == f"{rows[row.counts_for[0]].title}, unguided"
            )
        else:
            assert row.counts_for == ()
