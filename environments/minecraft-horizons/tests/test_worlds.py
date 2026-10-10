"""The worlds on a live server: a setting laid out, what the team began with not counted, and what it stores in a chest
it placed counted as its own (and a chest it did not place, not).

Needs Java and Node (and the network once, for Paper and the harness's packages).
"""

import shutil
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from minecraft_horizons.building import HUT, positions
from minecraft_horizons.worlds import HorizonWorlds, world
from minecraft_team.worlds import WINDOW_TICKS, run_window

CREW = ["ada", "ben"]
HANDLE = "test-world"


@pytest.fixture(scope="module")
async def worlds(tmp_path_factory: pytest.TempPathFactory) -> AsyncIterator[HorizonWorlds]:
    if shutil.which("java") is None or shutil.which("node") is None:
        pytest.skip("Java and Node are needed")
    made = HorizonWorlds(logs=tmp_path_factory.mktemp("logs"), size=1)
    yield made
    await made.close()


async def act(made: HorizonWorlds, action: dict[str, Any], agent: str = "ada") -> dict[str, Any]:
    """An agent's action and a window of game time; how it ended."""
    episode = made._world(HANDLE)  # pyright: ignore[reportPrivateUsage]
    await episode.harness.thaw()
    await episode.harness.act(agent, action)
    await run_window(episode.control, episode.harness, ticks=WINDOW_TICKS, settle=0.3)
    return (await episode.harness.observe(agent))["last_action"]


async def score(made: HorizonWorlds) -> dict[str, Any]:
    result = await made.call(HANDLE, "score", {}, effect_id="score", arguments_digest="")
    assert result.structured is not None
    return dict(result.structured)  # type: ignore[arg-type]


@pytest.mark.live
@pytest.mark.asyncio(loop_scope="module")
async def test_iron_stored_in_a_chest_the_team_placed_counts_and_what_it_began_with_does_not(
    worlds: HorizonWorlds,
) -> None:
    await worlds.create(HANDLE, world("iron-underground-5m", 12345, 7, CREW), {})
    episode = worlds._world(HANDLE)  # pyright: ignore[reportPrivateUsage]
    assert episode.held_at_start.get("stone_pickaxe") == 2 and episode.held_at_start.get("bread") == 16  # (the kit)
    began = await score(worlds)
    assert began["amount"] == 0.0 and began["reward"] == 0.0 and began["containers"] == 0

    here = (await episode.harness.observe("ada"))["self"]["position"]
    x, y, z = int(here["x"] // 1), int(here["y"] // 1), int(here["z"] // 1)
    await episode.control.drop_items(x, y, z, [{"item": "chest"}, {"item": "raw_iron", "count": 5}])
    await act(worlds, {"name": "wait"})  # (walking over them, or standing on them, picks them up)
    holding = await score(worlds)
    assert holding["amount"] == 5.0 and holding["held"].get("raw_iron") == 5, holding["held"]

    placed: dict[str, Any] | None = None
    spot = (x, y, z)
    for dx, dz in [(1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (0, 2)]:  # somewhere free beside ada
        placed = await act(worlds, {"name": "place_at", "item": "chest", "x": x + dx, "y": y, "z": z + dz})
        if placed["ok"]:
            spot = (x + dx, y, z + dz)
            break
    assert placed is not None and placed["ok"], placed
    stored = await act(
        worlds, {"name": "store", "item": "raw_iron", "count": 5, "x": spot[0], "y": spot[1], "z": spot[2]}
    )
    assert stored["ok"], stored
    after = await score(worlds)
    assert after["containers"] == 1 and after["stored"] == {"raw_iron": 5}, after
    assert after["amount"] == 5.0 and "raw_iron" not in after["held"]

    # A chest the team did not place (one the plugin set up) holds nothing of the team's.
    await episode.control.chest(x + 3, y, z + 3, [{"item": "raw_iron", "count": 9}])
    assert (await score(worlds))["amount"] == 5.0


@pytest.mark.live
@pytest.mark.asyncio(loop_scope="module")
async def test_a_speedrun_world_starts_with_no_progress_and_no_time_spent(worlds: HorizonWorlds) -> None:
    await worlds.delete(HANDLE)  # (the pool holds one world at a time)
    await worlds.create(HANDLE, world("nether-portal-kit-5m", 12345, 3, CREW), {})
    window = await worlds.call(HANDLE, "window", {}, effect_id="window", arguments_digest="")
    assert window.structured is not None and window.structured["done"] is False  # type: ignore[index]
    began = await score(worlds)
    assert began["objective"] == "nether" and began["amount"] == 0.0
    assert began["amount_parts"] == {"progress": 0.0, "time_left": 0.0}
    assert began["held_at_start"].get("obsidian") == 28  # (the portal kit, two agents' worth: not progress)


@pytest.mark.live
@pytest.mark.asyncio(loop_scope="module")
async def test_a_building_world_lays_out_its_sites_and_counts_materials_in_the_blueprint_s_places(
    worlds: HorizonWorlds,
) -> None:
    await worlds.delete(HANDLE)
    await worlds.create(HANDLE, world("huts-fresh-5m", 12345, 5, CREW), {})
    episode = worlds._world(HANDLE)  # pyright: ignore[reportPrivateUsage]
    brief = await worlds.call(HANDLE, "brief", {}, effect_id="brief", arguments_digest="")
    sites = cast(list[dict[str, int]], brief.structured["sites"])  # type: ignore[index]
    assert len(sites) == 16 and len(episode.sites) == 16
    first = episode.sites[0]
    floor = await episode.control.blocks(
        [(first.x + dx, first.y - 1, first.z + dz) for dx in range(5) for dz in range(5)]
    )
    assert set(floor) == {"smooth_stone"}
    assert set(await episode.control.blocks(positions([first]))) == {"air"}  # (cleared)
    assert (await score(worlds))["amount"] == 0.0
    for dx, dy, dz in HUT[:5]:
        await episode.control.set_block(first.x + dx, first.y + dy, first.z + dz, "oak_planks")
    dx, dy, dz = HUT[5]
    await episode.control.set_block(first.x + dx, first.y + dy, first.z + dz, "dirt")  # (not a building material)
    built = await score(worlds)
    assert built["amount"] == 5.0 and list(built["amount_parts"].values()) == [5.0]
