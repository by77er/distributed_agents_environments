"""Paper servers from templates, and the ground-truth plugin's tick control (live: needs Java and the network once)."""

import asyncio
import shutil

import pytest
from minecraft_swarm.control import Control
from minecraft_swarm.paper import (  # pyright: ignore[reportPrivateUsage]
    Installation,
    PaperServer,
    _merge,
    server_properties,
)

SEED = 12345


def test_configuration_overrides_merge_into_paper_defaults() -> None:
    defaults = {"anticheat": {"anti-xray": {"enabled": False, "engine-mode": 2, "hidden-blocks": ["gold_ore"]}}}
    overrides = {"anticheat": {"anti-xray": {"enabled": True, "hidden-blocks-also": ["diamond_ore", "gold_ore"]}}}
    assert _merge(defaults, overrides) == {
        "anticheat": {"anti-xray": {"enabled": True, "engine-mode": 2, "hidden-blocks": ["gold_ore", "diamond_ore"]}}
    }
    properties = server_properties()
    assert properties["online-mode"] == "false" and properties["server-ip"] == "127.0.0.1"


@pytest.fixture(scope="module")
def installation() -> Installation:
    if shutil.which("java") is None:
        pytest.skip("Java is not installed")
    return Installation()


async def test_a_server_freezes_steps_and_reports_ground_truth(installation: Installation) -> None:
    server = PaperServer(installation, seed=SEED)
    await server.start()
    control = Control(server.control_url)
    try:
        assert (await control.health())["version"] == "1.21.11"
        frozen = await control.freeze()
        await asyncio.sleep(0.5)
        assert (await control.ticks())["game_time"] == frozen["game_time"]  # nothing ticks while frozen
        stepped = await control.step(40)
        assert stepped["stepped"] == 40 and stepped["frozen"]
        ores = await control.ores(0, -58, 0, radius=24)
        assert ores and all(o["y"] < 16 for o in ores)
        assert (await control.state())["team_diamonds"] == 0
    finally:
        await control.close()
        await server.stop()
    assert not server.directory.exists()
