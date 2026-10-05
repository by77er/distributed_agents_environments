"""Paper servers from templates, and the ground-truth plugin's tick control (live: needs Java and the network once)."""

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from minecraft_team import paper
from minecraft_team.control import Control
from minecraft_team.paper import (
    PAPER_VERSION,
    Installation,
    PaperServer,
    merge_configuration,
    offline_uuid,
    operator_entries,
    server_properties,
)

SEED = 12345


def test_configuration_overrides_merge_into_paper_defaults() -> None:
    defaults = {"anticheat": {"anti-xray": {"enabled": False, "engine-mode": 2, "hidden-blocks": ["gold_ore"]}}}
    overrides = {"anticheat": {"anti-xray": {"enabled": True, "hidden-blocks-also": ["diamond_ore", "gold_ore"]}}}
    assert merge_configuration(defaults, overrides) == {
        "anticheat": {"anti-xray": {"enabled": True, "engine-mode": 2, "hidden-blocks": ["gold_ore", "diamond_ore"]}}
    }
    properties = server_properties()
    assert properties["online-mode"] == "false" and properties["server-ip"] == "127.0.0.1"


def test_operators_are_listed_by_their_offline_ids() -> None:
    # What an offline-mode server computes for the name (it logs "UUID of player By73 is ..." when they join).
    assert str(offline_uuid("By73")) == "73bc9dd2-78aa-3946-ba5b-5b7c56e2b546"
    assert operator_entries(["By73"]) == [
        {"uuid": "73bc9dd2-78aa-3946-ba5b-5b7c56e2b546", "name": "By73", "level": 4, "bypassesPlayerLimit": True}
    ]


@pytest.mark.skipif(shutil.which("npm") is None, reason="npm is needed")
def test_without_its_own_packages_the_harness_takes_them_installed_once_per_lock_under_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = tmp_path / "harness"  # (a version imported from git: its sources, and no node_modules)
    harness.mkdir()
    (harness / "package.json").write_text(json.dumps({"name": "h", "version": "1.0.0", "private": True}))
    lock = {"name": "h", "version": "1.0.0", "lockfileVersion": 3, "requires": True,
            "packages": {"": {"name": "h", "version": "1.0.0"}}}  # fmt: skip
    (harness / "package-lock.json").write_text(json.dumps(lock))
    monkeypatch.setattr(paper, "HARNESS", harness)
    installation = Installation(root=tmp_path / "cache")
    packages = installation.harness_packages()
    assert packages.parent.parent == tmp_path / "cache" / "harness" and (packages.parent / "ready").exists()
    assert installation.harness_packages() == packages  # (installed once)
    (harness / "node_modules").mkdir()
    assert installation.harness_packages() == harness / "node_modules"  # (its own, where they are installed)


@pytest.fixture(scope="module")
def installation() -> Installation:
    if shutil.which("java") is None:
        pytest.skip("Java is not installed")
    return Installation()


@pytest.mark.live
async def test_a_server_freezes_steps_and_reports_ground_truth(installation: Installation) -> None:
    server = PaperServer(installation, seed=SEED)
    await server.start()
    control = Control(server.control_url)
    try:
        assert (await control.health())["version"] == PAPER_VERSION
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
