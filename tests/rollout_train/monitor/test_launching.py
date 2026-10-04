"""The monitor from the heartbeats and the database alone: the machines, what the engines serve and how fast, finished
episodes read from the blob store a run's start names, and runs asked for from the page."""

import lzma
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import JsonValue

from rollout.contracts import RunEvent, RunEventType
from rollout.harness.blobs import FileBlobStore
from rollout_train.launcher import LAUNCHER
from rollout_train.launches import ASKED, STOPPED, STOPPING, launches_of
from rollout_train.ledger import FileLedger, Ledger
from rollout_train.monitor.system import System
from rollout_train.presence import presence_of
from rollout_train.record import GROUPS, STARTS, scope, table
from rollout_train.registry import Taken, registry_of
from rollout_train.rollouts.episodes import Episode, Outcome, Record, Trajectory
from rollout_train.rollouts.scheduler import EPISODES, runner_scope
from rollout_train.stores import location
from rollout_train.versions import Versions, new_id

pytest.importorskip("starlette")
from rollout_train.monitor.app import create_app

OFFERED: dict[str, Any] = {
    "profile": "one-gpu",
    "path": "/profiles/one-gpu.toml",
    "model": "m",
    "settings": {"trainer.learning_rate": 5e-5, "episodes_at_once": 6, "trainer.start": None},
}


async def a_run(ledger: Ledger, run: str, **start: JsonValue) -> None:
    fence = await ledger.take(scope(run))
    begun: JsonValue = {"from": None, "host": "far-away", "started": time.time() - 60, **start}
    await ledger.append(table(run, STARTS), str(fence.number), begun, fence)
    await ledger.append(table(run, GROUPS), "1", {"task": "t", "decided": time.time() - 60, "episodes": 1}, fence)


async def test_machines_engines_and_throughput_come_from_the_runners_heartbeats(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    await a_run(ledger, "train")
    heartbeats = presence_of(ledger)
    assert heartbeats is not None
    gpu: dict[str, Any] = {"name": "RTX", "used": 1, "total": 2, "busy": 0.5}
    machine: dict[str, Any] = {"memory": {"available": 1, "total": 2}, "accelerators": [gpu], "disk": None}
    served: dict[str, Any] = {"channel": "policy", "version": 0, "requests": 0}
    await heartbeats.beat("far/train", {"run": "train", "machine": machine, "channels": [{**served, "adapter": None}]})
    busy = {
        **served,
        "adapter": "kpqx",
        "version": 1,
        "requests": 4,
        "tokens_per_second": 80.0,
        "mean_concurrency": 2.0,
    }
    await heartbeats.beat("far/train", {"run": "train", "machine": machine, "channels": [busy], "places": 6})
    await heartbeats.beat("launcher/far", {"kind": LAUNCHER, "profiles": []})

    system = System(ledger=ledger)
    (run,) = (await system.snapshot())["runs"]
    (channel,) = run["channels"]
    assert channel["adapter"] == "kpqx" and channel["version"] == 1 and channel["throughput"][0]["requests"] == 4
    assert run["state"] == "running"  # (it beat just now, though its ledger is a minute old)
    machines = (await system.machines())["machines"]
    assert sorted(each["runner"] for each in machines) == ["far/train", "launcher/far"]
    (far,) = [each for each in machines if each["runner"] == "far/train"]
    assert far["alive"] and far["machine"]["accelerators"] == [gpu] and far["places"] == 6 and len(far["history"]) == 2
    figures = await system.statistics()
    (measured,) = [each for each in figures["runs"] if each["run"] == "train"]
    assert [each["channel"] for each in measured["inference"]] == ["policy"] and "machines" in figures
    published = [edge for edge in (await system.lineage())["runs"] if edge["run"] == "train"]
    assert published == [] or published[0]["latest"] in (None, "kpqx")


async def test_a_finished_episode_is_read_from_the_blob_store_its_runs_start_names(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    store = FileBlobStore(tmp_path / "shared-blobs")  # (an object store every machine reaches, say)
    await a_run(ledger, "far", directory="/on/another/machine", blobs=location({}, tmp_path / "shared-blobs"))
    at = datetime.now(UTC)
    created = RunEvent(run_id="r_far", seq=0, type=RunEventType.RUN_CREATED, payload={}, recorded_at=at)
    events = await store.put(lzma.compress((created.model_dump_json() + "\n").encode()), "application/x-xz")
    episode = Episode("far", 1, 1, "r_far", {"run": "far", "group": "1", "episode": "1"}, Outcome.COMPLETED,
                      trajectories={"ada": Trajectory([], {"default": 1.0})})  # fmt: skip
    fence = await ledger.take(runner_scope("far-runner"))
    await ledger.append(table("far", EPISODES), "1/1", Record(episode, events=events).to_json(), fence)

    shown = await System(ledger=ledger).episode("r_far")
    assert shown["source"] == "archive" and shown["ended"]["state"] == "completed"


async def launching(tmp_path: Path) -> tuple[FileLedger, httpx.AsyncClient]:
    ledger = FileLedger(tmp_path / "ledger")
    await a_run(ledger, "train")
    registry = registry_of(ledger)
    heartbeats = presence_of(ledger)
    assert registry is not None and heartbeats is not None
    await registry.create("taken name", "train")
    fence = await ledger.take(scope("train"))
    weights = tmp_path / "w"
    weights.mkdir()
    (weights / "adapter.bin").write_text("weights")
    version = await Versions(ledger, FileBlobStore(tmp_path / "blobs")).add(
        fence, new_id(), weights=weights, run="train"
    )
    await registry.bookmark("best", version.id)
    about: JsonValue = {"kind": LAUNCHER, "profiles": [OFFERED], "catalogs": ["c:c"], "at_once": 1, "playing": 0}
    await heartbeats.beat("launcher/far", about)
    transport = httpx.ASGITransport(app=create_app(str(tmp_path / "ledger"), beat=0.0))
    return ledger, httpx.AsyncClient(transport=transport, base_url="http://monitor")


async def test_a_run_is_asked_for_from_the_page_with_its_settings_and_stopped(tmp_path: Path) -> None:
    ledger, client = await launching(tmp_path)
    async with client:
        listed = (await client.get("/api/launches")).json()
        assert listed["launches"] == [] and [each["launcher"] for each in listed["launchers"]] == ["launcher/far"]
        asked = {"profile": "one-gpu", "catalog": "c:c", "name": "diamonds, again", "start": "best",
                 "settings": {"trainer.learning_rate": 3e-5}, "groups": 40}  # fmt: skip
        answer = await client.post("/api/launches", json=asked)
        assert answer.status_code == 200
        made = answer.json()["launch"]
        assert made["state"] == ASKED and made["asked"]["settings"] == {"trainer.learning_rate": 3e-5}
        assert made["asked"]["groups"] == 40
        refused = {
            "unknown profile": ({**asked, "profile": "eight-gpu"}, 404),
            "unknown catalog": ({**asked, "catalog": "other:catalog"}, 404),
            "taken name": ({**asked, "name": "taken name"}, 409),
            "no name": ({**asked, "name": " "}, 409),
            "unknown setting": ({**asked, "settings": {"episodes_at_onc": 2}}, 409),
            "no such version": ({**asked, "start": "nothing-like-it"}, 404),
            "missing fields": ({"profile": "one-gpu"}, 409),
        }
        for why, (body, status) in refused.items():
            assert (await client.post("/api/launches", json=body)).status_code == status, why
        assert (await client.post("/api/launches", content=b"not json")).status_code == 400
        stopped = (await client.post(f"/api/launches/{made['id']}/stop")).json()["launch"]
        assert stopped["state"] == STOPPED  # (not started yet: stopped at once)
        assert (await client.post(f"/api/launches/{made['id']}/stop")).status_code == 404  # (nothing going)
        assert (await client.post("/api/launches/launch_nothing/stop")).status_code == 404
        assert (await client.get("/api/machines")).status_code == 200

    launches = launches_of(ledger)
    assert launches is not None
    going = await launches.ask(next(iter(await launches.all())).asked)
    await launches.claim(going.id, "launcher/far")
    system = System(ledger=ledger)
    assert (await system.stop(going.id)).state == STOPPING  # (a run going is stopped by its launcher)


async def test_no_launch_is_asked_for_a_profile_no_launcher_alive_offers(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    await a_run(ledger, "train")
    system = System(ledger=ledger)
    with pytest.raises(KeyError, match="no launcher alive offers"):
        await system.launch({"profile": "one-gpu", "catalog": "c:c", "name": "fresh"})
    with pytest.raises(Taken):
        await system.launch({"profile": "one-gpu"})


async def test_a_launch_whose_launcher_stopped_beating_is_shown_lost(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    launches = launches_of(ledger)
    assert launches is not None
    from rollout_train.launches import RUNNING, Asked

    asked = await launches.ask(Asked(profile="one-gpu", catalog="words:catalog", name="quiet"))
    await launches.claim(asked.id, "launcher/gone")  # (a launcher that never beats)
    await launches.note(asked.id, state=RUNNING, pid=4242)
    waiting = await launches.ask(Asked(profile="one-gpu", catalog="words:catalog", name="waiting"))
    shown = {each["id"]: each for each in (await System(ledger=ledger).launches())["launches"]}
    assert shown[asked.id]["state"] == "lost" and "stopped beating" in shown[asked.id]["detail"]
    assert shown[waiting.id]["state"] == ASKED  # (not claimed: nobody's to lose)
    stored = next(each for each in await launches.all() if each.id == asked.id)
    assert stored.state == RUNNING  # shown lost, kept as it was
