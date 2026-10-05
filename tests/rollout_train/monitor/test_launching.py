"""The monitor from the heartbeats and the database alone: the machines, what the engines serve and how fast, finished
episodes read from the blob store a run's start names; and, with the cluster config, what a run can be asked for, runs
asked for from the page (checked first, refusals tied to their settings), stopped, and resumed."""

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
from rollout_train.checkpoints import new_id
from rollout_train.launches import STOPPING, SUBMITTED, TRAIN, launches_of
from rollout_train.ledger import FileLedger, Ledger
from rollout_train.monitor.system import System
from rollout_train.presence import presence_of
from rollout_train.record import ENDS, GROUPS, STARTS, scope, table
from rollout_train.rollouts.episodes import Episode, Outcome, Record, Trajectory
from rollout_train.rollouts.scheduler import EPISODES, runner_scope
from rollout_train.stores import Stores, location
from rollout_train.submitting import RayJobs
from tests.rollout_train.clusters import POLICY, WORDS, a_cluster
from tests.rollout_train.test_submitting import Jobs

pytest.importorskip("starlette")
from rollout_train.monitor.app import create_app


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
    await heartbeats.beat("run/waiting", {"kind": "run", "run": "waiting", "waiting": ["run/waiting/trainer (1 GPU)"]})

    system = System(ledger=ledger)
    (run,) = (await system.snapshot())["runs"]
    (channel,) = run["channels"]
    assert channel["adapter"] == "kpqx" and channel["version"] == 1 and channel["throughput"][0]["requests"] == 4
    assert run["state"] == "running"  # (it beat just now, though its ledger is a minute old)
    machines = await system.machines()
    assert [each["name"] for each in machines["runners"]] == ["far/train", "run/waiting"]
    assert "launchers" not in machines
    far, waiting = machines["runners"]
    assert waiting["waiting"] == ["run/waiting/trainer (1 GPU)"] and waiting["places"] == 0
    assert far["alive"] and far["places"] == 6 and far["channels"][0]["serving"] == "kpqx"
    (host,) = [each for each in machines["hosts"] if each["host"] == "far/train"]  # (its beats name no host)
    assert host["machine"]["accelerators"] == [gpu] and len(host["history"]) == 2
    figures = await system.statistics()
    (measured,) = [each for each in figures["runs"] if each["run"] == "train"]
    assert [each["channel"] for each in measured["inference"]] == ["policy"] and "machines" not in figures
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


async def launching(tmp_path: Path, jobs: Jobs) -> tuple[Stores, httpx.AsyncClient]:
    """A monitor over a cluster config's ledger that asks for runs on it, their jobs going to `jobs`; a run called
    `taken name`, a bookmark `best` and a preset `small` beside it."""
    cluster = a_cluster(tmp_path)
    stores = Stores.open(cluster)
    await stores.registry.create("taken name")
    fence = await stores.ledger.take(scope("elsewhere"))
    weights = tmp_path / "w"
    weights.mkdir()
    (weights / "adapter_config.json").write_text("{}")
    (weights / "adapter_model.safetensors").write_text("weights")
    checkpoint = await stores.checkpoints.add(fence, new_id(), weights=weights, run="elsewhere", base="tiny")
    await stores.registry.bookmark("best", checkpoint.id)
    await stores.presets.save("small", {**POLICY, "environment": WORDS}, "the test's")
    backends = {"ray": RayJobs("x", jobs)}
    app = create_app(f"sqlite:///{tmp_path}/ledger.db", beat=0.0, cluster=cluster, backends=backends)
    return stores, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://monitor")


async def test_the_page_is_offered_what_the_cluster_config_and_the_ledger_hold(tmp_path: Path) -> None:
    _, client = await launching(tmp_path, Jobs())
    async with client:
        offers = (await client.get("/api/offers")).json()
    assert offers["cluster"] == "test" and offers["submits"] == "ray"
    assert offers["environments"][0]["environment"] == WORDS
    (trainer,) = offers["trainers"]
    assert trainer["name"] == "steps" and trainer["format"] == "peft" and trainer["models"] == ["tiny"]
    assert any(each["key"] == "trainer.learning_rate" and each["changeable"] for each in trainer["settings"])
    (provider,) = offers["inference"]
    assert provider["name"] == "local" and provider["capabilities"]["token_exact"]
    assert [each["model"] for each in provider["models"]] == ["tiny"]
    assert (provider["allocation"], provider["concurrency"], trainer["allocation"]) == ("scheduled", None, "scheduled")
    assert (trainer["weights"], provider["weights"]) == (["lora"], ["lora", "full"])
    assert offers["pairs"] == [{"trainer": "steps", "inference": "local", "bridge": ["verbatim"]}]
    (preset,) = offers["presets"]
    assert preset["id"] == "small@1" and preset["settings"]["trainer.provider"] == "steps"
    plain = httpx.ASGITransport(app=create_app(str(tmp_path / "files"), beat=0.0))
    async with httpx.AsyncClient(transport=plain, base_url="http://monitor") as without:
        assert (await without.get("/api/offers")).json()["cluster"] is None  # (a monitor with no cluster config)
        refused = await without.post("/api/launches", json={"kind": TRAIN, "name": "x", "preset": "small"})
        assert refused.status_code == 409 and "cluster config" in refused.json()["error"]


async def test_a_run_is_checked_asked_for_from_the_page_and_stopped(tmp_path: Path) -> None:
    jobs = Jobs()
    stores, client = await launching(tmp_path, jobs)
    asked: dict[str, Any] = {
        "kind": TRAIN, "name": "words, again", "environment": WORDS, "preset": "small",
        "settings": {"trainer.learning_rate": 3e-5, "start": "best"},
    }  # fmt: skip
    wrong = {**asked, "settings": {"channels.policy.provider": "elsewhere", "trainer.rank": "big"}}
    async with client:
        checked = (await client.post("/api/launches/check", json=asked)).json()
        assert checked["refusals"] == [] and checked["preset"] == "small@1"
        assert checked["settings"]["trainer.learning_rate"] == 3e-5 and checked["settings"]["groups"] == 2
        refusals = (await client.post("/api/launches/check", json=wrong)).json()["refusals"]
        assert {each["key"] for each in refusals} >= {"channels.policy.provider", "trainer.rank"}
        answer = await client.post("/api/launches", json=asked)
        assert answer.status_code == 200, answer.text
        made = answer.json()["launch"]
        assert made["state"] == SUBMITTED and made["asked"]["preset"] == "small@1" and made["run"]
        assert made["job"] == jobs.submitted[0]["submission_id"]
        assert made["asked"]["settings"]["trainer.learning_rate"] == 3e-5
        assert made["asked"]["settings"]["weights"] == "lora"  # (unsaid: what its trainer makes, recorded so)
        refused = await client.post("/api/launches", json=wrong)
        assert refused.status_code == 422
        assert {each["key"] for each in refused.json()["refusals"]} >= {"channels.policy.provider", "trainer.rank"}
        taken = await client.post("/api/launches", json={**asked, "name": "taken name"})
        assert taken.status_code == 422 and [each["key"] for each in taken.json()["refusals"]] == ["name"]
        assert (await client.post("/api/launches", json={**asked, "name": " "})).status_code == 409
        assert (await client.post("/api/launches", json={**asked, "preset": "none-such"})).status_code == 404
        assert (await client.post("/api/launches", content=b"not json")).status_code == 400
        listed = (await client.get("/api/launches")).json()
        assert [each["id"] for each in listed["launches"]] == [made["id"]] and listed["submits"]
        stopping = (await client.post(f"/api/launches/{made['id']}/stop")).json()["launch"]
        assert stopping["state"] == STOPPING and jobs.stopped == [made["job"]]
        assert (await client.post("/api/launches/launch_nothing/stop")).status_code == 404
    launches = launches_of(stores.ledger)
    assert launches is not None and len(await launches.all()) == 1


async def test_a_run_is_paused_resumed_in_place_and_once_stopped_submitted_again_from_the_page(tmp_path: Path) -> None:
    jobs = Jobs()
    stores, client = await launching(tmp_path, jobs)
    ledger = stores.ledger
    entry = await stores.registry.create("words")
    fixed: dict[str, JsonValue] = {**POLICY, "kind": TRAIN, "name": "words", "environment": WORDS, "groups": 10}
    # (a start from before runs said what they train, with a setting that is no run setting now)
    recorded: dict[str, JsonValue] = {
        "fixed": {**fixed, "seed": 3},
        "changeable": {"groups_per_step": 2, "share": 1.0},
        "preset": "small@1",
    }
    begun: dict[str, JsonValue] = {"environment": WORDS, "run_settings": recorded}
    await a_run(ledger, entry.id, **begun)
    await a_run(ledger, f"{entry.id}-eval-2", kind="eval", by=entry.id, step=2)  # (an eval its schedule asked for)
    heartbeats = presence_of(ledger)
    assert heartbeats is not None
    await heartbeats.beat(f"run/{entry.id}", {"run": entry.id})

    async def states() -> dict[str, tuple[str, bool]]:
        runs = (await client.get("/api/system")).json()["runs"]
        return {run["run"]: (run["state"], run["pause"]) for run in runs}

    async with client:
        assert (await states())[entry.id] == ("running", False)
        settings = (await client.get(f"/api/runs/{entry.id}/settings")).json()
        assert settings["fixed"]["weights"] == "lora"  # (read as its trainer's)
        paused = (await client.post(f"/api/runs/{entry.id}/pause")).json()
        assert paused["desired"]["settings"] == {"paused": True}
        resumed = (await client.post(f"/api/runs/{entry.id}/resume")).json()["resumed"]
        assert resumed["how"] == "in place" and resumed["launch"] is None
        refused = await client.post(f"/api/runs/{entry.id}/resume")
        assert refused.status_code == 409 and "is running" in refused.json()["error"]
        fence = await ledger.take(scope(entry.id))  # (stopped: its newest start says so)
        await ledger.append(table(entry.id, STARTS), str(fence.number), {**begun, "started": time.time()}, fence)
        await ledger.append(table(entry.id, ENDS), str(fence.number), {"how": "stopped", "at": time.time()}, fence)
        launch = (await client.post(f"/api/runs/{entry.id}/resume")).json()["resumed"]["launch"]
        asked = launch["asked"]
        assert (asked["resumes"], launch["run"], asked["name"]) == (entry.id, entry.id, "words")
        assert asked["preset"] == "small@1" and launch["state"] == SUBMITTED and len(jobs.submitted) == 1
        given = asked["settings"]
        assert (given["groups"], given["groups_per_step"], given["seed"]) == (10, 2, 3)
        assert "share" not in given and given["weights"] == "lora"  # (no run setting now; said as it trains)
        assert (await client.post(f"/api/runs/{entry.id}/resume")).status_code == 409  # (being launched already)
        eval_refused = await client.post(f"/api/runs/{entry.id}-eval-2/resume")
        assert eval_refused.status_code == 409 and "resume that run" in eval_refused.json()["error"]
