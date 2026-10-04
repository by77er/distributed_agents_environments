"""The policies as a graph: forks and distillations between lines of versions, trainers, workers and evaluations."""

from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import JsonValue

from rollout_train.monitor.lineage import (
    MADE,
    OFF_POLICY,
    ON_POLICY,
    QUEUED,
    RESHARDING,
    ROLLING,
    SERVING,
    SUPERSEDED,
    TAKING,
    WRITTEN,
    lineage,
    mode,
)

NOW = 1_800_000_000.0


def version(policy: str, number: int, parent: str | None, made: float) -> dict[str, JsonValue]:
    return {"policy": policy, "number": number, "weights": {"files": {}}, "parent": parent, "made": made}


def step(policy: str, number: int, parent: str | None, decided: float, **more: JsonValue) -> dict[str, JsonValue]:
    return {"policy": policy, "number": number, "parent": parent, "groups": [number], "decided": decided, **more}


def tables() -> dict[str, dict[str, JsonValue]]:
    """A policy trained by a run, a fork of it, and a distillation of both into a new policy."""
    return {
        "policies/miner/versions": {
            str(n): version("miner", n, f"miner@{n - 1}" if n > 1 else None, n * 100.0) for n in (1, 2, 3)
        },
        "policies/miner/released": {"1": {"at": 900.0}},
        "runs/train/steps": {str(n): step("miner", n, None, n * 100.0 - 50) for n in (1, 2, 3)},
        "runs/train/results": {"1": {"time": 40.0, "segments": 3}, "2": {"time": 140.0, "segments": 3}},
        "policies/digger/versions": {
            "1": version("digger", 1, "miner@2", 400.0),
            "2": version("digger", 2, "digger@1", 500.0),
        },
        "runs/dig/steps": {
            "1": step("digger", 1, "miner@2", 350.0, trainer="shared"),
            "2": step("digger", 2, "digger@1", 450.0, trainer="shared"),
            "3": step("digger", 3, "digger@2", 950.0, trainer="shared"),
        },
        "policies/both/versions": {"1": version("both", 1, "miner@3", 700.0)},
        "policies/both/definition": {"definition": {"weights": "full", "base": "big"}},
        "policies/both/resharding": {"1": {"at": 710.0}},
        "runs/merge/plan": {
            "plan": {
                "kind": "distill",
                "student": "both",
                "from": "miner@3",
                "teachers": ["miner@3", "digger@2"],
                "data": {"sampled_by": ["miner", "digger"]},
                "objective": "likelihood",
            }
        },
        "runs/merge/steps": {"1": step("both", 1, "miner@3", 650.0, teachers=["miner@3", "digger@2"])},
        "trainers/shared/registered": {"1": {"weights": "lora", "base": "small", "policies": ["digger"]}},
        "trainers/shared/queue": {
            f"dig/{n}": {"run": "dig", "step": n, "policy": "digger", "makes": f"digger@{n}", "at": at}
            for n, at in ((1, 350.0), (2, 450.0), (3, 950.0))
        },
        "trainers/shared/taken": {"dig/1": {"began": 360.0}, "dig/2": {"began": 460.0}, "dig/3": {"began": 960.0}},
        "runs/dig/published": {"digger@1": {"at": 410.0}, "digger@2": {"at": 510.0}},
        "workers/a/registered": {"1": {"holds": {"base": "small"}, "adapters": 4}},
        "workers/a/loaded": {"digger@1/1": {"at": 420.0}, "digger@2/1": {"at": 520.0}},
        "workers/a/unloaded": {"digger@1/1": {"at": 520.0}},
        "workers/b/loaded": {"digger@1/1": {"at": 430.0}},
        "evaluations/suite/starts": {"1": {"task": "t1", "seed": 1}, "2": {"task": "t2", "seed": 1}},
        "evaluations/suite/miner@3/results": {
            "1-1": {"solved": True, "reward": 1.0},
            "2-1": {"solved": False, "reward": 0.2},
        },
        "evaluations/suite/model.big/subject": {"subject": {"kind": "model", "model": "big"}},
        "evaluations/suite/model.big/results": {"1-1": {"solved": True, "reward": 1.2}},
    }


def test_a_distillation_is_on_policy_when_the_student_samples_and_off_it_when_others_do() -> None:
    assert mode("both", ["both"]) == ON_POLICY and mode("both", ["both@4"]) == ON_POLICY
    assert mode("both", ["miner", "digger@2"]) == OFF_POLICY
    assert mode("both", ["both", "miner"]) == "mixed"


def test_the_graph_has_each_line_its_forks_and_its_distillations_and_what_trains_serves_and_evaluates_them() -> None:
    graph = lineage(tables(), now=NOW)
    policies = {policy["policy"]: policy for policy in graph["policies"]}
    assert [version["name"] for version in policies["miner"]["versions"]] == ["miner@1", "miner@2", "miner@3"]
    assert not policies["miner"]["versions"][0]["kept"] and policies["miner"]["versions"][0]["released"] == 900.0
    assert policies["digger"]["fork"] == "miner@2" and policies["miner"]["fork"] is None
    # A distillation's start is drawn into the distillation, not as a fork; its teachers are edges into it too.
    assert {(edge["kind"], edge["from"], edge["to"]) for edge in graph["edges"]} == {
        ("fork", "miner@2", "digger@1"),
        ("start", "miner@3", "merge"),
        ("teach", "miner@3", "merge"),
        ("teach", "digger@2", "merge"),
    }
    runs = {run["run"]: run for run in graph["runs"]}
    assert (
        runs["merge"]["kind"] == "distill"
        and runs["merge"]["mode"] == OFF_POLICY
        and runs["merge"]["versions"] == ["both@1"]
    )
    assert runs["train"]["kind"] == "train" and runs["train"]["groups"] == [
        [40.0, 1],
        [50.0, 0],
        [140.0, 1],
        [150.0, 0],
    ]
    assert runs["dig"]["latest"] == "digger@2"

    # The way to the engines: a run's latest serves once no worker serves an older version in its place.
    life = {version["name"]: version["life"]["state"] for policy in graph["policies"] for version in policy["versions"]}
    assert life["digger@2"] == ROLLING  # (worker b still serves digger@1)
    assert life["digger@1"] == SERVING and life["both@1"] == RESHARDING and life["miner@3"] == WRITTEN

    trainers = {trainer["trainer"]: trainer for trainer in graph["trainers"]}
    shared = trainers["shared"]
    assert [entry["state"] for entry in shared["queue"]] == [MADE, MADE, TAKING]
    assert shared["depth"][-1] == [960.0, 0, 1]
    implicit = trainers["train (the run's own)"]
    assert implicit["implicit"] and implicit["policies"] == ["miner"] and len(implicit["queue"]) == 3
    assert {worker["worker"]: worker["serving"] for worker in graph["workers"]} == {
        "a": ["digger@2"],
        "b": ["digger@1"],
    }

    (suite,) = graph["evaluations"]
    subjects = {subject["subject"]: subject for subject in suite["subjects"]}
    assert (
        subjects["miner@3"]["solved"] == 1
        and subjects["miner@3"]["played"] == 2
        and subjects["miner@3"]["version"] == "miner@3"
    )
    assert subjects["model.big"]["kind"] == "model" and subjects["model.big"]["version"] is None
    assert not graph["sample"] and not any(policy["sample"] for policy in graph["policies"])


def test_without_proposed_tables_a_runs_steps_stand_for_its_trainer_and_its_published_notes_for_what_it_serves() -> (
    None
):
    only = {name: records for name, records in tables().items() if name.startswith(("policies/miner/", "runs/train/"))}
    notes = [
        {"kind": "published", "job": "train", "channel": "policy", "adapter": "miner@2", "at": 260.0},
        {"kind": "published", "job": "train", "channel": "policy", "adapter": "miner@3", "at": 310.0},
        {"kind": "published", "job": "train", "channel": "policy", "adapter": "step-1", "at": 100.0},  # (not a version)
    ]
    graph = lineage(only, notes, now=NOW)
    life = {version["name"]: version["life"] for version in graph["policies"][0]["versions"]}
    assert life["miner@3"]["state"] == SERVING and life["miner@3"]["latest_of"] == "train"
    assert life["miner@2"]["state"] == SUPERSEDED and life["miner@1"]["state"] == WRITTEN
    assert [worker["worker"] for worker in graph["workers"]] == ["train engines"]
    assert [trainer["trainer"] for trainer in graph["trainers"]] == ["train (the run's own)"]


def test_the_sample_fixture_is_read_only_when_asked_for_is_marked_and_never_replaces_a_ledgers_table() -> None:
    own = tables()
    graph = lineage(own, sample=True, now=NOW)
    assert graph["sample"]
    policies = {policy["policy"]: policy for policy in graph["policies"]}
    assert not policies["miner"]["sample"] and {"scout", "crafter", "merged", "planner"} <= {
        name for name, policy in policies.items() if policy["sample"]
    }
    # What the sample shows: two distillations, a shared trainer with a queue, resharding, a roll-out, evaluations.
    modes = {run["mode"] for run in graph["runs"] if run["kind"] == "distill" and run["sample"]}
    assert modes == {ON_POLICY, OFF_POLICY}
    lora = next(trainer for trainer in graph["trainers"] if trainer["trainer"] == "lora-9b")
    assert [entry["state"] for entry in lora["queue"]].count(QUEUED) >= 2 and max(
        point[1] for point in lora["depth"]
    ) >= 2
    states = {version["life"]["state"] for policy in graph["policies"] for version in policy["versions"]}
    assert {RESHARDING, ROLLING, SERVING} <= states
    assert any(suite["sample"] and len(suite["subjects"]) > 3 for suite in graph["evaluations"])
    assert graph["routing"]["waiting"] and graph["routing"]["history"]
    every: list[Any] = [version["made"] for policy in graph["policies"] for version in policy["versions"]]
    assert all(0 < made <= NOW for made in every)  # (the fixture's times are moved to before now)


async def test_the_page_asks_for_the_graph_with_or_without_the_sample(tmp_path: Path) -> None:
    pytest.importorskip("starlette")
    from rollout_train.monitor.app import create_app

    transport = httpx.ASGITransport(app=create_app(tmp_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://monitor") as client:
        plain = (await client.get("/api/policies")).json()
        sampled = (await client.get("/api/policies?sample=1")).json()
    assert plain["policies"] == [] and not plain["sample"]
    assert sampled["sample"] and sampled["policies"] and all(policy["sample"] for policy in sampled["policies"])
    assert not (tmp_path / "ledger").exists()  # (a reader makes no ledger where none is)
