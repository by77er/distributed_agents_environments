"""The checkpoints as a graph: growing from base models, forks and distillations, trainers, workers and evaluations."""

from pathlib import Path

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
    SAYS,
    SERVING,
    SUPERSEDED,
    TAKING,
    WRITTEN,
    lineage,
    mode,
)

NOW = 1_800_000_000.0


def checkpoint(
    id: str, parents: list[str], depth: int, run: str, step: int, made: float, base: str = "small"
) -> dict[str, JsonValue]:
    return {
        "id": id,
        "weights": {"files": {}},
        "parents": list[JsonValue](parents),
        "depth": depth,
        "base": base,
        "run": run,
        "step": step,
        "made": made,
    }


def step(makes: str, parent: str | None, number: int, decided: float, **more: JsonValue) -> dict[str, JsonValue]:
    return {"makes": makes, "parent": parent, "groups": [number], "decided": decided, **more}


def tables() -> dict[str, dict[str, JsonValue]]:
    """A run's line of checkpoints, a run forked from it, a distillation of both, and a second line from the same
    base."""
    return {
        "checkpoints": {
            "minerone": checkpoint("minerone", [], 1, "train", 1, 100.0),
            "minertwo": checkpoint("minertwo", ["minerone"], 2, "train", 2, 200.0),
            "minerthree": checkpoint("minerthree", ["minertwo"], 3, "train", 3, 300.0),
            "diggerone": checkpoint("diggerone", ["minertwo"], 3, "dig", 1, 400.0),
            "diggertwo": checkpoint("diggertwo", ["diggerone"], 4, "dig", 2, 500.0),
            "bothone": checkpoint("bothone", ["minerthree", "diggertwo"], 4, "merge", 1, 700.0),
            "freshone": checkpoint("freshone", [], 1, "fresh", 1, 800.0),
        },
        "checkpoints/released": {"minerone": {"at": 900.0}},
        "runs/train/steps": {
            "1": step("minerone", None, 1, 50.0),
            "2": step("minertwo", "minerone", 2, 150.0),
            "3": step("minerthree", "minertwo", 3, 250.0),
        },
        "runs/train/results": {"1": {"time": 40.0, "segments": 3}, "2": {"time": 140.0, "segments": 3}},
        "runs/dig/starts": {"1": {"from": "minertwo"}},
        "runs/dig/steps": {
            "1": step("diggerone", "minertwo", 1, 350.0, trainer="shared"),
            "2": step("diggertwo", "diggerone", 2, 450.0, trainer="shared"),
            "3": step("diggerthree", "diggertwo", 3, 950.0, trainer="shared"),
        },
        "runs/fresh/steps": {"1": step("freshone", None, 1, 750.0)},
        "checkpoints/resharding": {"bothone": {"at": 710.0}},
        "runs/merge/plan": {
            "plan": {
                "kind": "distill",
                "from": "minerthree",
                "teachers": ["minerthree", "diggertwo"],
                "data": {"sampled_by": ["minerthree", "diggertwo"]},
                "objective": "likelihood",
            }
        },
        "runs/merge/steps": {"1": step("bothone", "minerthree", 1, 650.0, teachers=["minerthree", "diggertwo"])},
        "trainers/shared/registered": {"1": {"weights": "lora", "base": "small", "runs": ["dig"]}},
        "trainers/shared/queue": {
            f"dig/{n}": {"run": "dig", "step": n, "makes": makes, "at": at}
            for n, makes, at in ((1, "diggerone", 350.0), (2, "diggertwo", 450.0), (3, "diggerthree", 950.0))
        },
        "trainers/shared/taken": {"dig/1": {"began": 360.0}, "dig/2": {"began": 460.0}, "dig/3": {"began": 960.0}},
        "runs/dig/published": {"diggerone": {"at": 410.0}, "diggertwo": {"at": 510.0}},
        "workers/a/registered": {"1": {"holds": {"base": "small"}, "adapters": 4}},
        "workers/a/loaded": {"diggerone/1": {"at": 420.0}, "diggertwo/1": {"at": 520.0}},
        "workers/a/unloaded": {"diggerone/1": {"at": 520.0}},
        "workers/b/loaded": {"diggerone/1": {"at": 430.0}},
        "evaluations/suite/starts": {"1": {"task": "t1", "seed": 1}, "2": {"task": "t2", "seed": 1}},
        "evaluations/suite/minerthree/subject": {"subject": {"kind": "checkpoint", "checkpoint": "minerthree"}},
        "evaluations/suite/minerthree/results": {
            "1-1": {"solved": True, "reward": 1.0},
            "2-1": {"solved": False, "reward": 0.2},
        },
        "evaluations/suite/model.big/subject": {"subject": {"kind": "model", "model": "big"}},
        "evaluations/suite/model.big/results": {"1-1": {"solved": True, "reward": 1.2}},
    }


NAMES = {"runs": {"train": "miner", "dig": "digger"}, "bookmarks": {"best": "minerthree", "dig": "diggertwo"}}


def test_a_distillation_is_on_policy_when_the_student_samples_and_off_it_when_others_do() -> None:
    assert mode(["student"]) == ON_POLICY and mode([]) == ON_POLICY
    assert mode(["minerthree", "diggertwo"]) == OFF_POLICY
    assert mode(["student", "minerthree"]) == "mixed"
    assert all(SAYS[each] for each in (ON_POLICY, OFF_POLICY, "mixed"))


def test_every_version_grows_from_its_base_model_along_its_parents() -> None:
    graph = lineage(tables(), names=NAMES, now=NOW)
    edges = {(edge["kind"], edge["from"], edge["to"]) for edge in graph["edges"]}
    roots = {edge["to"] for edge in graph["edges"] if edge["kind"] == "base"}
    # Every checkpoint with no parent hangs from its base; two lines from the same base share that root.
    no_parent = {each["id"] for each in graph["checkpoints"] if not each["parents"]}
    assert roots == no_parent == {"minerone", "freshone"}
    assert {("base", "base:small", "minerone"), ("base", "base:small", "freshone")} <= edges
    assert graph["bases"] == ["small"]
    assert {
        ("trained", "minerone", "minertwo"),
        ("trained", "minertwo", "diggerone"),  # (the run `dig` forks at minertwo)
        ("trained", "minerthree", "bothone"),
        ("learned", "diggertwo", "bothone"),
    } <= edges
    shown = {each["id"]: each for each in graph["checkpoints"]}
    assert shown["diggertwo"]["by"] == {"run": "dig", "name": "digger", "kind": "train", "step": 2}
    assert shown["diggertwo"]["depth"] == 4 and shown["diggertwo"]["short"] == "diggert"
    assert shown["minerthree"]["bookmarks"] == ["best"] and shown["diggertwo"]["bookmarks"] == ["dig"]
    assert not shown["minerone"]["kept"] and shown["minerone"]["released"] == 900.0
    assert graph["bookmarks"] == NAMES["bookmarks"] and graph["outside"] == []


def test_the_graph_has_each_runs_versions_its_distillations_and_what_trains_serves_and_evaluates_them() -> None:
    graph = lineage(tables(), names=NAMES, now=NOW)
    edges = {(edge["kind"], edge["from"], edge["to"]) for edge in graph["edges"]}
    assert {
        ("start", "minerthree", "merge"),
        ("teach", "minerthree", "merge"),
        ("teach", "diggertwo", "merge"),
    } <= edges
    runs = {run["run"]: run for run in graph["runs"]}
    merge = runs["merge"]
    assert merge["kind"] == "distill" and merge["mode"] == OFF_POLICY and merge["checkpoints"] == ["bothone"]
    assert merge["says"] == SAYS[OFF_POLICY] and "teachers' samples" in merge["says"]
    assert all(edge["says"] == SAYS[OFF_POLICY] for edge in graph["edges"] if edge["kind"] in ("teach", "start"))
    assert runs["dig"]["from"] == "minertwo" and runs["dig"]["name"] == "digger" and runs["train"]["name"] == "miner"
    assert (
        runs["train"]["kind"] == "train"
        and runs["train"]["says"] is None
        and runs["train"]["groups"]
        == [
            [40.0, 1],
            [50.0, 0],
            [140.0, 1],
            [150.0, 0],
        ]
    )
    assert runs["dig"]["latest"] == "diggertwo"

    # The way to the engines: a run's latest serves once no worker serves an older checkpoint in its place.
    life = {each["id"]: each["life"]["state"] for each in graph["checkpoints"]}
    assert life["diggertwo"] == ROLLING  # (worker b still serves diggerone)
    assert life["diggerone"] == SERVING and life["bothone"] == RESHARDING and life["minerthree"] == WRITTEN

    trainers = {trainer["trainer"]: trainer for trainer in graph["trainers"]}
    shared = trainers["shared"]
    assert [entry["state"] for entry in shared["queue"]] == [MADE, MADE, TAKING]
    assert shared["depth"][-1] == [960.0, 0, 1]
    implicit = trainers["miner (the run's own)"]
    assert implicit["implicit"] and implicit["runs"] == ["train"] and len(implicit["queue"]) == 3
    assert implicit["base"] == "small"
    assert {worker["worker"]: worker["serving"] for worker in graph["workers"]} == {
        "a": ["diggertwo"],
        "b": ["diggerone"],
    }

    (suite,) = graph["evaluations"]
    subjects = {subject["subject"]: subject for subject in suite["subjects"]}
    assert (
        subjects["minerthree"]["solved"] == 1
        and subjects["minerthree"]["played"] == 2
        and subjects["minerthree"]["checkpoint"] == "minerthree"
    )
    assert subjects["model.big"]["kind"] == "model" and subjects["model.big"]["checkpoint"] is None
    assert not graph["sample"] and not any(each["sample"] for each in graph["checkpoints"])


def test_without_proposed_tables_a_runs_steps_stand_for_its_trainer_and_its_published_notes_for_what_it_serves() -> (
    None
):
    only = {name: records for name, records in tables().items() if name.startswith(("checkpoints", "runs/train/"))}
    only["checkpoints"] = {key: each for key, each in only["checkpoints"].items() if key.startswith("miner")}
    notes = [
        {"kind": "published", "run": "train", "channel": "policy", "adapter": "minertwo", "at": 260.0},
        {"kind": "published", "run": "train", "channel": "policy", "adapter": "minerthree", "at": 310.0},
    ]
    graph = lineage(only, notes, now=NOW)
    life = {each["id"]: each["life"] for each in graph["checkpoints"]}
    assert life["minerthree"]["state"] == SERVING and life["minerthree"]["latest_of"] == "train"
    assert life["minertwo"]["state"] == SUPERSEDED and life["minerone"]["state"] == WRITTEN
    assert [worker["worker"] for worker in graph["workers"]] == ["train engines"]
    assert [trainer["trainer"] for trainer in graph["trainers"]] == ["train (the run's own)"]


def test_a_runs_own_trainer_and_resharding_follow_what_its_checkpoints_are() -> None:
    graph = lineage(
        {
            "checkpoints": {
                "fullone": checkpoint("fullone", [], 1, "big", 1, 100.0) | {"kind": "full"},
                "adapterone": checkpoint("adapterone", [], 1, "small", 1, 100.0),
            },
            "runs/big/steps": {"1": step("fullone", None, 1, 50.0)},
            "runs/small/steps": {"1": step("adapterone", None, 1, 50.0)},
            "runs/idle/steps": {"1": step("nothingyet", None, 1, 50.0)},
        },
        now=NOW,
    )
    assert {trainer["runs"][0]: trainer["weights"] for trainer in graph["trainers"]} == {
        "big": "full",
        "small": "lora",
        "idle": None,  # (it has made nothing to say)
    }
    shown = {each["id"]: each for each in graph["checkpoints"]}
    assert (shown["fullone"]["kind"], shown["fullone"]["life"]["reshard"]) == ("full", True)
    assert (shown["adapterone"]["kind"], shown["adapterone"]["life"]["reshard"]) == ("lora", False)


def test_the_sample_fixture_is_read_only_when_asked_for_is_marked_and_never_replaces_a_ledgers_table() -> None:
    own = tables()
    graph = lineage(own, names=NAMES, sample=True, now=NOW)
    assert graph["sample"]
    shown = {each["id"]: each for each in graph["checkpoints"]}
    assert not shown["minerthree"]["sample"] and sum(each["sample"] for each in shown.values()) >= 20
    names = {run["name"] for run in graph["runs"] if run["sample"]}
    assert {"scout", "crafter", "planner"} <= names and "miner" in {run["name"] for run in graph["runs"]}
    # What the sample shows: two distillations, a shared trainer with a queue, resharding, a roll-out, evaluations.
    distilled = [run for run in graph["runs"] if run["kind"] == "distill" and run["sample"]]
    assert {run["mode"] for run in distilled} == {ON_POLICY, OFF_POLICY}
    assert all(run["says"] == SAYS[run["mode"]] for run in distilled)
    lora = next(trainer for trainer in graph["trainers"] if trainer["trainer"] == "lora-9b")
    assert [entry["state"] for entry in lora["queue"]].count(QUEUED) >= 2 and max(
        point[1] for point in lora["depth"]
    ) >= 2
    states = {each["life"]["state"] for each in graph["checkpoints"]}
    assert {RESHARDING, ROLLING, SERVING} <= states
    assert any(suite["sample"] and len(suite["subjects"]) > 3 for suite in graph["evaluations"])
    assert graph["routing"]["waiting"] and graph["routing"]["history"]
    assert all(0 < each["made"] <= NOW for each in shown.values())  # (the fixture's times are moved to before now)
    # Nothing dangles but the real ledger's checkpoints the sample forks from; lines from one base share its root.
    assert graph["outside"] and all(name.startswith("curriculum-") for name in graph["outside"])
    referenced = {edge["from"] for edge in graph["edges"]} | {edge["to"] for edge in graph["edges"]}
    runs = {run["run"] for run in graph["runs"]}
    dangling = {each for each in referenced if each not in shown and each not in runs and not each.startswith("base:")}
    assert dangling == set(graph["outside"])
    assert {"cyankiwi/Qwen3.5-9B-AWQ-4bit", "Qwen/Qwen3.5-27B"} <= {
        each["base"] for each in shown.values() if each["sample"]
    }
    sample_marks = {"scout", "crafter", "merged", "planner"}
    assert sample_marks <= set(graph["bookmarks"]) and graph["bookmarks"]["best"] == "minerthree"
    assert all(mark in shown[checkpoint]["bookmarks"] for mark, checkpoint in graph["bookmarks"].items())


async def test_the_page_asks_for_the_graph_with_or_without_the_sample(tmp_path: Path) -> None:
    pytest.importorskip("starlette")
    from rollout_train.monitor.app import create_app

    transport = httpx.ASGITransport(app=create_app(tmp_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://monitor") as client:
        plain = (await client.get("/api/checkpoints")).json()
        sampled = (await client.get("/api/checkpoints?sample=1")).json()
    assert plain["checkpoints"] == [] and plain["bases"] == [] and not plain["sample"]
    assert sampled["sample"] and sampled["checkpoints"] and all(each["sample"] for each in sampled["checkpoints"])
    assert not (tmp_path / "ledger").exists()  # (a reader makes no ledger where none is)
