"""The checkpoints as a graph: growing from base models, forks and merges, the runs' trainers, their engines and
evaluations."""

from pathlib import Path

import httpx
import pytest
from pydantic import JsonValue

from rollout_train.monitor.lineage import (
    MADE,
    RESHARDING,
    SERVING,
    SUPERSEDED,
    TAKING,
    WRITTEN,
    lineage,
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
    """A run's line of checkpoints, a run forked from it, a merge of both, a second line from the same base, and a
    suite two subjects played."""
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
            "1": step("diggerone", "minertwo", 1, 350.0),
            "2": step("diggertwo", "diggerone", 2, 450.0),
            "3": step("diggerthree", "diggertwo", 3, 950.0),
        },
        "runs/fresh/steps": {"1": step("freshone", None, 1, 750.0)},
        "checkpoints/resharding": {"bothone": {"at": 710.0}},
        "runs/merge/steps": {"1": step("bothone", "minerthree", 1, 650.0)},
        "evaluations/suite/suite": {
            "suite": {"entries": [{"starts": [{"task": "t1", "seed": 1}, {"task": "t2", "seed": 1}]}]}
        },
        "evaluations/suite/minerthree/subject": {
            "subject": {"kind": "checkpoint", "checkpoint": "minerthree", "version": "suite@1"}
        },
        "evaluations/suite/minerthree/results": {
            "1-1": {"solved": True, "reward": 1.0},
            "2-1": {"solved": False, "reward": 0.2},
        },
        "evaluations/suite/model.big/subject": {"subject": {"kind": "model", "model": "big", "version": "suite@1"}},
        "evaluations/suite/model.big/results": {"1-1": {"solved": True, "reward": 1.2}},
    }


NAMES = {"runs": {"train": "miner", "dig": "digger"}, "bookmarks": {"best": "minerthree", "dig": "diggertwo"}}
NOTES = [
    {"kind": "published", "run": "dig", "channel": "policy", "adapter": "diggerone", "at": 410.0},
    {"kind": "published", "run": "dig", "channel": "policy", "adapter": "diggertwo", "at": 510.0},
]


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
    assert shown["diggertwo"]["by"] == {"run": "dig", "name": "digger", "step": 2}
    assert shown["diggertwo"]["depth"] == 4 and shown["diggertwo"]["short"] == "diggert"
    assert shown["minerthree"]["bookmarks"] == ["best"] and shown["diggertwo"]["bookmarks"] == ["dig"]
    assert not shown["minerone"]["kept"] and shown["minerone"]["released"] == 900.0
    assert graph["bookmarks"] == NAMES["bookmarks"] and graph["outside"] == []


def test_the_graph_has_each_runs_checkpoints_and_what_trains_serves_and_evaluates_them() -> None:
    graph = lineage(tables(), NOTES, names=NAMES, now=NOW)
    runs = {run["run"]: run for run in graph["runs"]}
    assert runs["merge"]["checkpoints"] == ["bothone"] and runs["merge"]["from"] == "minerthree"
    assert runs["dig"]["from"] == "minertwo" and runs["dig"]["name"] == "digger" and runs["train"]["name"] == "miner"
    assert runs["train"]["groups"] == [[40.0, 1], [50.0, 0], [140.0, 1], [150.0, 0]]
    assert runs["dig"]["latest"] == "diggertwo"

    # The way to the engines: what a run published serves until it publishes the next.
    life = {each["id"]: each["life"]["state"] for each in graph["checkpoints"]}
    assert life["diggertwo"] == SERVING and life["diggerone"] == SUPERSEDED
    assert life["bothone"] == RESHARDING and life["minerthree"] == WRITTEN

    trainers = {trainer["trainer"]: trainer for trainer in graph["trainers"]}
    digger = trainers["digger (the run's own)"]
    assert [entry["state"] for entry in digger["queue"]] == [MADE, MADE, TAKING]
    assert digger["runs"] == ["dig"] and digger["base"] == "small"
    assert {worker["worker"]: worker["serving"] for worker in graph["workers"]} == {"dig engines": ["diggertwo"]}

    (suite,) = graph["evaluations"]
    subjects = {subject["subject"]: subject for subject in suite["subjects"]}
    assert (
        subjects["minerthree"]["solved"] == 1
        and subjects["minerthree"]["played"] == 2
        and subjects["minerthree"]["checkpoint"] == "minerthree"
    )
    assert subjects["model.big"]["kind"] == "model" and subjects["model.big"]["checkpoint"] is None


def test_a_runs_steps_stand_for_its_trainer_and_its_published_notes_for_what_it_serves() -> None:
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
    (trainer,) = graph["trainers"]
    assert trainer["trainer"] == "train (the run's own)" and trainer["runs"] == ["train"]
    assert [entry["state"] for entry in trainer["queue"]] == [MADE, MADE, MADE]


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


async def test_the_page_asks_for_the_graph(tmp_path: Path) -> None:
    pytest.importorskip("starlette")
    from rollout_train.monitor.app import create_app

    transport = httpx.ASGITransport(app=create_app(tmp_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://monitor") as client:
        graph = (await client.get("/api/checkpoints")).json()
    assert graph["checkpoints"] == [] and graph["bases"] == []
    assert not (tmp_path / "ledger").exists()  # (a reader makes no ledger where none is)
