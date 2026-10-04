"""Every run of a shared ledger: where each one's episodes are read, and whether it is running."""

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import JsonValue

from rollout_train.layout import FEED, JOBS
from rollout_train.ledger import Ledger
from rollout_train.monitor import RunFeed, System
from rollout_train.monitor.system import GONE, IDLE, RELAYED, RUNNING
from rollout_train.record import GROUPS, RESULTS, STARTS, scope, table
from rollout_train.rollouts.episodes import Episode, Outcome, Record, Trajectory
from rollout_train.rollouts.jobs import EPISODES, TICKETS


async def started(ledger: Ledger, run: str, at: float, **where: JsonValue) -> None:
    """A run that started at `at` and played one group, with its result written then."""
    fence = await ledger.take(scope(run))
    await ledger.append(
        table(run, STARTS), str(fence.number), {"policy": run, "host": "here", "started": at, **where}, fence
    )
    await ledger.append(table(run, GROUPS), "1", {"task": "t1", "title": "one", "decided": at}, fence)
    result: JsonValue = {"time": at + 1, "rewards": [1.0], "solved": [True], "segments": 0, "skipped": "alone"}
    await ledger.append(table(run, RESULTS), "1", result, fence)


def played(directory: Path, run: str, run_id: str, at: float) -> None:
    """A run's directory with its job's log (one group asked for, one episode ended) and a feed that measured its
    engines once; every file written at `at`."""
    log = directory / JOBS / run
    log.mkdir(parents=True)
    (log / TICKETS).write_text(
        json.dumps({"id": f"t_{run}", "parameters": None, "labels": {"group": "1"}, "count": 1}) + "\n"
    )
    trajectories = {"ada": Trajectory([], {"default": 1.0})}
    episode = Episode(1, run, f"t_{run}", run_id, {"episode": "1", "group": "1", "job": run}, Outcome.COMPLETED,
                      trajectories=trajectories)  # fmt: skip
    (log / EPISODES).write_text(json.dumps(Record(episode, sampled={"ada": 10}).to_json()) + "\n")
    feed = RunFeed(directory / FEED)
    feed.on_job(
        {"kind": "inference", "at": at, "channel": "policy", "tokens_per_second": 50.0, "mean_concurrency": 2.0}
    )
    feed.close()
    for path in [*log.iterdir(), *(directory / FEED).iterdir()]:
        os.utime(path, (at, at))


async def test_one_monitor_shows_every_run_of_a_database_each_from_its_own_directory(tmp_path: Path) -> None:
    pytest.importorskip("sqlalchemy")
    from rollout_train.database import DatabaseLedger

    now = time.time()
    ledger = DatabaseLedger(f"sqlite:///{tmp_path}/ledger.db")
    busy, quiet = tmp_path / "runs" / "busy", tmp_path / "runs" / "quiet"
    await started(ledger, "busy", now - 60, directory=str(busy))
    await started(ledger, "quiet", now - 3600, directory=str(quiet))
    await started(ledger, "gone", now - 5 * 3600, directory="/nowhere/gone")
    played(busy, "busy", "r_busy", now - 30)
    played(quiet, "quiet", "r_quiet", now - 3600)

    system = System(ledger=ledger)
    snapshot = await system.snapshot()
    runs = {run["run"]: run for run in snapshot["runs"]}
    assert [run["run"] for run in snapshot["runs"]] == ["busy", "quiet", "gone"]  # (running ones first)
    assert [runs[name]["state"] for name in ("busy", "quiet", "gone")] == [RUNNING, IDLE, GONE]
    assert runs["busy"]["episodes_at"] == "here" and runs["busy"]["job"]["episodes"] == 1
    assert runs["busy"]["directory"] == str(busy) and runs["busy"]["host"] == "here"
    assert runs["gone"]["episodes_at"] is None and runs["gone"]["job"] is None and runs["gone"]["done"]
    assert snapshot["directory"] is None and snapshot["ledger_at"] == ledger.url
    assert sorted(job["job"] for job in snapshot["jobs"]) == ["busy", "quiet"]

    group = await system.group("quiet", 1)
    assert (
        group is not None
        and group["episodes_at"] == "here"
        and [each["run_id"] for each in group["episodes"]] == ["r_quiet"]
    )
    episode = await system.episode("r_busy")
    assert episode["ended"] is not None and episode["labels"]["job"] == "busy"

    figures = await system.statistics()
    measured = {run["run"]: run["inference"] for run in figures["runs"]}
    assert [channel["channel"] for channel in measured["busy"]] == ["policy"] and measured["gone"] == []
    ledger.close()


class Elsewhere:
    """A monitor on another machine, as httpx's transport: what it serves of its run `far`, and what it was asked."""

    def __init__(self, answering: bool = True) -> None:
        self.answering = answering
        self.asked: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.asked.append(request)
        if not self.answering:
            raise httpx.ConnectError("no route to host", request=request)
        path = request.url.path
        episode: dict[str, Any] = {"run_id": "r_far", "episode": "1", "state": "running", "samples": 3}
        if path == "/api/system":
            far = {"run": "far", "open": [{"number": 1, "stage": "playing", "episodes": [episode]}], "done": [],
                   "job": {"job": "far", "episodes": 0}, "channels": [], "written": time.time()}  # fmt: skip
            return httpx.Response(200, json={"runs": [far]})
        if path == "/api/groups/far/1":
            return httpx.Response(200, json={"number": 1, "run": "far", "stage": "playing", "episodes": [episode]})
        if path == "/api/episodes/r_far":
            return httpx.Response(200, json={"run_id": "r_far", "source": "feed", "lines": [{"kind": "sample"}]})
        if path == "/api/runs":
            return httpx.Response(200, json=[{"run_id": "r_far", "labels": {}, "started": 1.0}])
        return httpx.Response(404, json={"error": "no such thing"})


async def test_a_run_on_another_machine_has_its_episodes_from_the_monitor_its_start_names(tmp_path: Path) -> None:
    from rollout_train.ledger import FileLedger

    ledger = FileLedger(tmp_path / "ledger")
    await started(ledger, "far", time.time() - 3600, directory="/on/runner-2/far", address="http://runner-2:8765/")
    elsewhere = Elsewhere()
    system = System(ledger=ledger, client=httpx.Client(transport=httpx.MockTransport(elsewhere)))

    (run,) = (await system.snapshot())["runs"]
    assert run["episodes_at"] == "http://runner-2:8765" and run["reached"] and run["state"] == RUNNING
    assert run["open"][0]["episodes"][0]["run_id"] == "r_far" and run["job"]["job"] == "far"
    group = await system.group("far", 1)
    assert group is not None and group["stage"] == "playing" and group["episodes_at"] == "http://runner-2:8765"
    assert (await system.episode("r_far"))["source"] == "feed"
    assert [each["run_id"] for each in system.feeds()] == ["r_far"]
    assert elsewhere.asked and all(request.headers.get(RELAYED) == "1" for request in elsewhere.asked)

    # Asked by another monitor, it answers from its own machine: it asks nobody.
    count = len(elsewhere.asked)
    (relayed,) = (await system.snapshot(relayed=True))["runs"]
    assert relayed["episodes_at"] is None and relayed["open"] == [] and len(elsewhere.asked) == count
    assert (await system.episode("r_far", relayed=True))["source"] is None


async def test_a_monitor_elsewhere_that_does_not_answer_leaves_what_the_ledger_has(tmp_path: Path) -> None:
    from rollout_train.ledger import FileLedger

    ledger = FileLedger(tmp_path / "ledger")
    await started(ledger, "far", time.time() - 5 * 3600, address="http://runner-2:8765")
    elsewhere = Elsewhere(answering=False)
    system = System(ledger=ledger, client=httpx.Client(transport=httpx.MockTransport(elsewhere)))
    (run,) = (await system.snapshot())["runs"]
    assert run["reached"] is False and run["state"] == GONE and run["done"] and run["open"] == []
    group = await system.group("far", 1)
    assert group is not None and group["stage"] == "done" and group["episodes"] == []
    asked = len(elsewhere.asked)
    await system.snapshot()
    assert len(elsewhere.asked) == asked  # (it is left a while before it is asked again)


def test_the_monitor_opens_a_database_a_ledgers_directory_or_a_runs_directory(tmp_path: Path) -> None:
    pytest.importorskip("starlette")
    pytest.importorskip("sqlalchemy")
    from rollout_train.ledger import FENCES
    from rollout_train.monitor.app import watched

    database = watched(f"sqlite:///{tmp_path}/shared.db")
    assert database.directory is None and database.ledger == f"sqlite:///{tmp_path}/shared.db"
    (tmp_path / "files").mkdir()
    (tmp_path / "files" / FENCES).write_text("{}")
    files = watched(tmp_path / "files")
    assert files.directory is None and files.ledger == str(tmp_path / "files")
    run = watched(tmp_path / "run")
    assert run.directory == (tmp_path / "run").resolve() and run.ledger == str((tmp_path / "run" / "ledger").resolve())
