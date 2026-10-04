"""Evals from a checkpoint's point of view: every eval it had, by hand or by a schedule, and its scores along its line
from the base model, across runs and merges; a run's settings as the page reads and changes them; and a run's state as
its heartbeats say."""

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import JsonValue

from rollout.harness.blobs import FileBlobStore
from rollout_train.checkpoints import Checkpoint, Checkpoints, new_id
from rollout_train.evals import EVAL, make_suite, subject_table
from rollout_train.ledger import FileLedger, Ledger
from rollout_train.monitor.scores import BY_HAND, BY_SCHEDULE, path_of
from rollout_train.monitor.system import GONE, RUNNING, System
from rollout_train.presence import Beat
from rollout_train.record import EVALS, GROUPS, RESULTS, STARTS, STEPS, scope, table
from rollout_train.registry import registry_of
from rollout_train.rollouts.episodes import Episode, Outcome, Record
from rollout_train.rollouts.scheduler import EPISODES, runner_scope
from rollout_train.settings import desired_settings_of
from tests.rollout_train.rollouts.games import words

pytest.importorskip("starlette")
from rollout_train.monitor.app import create_app

CATALOG = "tests.rollout_train.rollouts.games:words"


async def a_checkpoint(
    checkpoints: Checkpoints, directory: Path, run: str | None, step: int | None, parents: list[str], kind: str = "lora"
) -> Checkpoint:
    weights = directory / new_id()
    weights.mkdir(parents=True)
    (weights / "adapter.bin").write_text("weights")
    fence = await checkpoints.ledger.take(scope(run or "merges"))
    base = "tiny"
    return await checkpoints.add(fence, new_id(), weights=weights, run=run, step=step, base=base, parents=parents,
                                 kind=kind)  # fmt: skip


async def an_eval(
    ledger: Ledger,
    run: str,
    subject: str | None,
    solved: list[bool],
    *,
    suite: str = "words-v1",
    by: str | None = None,
    step: int | None = None,
) -> None:
    """An eval that played `suite` (two starts) with `subject` (none: the base model), one episode each start."""
    fence = await ledger.take(scope(run))
    begun: JsonValue = {"kind": EVAL, "suite": suite, "checkpoint": subject, "from": None, "started": time.time()}
    if by is not None:
        begun = {**begun, "by": by, "step": step}
    await ledger.append(table(run, STARTS), str(fence.number), begun, fence)
    who: JsonValue = {"kind": "checkpoint" if subject else "model", "checkpoint": subject, "model": "tiny",
                      "episodes": 1, "asked_by": BY_HAND if by is None else "by its run's schedule"}  # fmt: skip
    await ledger.append(subject_table(suite, run, "subject"), "subject", who, fence)
    for number, outcome in enumerate(solved, start=1):
        await ledger.append(table(run, GROUPS), str(number), {"task": "say-yes", "episodes": 1}, fence)
        result: JsonValue = {"reward": 1.0 if outcome else 0.0, "solved": outcome, "time": time.time()}
        await ledger.append(subject_table(suite, run, "results"), f"{number}-1", result, fence)
        await ledger.append(table(run, RESULTS), str(number), {"time": time.time()}, fence)
        episode = Episode(run, number, 1, f"r_{run}_{number}", {}, Outcome.COMPLETED, info={"solved": outcome})
        record: JsonValue = Record(episode).to_json()
        await ledger.append(table(run, EPISODES), f"{number}/1", record, await ledger.take(runner_scope(run)))
    if by is not None:
        trained = await ledger.take(scope(by))
        said: JsonValue = {"suite": suite, "checkpoint": subject, "run": run, "played": len(solved)}
        await ledger.append(table(by, EVALS), str(step), said, trained)


async def a_line(tmp_path: Path) -> tuple[FileLedger, dict[str, Checkpoint]]:
    """A LoRA run of two steps, a merge of its second checkpoint, and an adapter stacked on the merge; evals of the base
    model, of each checkpoint by hand, and of the first step's by its run's schedule."""
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    checkpoints = Checkpoints(ledger, blobs)
    await make_suite(ledger, "words-v1", CATALOG, words, rows=["say-yes", "say-no"], seeds=[1])
    first = await a_checkpoint(checkpoints, tmp_path, "lora-run", 1, [])
    second = await a_checkpoint(checkpoints, tmp_path, "lora-run", 2, [first.id])
    merged = await a_checkpoint(checkpoints, tmp_path, None, None, [second.id], kind="full")
    stacked = await a_checkpoint(checkpoints, tmp_path, "stacked-run", 1, [merged.id])
    elsewhere = await a_checkpoint(checkpoints, tmp_path, "other-run", 1, [])  # (not on the line)
    await an_eval(ledger, "eval-base", None, [False, False])
    await an_eval(ledger, "eval-first", first.id, [True, False], by="lora-run", step=1)
    await an_eval(ledger, "eval-second", second.id, [True, True])
    await an_eval(ledger, "eval-second-again", second.id, [False, True])
    await an_eval(ledger, "eval-stacked", stacked.id, [True, True], suite="words-v1")
    await an_eval(ledger, "eval-elsewhere", elsewhere.id, [True, True])
    registry = registry_of(ledger)
    assert registry is not None
    await registry.bookmark("merged", merged.id)
    return ledger, {"first": first, "second": second, "merged": merged, "stacked": stacked, "elsewhere": elsewhere}


async def test_a_checkpoint_lists_every_eval_it_had_by_hand_or_by_its_runs_schedule(tmp_path: Path) -> None:
    _, made = await a_line(tmp_path)
    transport = httpx.ASGITransport(app=create_app(str(tmp_path / "ledger"), beat=0.0))
    async with httpx.AsyncClient(transport=transport, base_url="http://monitor") as client:
        second: Any = (await client.get(f"/api/checkpoints/{made['second'].id[:6]}/evals")).json()  # (by its start)
        first: Any = (await client.get(f"/api/checkpoints/{made['first'].id}/evals")).json()
        assert (await client.get("/api/checkpoints/nothing/evals")).status_code == 404
    assert second["checkpoint"] == made["second"].id
    assert [each["run"] for each in second["evals"]] == ["eval-second-again", "eval-second"]  # (newest first)
    again = second["evals"][0]
    assert (again["suite"], again["asked_by"], again["played"], again["expected"], again["episodes"]) == (
        "words-v1", BY_HAND, 2, 2, 1,
    )  # fmt: skip
    assert (again["solved"], again["share"], again["reward"], again["done"]) == (1, 0.5, 0.5, True)
    (scheduled,) = first["evals"]
    assert (scheduled["asked_by"], scheduled["by"], scheduled["step"]) == (BY_SCHEDULE, "lora-run", 1)


async def test_a_checkpoints_line_runs_from_the_base_model_through_merges_with_each_points_scores(
    tmp_path: Path,
) -> None:
    _, made = await a_line(tmp_path)
    transport = httpx.ASGITransport(app=create_app(str(tmp_path / "ledger"), beat=0.0))
    async with httpx.AsyncClient(transport=transport, base_url="http://monitor") as client:
        path: Any = (await client.get(f"/api/checkpoints/{made['stacked'].id}/path")).json()
    points = path["points"]
    assert [point["id"] for point in points] == [None, *(made[name].id for name in ("first", "second", "merged",
                                                                                     "stacked"))]  # fmt: skip
    assert [point["depth"] for point in points] == [0, 1, 2, 3, 4]
    assert [point["kind"] for point in points] == ["model", "lora", "lora", "full", "lora"]
    assert [point["run"] for point in points] == [None, "lora-run", "lora-run", None, "stacked-run"]
    assert points[0]["model"] == "tiny" and points[3]["bookmarks"] == ["merged"]
    scores = [point["scores"].get("words-v1") for point in points]
    assert [score["solved"] if score else None for score in scores] == [0.0, 0.5, 0.75, None, 1.0]
    assert scores[2]["played"] == 4 and sorted(scores[2]["evals"]) == ["eval-second", "eval-second-again"]  # (pooled)
    assert path["suites"] == [{"suite": "words-v1", "catalog": CATALOG}]
    alone = path_of({}, {made["first"].id: made["first"]}, made["first"].id)
    assert alone["points"][0]["short"] == "base" and alone["suites"] == []


async def test_a_runs_settings_are_read_and_changed_from_the_page(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    await make_suite(ledger, "words-v1", CATALOG, words, rows=["say-yes"], seeds=[1])
    fence = await ledger.take(scope("train"))
    changeable: dict[str, JsonValue] = {"groups_per_step": 4, "evals.suite": None, "evals.every": 1,
                                        "evals.episodes": 1, "trainer.learning_rate": 5e-5}  # fmt: skip
    recorded: JsonValue = {"fixed": {"model": "tiny", "trainer.rank": 8}, "changeable": changeable}
    begun: JsonValue = {"from": None, "started": time.time(), "settings": recorded}
    await ledger.append(table("train", STARTS), "1", begun, fence)
    for key, rate in (("1", 5e-5), ("2", 5e-5), ("3", 3e-5)):
        step: JsonValue = {
            "groups": [int(key)],
            "makes": new_id(),
            "settings": {**changeable, "trainer.learning_rate": rate},
        }
        await ledger.append(table("train", STEPS), key, step, fence)
    transport = httpx.ASGITransport(app=create_app(str(tmp_path / "ledger"), beat=0.0))
    async with httpx.AsyncClient(transport=transport, base_url="http://monitor") as client:
        said: Any = (await client.get("/api/runs/train/settings")).json()
        assert said["fixed"] == {"model": "tiny", "trainer.rank": 8} and said["changeable"] == changeable
        assert said["now"]["trainer.learning_rate"] == 3e-5 and said["desired"] == {}
        assert said["changes"] == [{"step": 3, "changed": {"trainer.learning_rate": 3e-5}}]
        wanted = {"evals.suite": "words-v1", "evals.every": 2, "trainer.learning_rate": 1e-5}
        answer = await client.post("/api/runs/train/settings", json={"settings": wanted})
        assert answer.status_code == 200, answer.text
        assert (await client.get("/api/runs/train/settings")).json()["desired"] == wanted
        for refused in ({"trainer.rank": 16}, {"model": "big"}, {"evals.every": 0}, {"evals.suite": "no-such-suite"},
                        {"trainer.learning_rate": [1]}):  # fmt: skip
            answer = await client.post("/api/runs/train/settings", json={"settings": refused})
            assert answer.status_code == 409, refused
        assert (await client.post("/api/runs/nothing/settings", json={"settings": {}})).status_code == 404
        assert (await client.post("/api/runs/train/settings", json={"wanted": {}})).status_code == 400
    desired = desired_settings_of(ledger)
    assert desired is not None and (found := await desired.desired("train")) is not None and found.settings == wanted


async def test_a_run_whose_runner_stopped_beating_has_ended_and_serves_nothing(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    now = time.time()
    for run in ("alive", "stopped"):
        fence = await ledger.take(scope(run))
        await ledger.append(table(run, STARTS), "1", {"from": None, "started": now - 60}, fence)
    serving: dict[str, Any] = {"channel": "policy", "adapter": "kpqx", "version": 1, "requests": 2}
    beats = [
        Beat("here/alive", now, {"run": "alive"}, [{"at": now, "channels": [serving]}]),
        Beat("here/stopped", now - 600, {"run": "stopped"}, [{"at": now - 600, "channels": [serving]}]),
    ]
    (tmp_path / "ledger" / "presence.json").write_text(json.dumps([asdict(each) for each in beats]))
    runs = {run["run"]: run for run in (await System(ledger=ledger).snapshot())["runs"]}
    assert runs["alive"]["state"] == RUNNING and runs["alive"]["channels"][0]["adapter"] == "kpqx"
    assert runs["stopped"]["state"] == GONE and runs["stopped"]["channels"] == []  # (though it wrote a minute ago)


async def test_an_eval_that_played_every_start_has_ended_and_a_scheduled_one_is_as_its_run(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    now = time.time()
    fence = await ledger.take(scope("train"))
    await ledger.append(table("train", STARTS), "1", {"from": None, "started": now - 60}, fence)
    await an_eval(ledger, "done", None, [True, False])
    unfinished = await ledger.take(scope("going"))
    begun: JsonValue = {"kind": EVAL, "suite": "words-v1", "by": "train", "step": 1, "started": now - 5 * 3600}
    await ledger.append(table("going", STARTS), "1", begun, unfinished)
    await ledger.append(table("going", GROUPS), "1", {"task": "say-yes", "episodes": 1, "decided": now - 5 * 3600},
                        unfinished)  # fmt: skip
    beats = [Beat("here/train", now, {"run": "train"}, [])]
    (tmp_path / "ledger" / "presence.json").write_text(json.dumps([asdict(each) for each in beats]))
    runs = {run["run"]: run for run in (await System(ledger=ledger).snapshot())["runs"]}
    assert runs["done"]["state"] == GONE  # (it wrote just now, and played every start)
    assert runs["going"]["state"] == RUNNING and runs["going"]["by"] == "train"  # (quiet for hours, its run beats)
