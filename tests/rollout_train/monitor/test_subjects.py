"""Evals by subject: every checkpoint and base model that has had an eval, and each one's history across suites,
versions and environments."""

import time
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import JsonValue

from rollout_train.evals import EVAL, edit_suite, subject_table, suite_entry
from rollout_train.ledger import Ledger
from rollout_train.monitor.scores import BY_HAND, BY_SCHEDULE, CHECKPOINT, MODEL
from rollout_train.record import EVALS, GROUPS, STARTS, scope, table
from rollout_train.rollouts.episodes import Episode, Outcome, Record
from rollout_train.rollouts.scheduler import EPISODES, runner_scope
from tests.rollout_train.monitor.test_scores import ENVIRONMENT, a_line
from tests.rollout_train.rollouts.games import guessing, words

pytest.importorskip("starlette")
from rollout_train.monitor.app import create_app

GUESSING = "tests.rollout_train.rollouts.games:guessing"


async def an_eval_of_two(ledger: Ledger, run: str, subject: str, solved: list[bool], *, by: str, step: int) -> None:
    """An eval of `subject` that played version 2 of words-v1 (two entries, a start each) in a part each, one episode
    each start, asked for by `by`'s schedule at `step`."""
    fence = await ledger.take(scope(run))
    parts = [f"{run}-1", f"{run}-2"]
    begun: JsonValue = {"kind": EVAL, "suite": "words-v1", "version": "words-v1@2", "checkpoint": subject,
                        "started": time.time(), "parts": cast(JsonValue, parts), "by": by, "step": step}  # fmt: skip
    await ledger.append(table(run, STARTS), str(fence.number), begun, fence)
    who: JsonValue = {"kind": "checkpoint", "checkpoint": subject, "model": "tiny", "episodes": 1,
                      "asked_by": "by its run's schedule", "version": "words-v1@2",
                      "parts": [{"environment": environment, "run": part, "episodes": 1}
                                for environment, part in zip([ENVIRONMENT, GUESSING], parts, strict=True)]}  # fmt: skip
    await ledger.append(subject_table("words-v1", run, "subject"), "subject", who, fence)
    for number, (part, outcome) in enumerate(zip(parts, solved, strict=True), start=1):
        its = await ledger.take(scope(part))
        own: JsonValue = {"kind": EVAL, "suite": "words-v1", "part_of": run, "entry": number, "started": time.time()}
        await ledger.append(table(part, STARTS), str(its.number), own, its)
        await ledger.append(table(part, GROUPS), "1", {"task": "a-row", "episodes": 1}, its)
        result: JsonValue = {"reward": 1.0 if outcome else 0.0, "solved": outcome, "time": time.time()}
        await ledger.append(subject_table("words-v1", run, "results"), f"{number}-1", result, fence)
        episode = Episode(part, 1, 1, f"r_{part}_1", {}, Outcome.COMPLETED, info={"solved": outcome})
        record: JsonValue = Record(episode).to_json()
        await ledger.append(table(part, EPISODES), "1/1", record, await ledger.take(runner_scope(part)))
    trained = await ledger.take(scope(by))
    said: JsonValue = {"suite": "words-v1", "checkpoint": subject, "run": run, "played": len(solved)}
    await ledger.append(table(by, EVALS), str(step), said, trained)


async def read(tmp_path: Path, *paths: str) -> list[httpx.Response]:
    transport = httpx.ASGITransport(app=create_app(str(tmp_path / "ledger"), beat=0.0))
    async with httpx.AsyncClient(transport=transport, base_url="http://monitor") as client:
        return [await client.get(path) for path in paths]


async def test_every_subject_with_an_eval_is_listed_the_one_evaluated_last_first(tmp_path: Path) -> None:
    _, made = await a_line(tmp_path)
    (answer,) = await read(tmp_path, "/api/evals/subjects")
    subjects: list[dict[str, Any]] = answer.json()["subjects"]
    assert [(each["kind"], each["id"]) for each in subjects] == [
        (CHECKPOINT, made["elsewhere"].id),
        (CHECKPOINT, made["stacked"].id),
        (CHECKPOINT, made["second"].id),
        (CHECKPOINT, made["first"].id),
        (MODEL, "tiny"),
    ]  # (the merge had no eval: not a subject)
    second = subjects[2]
    assert second["evals"] == ["eval-second-again", "eval-second"]  # (newest first)
    assert (second["run"], second["name"], second["step"], second["suites"], second["playing"]) == (
        "lora-run", "lora-run", 2, ["words-v1"], 0,
    )  # fmt: skip
    assert second["short"] == made["second"].id[: len(second["short"])] and second["base"] == "tiny"
    model = subjects[-1]
    assert (model["short"], model["run"], model["step"], model["bookmarks"], model["evals"]) == (
        "tiny", None, None, [], ["eval-base"],
    )  # fmt: skip


async def test_a_checkpoints_history_is_every_eval_it_had_across_versions_and_environments(tmp_path: Path) -> None:
    ledger, made = await a_line(tmp_path)
    second = made["second"]
    two = [suite_entry(ENVIRONMENT, words, rows=["say-yes"], seeds=[1]),
           suite_entry(GUESSING, guessing, rows=["guess-apple"], seeds=[7])]  # fmt: skip
    await edit_suite(ledger, "words-v1", two)
    await an_eval_of_two(ledger, "eval-second-v2", second.id, [True, False], by="lora-run", step=3)
    by_id, by_start, nothing = await read(
        tmp_path,
        f"/api/evals/checkpoint/{second.id}",
        f"/api/evals/checkpoint/{second.id[:6]}",
        "/api/evals/checkpoint/nothing",
    )
    history: dict[str, Any] = by_id.json()
    assert by_start.json() == history and nothing.status_code == 404
    assert history["subject"]["id"] == second.id and history["subject"]["kind"] == CHECKPOINT
    evals = history["evals"]
    assert [each["run"] for each in evals] == ["eval-second-v2", "eval-second-again", "eval-second"]
    newest = evals[0]
    assert (newest["version"], newest["asked_by"], newest["by"], newest["step"]) == ("words-v1@2", BY_SCHEDULE,
                                                                                     "lora-run", 3)  # fmt: skip
    assert [entry["environment"] for entry in newest["entries"]] == [ENVIRONMENT, GUESSING]
    assert [(entry["played"], entry["share"], entry["reward"]) for entry in newest["entries"]] == [
        (1, 1.0, 1.0), (1, 0.0, 0.0),
    ]  # fmt: skip
    older = evals[1]
    assert (older["version"], older["asked_by"], older["share"], older["reward"]) == ("words-v1@1", BY_HAND, 0.5, 0.5)
    assert all("rewards" not in each and all("rewards" not in entry for entry in each["entries"]) for each in evals)


async def test_a_base_models_history_is_the_evals_of_it_by_name(tmp_path: Path) -> None:
    await a_line(tmp_path)
    model, unknown = await read(tmp_path, "/api/evals/model/tiny", "/api/evals/model/someone/else")
    history: dict[str, Any] = model.json()
    assert history["subject"] == {**history["subject"], "kind": MODEL, "id": "tiny", "short": "tiny", "run": None}
    ((only),) = history["evals"]
    assert (only["run"], only["kind"], only["model"], only["checkpoint"], only["share"]) == (
        "eval-base", MODEL, "tiny", None, 0.0,
    )  # fmt: skip
    assert unknown.status_code == 404


async def test_a_checkpoint_or_base_model_with_no_eval_has_an_empty_history(tmp_path: Path) -> None:
    _, made = await a_line(tmp_path)
    (merged,) = await read(tmp_path, f"/api/evals/checkpoint/{made['merged'].id}")
    history: dict[str, Any] = merged.json()
    assert history["evals"] == [] and history["subject"]["bookmarks"] == ["merged"]
    assert history["subject"]["started"] is None and history["subject"]["kind"] == CHECKPOINT
