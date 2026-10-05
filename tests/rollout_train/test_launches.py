"""Runs asked for from anywhere: a launch records what was asked and the run it is, and moves only as its states
allow and its writer expects; a launch stored when launches named profiles reads as its run settings."""

import asyncio
import json
from pathlib import Path

import pytest

from rollout_train.database import DatabaseLedger
from rollout_train.launches import (
    ASKED,
    ENDED,
    EVAL,
    FAILED,
    MOVES,
    OPEN,
    RUNNING,
    STOPPED,
    STOPPING,
    SUBMITTED,
    TRAIN,
    Asked,
    as_launch,
    launch_of,
    launches_of,
)
from rollout_train.ledger import FileLedger, Ledger


def ledgers(tmp_path: Path) -> list[Ledger]:
    return [FileLedger(tmp_path / "files"), DatabaseLedger(f"sqlite:///{tmp_path / 'ledger.db'}")]


@pytest.mark.parametrize("kind", [0, 1], ids=["files", "database"])
async def test_a_launch_records_what_was_asked_and_the_job_it_became(tmp_path: Path, kind: int) -> None:
    launches = launches_of(ledgers(tmp_path)[kind])
    assert launches is not None and await launches.all() == []
    asked = Asked(TRAIN, "diamonds", {"environment": "minecraft_team.environment:environment"}, "one-gpu@2")
    first = await launches.ask(asked, "run_1")
    second = await launches.ask(Asked(EVAL, "later", {"eval.suite": "math"}), "run_2")
    assert first.state == ASKED and first.id.startswith("launch_") and first.asked == asked and first.run == "run_1"
    assert first.asked.environment == "minecraft_team.environment:environment" and second.asked.environment is None
    assert [each.id for each in await launches.all()] == [second.id, first.id]  # (newest first)
    submitted = await launches.note(first.id, expect=(ASKED,), state=SUBMITTED, job="run-1", backend="ray")
    assert (submitted.state, submitted.job, submitted.backend, submitted.run) == (SUBMITTED, "run-1", "ray", "run_1")
    assert await launch_of(launches, first.id) == submitted
    with pytest.raises(KeyError):
        await launch_of(launches, "launch_none")


@pytest.mark.parametrize("kind", [0, 1], ids=["files", "database"])
async def test_a_launch_moves_only_as_its_states_allow_and_as_its_writer_expects(tmp_path: Path, kind: int) -> None:
    launches = launches_of(ledgers(tmp_path)[kind])
    assert launches is not None
    asked = await launches.ask(Asked(TRAIN, "run"), "r")
    assert (await launches.note(asked.id, state=ENDED)).state == ASKED  # (nothing started: not ended)
    assert (await launches.note(asked.id, state=SUBMITTED, job="run-x")).state == SUBMITTED
    stopping = await launches.note(asked.id, expect=(SUBMITTED, RUNNING), state=STOPPING)  # a stop, while it starts
    assert stopping.state == STOPPING
    refused = await launches.note(asked.id, expect=(ASKED, SUBMITTED), state=RUNNING)  # its driver's, after it
    assert refused == stopping  # (nothing written)
    assert (await launches.note(asked.id, state=RUNNING)).state == STOPPING  # (nor without an expectation)
    assert (await launches.note(asked.id, expect=OPEN, detail="waits")).detail == "waits"  # (its details, as it is)
    assert (await launches.note(asked.id, state=STOPPED, detail="stopped")).state == STOPPED
    for state in (ASKED, SUBMITTED, RUNNING, STOPPING, ENDED, FAILED):  # a launch that finished goes nowhere
        assert (await launches.note(asked.id, state=state)).state == STOPPED
    with pytest.raises(KeyError):
        await launches.note("launch_none", state=STOPPED)


def test_every_state_a_launch_goes_to_is_written_down() -> None:
    assert set(MOVES) == {ASKED, SUBMITTED, RUNNING, STOPPING} == set(OPEN)  # (the finished ones go nowhere)
    reached = {ASKED} | {state for moves in MOVES.values() for state in moves}
    assert reached == {ASKED, SUBMITTED, RUNNING, STOPPING, ENDED, FAILED, STOPPED}
    assert all(ASKED not in moves for moves in MOVES.values())  # (nothing goes back)
    assert RUNNING not in MOVES[STOPPING] and SUBMITTED not in MOVES[RUNNING]
    assert RUNNING in MOVES[ASKED]  # (a driver that starts before its submitter notes its job)


@pytest.mark.parametrize("kind", [0, 1], ids=["files", "database"])
async def test_concurrent_stops_and_starts_leave_a_launch_stopping_never_running(tmp_path: Path, kind: int) -> None:
    launches = launches_of(ledgers(tmp_path)[kind])
    assert launches is not None
    for _ in range(10):
        asked = await launches.ask(Asked(TRAIN, "run"), "r")
        await launches.note(asked.id, state=SUBMITTED, job="run-x")
        await asyncio.gather(
            launches.note(asked.id, expect=(SUBMITTED, RUNNING), state=STOPPING),
            launches.note(asked.id, expect=(ASKED, SUBMITTED), state=RUNNING),
        )
        now = await launch_of(launches, asked.id)
        assert now.state == STOPPING  # (whichever came first: a stop is never overwritten by RUNNING)


def test_a_launch_stored_when_launches_named_profiles_reads_as_its_run_settings(tmp_path: Path) -> None:
    trained = as_launch({
        "id": "launch_1", "at": 1.0, "state": "claimed", "launcher": "launcher/here", "directory": "/runs/team-8",
        "pid": 12, "job": None, "detail": None, "updated": 2.0,
        "asked": {
            "profile": "one-gpu", "environment": "minecraft_team.environment:environment", "name": "team-8",
            "start": "team-7:31", "bookmark": "best", "groups": 30, "groups_per_step": 4, "seed": 0,
            "settings": {"trainer.learning_rate": 3e-5}, "kind": "run", "suite": None, "episodes": None,
            "model": None, "environments": [], "resumes": None, "directory": None,
        },
    })  # fmt: skip
    assert (trained.state, trained.asked.kind, trained.asked.name, trained.run) == (SUBMITTED, TRAIN, "team-8", None)
    assert trained.asked.settings == {
        "environment": "minecraft_team.environment:environment", "start": "team-7:31", "bookmark": "best",
        "groups": 30, "groups_per_step": 4, "seed": 0, "trainer.learning_rate": 3e-5,
    }  # fmt: skip
    evaluated = as_launch({
        "id": "launch_2", "at": 1.0, "state": "ended",
        "asked": {"profile": "one-gpu", "environment": "a:b", "name": "math on 31", "kind": "eval", "suite": "math@2",
                  "episodes": 2, "start": "team-7:31"},
    })  # fmt: skip
    assert evaluated.asked.kind == EVAL and evaluated.state == ENDED
    assert evaluated.asked.settings == {"environment": "a:b", "start": "team-7:31", "eval.suite": "math@2",
                                        "eval.episodes": 2}  # fmt: skip
    stored = FileLedger(tmp_path / "ledger")
    (tmp_path / "ledger").mkdir()
    (tmp_path / "ledger" / "launches.json").write_text(json.dumps([{
        "id": "launch_3", "at": 1.0, "state": "running", "launcher": "launcher/here",
        "asked": {"profile": "p", "environment": "a:b", "name": "old"},
    }]))  # fmt: skip
    launches = launches_of(stored)
    assert launches is not None
    (read,) = asyncio.run(launches.all())
    assert read.asked.name == "old" and read.asked.settings["environment"] == "a:b" and read.state == RUNNING
