"""A run survives `kill -9`: it resumes in a new process with no duplicated model calls or events."""

import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from rollout.contracts import RunEventType
from rollout_durable.store import RunStore

CHILD = Path(__file__).with_name("crash_child.py")


def child(mode: str, directory: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, str(CHILD), mode, str(directory)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def ledger(directory: Path) -> list[str]:
    path = directory / "ledger.jsonl"
    return [json.loads(line)["effect_id"] for line in path.read_text().splitlines()] if path.exists() else []


def test_a_run_resumes_after_the_process_is_killed(tmp_path: Path) -> None:
    first = child("start", tmp_path)
    deadline = time.monotonic() + 60
    while len(ledger(tmp_path)) < 3:  # the third model call is in flight
        assert time.monotonic() < deadline, "the first process never reached its third model call"
        assert first.poll() is None, first.stdout.read() if first.stdout else ""
        time.sleep(0.1)
    os.kill(first.pid, signal.SIGKILL)
    first.wait()
    calls_before_crash = ledger(tmp_path)

    second = child("resume", tmp_path)
    output, _ = second.communicate(timeout=120)
    assert second.returncode == 0, output
    outcome = json.loads(output.split("OUTCOME ", 1)[1].splitlines()[0])
    assert outcome["status"] == "completed"

    calls = Counter(ledger(tmp_path))
    assert len(calls) == 5, calls  # one model effect per turn
    duplicated = [effect for effect, count in calls.items() if count > 1]
    assert duplicated in ([], [calls_before_crash[-1]]), calls  # only the call in flight at the crash may repeat

    store = RunStore(tmp_path / "state" / "runs.sqlite")
    events = store.events("r_crashtest")
    assert [event.seq for event in events] == list(range(len(events)))  # gapless, no duplicates
    assert sum(event.type is RunEventType.RUN_CREATED for event in events) == 1
    replies = [event.payload["payload"] for event in events if event.type is RunEventType.OUTPUT_EMITTED]  # type: ignore[index, call-overload]
    assert replies == [f"reply to turn {turn}" for turn in range(1, 6)]
    assert events[-1].type is RunEventType.RUN_COMPLETED
    store.close()


GUARD_CHILD = Path(__file__).with_name("guard_child.py")


def test_a_command_interrupted_by_a_crash_is_not_run_again(tmp_path: Path) -> None:
    def start(mode: str) -> subprocess.Popen[str]:
        return subprocess.Popen(
            [sys.executable, str(GUARD_CHILD), mode, str(tmp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def entries() -> list[dict[str, str]]:
        path = tmp_path / "ledger.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    first = start("start")
    deadline = time.monotonic() + 60
    while not any(entry["operation"] == "execute" for entry in entries()):  # the command is running
        assert time.monotonic() < deadline and first.poll() is None, "the command never started"
        time.sleep(0.1)
    os.kill(first.pid, signal.SIGKILL)
    first.wait()

    second = start("resume")
    output, _ = second.communicate(timeout=120)
    assert second.returncode == 0, output
    assert json.loads(output.split("OUTCOME ", 1)[1].splitlines()[0])["status"] == "completed"

    operations = [entry["operation"] for entry in entries()]
    assert operations.count("create") == 1  # the creation was recorded before the crash
    assert operations.count("execute") == 1  # guarded: never started a second time
    assert operations.count("destroy") == 1  # the run's environment is released when it ends

    store = RunStore(tmp_path / "state" / "runs.sqlite")
    events = store.events("r_guardtest")
    completed = [e.payload for e in events if e.type is RunEventType.EFFECT_COMPLETED]
    assert any(isinstance(p, dict) and p.get("status") == "outcome_unknown" for p in completed)
    results = [e.payload["payload"] for e in events if e.type is RunEventType.OUTPUT_EMITTED]  # type: ignore[index, call-overload]
    assert results == ["outcome unknown"]
    store.close()
