"""Processes a dead driver left behind are found and ended."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from rollout.processes import children, end_orphans, note_processes

NAMED = (
    "import ctypes, sys, time; ctypes.CDLL('libc.so.6').prctl(15, sys.argv[1].encode()); "
    "print('ready', flush=True); time.sleep(60)"
)


def named(name: str) -> subprocess.Popen[bytes]:
    """A process that calls itself `name`, as an engine core calls itself `VLLM::EngineCore`."""
    process = subprocess.Popen([sys.executable, "-c", NAMED, name], stdout=subprocess.PIPE)
    assert process.stdout is not None and process.stdout.readline().strip() == b"ready"
    return process


def gone() -> int:
    """The id of a process that has ended."""
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def test_a_process_a_dead_driver_left_is_ended_and_nothing_else_is(tmp_path: Path) -> None:
    engine, other = named("VLLM::EngineCore"), named("something-else")
    record = tmp_path / "engine.json"
    try:
        assert children("VLLM::Engine") == [engine.pid]  # this process's children, by name
        note_processes(record, [engine.pid])
        assert json.loads(record.read_text()) == {
            "owner": os.getpid(),
            "processes": {str(engine.pid): "VLLM::EngineCor"},
        }
        assert end_orphans(record) == []  # its owner (this process) is alive: it is no orphan

        noted = {str(engine.pid): "VLLM::EngineCor", str(other.pid): "VLLM::EngineCor"}  # (the second id was reused)
        record.write_text(json.dumps({"owner": gone(), "processes": noted}))
        assert end_orphans(record) == [engine.pid]  # the one that is still what was noted, and only that one
        assert engine.wait(timeout=5) == -signal.SIGKILL
        time.sleep(0.1)
        assert other.poll() is None
        assert end_orphans(record) == [] and end_orphans(tmp_path / "none.json") == []
    finally:
        for process in (engine, other):
            process.kill()
            process.wait()
