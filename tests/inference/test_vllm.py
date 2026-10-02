"""Engine processes a dead driver left behind are found and ended (no engine is started here)."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from rollout.inference.vllm import end_orphaned_engines, engine_processes, note_engines

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


def test_an_engine_a_dead_driver_left_is_ended_and_nothing_else_is(tmp_path: Path) -> None:
    engine, other = named("VLLM::EngineCore"), named("something-else")
    record = tmp_path / "engine.json"
    try:
        assert engine_processes() == [engine.pid]  # this process's engine cores, by name
        note_engines(record)
        assert json.loads(record.read_text()) == {"owner": os.getpid(), "engines": [engine.pid]}
        assert end_orphaned_engines(record) == []  # its owner (this process) is alive: it is no orphan

        record.write_text(json.dumps({"owner": gone(), "engines": [engine.pid, other.pid]}))
        assert end_orphaned_engines(record) == [engine.pid]  # the one named as an engine, and only that one
        assert engine.wait(timeout=5) == -signal.SIGKILL
        time.sleep(0.1)
        assert other.poll() is None
        assert end_orphaned_engines(record) == [] and end_orphaned_engines(tmp_path / "none.json") == []
    finally:
        for process in (engine, other):
            process.kill()
            process.wait()
