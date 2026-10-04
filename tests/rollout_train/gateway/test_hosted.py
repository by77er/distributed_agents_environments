"""A runner that records through a gateway elsewhere whose own engines serve a channel: one profile, two processes.
`rollout gateway PROFILE` starts the channel's engine (a fake one) in its process; `rollout eval PROFILE SUITE` with the
profile's `[gateway] url` starts none, learns what the channel guarantees from the gateway, and plays the suite there,
the turns recorded by the gateway in the ledger and blob store the two share."""

import asyncio
import contextlib
import os
import socket
import subprocess
import sys
import time
from collections.abc import Generator
from pathlib import Path

import httpx
import pytest

from rollout.contracts import CapabilityContract, ModelEndpointError
from rollout.harness.blobs import FileBlobStore
from rollout_train.evals import make_suite, suite_entry
from rollout_train.gateway import TurnStore
from rollout_train.ledger import FileLedger
from rollout_train.monitor.system import System
from rollout_train.profile import Profile
from tests.rollout_train.gateway.support import SECRETS
from tests.rollout_train.rollouts.games import words

pytest.importorskip("uvicorn")

ROOT = Path(__file__).resolve().parents[3]
ENVIRONMENT = "tests.rollout_train.rollouts.games:words"
PROFILE = """
directory = "{directory}/run"
ledger = "{directory}/ledger"

[blobs]
kind = "rollout.harness.blobs:FileBlobStore"
directory = "{directory}/blobs"

[channels.policy]
model = "echo"
renderer = "rollout_train.testing:plain_renderer"
engine = "tests.rollout_train.gateway.support:echo_engine"
thinking_tokens = 48
answer_tokens = 16

[gateway]
url = "http://127.0.0.1:{port}"
listen = "127.0.0.1:{port}"
keys = "{directory}/keys"
"""


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def a_profile(directory: Path, port: int, text: str = PROFILE) -> Path:
    (directory / "keys").write_text("".join(f"{name} {secret}\n" for name, secret in SECRETS))
    path = directory / "profile.toml"
    path.write_text(text.format(directory=directory, port=port))
    return path


@contextlib.contextmanager
def gateway(profile: Path, port: int) -> Generator[subprocess.Popen[bytes]]:
    """`rollout gateway PROFILE`, in a process of its own, once it is ready."""
    started = subprocess.Popen(
        [sys.executable, "-m", "rollout_train.cli", "gateway", str(profile)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    try:
        deadline = time.monotonic() + 60
        while True:
            with contextlib.suppress(httpx.TransportError):
                if httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=1).status_code == 200:
                    break
            if time.monotonic() > deadline or started.poll() is not None:
                raise RuntimeError("the gateway did not start")
            time.sleep(0.2)
        yield started
    finally:
        started.kill()
        started.wait()


def test_a_runner_plays_a_suite_on_a_channel_the_gateway_elsewhere_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rollout_train.cli import main

    port = free_port()
    profile = a_profile(tmp_path, port)
    ledger = FileLedger(tmp_path / "ledger")
    entry = suite_entry(ENVIRONMENT, words, rows=None, seeds=[1], thinking_tokens=32, answer_tokens=8)
    asyncio.run(make_suite(ledger, "words-v1", [entry]))
    with gateway(profile, port):
        said = httpx.get(f"http://127.0.0.1:{port}/v1/models").json()["data"]
        assert [(each["id"], each["contract"]) for each in said] == [
            ("policy", {"context_limit": 32_768, "max_output_tokens": 64})
        ]  # (the channel's own room: 48 of thinking and 16 of answer)

        async def opened() -> None:
            async with Profile.load(profile).open(training=False) as platform:
                assert platform.channels == {} and platform.gateway is None  # (no engine, no gateway, here)
                assert platform.recorder.contracts == {
                    "policy": CapabilityContract(context_limit=32_768, max_output_tokens=64)
                }

        asyncio.run(opened())
        arguments = [str(profile), "words-v1", "--directory", str(tmp_path / "eval"), "--name", "words-through"]
        monkeypatch.setattr("sys.argv", ["rollout", "eval", *arguments])
        with pytest.raises(SystemExit) as exited:
            main()
        assert exited.value.code == 0
    assert capsys.readouterr().out.startswith(f"words-v1@1 {ENVIRONMENT}: ")
    (listed,) = asyncio.run(System(ledger=ledger).evals())["evals"]
    assert (listed["name"], listed["checkpoint"], listed["played"], listed["done"]) == ("words-through", None, 3, True)

    async def recorded() -> None:  # (by the gateway, in the blob store the two share, as the suite's entry limits it)
        store = TurnStore(ledger, FileBlobStore(tmp_path / "blobs"))
        turns = [
            turn
            for name in await ledger.tables()
            if name.startswith(f"runs/{listed['run']}/turns/")
            for turn in await store.turns(listed["run"], name.rsplit("/", 1)[-1])
        ]
        assert turns and all(turn.checkpoint == "echo" for turn in turns)
        assert all(len(turn.completion) <= 40 for turn in turns)  # (32 of thinking and 8 of answer, at most)

    asyncio.run(recorded())


def test_a_runner_refuses_a_channel_the_gateway_elsewhere_does_not_host(tmp_path: Path) -> None:
    port = free_port()
    profile = a_profile(tmp_path, port)
    other = tmp_path / "other"
    other.mkdir()
    runner = a_profile(other, port, PROFILE.replace("[channels.policy]", "[channels.judge]"))

    async def opened() -> None:
        async with Profile.load(runner).open(training=False):
            pass

    with gateway(profile, port), pytest.raises(ModelEndpointError, match="hosts no channel judge"):
        asyncio.run(opened())


def test_a_channel_the_gateway_elsewhere_hosts_is_not_trained(tmp_path: Path) -> None:
    trained = PROFILE + '\n[trainer]\nkind = "tests.rollout_train.test_profile:Steps"\nchannel = "policy"\n'
    profile = Profile.load(a_profile(tmp_path, free_port(), trained))
    assert profile.hosted == ["policy"]

    async def opened() -> None:
        async with profile.open():
            pass

    with pytest.raises(ValueError, match="hosts samples what its own engines serve, so it cannot be trained"):
        asyncio.run(opened())
