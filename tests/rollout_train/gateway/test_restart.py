"""A runner killed in the middle of an episode, recording through the gateway in its own process: started again, it
adopts the run, the run asks again for the sample whose reply it never heard and gets the recorded turn back, nothing is
sampled twice, and the episode trains on the same segments as one played in a single process."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from rollout.harness import ModelBinding, RecordedModel, RunBinding, agent_program
from rollout.harness.blobs import FileBlobStore
from rollout.local import LocalRunner
from rollout_train.gateway import TurnStore
from rollout_train.ledger import FileLedger
from rollout_train.record import GROUPS, scope, table
from rollout_train.rollouts import Episode, EpisodeRunner, Plan, episodes_of, plan, playing
from rollout_train.rollouts.scheduler import ADOPTED, CLAIMS, INTERRUPTED
from rollout_train.testing import recording
from tests.rollout_train.gateway.restart_child import STEPS, STUCK_AFTER, Walk
from tests.rollout_train.gateway.support import echo_channel

pytest.importorskip("rollout_durable")

ROOT = Path(__file__).resolve().parents[3]
BINDING = RunBinding(models={"policy": ModelBinding(recorded=RecordedModel(channel="policy"))})


def child(mode: str, directory: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", "tests.rollout_train.gateway.restart_child", mode, str(directory)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


async def ask(ledger: FileLedger) -> None:
    """A run that asks for one episode of the walk."""
    fence = await ledger.take(scope("train"))
    await plan(ledger, "train", Plan(agent_program(Walk), BINDING), fence)
    await ledger.append(table("train", GROUPS), "1", {"parameters": {}, "episodes": 1, "decided": 1.0}, fence)


def generations(directory: Path) -> list[list[int]]:
    path = directory / "generations.jsonl"
    return [json.loads(line)["prompt"] for line in path.read_text().splitlines()] if path.exists() else []


async def in_one_process(directory: Path) -> Episode:
    """The same episode, played by a runner that is never stopped."""
    ledger, blobs = FileLedger(directory / "ledger"), FileBlobStore(directory / "blobs")
    await ask(ledger)
    endpoints = recording(echo_channel(), ledger=ledger, blobs=blobs)
    async with playing(
        EpisodeRunner("alone", ledger, LocalRunner(recorder=endpoints), endpoints, blobs, 1, every=0.02)
    ):
        (episode,) = await episodes_of(ledger, blobs, "train", 1, 1, every=0.02)
    return episode


def trained(episode: Episode) -> list[Any]:
    """What an episode trains on: each segment's tokens, logprobs and sampled spans (by place and version)."""
    return [
        (segment.tokens, segment.logprobs, [(span.start, span.end, span.version) for span in segment.spans])
        for segment in episode.trajectories["policy"].segments
    ]


def test_a_runner_killed_mid_episode_adopts_it_and_nothing_is_sampled_twice(tmp_path: Path) -> None:
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    asyncio.run(ask(ledger))
    first = child("start", tmp_path)
    deadline = time.monotonic() + 90
    while not (tmp_path / "recorded").exists():  # the third turn is recorded; its reply never reaches the run
        assert time.monotonic() < deadline, "the first runner never recorded its third turn"
        assert first.poll() is None, first.stdout.read() if first.stdout else ""
        time.sleep(0.05)
    first.send_signal(signal.SIGKILL)
    first.wait()
    claims: Any = asyncio.run(ledger.read(table("train", CLAIMS)))
    run_id = str(claims["1/1/1"]["run_id"])
    store = TurnStore(ledger, blobs)
    assert len(asyncio.run(store.turns("train", run_id))) == len(generations(tmp_path)) == STUCK_AFTER

    second = child("resume", tmp_path)
    output, _ = second.communicate(timeout=180)
    assert second.returncode == 0 and "ENDED" in output, output

    (episode,) = asyncio.run(episodes_of(ledger, blobs, "train", 1, 1))
    assert list(asyncio.run(ledger.read(table("train", CLAIMS)))) == ["1/1/1"]  # no new attempt: it was adopted
    assert len(asyncio.run(ledger.read(table("train", ADOPTED)))) == 1
    assert asyncio.run(ledger.read(table("train", INTERRUPTED))) == {}
    assert episode.run_id == run_id and episode.reward == 1.0 and episode.excluded is None and episode.trainable
    sampled = generations(tmp_path)
    assert len(sampled) == STEPS and len({tuple(prompt) for prompt in sampled}) == STEPS  # each turn sampled once
    turns = asyncio.run(store.turns("train", run_id))
    assert len(turns) == STEPS and len({turn.effect_id for turn in turns}) == STEPS  # and recorded once
    assert trained(episode) == trained(asyncio.run(in_one_process(tmp_path / "alone")))
