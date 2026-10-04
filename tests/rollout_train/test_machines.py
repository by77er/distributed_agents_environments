"""A run on three machines in one process, each opened from a profile of its own, sharing a scratch ledger and blob
store: a trainer whose channel's engines serve elsewhere, an engine host that follows what the run says its channel
should serve and serves it over HTTP, and a runner that routes its sessions there and records them."""

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Coroutine
from pathlib import Path
from typing import Any

import pytest

from rollout.environment import binding_for
from rollout_train import train
from rollout_train.hosting import host_engines, run_episodes
from rollout_train.inference import remote
from rollout_train.profile import Profile
from rollout_train.rollouts import Record, loaded
from rollout_train.rollouts.scheduler import EPISODES
from rollout_train.serving import SERVING, wanted
from tests.rollout_train.machines import free_port, yes

SHARED = """
ledger = "{root}/ledger"

[blobs]
kind = "rollout.harness.blobs:FileBlobStore"
directory = "{root}/blobs"
"""

ROUTED = """
[channels.policy]
model = "a-checkpoint"
renderer = "rollout_train.testing:plain_renderer"
engine = "rollout_train.inference:RemoteEngine"   # its replicas serve on the engine host
replicas = "heartbeats"
"""

TRAINER = (
    """
directory = "{root}/trainer"
episodes_at_once = 0                              # its own runner plays nothing: the runner machine's does
"""
    + SHARED
    + ROUTED
    + """
[trainer]
kind = "tests.rollout_train.test_profile:Steps"
channel = "policy"
segment_tokens = 900
segments_per_step = 3
"""
)

ENGINES = (
    """
directory = "{root}/gpu-1"
"""
    + SHARED
    + """
[channels.policy]
model = "a-checkpoint"
renderer = "rollout_train.testing:plain_renderer"
engine = "tests.rollout_train.machines:saying_engine"
"""
)

RUNNER = (
    """
directory = "{root}/runner"
episodes_at_once = 4
"""
    + SHARED
    + ROUTED
)


def profile(tmp_path: Path, text: str, name: str) -> Profile:
    path = tmp_path / f"{name}.toml"
    path.write_text(text.format(root=tmp_path))
    return Profile.load(path)


@contextlib.asynccontextmanager
async def going(work: Coroutine[Any, Any, None]) -> AsyncGenerator[asyncio.Task[None]]:
    task = asyncio.create_task(work)
    try:
        yield task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_a_run_trains_on_one_machine_serves_on_another_and_plays_on_a_third(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(remote, "EVERY", 0.05)  # (the runner asks what the replicas serve more often than it would)
    async with profile(tmp_path, TRAINER, "trainer").open() as platform:
        assert platform.channels == {} and platform.trainer is not None  # (no engine here)
        run = platform.run.id
        engines = host_engines(profile(tmp_path, ENGINES, "engines"), run, port=free_port(), name="gpu-1", every=0.1)
        async with going(engines), going(run_episodes(profile(tmp_path, RUNNER, "runner"), [platform.run.name])):
            await asyncio.wait_for(
                train(
                    yes, platform.trainer, platform.checkpoints, base="a-checkpoint", channel="policy",
                    directory=tmp_path / "trainer" / "checkpoints", publish=platform.publish, run=run, groups=6,
                    groups_per_step=1, episodes_at_once=4, binding=binding_for(yes, "policy", {}),
                ),
                60,
            )  # fmt: skip
        made = {checkpoint.id: checkpoint for checkpoint in await platform.checkpoints.all()}
        ledger, blobs = platform.ledger, platform.blobs
        said = await wanted(ledger, run, "policy")
        newest = max(made.values(), key=lambda checkpoint: checkpoint.depth)
        assert said is not None and (said.checkpoint, said.depth, said.sequence) == (newest.id, newest.depth, 900)
        assert len(await ledger.read(f"runs/{run}/{SERVING}")) == len(made) + 1  # (the base model, then each made)
        stamped: set[tuple[str, int]] = set()
        for line in (await ledger.read(f"runs/{run}/{EPISODES}")).values():
            episode = await loaded(Record.from_json(line), blobs)  # type: ignore[arg-type]
            for segment in episode.trajectories["policy"].segments:
                text = "".join(map(chr, segment.tokens))
                for span in segment.spans:
                    sampled_by = text[span.start : span.end].split()[0]  # (the engine says which weights sampled it)
                    assert span.version == (made[sampled_by].depth if sampled_by in made else 0), sampled_by
                    stamped.add((sampled_by, span.version))
        assert ("base", 0) in stamped and len(stamped) > 1  # (later groups were played by checkpoints the run made)
