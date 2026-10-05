"""A run's job exits as its run ended: a run that fails fails its job, and so does every job its RayJob submits again
(`backoffLimit`) once the launch has failed, so that KubeRay marks the RayJob failed; a launch that ended or stopped
exits 0."""

import asyncio
from pathlib import Path

import pytest

from rollout_train import jobs
from rollout_train.launches import ASKED, ENDED, FAILED, RUNNING, STOPPED, launch_of, launches_of
from rollout_train.run_settings import RunSettings
from rollout_train.stores import Stores
from rollout_train.submitting import ask
from tests.rollout_train.clusters import POLICY, WORDS, a_cluster


def a_launch(root: Path) -> tuple[Path, str]:
    """A cluster config under `root`, and a launch asked for on it."""
    cluster = a_cluster(root)
    settings = RunSettings({**POLICY, "kind": "train", "name": "failing", "environment": WORDS})
    launch = asyncio.run(ask(settings, Stores.open(cluster).ledger))
    return root / "cluster.toml", launch.id


def connected(address: str) -> None:
    """In place of joining a Ray cluster: these jobs end before they ask Ray for anything."""


def state_of(root: Path, launch: str) -> str:
    launches = launches_of(Stores.open(a_cluster(root)).ledger)
    assert launches is not None
    return asyncio.run(launch_of(launches, launch)).state


def test_a_failed_run_fails_its_job_and_every_job_submitted_again_after_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, launch = a_launch(tmp_path)
    monkeypatch.setattr("rollout_train.ray_cluster.connect", connected)

    async def failing(run: jobs.Run) -> None:
        raise RuntimeError("run/RUN/engine/policy/0 did not start: vLLM refused its options")

    monkeypatch.setattr(jobs, "ran", failing)
    with pytest.raises(RuntimeError, match="did not start"):  # (the process exits 1, with the traceback)
        jobs.main([launch, "--cluster", str(path)])
    assert state_of(tmp_path, launch) == FAILED
    with pytest.raises(SystemExit) as again:  # (KubeRay's next attempt: the launch has failed, so the job fails)
        jobs.main([launch, "--cluster", str(path)])
    assert again.value.code == 1


@pytest.mark.parametrize("finished", [ENDED, STOPPED])
def test_a_job_submitted_again_after_its_run_ended_or_stopped_exits_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finished: str
) -> None:
    path, launch = a_launch(tmp_path)
    monkeypatch.setattr("rollout_train.ray_cluster.connect", connected)
    launches = launches_of(Stores.open(a_cluster(tmp_path)).ledger)
    assert launches is not None
    asyncio.run(launches.note(launch, expect=(ASKED,), state=RUNNING))
    asyncio.run(launches.note(launch, expect=(RUNNING,), state=finished))
    with pytest.raises(SystemExit) as exited:
        jobs.main([launch, "--cluster", str(path)])
    assert exited.value.code == 0 and state_of(tmp_path, launch) == finished
