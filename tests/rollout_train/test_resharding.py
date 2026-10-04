"""Resharding a checkpoint into its engines' layout, here and as a Ray task; and a launcher whose runs are Ray jobs."""

import asyncio
import shutil
from pathlib import Path
from typing import Any

import pytest

from rollout.harness.blobs import FileBlobStore
from rollout_train.checkpoints import Checkpoints, Retention
from rollout_train.launcher import OUTPUT, Launcher
from rollout_train.launches import CLAIMED, ENDED, FAILED, RUNNING, STOPPED, STOPPING, Asked, launches_of
from rollout_train.ledger import FileLedger
from rollout_train.presence import presence_of
from rollout_train.ray_cluster import prepare
from rollout_train.record import scope
from rollout_train.resharding import RESHARDED, RESHARDING, VERBATIM, on_ray, reshard, resharded
from tests.rollout_train.test_launches import profiles


async def a_version(tmp_path: Path) -> tuple[Checkpoints, Any, str]:
    """A ledger and blob store with one checkpoint, its weights two files."""
    checkpoints = Checkpoints(FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs"))
    fence = await checkpoints.ledger.take(scope("run"))
    weights = tmp_path / "made" / "weights"
    weights.mkdir(parents=True)
    (weights / "adapter_config.json").write_text('{"r": 8}')
    (weights / "adapter_model.safetensors").write_bytes(b"\x00" * 64)
    made = await checkpoints.add(fence, "kpqxrmtzwvolxqvu", weights=weights, run="run", step=1, base="tiny")
    return checkpoints, fence, made.id


async def test_a_version_is_resharded_once_into_its_engines_layout(tmp_path: Path) -> None:
    checkpoints, fence, checkpoint = await a_version(tmp_path)
    manifest = await reshard(checkpoints, fence, checkpoint, VERBATIM, tmp_path / "scratch")
    assert sorted(manifest.files) == ["adapter_config.json", "adapter_model.safetensors"]
    made = await checkpoints.checkpoint(checkpoint)
    assert made.weights is not None
    assert {name: blob.sha256 for name, blob in manifest.files.items()} == {
        name: blob.sha256 for name, blob in made.weights.files.items()
    }  # (verbatim: the same files, so the same blobs)
    assert (
        list(await checkpoints.ledger.read(RESHARDING))
        == [checkpoint]
        == list(await checkpoints.ledger.read(RESHARDED))
    )
    assert await resharded(checkpoints.ledger, checkpoint) == manifest
    again = await reshard(checkpoints, fence, checkpoint, VERBATIM, tmp_path / "scratch")
    assert again == manifest and len(await checkpoints.ledger.read(RESHARDING)) == 1  # not resharded again
    assert not list((tmp_path / "scratch").iterdir())  # what it wrote there is gone


async def test_a_released_version_cannot_be_resharded(tmp_path: Path) -> None:
    checkpoints, fence, checkpoint = await a_version(tmp_path)
    second = tmp_path / "second"
    second.mkdir()
    (second / "w").write_bytes(b"1")
    later = await checkpoints.add(fence, "zzzzzzzzzzzzzzzz", weights=second, run="run", step=2, parents=[checkpoint])
    await checkpoints.thin(fence, "run", Retention(recent=1, every=0), keep={later.id})
    with pytest.raises(ValueError, match="weights were deleted"):
        await reshard(checkpoints, fence, checkpoint, VERBATIM, tmp_path / "scratch")


async def test_a_reshard_runs_as_a_ray_task(tmp_path: Path) -> None:
    prepare()  # (before Ray is imported: workers run in this environment)
    ray = pytest.importorskip("ray")
    checkpoints, fence, checkpoint = await a_version(tmp_path)
    sessions = Path.home() / ".cache" / "ray-tests"  # (on disk: /tmp may be memory)
    ray.init(
        num_cpus=1, object_store_memory=80 * 2**20, include_dashboard=False, log_to_driver=False,
        _temp_dir=str(sessions),
    )  # fmt: skip
    try:
        ledger_at = {"directory": str(tmp_path / "ledger")}
        blobs_at = {"kind": "rollout.harness.blobs:FileBlobStore", "directory": str(tmp_path / "blobs")}
        manifest = await on_ray(ledger_at, blobs_at, fence, checkpoint, VERBATIM)
    finally:
        ray.shutdown()
        shutil.rmtree(sessions, ignore_errors=True)
    assert sorted(manifest.files) == ["adapter_config.json", "adapter_model.safetensors"]
    assert await resharded(checkpoints.ledger, checkpoint) == manifest  # (noted by the task, in the ledger)


class Jobs:
    """A stand-in for Ray's job client: each job runs `steps` polls, then ends as `ends`."""

    def __init__(self, ends: str = "SUCCEEDED", steps: int = 2) -> None:
        self.submitted: list[dict[str, Any]] = []
        self.stopped: list[str] = []
        self.ends, self.steps, self.polls = ends, steps, 0

    def submit_job(self, **given: Any) -> str:
        self.submitted.append(given)
        return str(given["submission_id"])

    def get_job_status(self, job: str) -> Any:
        from ray.job_submission import JobStatus

        self.polls += 1
        if job in self.stopped:
            return JobStatus.STOPPED
        return JobStatus.RUNNING if self.polls <= self.steps else JobStatus(self.ends)

    def get_job_logs(self, job: str) -> str:
        return f"the output of {job}\nTraceback: it broke" if self.ends == "FAILED" else f"the output of {job}"

    def stop_job(self, job: str) -> bool:
        self.stopped.append(job)
        return True


async def ray_launcher(tmp_path: Path, jobs: Jobs) -> tuple[Launcher, Any]:
    pytest.importorskip("ray")
    ledger = FileLedger(tmp_path / "ledger")
    launches, heartbeats = launches_of(ledger), presence_of(ledger)
    assert launches is not None and heartbeats is not None
    offered = tmp_path / "profiles" if (tmp_path / "profiles").exists() else profiles(tmp_path)
    found = Launcher(
        "launcher/here", launches, heartbeats, offered, [], tmp_path / "runs",
        ray="http://127.0.0.1:8265", gpus=1.0, every=0.01,
    )  # fmt: skip
    found._client = lambda: jobs  # type: ignore[method-assign]  # (no cluster: the stand-in)
    found._offered = launching_offered(found)  # pyright: ignore[reportPrivateUsage]
    return found, launches


def launching_offered(found: Launcher) -> list[dict[str, Any]]:
    from rollout_train.launcher import offered

    return offered(found.profiles)


async def until_state(launches: Any, id: str, *states: str) -> Any:
    async with asyncio.timeout(5):
        while (launch := next(each for each in await launches.all() if each.id == id)).state not in states:  # noqa: ASYNC110
            await asyncio.sleep(0.01)
    return launch


@pytest.mark.parametrize(("ends", "state"), [("SUCCEEDED", ENDED), ("FAILED", FAILED)])
async def test_a_launchers_run_is_a_ray_job_that_asks_for_a_gpu(tmp_path: Path, ends: str, state: str) -> None:
    jobs = Jobs(ends)
    found, launches = await ray_launcher(tmp_path, jobs)
    asked = await launches.ask(Asked(profile="small", catalog="c:c", name="on ray"))
    await found._step()  # pyright: ignore[reportPrivateUsage]
    (submitted,) = jobs.submitted
    assert submitted["entrypoint_num_gpus"] == 1.0 and submitted["submission_id"] == f"run-{asked.id}"
    assert "rollout_train.cli train" in submitted["entrypoint"] and "--name 'on ray'" in submitted["entrypoint"]
    assert (await until_state(launches, asked.id, CLAIMED, RUNNING)).job == f"run-{asked.id}"
    done = await until_state(launches, asked.id, ENDED, FAILED)
    assert done.state == state and (state == ENDED or "it broke" in str(done.detail))
    written = await asyncio.to_thread(Path(str(done.directory), OUTPUT).read_text)
    assert written.startswith(f"the output of run-{asked.id}")


async def test_a_ray_job_is_stopped_and_followed_again_by_a_launcher_started_again(tmp_path: Path) -> None:
    jobs = Jobs(steps=10_000)
    found, launches = await ray_launcher(tmp_path, jobs)
    asked = await launches.ask(Asked(profile="small", catalog="c:c", name="long"))
    await found._step()  # pyright: ignore[reportPrivateUsage]
    await until_state(launches, asked.id, RUNNING)
    for task in list(found._watching):  # pyright: ignore[reportPrivateUsage]  (the launcher is gone)
        task.cancel()
    again, _ = await ray_launcher(tmp_path, jobs)
    await again._adopt()  # pyright: ignore[reportPrivateUsage]
    await launches.note(asked.id, state=STOPPING)
    await again._step()  # pyright: ignore[reportPrivateUsage]
    assert jobs.stopped == [f"run-{asked.id}"]
    assert (await until_state(launches, asked.id, STOPPED)).detail == f"stopped (Ray job run-{asked.id})"
