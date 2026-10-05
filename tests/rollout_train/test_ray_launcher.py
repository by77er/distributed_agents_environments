"""A launcher whose runs are Ray jobs: each asks for a GPU, is followed until it ends, and is stopped by a launcher
started again; a run on a published environment is a job in its version's runtime environment."""

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from rollout_train.launcher import OUTPUT, Launcher
from rollout_train.launches import ASKED, CLAIMED, ENDED, FAILED, RUNNING, STOPPED, STOPPING, Asked, launches_of
from rollout_train.ledger import FileLedger
from rollout_train.presence import presence_of
from rollout_train.published import FileEnvironmentVersions
from tests.rollout_train.sources import a_version
from tests.rollout_train.support import profiles


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
    asked = await launches.ask(Asked(profile="small", environment="c:c", name="on ray"))
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
    asked = await launches.ask(Asked(profile="small", environment="c:c", name="long"))
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


async def test_a_run_on_a_published_environment_is_a_job_in_its_versions_runtime_environment(tmp_path: Path) -> None:
    jobs = Jobs(steps=10_000)
    found, launches = await ray_launcher(tmp_path, jobs)
    versions = FileEnvironmentVersions(tmp_path / "ledger" / "environment_versions")
    version, other = await versions.record(a_version()), await versions.record(a_version("two"))
    found.versions, found.at_once = versions, 3
    unknown = await launches.ask(Asked(profile="small", environment=f"words@{'0' * 64}", name="unknown"))
    both = [version.reference, other.reference]
    mixed = await launches.ask(Asked(profile="small", environment=both[0], name="mixed", environments=both[1:]))
    asked = await launches.ask(Asked(profile="small", environment=version.reference, name="published"))
    await found._step()  # pyright: ignore[reportPrivateUsage]
    submitted = {each["metadata"]["name"]: each for each in jobs.submitted}
    assert set(submitted) == {"published"}  # (the unknown version is not claimed; the mixed launch fails)
    assert submitted["published"]["runtime_env"] == version.runtime_env
    entrypoint = submitted["published"]["entrypoint"]
    assert entrypoint.startswith("exec python -m rollout_train.cli train ") and version.reference in entrypoint
    states = {each.id: each for each in await launches.all()}
    assert (states[unknown.id].state, states[asked.id].state, states[mixed.id].state) == (ASKED, CLAIMED, FAILED)
    assert "several versions" in str(states[mixed.id].detail)
    await found._beat()  # pyright: ignore[reportPrivateUsage]
    heartbeats = presence_of(FileLedger(tmp_path / "ledger"))
    assert heartbeats is not None
    (beat,) = await heartbeats.beats()
    assert set(cast(list[str], beat.about["environments"])) == set(both)
    assert {each["environment"] for each in cast(list[dict[str, Any]], beat.about["published"])} == set(both)
    found.ray = None  # (without Ray, no published environment is offered or played)
    assert not await found._plays(states[asked.id])  # pyright: ignore[reportPrivateUsage]
