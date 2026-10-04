# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (Ray is partly untyped.)
"""The test session's Ray is a local instance of its own: on disk, 4 CPUs, a dashboard that takes jobs, and workers
in this environment."""

import os
import sys
import time
from pathlib import Path

from tests.local_ray import RAY_TESTS, LocalRay, submitted


def test_the_sessions_ray_is_its_own_local_instance(local_ray: LocalRay) -> None:
    import ray

    assert local_ray.directory.parent == RAY_TESTS and local_ray.directory.is_dir()
    assert local_ray.directory.name.endswith(f"_{os.getpid()}")  # (a session's own)
    assert ray.cluster_resources()["CPU"] == 4
    context = ray.get_runtime_context()
    assert context.gcs_address == local_ray.address
    (node,) = ray.nodes()
    assert Path(str(node["ObjectStoreSocketName"])).is_relative_to(local_ray.directory)  # (its files, not another's)
    assert "RAY_ADDRESS" not in os.environ


def test_its_workers_run_in_this_environment(local_ray: LocalRay) -> None:
    import ray

    def told() -> tuple[str | None, str]:
        return os.environ.get("RAY_ENABLE_UV_RUN_RUNTIME_ENV"), sys.executable

    task = ray.remote(num_cpus=1)(told)
    assert ray.get(task.remote()) == ("0", sys.executable)


def test_its_dashboard_takes_jobs(local_ray: LocalRay) -> None:
    from ray.job_submission import JobStatus, JobSubmissionClient

    client = JobSubmissionClient(local_ray.dashboard)
    job = submitted(client, entrypoint=f"{sys.executable} -c 'print(6 * 7)'")
    deadline = time.monotonic() + 60
    while (status := client.get_job_status(job)) not in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED):
        assert time.monotonic() < deadline, f"the job is still {status}"
        time.sleep(0.2)
    assert status == JobStatus.SUCCEEDED and "42" in client.get_job_logs(job)
