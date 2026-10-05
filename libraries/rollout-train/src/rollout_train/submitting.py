"""Starting a run's job, and reading how it goes: one function, `submit`, wherever the run is asked for.

`submit(settings, cluster, ledger, …)` records a launch (`rollout_train.launches`: what was asked, and the run it is,
registered under its name unless it resumes one) and starts the job its run runs in, `python -m rollout_train.jobs
LAUNCH` (`rollout_train.jobs`), handed the cluster config as JSON (`ROLLOUT_CLUSTER_JSON`). Where the job goes is the
cluster config's to say (`backend_of`):

- **Ray's job API** (`RayJobs`), on a cluster without `[kubernetes]`: a Ray job submitted to `[ray] jobs`, its
  submission id `run-LAUNCH`, asking for one CPU for its driver. A run on a published environment
  (`rollout_train.published`) is submitted in its version's Ray runtime environment, so its driver imports the
  environment from the version's source.
- **A RayJob** (`RayJobResources`), on a cluster with `[kubernetes]`: a RayJob custom resource made from the template
  `[kubernetes] rayjob` names (its Ray cluster, image, volumes, `backoffLimit`), with the run's entrypoint, runtime
  environment and metadata filled in (`rendered`), created in `[kubernetes] namespace` through the API server
  (`KubernetesApi`). KubeRay starts a Ray cluster for it, runs the driver there, and removes the cluster when it ends.

A job's driver notes on its launch when it runs and how it ends; `followed` reads the job's status for a launch that is
going and notes what the driver could not (a job that waits for its resources, one that died without a word), and
`stopped` asks a job to stop.
"""

import asyncio
import contextlib
import copy
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import httpx
from pydantic import JsonValue

from rollout_train.cluster import HANDED, Cluster, KubernetesSection
from rollout_train.launches import (
    ASKED,
    ENDED,
    FAILED,
    OPEN,
    RUNNING,
    STOPPED,
    STOPPING,
    SUBMITTED,
    Asked,
    Launch,
    Launches,
    launches_of,
)
from rollout_train.ledger import Ledger
from rollout_train.published import environment_versions_of, is_published
from rollout_train.registry import registry_of
from rollout_train.run_settings import RunSettings

__all__ = [
    "Backend",
    "JobState",
    "KubernetesApi",
    "RayJobResources",
    "RayJobs",
    "ask",
    "backend_of",
    "entrypoint_of",
    "followed",
    "job_name",
    "rendered",
    "runtime_env_of",
    "start",
    "stopped",
    "submit",
]

TAIL = 2000
"""Characters of a failed job's output or message kept as why it failed."""
GROUP = "ray.io"
VERSION = "v1"
"""The RayJob custom resource's API group and version (KubeRay 1.x)."""


@dataclass(frozen=True)
class JobState:
    """How a job goes, in a launch's states (`submitted`, `running`, `ended`, `failed`, `stopped`), with why."""

    state: str
    detail: str | None = None


class Backend(Protocol):
    """Where runs' jobs go: Ray's job API, or RayJobs on Kubernetes."""

    name: str

    async def start(self, launch: Launch, entrypoint: str, runtime_env: Mapping[str, JsonValue]) -> str:
        """Start a launch's job; its name (a Ray job's submission id, a RayJob's name)."""
        ...

    async def status(self, job: str) -> JobState:
        """How a job goes now."""
        ...

    async def stop(self, job: str) -> None:
        """Ask a job to stop: its driver is interrupted, and notes its run stopped."""
        ...


def job_name(launch: Launch) -> str:
    """The name of a launch's job: `run-` and its id, lowercase (a Kubernetes name: letters, digits and dashes)."""
    return "run-" + re.sub(r"[^a-z0-9]+", "-", launch.id.removeprefix("launch_").lower()).strip("-")


class RayJobs:
    """Runs' jobs as Ray jobs, submitted to the job server at `address` (`client`: a `JobSubmissionClient`, or one
    like it)."""

    name = "ray"

    def __init__(self, address: str, client: Any = None) -> None:
        self.address = address
        self._client = client

    def client(self) -> Any:
        if self._client is None:
            from rollout_train.ray_cluster import prepare

            prepare()
            from ray.job_submission import JobSubmissionClient

            self._client = JobSubmissionClient(self.address)
        return self._client

    async def start(self, launch: Launch, entrypoint: str, runtime_env: Mapping[str, JsonValue]) -> str:
        metadata = {"kind": launch.asked.kind, "launch": launch.id, "name": launch.asked.name}
        return str(
            await asyncio.to_thread(
                self.client().submit_job, entrypoint=entrypoint, submission_id=job_name(launch),
                runtime_env=dict(runtime_env), entrypoint_num_cpus=1, metadata=metadata,
            )
        )  # fmt: skip

    async def status(self, job: str) -> JobState:
        from ray.job_submission import JobStatus

        client = self.client()
        try:
            info = await asyncio.to_thread(client.get_job_info, job)
        except RuntimeError as error:  # (a job the server does not know: its cluster was started again)
            return JobState(FAILED, f"Ray job {job} is not known to the job server: {error}")
        status = info.status
        message = str(getattr(info, "message", "") or "")
        if status == JobStatus.PENDING:
            return JobState(SUBMITTED, message or "waits for Ray to start its driver")
        if status == JobStatus.RUNNING:
            return JobState(RUNNING)
        if status == JobStatus.SUCCEEDED:
            return JobState(ENDED, "ended")
        if status == JobStatus.STOPPED:
            return JobState(STOPPED, f"stopped (Ray job {job})")
        output = ""
        with contextlib.suppress(Exception):
            output = str(await asyncio.to_thread(client.get_job_logs, job))
        return JobState(FAILED, f"Ray job {job} failed: {(output or message)[-TAIL:].strip()}")

    async def stop(self, job: str) -> None:
        with contextlib.suppress(Exception):  # (a job that ended already)
            await asyncio.to_thread(self.client().stop_job, job)


class KubernetesApi:
    """What makes, reads and deletes RayJobs: the API server at `base`, with the service account's token and CA (by
    default the pod's own), over `transport` where given (a test's)."""

    ACCOUNT = Path("/var/run/secrets/kubernetes.io/serviceaccount")

    def __init__(
        self,
        base: str = "https://kubernetes.default.svc",
        *,
        token: str | None = None,
        ca: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base = base.rstrip("/")
        self._token = token
        self._ca = ca
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        token = self._token
        if token is None and (self.ACCOUNT / "token").exists():
            token = (self.ACCOUNT / "token").read_text().strip()
        ca = self._ca or (str(self.ACCOUNT / "ca.crt") if (self.ACCOUNT / "ca.crt").exists() else None)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        if self._transport is not None:
            return httpx.AsyncClient(base_url=self.base, headers=headers, transport=self._transport)
        return httpx.AsyncClient(base_url=self.base, headers=headers, verify=ca or True, timeout=30.0)

    def _path(self, namespace: str, name: str | None = None) -> str:
        found = f"/apis/{GROUP}/{VERSION}/namespaces/{namespace}/rayjobs"
        return f"{found}/{name}" if name else found

    async def create(self, namespace: str, resource: Mapping[str, Any]) -> dict[str, Any]:
        async with self._client() as client:
            response = await client.post(self._path(namespace), json=dict(resource))
        if response.status_code >= 300:
            raise RuntimeError(f"the API server refused the RayJob ({response.status_code}): {response.text[:TAIL]}")
        return cast(dict[str, Any], response.json())

    async def get(self, namespace: str, name: str) -> dict[str, Any] | None:
        async with self._client() as client:
            response = await client.get(self._path(namespace, name))
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    async def delete(self, namespace: str, name: str) -> None:
        async with self._client() as client:
            response = await client.request("DELETE", self._path(namespace, name), json={"propagationPolicy":
                                                                                           "Background"})  # fmt: skip
        if response.status_code not in (200, 202, 404):
            response.raise_for_status()


def rendered(
    template: Mapping[str, Any], launch: Launch, entrypoint: str, runtime_env: Mapping[str, JsonValue], namespace: str
) -> dict[str, Any]:
    """The RayJob of a launch's job, made from `template` (a RayJob as YAML reads it: its Ray cluster, image, volumes,
    retries): its name and labels, its entrypoint and the driver's CPU, its runtime environment (as YAML, as KubeRay
    takes it), its job's submission id and metadata. Everything else is the template's."""
    import yaml

    made: dict[str, Any] = copy.deepcopy(dict(template))
    name = job_name(launch)
    made["apiVersion"] = f"{GROUP}/{VERSION}"
    made["kind"] = "RayJob"
    metadata = dict(made.get("metadata") or {})
    metadata.pop("generateName", None)
    labels = dict(metadata.get("labels") or {})
    labels |= {"app.kubernetes.io/managed-by": "rollout", "rollout/launch": launch.id.lower().replace("_", "-"),
               "rollout/kind": launch.asked.kind}  # fmt: skip
    metadata |= {"name": name, "namespace": namespace, "labels": labels}
    made["metadata"] = metadata
    spec = dict(made.get("spec") or {})
    spec["entrypoint"] = entrypoint
    spec["entrypointNumCpus"] = 1
    spec["jobId"] = name
    spec["runtimeEnvYAML"] = yaml.safe_dump(json.loads(json.dumps(dict(runtime_env))), sort_keys=True)
    spec["metadata"] = {"kind": launch.asked.kind, "launch": launch.id, "name": launch.asked.name}
    made["spec"] = spec
    return made


class RayJobResources:
    """Runs' jobs as RayJobs in a Kubernetes namespace, each made from the cluster config's template
    (`[kubernetes]`)."""

    name = "kubernetes"

    def __init__(self, section: KubernetesSection, api: KubernetesApi | None = None) -> None:
        self.section = section
        self.api = api or KubernetesApi(section.api)

    def template(self) -> dict[str, Any]:
        import yaml

        loaded: Any = yaml.safe_load(Path(self.section.rayjob).read_text())
        if not isinstance(loaded, dict):
            raise ValueError(f"{self.section.rayjob} is not a RayJob (a YAML mapping)")
        return cast(dict[str, Any], loaded)

    async def start(self, launch: Launch, entrypoint: str, runtime_env: Mapping[str, JsonValue]) -> str:
        template = await asyncio.to_thread(self.template)
        resource = rendered(template, launch, entrypoint, runtime_env, self.section.namespace)
        made = await self.api.create(self.section.namespace, resource)
        return str(made.get("metadata", {}).get("name") or resource["metadata"]["name"])

    async def status(self, job: str) -> JobState:
        found = await self.api.get(self.section.namespace, job)
        if found is None:
            return JobState(FAILED, f"its RayJob {job} is gone")
        status = cast(dict[str, Any], found.get("status") or {})
        job_status = str(status.get("jobStatus") or "")
        deployment = str(status.get("jobDeploymentStatus") or "")
        message = str(status.get("message") or status.get("reason") or "")
        if job_status == "SUCCEEDED" or (deployment == "Complete" and job_status in ("", "SUCCEEDED")):
            return JobState(ENDED, "ended")
        if job_status == "STOPPED":
            return JobState(STOPPED, f"stopped (RayJob {job})")
        if job_status == "FAILED" or deployment == "Failed":
            return JobState(FAILED, f"RayJob {job} failed: {message[-TAIL:]}".rstrip(": "))
        if job_status == "RUNNING":
            return JobState(RUNNING)
        waiting = {"": "its Ray cluster is being made", "Initializing": "its Ray cluster is starting",
                   "Waiting": "waits for its Ray cluster", "Retrying": "its job is started again"}  # fmt: skip
        return JobState(SUBMITTED, message or waiting.get(deployment, f"its RayJob is {deployment.lower()}"))

    async def stop(self, job: str) -> None:
        await self.api.delete(self.section.namespace, job)


def backend_of(cluster: Cluster) -> Backend:
    """Where a cluster's runs' jobs go: RayJobs where its config has `[kubernetes]`, else Ray's job API at `[ray]
    jobs`."""
    if cluster.kubernetes is not None:
        return RayJobResources(cluster.kubernetes)
    return RayJobs(cluster.ray.jobs)


def entrypoint_of(launch: Launch, cluster: Cluster) -> str:
    """What a launch's job runs: `python -m rollout_train.jobs LAUNCH` in the interpreter its run starts in (its
    environment's own, for one in a project's Python; the cluster's `[ray] python` otherwise)."""
    import shlex

    environment = launch.asked.environment
    python = cluster.ray.python
    interpreter = "python" if python == "platform" else python
    if environment is not None and not is_published(environment):
        section = cluster.environments.get(environment)
        if section is not None and section.runs_in is not None:
            interpreter = section.runs_in
    return shlex.join([interpreter, "-m", "rollout_train.jobs", launch.id])


async def runtime_env_of(launch: Launch, cluster: Cluster, ledger: Ledger) -> dict[str, JsonValue]:
    """The Ray runtime environment of a launch's job: the cluster config as JSON (`ROLLOUT_CLUSTER_JSON`), in the
    runtime environment of the published version it plays, if it plays one. Raises `KeyError` for a published version
    the ledger does not keep."""
    handed: dict[str, JsonValue] = {"env_vars": {HANDED: json.dumps(dict(cluster.described))}}
    environment = launch.asked.environment
    if environment is None or not is_published(environment):
        return handed
    versions = environment_versions_of(ledger)
    version = await versions.get(environment) if versions is not None else None
    if version is None:
        raise KeyError(f"there is no published environment {environment}")
    found: dict[str, JsonValue] = dict(version.runtime_env)
    variables = dict(cast(Mapping[str, JsonValue], found.get("env_vars") or {}))
    found["env_vars"] = {**variables, HANDED: json.dumps(dict(cluster.described))}
    return found


async def ask(
    settings: RunSettings,
    ledger: Ledger,
    *,
    preset: str | None = None,
    resumes: str | None = None,
) -> Launch:
    """Record a launch of a run with these settings (its kind and name among them): of the run it resumes, else a run
    registered now under its name. Raises `ValueError` (`rollout_train.registry.Taken`) for a name another run has."""
    launches = launches_of(ledger)
    if launches is None:
        raise KeyError("this ledger keeps no launches")
    name = str(settings["name"] or "")
    if resumes is None:
        registry = registry_of(ledger)
        if registry is None:
            raise KeyError("this ledger keeps no registry of runs")
        run = (await registry.create(name)).id
    else:
        run = resumes
    given = {key: value for key, value in settings.values.items() if key not in ("kind", "name")}
    return await launches.ask(Asked(settings.kind, name, given, preset, resumes), run)


async def start(launch: Launch, cluster: Cluster, ledger: Ledger, backend: Backend | None = None) -> Launch:
    """Start a recorded launch's job; the launch, submitted (or failed, saying why the job could not be made)."""
    launches = launches_of(ledger)
    assert launches is not None
    backend = backend or backend_of(cluster)
    try:
        runtime_env = await runtime_env_of(launch, cluster, ledger)
        job = await backend.start(launch, entrypoint_of(launch, cluster), runtime_env)
    except Exception as error:  # (a job that cannot be made is a failed launch)
        return await launches.note(launch.id, expect=(ASKED,), state=FAILED, detail=f"{type(error).__name__}: {error}")
    noted = await launches.note(launch.id, expect=(ASKED,), state=SUBMITTED, job=job, backend=backend.name)
    if noted.state == SUBMITTED:
        return noted
    if noted.state in (STOPPING, STOPPED):  # (stopped while its job was made: the job is stopped too)
        await backend.stop(job)
    return await launches.note(launch.id, job=job, backend=backend.name)  # (its driver may have started already)


async def submit(
    settings: RunSettings,
    cluster: Cluster,
    ledger: Ledger,
    *,
    preset: str | None = None,
    resumes: str | None = None,
    backend: Backend | None = None,
) -> Launch:
    """Record a launch of a run with these settings and start its job (`ask`, then `start`)."""
    launch = await ask(settings, ledger, preset=preset, resumes=resumes)
    return await start(launch, cluster, ledger, backend)


def _backend(launch: Launch, cluster: Cluster | None, backends: Mapping[str, Backend] | None) -> Backend | None:
    if launch.job is None:
        return None
    kind = launch.backend or "ray"
    if backends is not None and kind in backends:
        return backends[kind]
    if cluster is None:
        return None
    if kind == "kubernetes":
        return RayJobResources(cluster.kubernetes) if cluster.kubernetes is not None else None
    return RayJobs(cluster.ray.jobs)


async def followed(
    launch: Launch,
    launches: Launches,
    cluster: Cluster | None = None,
    *,
    backends: Mapping[str, Backend] | None = None,
) -> Launch:
    """A launch that is going, with what its job's status says noted: a job that waits (with why), runs, ended, failed
    or stopped without its driver saying so. A launch whose job cannot be read is as it was."""
    if launch.state not in OPEN or launch.state == ASKED:
        return launch
    backend = _backend(launch, cluster, backends)
    if backend is None or launch.job is None:
        return launch
    try:
        said = await backend.status(launch.job)
    except Exception:  # (the job server or the API server does not answer now: read again later)
        return launch
    if said.state == SUBMITTED:
        if launch.state == SUBMITTED and said.detail and said.detail != launch.detail:
            return await launches.note(launch.id, expect=(SUBMITTED,), detail=said.detail)
        return launch
    if said.state == RUNNING:
        if launch.state == SUBMITTED:
            return await launches.note(launch.id, expect=(SUBMITTED,), state=RUNNING)
        return launch
    if said.state == ENDED and launch.state == STOPPING:
        return await launches.note(launch.id, expect=(STOPPING,), state=STOPPED, detail="stopped")
    return await launches.note(launch.id, expect=OPEN, state=said.state, detail=said.detail)


async def stopped(
    launch: Launch,
    launches: Launches,
    cluster: Cluster | None = None,
    *,
    backends: Mapping[str, Backend] | None = None,
) -> Launch:
    """Ask a launch to stop: one whose job was not made yet is stopped at once; a job going is asked to stop, and its
    driver notes its run stopped. Raises `KeyError` for a launch that is not going."""
    if launch.state not in OPEN:
        raise KeyError(f"there is no launch {launch.id} going")
    if launch.state == ASKED:
        noted = await launches.note(launch.id, expect=(ASKED,), state=STOPPED, detail="stopped")
        if noted.state == STOPPED:
            return noted
        launch = noted
    noted = await launches.note(launch.id, expect=(SUBMITTED, RUNNING, STOPPING), state=STOPPING)
    if noted.state != STOPPING:
        raise KeyError(f"there is no launch {launch.id} going")
    backend = _backend(noted, cluster, backends)
    if backend is not None and noted.job is not None:
        await backend.stop(noted.job)
    return noted
