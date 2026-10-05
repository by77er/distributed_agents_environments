"""Starting a run's job, and reading how it goes: one function, `submit`, wherever the run is asked for.

`submit(settings, cluster, ledger, …)` records a launch (`rollout_train.launches`: what was asked, and the run it is,
registered under its name unless it resumes one) and starts the job its run runs in, `python -m rollout_train.jobs
LAUNCH` (`rollout_train.jobs`), handed the cluster config as JSON (`ROLLOUT_CLUSTER_JSON`). Where the job goes is the
cluster config's to say (`backend_of`):

- **Ray's job API** (`RayJobs`), on a cluster without `[kubernetes]`: a Ray job submitted to `[ray] jobs`, its
  submission id `run-LAUNCH`, asking for its driver's CPUs (`rollout_train.demand`: the loop, its runners and sandbox
  pools). A run on a published environment (`rollout_train.published`) is submitted in its version's Ray runtime
  environment, so its driver imports the environment from the version's source.
- **A RayJob** (`RayJobResources`), on a cluster with `[kubernetes]`: a RayJob custom resource made from the template
  `[kubernetes] rayjob` names (its Ray cluster, image, volumes, `backoffLimit`), with the run's entrypoint, runtime
  environment and metadata filled in and its Ray cluster sized from the run's demand (`rendered`), created in
  `[kubernetes] namespace` through the API server (`KubernetesApi`). With `[kubernetes] queue`, it is made suspended
  and labelled with Kueue's queue, and Kueue starts it when the queue's quota holds all of it. KubeRay starts a Ray
  cluster for it, runs the driver there, and removes the cluster when it ends.

A job's driver notes on its launch when it runs and how it ends; `followed` reads the job's status for a launch that is
going and notes what the driver could not (a job that waits for its resources, one that died without a word), and
`stopped` asks a job to stop. A RayJob that Kueue holds says so (`waits for admission`, with Kueue's reason).
"""

import asyncio
import contextlib
import copy
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import httpx
from pydantic import JsonValue

from rollout_train.cluster import HANDED, Cluster, KubernetesSection
from rollout_train.demand import Demand, Pod, Resources, demand, pods
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
    "demand_of",
    "entrypoint_of",
    "followed",
    "job_name",
    "rendered",
    "runtime_env_of",
    "sized",
    "start",
    "stopped",
    "submit",
]

TAIL = 2000
"""Characters of a failed job's output or message kept as why it failed."""
GROUP = "ray.io"
VERSION = "v1"
"""The RayJob custom resource's API group and version (KubeRay 1.x)."""
QUEUE_LABEL = "kueue.x-k8s.io/queue-name"
"""The label that names the Kueue LocalQueue a RayJob is admitted through."""
JOB_UID_LABEL = "kueue.x-k8s.io/job-uid"
"""The label Kueue gives a job's Workload: the job's uid."""
KUEUE = "kueue.x-k8s.io/v1beta2"
"""Kueue's API group and version (Kueue 0.15 and after)."""
GPU = "nvidia.com/gpu"


@dataclass(frozen=True)
class JobState:
    """How a job goes, in a launch's states (`submitted`, `running`, `ended`, `failed`, `stopped`), with why."""

    state: str
    detail: str | None = None


class Backend(Protocol):
    """Where runs' jobs go: Ray's job API, or RayJobs on Kubernetes."""

    name: str

    async def start(
        self, launch: Launch, entrypoint: str, runtime_env: Mapping[str, JsonValue], asked: Demand | None = None
    ) -> str:
        """Start a launch's job, sized for what its run needs (`asked`); its name (a Ray job's submission id, a
        RayJob's name)."""
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

    async def start(
        self, launch: Launch, entrypoint: str, runtime_env: Mapping[str, JsonValue], asked: Demand | None = None
    ) -> str:
        metadata = {"kind": launch.asked.kind, "launch": launch.id, "name": launch.asked.name}
        cpus = asked.driver.cpus if asked is not None else 1
        return str(
            await asyncio.to_thread(
                self.client().submit_job, entrypoint=entrypoint, submission_id=job_name(launch),
                runtime_env=dict(runtime_env), entrypoint_num_cpus=cpus, metadata=metadata,
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
    """What makes, reads and deletes RayJobs, and reads Kueue's objects: the API server at `base`, with the service
    account's token and CA (by default the pod's own), over `transport` where given (a test's)."""

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

    async def workloads(self, namespace: str, uid: str) -> list[dict[str, Any]]:
        """Kueue's Workloads of a job, by the job's uid (none where Kueue is not installed)."""
        async with self._client() as client:
            response = await client.get(f"/apis/{KUEUE}/namespaces/{namespace}/workloads",
                                        params={"labelSelector": f"{JOB_UID_LABEL}={uid}"})  # fmt: skip
        if response.status_code == 404:
            return []
        response.raise_for_status()
        return cast(list[dict[str, Any]], response.json().get("items") or [])

    async def put_secret(self, namespace: str, name: str, data: Mapping[str, bytes]) -> None:
        """Make the Secret `name` hold `data` (its keys and their bytes), in place of what it held, or made anew."""
        import base64

        labels = {"app.kubernetes.io/part-of": "rollout"}
        resource = {"apiVersion": "v1", "kind": "Secret", "type": "Opaque",
                    "metadata": {"name": name, "namespace": namespace, "labels": labels},
                    "data": {key: base64.b64encode(value).decode() for key, value in data.items()}}  # fmt: skip
        path = f"/api/v1/namespaces/{namespace}/secrets"
        async with self._client() as client:
            response = await client.put(f"{path}/{name}", json=resource)
            if response.status_code == 404:
                response = await client.post(path, json=resource)
        if response.status_code >= 300:
            raise RuntimeError(f"the API server refused the Secret {name} ({response.status_code}): "
                               f"{response.text[:TAIL]}")  # fmt: skip

    async def read(self, path: str) -> dict[str, Any] | None:
        """What the API server answers at `path` (`/apis/GROUP/VERSION/...`): none where it is not found. Raises
        `RuntimeError` for any other refusal (a resource the account may not read, an API that is not served)."""
        async with self._client() as client:
            response = await client.get(path)
        if response.status_code == 404:
            return None
        if response.status_code >= 300:
            raise RuntimeError(f"the API server refused {path} ({response.status_code}): {response.text[:TAIL]}")
        return cast(dict[str, Any], response.json())

    async def delete(self, namespace: str, name: str) -> None:
        async with self._client() as client:
            response = await client.request("DELETE", self._path(namespace, name), json={"propagationPolicy":
                                                                                           "Background"})  # fmt: skip
        if response.status_code not in (200, 202, 404):
            response.raise_for_status()


DEADLINE_GRACE = 1800
"""Seconds past a run's `limits.hours` its RayJob may go on (its driver ends it at its hours, and releases its pods)
before Kubernetes stops it (`activeDeadlineSeconds`)."""


def rendered(
    template: Mapping[str, Any],
    launch: Launch,
    entrypoint: str,
    runtime_env: Mapping[str, JsonValue],
    namespace: str,
    *,
    asked: Demand | None = None,
    queue: str | None = None,
) -> dict[str, Any]:
    """The RayJob of a launch's job, made from `template` (a RayJob as YAML reads it: its Ray cluster, image, volumes,
    retries): its name and labels, its entrypoint and the driver's CPUs, its runtime environment (as YAML, as KubeRay
    takes it), its job's submission id and metadata. With `asked`, its Ray cluster is sized from the run's demand
    (`sized`). With `queue`, it is labelled with Kueue's queue and made suspended: Kueue starts it once it admits it.
    Everything else is the template's."""
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
    if queue is not None:
        labels[QUEUE_LABEL] = queue
    metadata |= {"name": name, "namespace": namespace, "labels": labels}
    made["metadata"] = metadata
    spec = dict(made.get("spec") or {})
    if queue is not None:
        spec["suspend"] = True
    spec["entrypoint"] = entrypoint
    spec["entrypointNumCpus"] = asked.driver.cpus if asked is not None else 1
    if asked is not None:
        spec["rayClusterSpec"] = sized(spec.get("rayClusterSpec") or {}, asked)
    spec["jobId"] = name
    spec["runtimeEnvYAML"] = yaml.safe_dump(json.loads(json.dumps(dict(runtime_env))), sort_keys=True)
    spec["metadata"] = {"kind": launch.asked.kind, "launch": launch.id, "name": launch.asked.name}
    hours = launch.asked.settings.get("limits.hours")
    if isinstance(hours, int | float) and not isinstance(hours, bool):  # (the driver ends it at its hours: this, after)
        spec["activeDeadlineSeconds"] = int(float(hours) * 3600 + DEADLINE_GRACE)
    made["spec"] = spec
    return made


def _quantity(text: object) -> float:
    """A Kubernetes quantity as a number: CPUs (`500m` is 0.5), or memory in GiB (`14Gi`, `512Mi`, `2G`)."""
    said = str(text).strip()
    units = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
    for suffix, factor in units.items():
        if said.endswith(suffix):
            return float(said.removesuffix(suffix)) * factor / 2**30
    if said.endswith("m"):
        return float(said.removesuffix("m")) / 1000
    return float(said)


def _bound(limits: Mapping[str, Any]) -> tuple[Resources, list[str]]:
    """The most one pod may have, from a container's limits, and which resources they bound."""
    known: list[str] = []
    cpus = memory = gpus = 0.0
    if "cpu" in limits:
        cpus, known = _quantity(limits["cpu"]), [*known, "cpus"]
    if "memory" in limits:
        memory, known = _quantity(limits["memory"]), [*known, "memory_gib"]
    if GPU in limits:
        gpus, known = float(limits[GPU]), [*known, "gpus"]
    return Resources(cpus, memory, gpus), known


def _mib(gib: float) -> str:
    return f"{math.ceil(gib * 1024)}Mi"


def _cpus(cpus: float) -> str:
    return f"{round(cpus, 3):g}"


def _pod(template: Mapping[str, Any], pod: Pod, *, container: str | None = None) -> dict[str, Any]:
    """A pod template sized for `pod`: its first container asks for what it needs (CPUs and memory, and whole GPUs
    where it holds any), its limits kept where they are larger (the template's are the most a pod may have)."""
    made: dict[str, Any] = copy.deepcopy(dict(template))
    spec = made.setdefault("spec", {})
    containers: list[dict[str, Any]] = spec.setdefault("containers", [{}])
    first = containers[0]
    if container is not None:
        first["name"] = container
    resources = dict(first.get("resources") or {})
    requests = dict(resources.get("requests") or {})
    limits = dict(resources.get("limits") or {})
    asked = pod.requests
    requests |= {"cpu": _cpus(asked.cpus), "memory": _mib(asked.memory_gib)}
    if "memory" in limits and _quantity(limits["memory"]) < asked.memory_gib:
        limits["memory"] = _mib(asked.memory_gib)
    if "cpu" in limits and _quantity(limits["cpu"]) < asked.cpus:
        limits["cpu"] = _cpus(asked.cpus)
    if pod.ray.gpus:
        requests[GPU] = limits[GPU] = int(pod.ray.gpus)
    else:
        requests.pop(GPU, None)
        limits.pop(GPU, None)
    resources["requests"] = requests
    if limits:
        resources["limits"] = limits
    else:
        resources.pop("limits", None)
    first["resources"] = resources
    return made


def _started(given: Mapping[str, Any] | None, pod: Pod) -> dict[str, Any]:
    """`rayStartParams` for a pod: Ray starts its node with the pod's CPUs, GPUs and custom resources."""
    params = dict(given or {})
    params["num-cpus"] = str(int(pod.ray.cpus))
    params["num-gpus"] = str(int(pod.ray.gpus))
    if pod.ray.custom:
        params["resources"] = json.dumps(json.dumps(dict(pod.ray.custom), sort_keys=True))
    return params


def sized(cluster: Mapping[str, Any], asked: Demand) -> dict[str, Any]:
    """A RayJob's Ray cluster (`rayClusterSpec`) sized from a run's demand (`rollout_train.demand.pods`): its head pod
    asks for the driver's and the placement group's resources and `HEADROOM`, where that fits the head's limits (one
    node's worth); else the head holds the driver, the trainer's bundle and the bridge's, and each engine host's bundle
    is a worker pod of a worker group made from the head's template. Ray starts each node with what its pod holds
    (`rayStartParams`)."""
    made: dict[str, Any] = copy.deepcopy(dict(cluster))
    head = dict(made.get("headGroupSpec") or {})
    template = dict(head.get("template") or {})
    containers = cast(list[dict[str, Any]], dict(template.get("spec") or {}).get("containers") or [{}])
    most, known = _bound(dict(dict(containers[0].get("resources") or {}).get("limits") or {}))
    first, *workers = pods(asked, most if known else None, known=known)
    head["template"] = _pod(template, first)
    head["rayStartParams"] = _started(head.get("rayStartParams"), first)
    made["headGroupSpec"] = head
    groups = list(made.get("workerGroupSpecs") or [])
    for each in workers:
        params = {key: value for key, value in dict(head.get("rayStartParams") or {}).items()
                  if key not in ("dashboard-host",)}  # fmt: skip
        groups.append({
            "groupName": each.group, "replicas": each.replicas, "minReplicas": each.replicas,
            "maxReplicas": each.replicas, "rayStartParams": _started(params, each),
            "template": _pod(template, each, container="ray-worker"),
        })  # fmt: skip
    if groups:
        made["workerGroupSpecs"] = groups
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

    async def start(
        self, launch: Launch, entrypoint: str, runtime_env: Mapping[str, JsonValue], asked: Demand | None = None
    ) -> str:
        template = await asyncio.to_thread(self.template)
        resource = rendered(template, launch, entrypoint, runtime_env, self.section.namespace, asked=asked,
                            queue=self.section.queue)  # fmt: skip
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
        if dict(found.get("spec") or {}).get("suspend"):
            return JobState(SUBMITTED, await self._admission(found))
        waiting = {"": "its Ray cluster is being made", "Initializing": "its Ray cluster is starting",
                   "Waiting": "waits for its Ray cluster", "Retrying": "its job is started again"}  # fmt: skip
        return JobState(SUBMITTED, message or waiting.get(deployment, f"its RayJob is {deployment.lower()}"))

    async def _admission(self, found: Mapping[str, Any]) -> str:
        """Why Kueue holds a suspended RayJob: its queue, and what its Workload says (the quota it waits for)."""
        metadata = dict(found.get("metadata") or {})
        queue = dict(metadata.get("labels") or {}).get(QUEUE_LABEL)
        said = f"waits for admission by Kueue (queue {queue})" if queue else "waits for admission by Kueue"
        uid = metadata.get("uid")
        if not uid:
            return said
        try:
            workloads = await self.api.workloads(self.section.namespace, str(uid))
        except Exception:  # (Kueue's API is not readable here: its queue is enough)
            return said
        for workload in workloads:
            for condition in cast(list[dict[str, Any]], dict(workload.get("status") or {}).get("conditions") or []):
                message = str(condition.get("message") or "")
                if (
                    condition.get("type") in ("QuotaReserved", "Admitted")
                    and condition.get("status") == "False"
                    and message
                ):
                    return f"{said}: {message}"
        return said

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


def demand_of(launch: Launch, cluster: Cluster) -> Demand:
    """What a launch's run needs (`rollout_train.demand`)."""
    asked = launch.asked
    return demand(RunSettings({**asked.settings, "kind": asked.kind, "name": asked.name}), cluster)


async def start(launch: Launch, cluster: Cluster, ledger: Ledger, backend: Backend | None = None) -> Launch:
    """Start a recorded launch's job; the launch, submitted (or failed, saying why the job could not be made)."""
    launches = launches_of(ledger)
    assert launches is not None
    backend = backend or backend_of(cluster)
    try:
        runtime_env = await runtime_env_of(launch, cluster, ledger)
        asked = demand_of(launch, cluster)
        job = await backend.start(launch, entrypoint_of(launch, cluster), runtime_env, asked)
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
    """Record a launch of a run with these settings and start its job (`ask`, then `start`), with what follows from its
    settings said: what it trains and each channel's renderer (`rollout_train.validation.completed`)."""
    from rollout_train.validation import completed

    launch = await ask(completed(settings, cluster), ledger, preset=preset, resumes=resumes)
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
