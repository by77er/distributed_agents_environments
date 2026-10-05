"""Starting a run's job: `submit` records the launch and the run, and starts `python -m rollout_train.jobs LAUNCH` as a
Ray job (handed the cluster config, in a published version's runtime environment where it plays one, asking for its
driver's CPUs) or as a RayJob made from the cluster's template, sized from the run's demand and, with Kueue, suspended
in its queue; `followed` notes what the job's status says (a RayJob Kueue holds waits for admission, saying why), and
`stopped` stops it."""

import json
import tomllib
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import yaml
from pydantic import JsonValue

from rollout_train.cluster import HANDED, Cluster, load, parsed
from rollout_train.demand import demand
from rollout_train.launches import ENDED, FAILED, RUNNING, STOPPED, STOPPING, SUBMITTED, TRAIN, launch_of, launches_of
from rollout_train.published import environment_versions_of
from rollout_train.run_settings import RunSettings
from rollout_train.stores import Stores
from rollout_train.submitting import (
    KubernetesApi,
    RayJobResources,
    RayJobs,
    entrypoint_of,
    followed,
    job_name,
    rendered,
    stopped,
    submit,
)
from tests.rollout_train.sources import a_version

CLUSTER = """
name = "here"
[ledger]
url = "sqlite:///{root}/ledger.db"
[blobs]
directory = "{root}/blobs"
[environments."tests.rollout_train.rollouts.games:words"]
python = "platform"
[environments."rollout_verifiers.environments:gsm8k"]
project = "{root}/verifiers"
"""
KUBERNETES = """
[kubernetes]
namespace = "rollout"
rayjob = "rayjob.yaml"
api = "https://kubernetes.test"
"""
QUEUE = """queue = "runs"
"""
ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = parsed(tomllib.loads((ROOT / "deploy" / "clusters" / "example.toml").read_text()))
ACCEPTANCE: dict[str, JsonValue] = {
    "kind": TRAIN, "environment": "rollout_verifiers.environments:gsm8k", "trainer.provider": "tinker-lora",
    "channels.policy.provider": "local-vllm", "channels.policy.model": "Qwen/Qwen3.5-4B",
    "channels.policy.renderer": "rollout_qwen:qwen35",
}  # fmt: skip
"""The acceptance run's shape on the example cluster: a local engine host and the peft-from-tinker bridge."""
TEMPLATE: dict[str, Any] = {
    "apiVersion": "ray.io/v1",
    "kind": "RayJob",
    "metadata": {"generateName": "run-", "labels": {"app": "run"}},
    "spec": {
        "shutdownAfterJobFinishes": True,
        "ttlSecondsAfterFinished": 600,
        "backoffLimit": 2,
        "rayClusterSpec": {"rayVersion": "2.59.0", "headGroupSpec": {"template": {"spec": {"containers": [
            {"name": "ray-head", "image": "localhost:30500/rollout-platform:dev"}
        ]}}}},
    },
}  # fmt: skip
WORDS = "tests.rollout_train.rollouts.games:words"


def a_cluster(root: Path, *, kubernetes: bool = False, queue: bool = False) -> Cluster:
    (root / "verifiers").mkdir(exist_ok=True)
    path = root / "cluster.toml"
    path.write_text(CLUSTER.format(root=root) + (KUBERNETES if kubernetes else "") + (QUEUE if queue else ""))
    (root / "rayjob.yaml").write_text(yaml.safe_dump(TEMPLATE))
    return load(path)


class Jobs:
    """A stand-in for Ray's job client: each job is in `status`, with `message` and `logs`."""

    def __init__(self) -> None:
        self.submitted: list[dict[str, Any]] = []
        self.stopped: list[str] = []
        self.status, self.message, self.logs = "PENDING", "", ""

    def submit_job(self, **given: Any) -> str:
        self.submitted.append(given)
        return str(given["submission_id"])

    def get_job_info(self, job: str) -> Any:
        from ray.job_submission import JobStatus

        class Info:
            status = JobStatus(self.status)
            message = self.message

        return Info()

    def get_job_logs(self, job: str) -> str:
        return self.logs

    def stop_job(self, job: str) -> bool:
        self.stopped.append(job)
        return True


def settings(**more: Any) -> RunSettings:
    return RunSettings({"kind": TRAIN, "name": "words-1", "environment": WORDS, **more})


async def test_a_run_is_submitted_as_a_ray_job_handed_the_cluster_config(tmp_path: Path) -> None:
    cluster, jobs = a_cluster(tmp_path), Jobs()
    stores = Stores.open(cluster)
    launch = await submit(settings(groups=3), cluster, stores.ledger, preset="small@2", backend=RayJobs("x", jobs))
    assert launch.state == SUBMITTED and launch.backend == "ray" and launch.job == job_name(launch)
    assert launch.asked.settings == {"environment": WORDS, "groups": 3} and launch.asked.preset == "small@2"
    (run,) = await stores.registry.runs()
    assert (run.id, run.name) == (launch.run, "words-1")  # (registered when it was asked for)
    (given,) = jobs.submitted
    assert given["entrypoint"] == f"python -m rollout_train.jobs {launch.id}"
    assert given["entrypoint_num_cpus"] == 0.5  # (its driver's: its loop, gateway and runner)
    assert json.loads(given["runtime_env"]["env_vars"][HANDED]) == dict(cluster.described)
    assert given["submission_id"] == launch.job and given["metadata"]["launch"] == launch.id
    gsm8k = await submit(settings(environment="rollout_verifiers.environments:gsm8k", name="gsm8k"), cluster,
                         stores.ledger, backend=RayJobs("x", jobs))  # fmt: skip
    assert entrypoint_of(gsm8k, cluster).startswith(f"{tmp_path}/verifiers/.venv/bin/python -m rollout_train.jobs")
    with pytest.raises(ValueError, match="words-1"):  # (a name another run has)
        await submit(settings(), cluster, stores.ledger, backend=RayJobs("x", jobs))


async def test_a_run_on_a_published_environment_is_a_job_in_its_versions_runtime_environment(tmp_path: Path) -> None:
    cluster, jobs = a_cluster(tmp_path), Jobs()
    stores = Stores.open(cluster)
    versions = environment_versions_of(stores.ledger)
    assert versions is not None
    version = await versions.record(a_version())
    launch = await submit(settings(environment=version.reference), cluster, stores.ledger, backend=RayJobs("x", jobs))
    (given,) = jobs.submitted
    environment = given["runtime_env"]
    assert environment["working_dir"] == version.runtime_env["working_dir"]
    own = cast(dict[str, Any], version.runtime_env.get("env_vars") or {})
    assert environment["env_vars"][HANDED] and set(own) <= set(environment["env_vars"])
    unknown = await submit(settings(environment=f"words@{'0' * 64}", name="other"), cluster, stores.ledger,
                           backend=RayJobs("x", jobs))  # fmt: skip
    assert unknown.state == FAILED and "no published environment" in str(unknown.detail) and launch.job


async def test_what_a_ray_job_says_is_noted_on_its_launch(tmp_path: Path) -> None:
    cluster, jobs = a_cluster(tmp_path), Jobs()
    stores = Stores.open(cluster)
    launches = launches_of(stores.ledger)
    assert launches is not None
    backends = {"ray": RayJobs("x", jobs)}
    launch = await submit(settings(), cluster, stores.ledger, backend=backends["ray"])
    jobs.message = "Job has not started yet. It may be waiting for resources (CPUs, GPUs)"
    waiting = await followed(launch, launches, cluster, backends=backends)
    assert waiting.state == SUBMITTED and waiting.detail == jobs.message
    jobs.status = "RUNNING"
    assert (await followed(waiting, launches, cluster, backends=backends)).state == RUNNING
    jobs.status, jobs.logs = "FAILED", "the output\nTraceback: it broke"
    failed = await followed(await launch_of(launches, launch.id), launches, cluster, backends=backends)
    assert failed.state == FAILED and "it broke" in str(failed.detail)
    other = await submit(settings(name="other"), cluster, stores.ledger, backend=backends["ray"])
    jobs.status = "SUCCEEDED"
    assert (await followed(other, launches, cluster, backends=backends)).state == ENDED
    third = await submit(settings(name="third"), cluster, stores.ledger, backend=backends["ray"])
    stopping = await stopped(third, launches, cluster, backends=backends)
    assert stopping.state == STOPPING and jobs.stopped == [third.job]
    jobs.status = "STOPPED"
    assert (await followed(stopping, launches, cluster, backends=backends)).state == STOPPED


async def test_a_launch_whose_job_cannot_be_made_fails_saying_why(tmp_path: Path) -> None:
    cluster = a_cluster(tmp_path)
    stores = Stores.open(cluster)

    class Refusing(Jobs):
        def submit_job(self, **given: Any) -> str:
            raise RuntimeError("the job server is not there")

    launch = await submit(settings(), cluster, stores.ledger, backend=RayJobs("x", Refusing()))
    assert launch.state == FAILED and "the job server is not there" in str(launch.detail)


def test_a_rayjob_is_made_from_the_clusters_template(tmp_path: Path) -> None:
    from rollout_train.launches import Asked, new_launch

    launch = new_launch(Asked(TRAIN, "words 1", {"environment": WORDS}), "run_1")
    environment: dict[str, JsonValue] = {"env_vars": {HANDED: "{}"}, "working_dir": "s3://bucket/blobs/x.zip"}
    made = rendered(TEMPLATE, launch, f"python -m rollout_train.jobs {launch.id}", environment, "rollout")
    assert made["apiVersion"] == "ray.io/v1" and made["kind"] == "RayJob"
    name = made["metadata"]["name"]
    assert name == job_name(launch) and name.startswith("run-") and name == name.lower() and len(name) <= 63
    assert made["metadata"]["namespace"] == "rollout" and "generateName" not in made["metadata"]
    assert made["metadata"]["labels"]["app"] == "run" and made["metadata"]["labels"]["rollout/kind"] == TRAIN
    spec = made["spec"]
    assert spec["entrypoint"] == f"python -m rollout_train.jobs {launch.id}" and spec["entrypointNumCpus"] == 1
    assert yaml.safe_load(spec["runtimeEnvYAML"]) == environment and spec["jobId"] == name
    assert spec["metadata"] == {"kind": TRAIN, "launch": launch.id, "name": "words 1"}
    assert (
        spec["backoffLimit"] == 2 and spec["rayClusterSpec"] == TEMPLATE["spec"]["rayClusterSpec"]
    )  # (the template's)
    assert TEMPLATE["metadata"].get("generateName") == "run-"  # (the template itself is left as it was)


def test_a_rayjob_is_sized_from_the_runs_demand_and_suspended_in_kueues_queue() -> None:
    from rollout_train.launches import Asked, new_launch

    launch = new_launch(Asked(TRAIN, "gsm8k", {"environment": ACCEPTANCE["environment"]}), "run_1")
    template = json.loads(json.dumps(TEMPLATE))
    (container,) = template["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"]
    container["resources"] = {"requests": {"cpu": "1", "memory": "4Gi"},
                              "limits": {"memory": "14Gi", "nvidia.com/gpu": 1}}  # fmt: skip
    asked = demand(RunSettings(ACCEPTANCE), EXAMPLE)
    made = rendered(template, launch, "python -m rollout_train.jobs L", {}, "rollout", asked=asked, queue="runs")
    assert made["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "runs" and made["spec"]["suspend"] is True
    spec = made["spec"]
    assert spec["entrypointNumCpus"] == 0.5  # (the driver: its loop, gateway and runner)
    head = spec["rayClusterSpec"]["headGroupSpec"]
    assert head["rayStartParams"]["num-cpus"] == "4" and head["rayStartParams"]["num-gpus"] == "1"
    (sized,) = head["template"]["spec"]["containers"]
    assert sized["resources"] == {  # (its parts' 3.5 CPUs and 3 GiB, Ray starting with 4, and room for Ray's own)
        "requests": {"cpu": "3.75", "memory": "5120Mi", "nvidia.com/gpu": 1},
        "limits": {"memory": "14Gi", "nvidia.com/gpu": 1},
    }
    assert "workerGroupSpecs" not in spec["rayClusterSpec"]
    alone = rendered(template, launch, "x", {}, "rollout", asked=demand(RunSettings({**ACCEPTANCE,
                     "channels.policy.provider": "tinker"}), EXAMPLE))  # fmt: skip
    (tinker,) = alone["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"]
    assert tinker["resources"] == {"requests": {"cpu": "0.75", "memory": "4096Mi"}, "limits": {"memory": "14Gi"}}
    assert "suspend" not in alone["spec"] and "kueue.x-k8s.io/queue-name" not in alone["metadata"]["labels"]
    two = rendered(template, launch, "x", {}, "rollout", asked=demand(RunSettings({**ACCEPTANCE,
                   "channels.policy.replicas": 2}), EXAMPLE))  # fmt: skip
    cluster = two["spec"]["rayClusterSpec"]
    assert cluster["headGroupSpec"]["rayStartParams"]["num-gpus"] == "0"  # (the driver and the bridge)
    (engines,) = cluster["workerGroupSpecs"]
    assert (engines["groupName"], engines["replicas"], engines["maxReplicas"]) == ("engines-0", 2, 2)
    assert engines["rayStartParams"] == {"num-cpus": "1", "num-gpus": "1"}
    (worker,) = engines["template"]["spec"]["containers"]
    assert worker["name"] == "ray-worker" and worker["resources"]["requests"]["nvidia.com/gpu"] == 1


class FakeKubernetes:
    """The RayJob part of a Kubernetes API server, in memory: what was created, read and deleted; and Kueue's
    Workloads, by their job's uid."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.deleted: list[str] = []
        self.tokens: set[str] = set()
        self.workloads: list[dict[str, Any]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.tokens.add(request.headers.get("authorization", ""))
        if request.url.path == "/apis/kueue.x-k8s.io/v1beta2/namespaces/rollout/workloads":
            uid = request.url.params["labelSelector"].removeprefix("kueue.x-k8s.io/job-uid=")
            items = [each for each in self.workloads if each["metadata"]["labels"]["kueue.x-k8s.io/job-uid"] == uid]
            return httpx.Response(200, json={"items": items})
        prefix = "/apis/ray.io/v1/namespaces/rollout/rayjobs"
        if not request.url.path.startswith(prefix):
            return httpx.Response(404, json={"kind": "Status", "reason": "NotFound"})
        name = request.url.path.removeprefix(prefix).strip("/")
        if request.method == "POST":
            body = json.loads(request.content)
            name = body["metadata"]["name"]
            if name in self.jobs:
                return httpx.Response(409, json={"reason": "AlreadyExists"})
            self.jobs[name] = body
            return httpx.Response(201, json=body)
        if name not in self.jobs:
            return httpx.Response(404, json={"reason": "NotFound"})
        if request.method == "DELETE":
            self.deleted.append(name)
            del self.jobs[name]
            return httpx.Response(200, json={"status": "Success"})
        return httpx.Response(200, json=self.jobs[name])


async def test_a_run_on_kubernetes_is_a_rayjob_made_read_and_deleted_through_the_api_server(tmp_path: Path) -> None:
    cluster = a_cluster(tmp_path, kubernetes=True)
    assert cluster.kubernetes is not None and cluster.kubernetes.rayjob == str(tmp_path / "rayjob.yaml")
    stores = Stores.open(cluster)
    launches = launches_of(stores.ledger)
    assert launches is not None
    server = FakeKubernetes()
    api = KubernetesApi(cluster.kubernetes.api, token="the-token", transport=httpx.MockTransport(server.handle))
    backends = {"kubernetes": RayJobResources(cluster.kubernetes, api)}
    launch = await submit(settings(), cluster, stores.ledger, backend=backends["kubernetes"])
    assert launch.state == SUBMITTED and launch.backend == "kubernetes" and server.tokens == {"Bearer the-token"}
    made = server.jobs[str(launch.job)]
    assert made["spec"]["entrypoint"] == f"python -m rollout_train.jobs {launch.id}"
    assert yaml.safe_load(made["spec"]["runtimeEnvYAML"])["env_vars"][HANDED]
    made["status"] = {"jobDeploymentStatus": "Initializing"}
    waiting = await followed(launch, launches, cluster, backends=backends)
    assert waiting.state == SUBMITTED and waiting.detail == "its Ray cluster is starting"
    made["status"] = {"jobDeploymentStatus": "Running", "jobStatus": "RUNNING"}
    running = await followed(waiting, launches, cluster, backends=backends)
    assert running.state == RUNNING
    made["status"] = {"jobDeploymentStatus": "Complete", "jobStatus": "SUCCEEDED"}
    assert (await followed(running, launches, cluster, backends=backends)).state == ENDED
    other = await submit(settings(name="other"), cluster, stores.ledger, backend=backends["kubernetes"])
    stopping = await stopped(other, launches, cluster, backends=backends)
    assert stopping.state == STOPPING and server.deleted == [other.job]
    gone = await followed(stopping, launches, cluster, backends=backends)
    assert gone.state == FAILED and "is gone" in str(gone.detail)


async def test_a_rayjob_kueue_holds_waits_for_admission_and_says_why(tmp_path: Path) -> None:
    cluster = a_cluster(tmp_path, kubernetes=True, queue=True)
    assert cluster.kubernetes is not None and cluster.kubernetes.queue == "runs"
    stores = Stores.open(cluster)
    launches = launches_of(stores.ledger)
    assert launches is not None
    server = FakeKubernetes()
    api = KubernetesApi(cluster.kubernetes.api, token="t", transport=httpx.MockTransport(server.handle))
    backends = {"kubernetes": RayJobResources(cluster.kubernetes, api)}
    launch = await submit(settings(), cluster, stores.ledger, backend=backends["kubernetes"])
    made = server.jobs[str(launch.job)]
    assert made["spec"]["suspend"] is True and made["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "runs"
    held = await followed(launch, launches, cluster, backends=backends)
    assert held.state == SUBMITTED and held.detail == "waits for admission by Kueue (queue runs)"
    made["metadata"]["uid"] = "u-1"
    pending = "couldn't assign flavors to pod set head: insufficient unused quota for nvidia.com/gpu in flavor rollout"
    server.workloads.append({
        "metadata": {"labels": {"kueue.x-k8s.io/job-uid": "u-1"}},
        "status": {"conditions": [{"type": "QuotaReserved", "status": "False", "reason": "Pending",
                                   "message": pending}]},
    })  # fmt: skip
    why = await followed(held, launches, cluster, backends=backends)
    assert why.detail == f"waits for admission by Kueue (queue runs): {pending}"
    made["spec"]["suspend"] = False  # (admitted)
    made["status"] = {"jobDeploymentStatus": "Running", "jobStatus": "RUNNING"}
    assert (await followed(why, launches, cluster, backends=backends)).state == RUNNING
