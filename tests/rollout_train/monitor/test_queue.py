"""The queue topic: how the runs share what the cluster gives them. With Kueue, from a fake API server's LocalQueue,
ClusterQueue and Workloads (admitted, pending, finished, of another queue), in the order Kueue's visibility API says
where it is served and by when each was made where it is not, each Workload's run found by its RayJob's launch;
without, from the runs' drivers' beats, with the Ray cluster's totals where the monitor is connected to Ray."""

import datetime
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import yaml

from rollout_train.cluster import Cluster, load
from rollout_train.launches import TRAIN
from rollout_train.monitor.queue import from_ray, quantity, workload_requests
from rollout_train.monitor.system import System
from rollout_train.presence import STALE, Beat, presence_of
from rollout_train.run_settings import RunSettings
from rollout_train.stores import Stores
from rollout_train.submitting import KubernetesApi, RayJobResources, submit

pytest.importorskip("starlette")
from tests.rollout_train.support import monitor_client

WORDS = "tests.rollout_train.rollouts.games:words"
CLUSTER = """
name = "here"
[ledger]
url = "sqlite:///{root}/ledger.db"
[blobs]
directory = "{root}/blobs"
[environments."tests.rollout_train.rollouts.games:words"]
python = "platform"
[kubernetes]
namespace = "rollout"
rayjob = "rayjob.yaml"
api = "https://kubernetes.test"
queue = "runs"
"""
TEMPLATE: dict[str, Any] = {
    "apiVersion": "ray.io/v1", "kind": "RayJob", "metadata": {"labels": {"app": "run"}},
    "spec": {"rayClusterSpec": {"headGroupSpec": {"template": {"spec": {"containers": [{"name": "ray-head"}]}}}}},
}  # fmt: skip
KUEUE = "/apis/kueue.x-k8s.io/v1beta2"
VISIBILITY = "/apis/visibility.kueue.x-k8s.io/v1beta2/clusterqueues/rollout/pendingworkloads"
GIB = 2**30
CLUSTER_QUEUE: dict[str, Any] = {  # (as the live cluster's reads, with two runs' quota reserved)
    "apiVersion": "kueue.x-k8s.io/v1beta2", "kind": "ClusterQueue", "metadata": {"name": "rollout"},
    "spec": {"queueingStrategy": "BestEffortFIFO", "resourceGroups": [{
        "coveredResources": ["cpu", "memory", "nvidia.com/gpu"],
        "flavors": [{"name": "rollout", "resources": [
            {"name": "cpu", "nominalQuota": "12"}, {"name": "memory", "nominalQuota": "16Gi"},
            {"name": "nvidia.com/gpu", "nominalQuota": "1"},
        ]}],
    }]},
    "status": {
        "admittedWorkloads": 2, "pendingWorkloads": 2, "reservingWorkloads": 2,
        "flavorsReservation": [{"name": "rollout", "resources": [
            {"name": "cpu", "total": "8500m", "borrowed": "0"}, {"name": "memory", "total": "12Gi", "borrowed": "0"},
            {"name": "nvidia.com/gpu", "total": "1", "borrowed": "0"},
        ]}],
    },
}  # fmt: skip
PENDING = (
    "couldn't assign flavors to pod set head: insufficient unused quota for nvidia.com/gpu in flavor rollout, 1 more "
    "needed"
)


def workload(name: str, job: str, made: str, *, queue: str = "runs", admitted: str | None = None,
             usage: dict[str, str] | None = None, requests: dict[str, str] | None = None,
             finished: bool = False) -> dict[str, Any]:  # fmt: skip
    """A Workload as Kueue 0.19 writes it for a RayJob: admitted (at `admitted`, with what each pod set was given) or
    pending (with what Kueue counted of it, and why it waits)."""
    conditions: list[dict[str, Any]] = []
    status: dict[str, Any] = {"conditions": conditions}
    if admitted is not None:
        conditions += [
            {"type": "QuotaReserved", "status": "True", "reason": "QuotaReserved", "lastTransitionTime": admitted,
             "message": "Quota reserved in ClusterQueue rollout"},
            {"type": "Admitted", "status": "True", "reason": "Admitted", "lastTransitionTime": admitted,
             "message": "The workload is admitted"},
        ]  # fmt: skip
        status["admission"] = {"clusterQueue": "rollout", "podSetAssignments": [
            {"name": "head", "count": 1, "flavors": {"cpu": "rollout"}, "resourceUsage": usage or {}},
            {"name": "submitter", "count": 1, "resourceUsage": {"cpu": "500m", "memory": "200Mi"}},
        ]}  # fmt: skip
    else:
        conditions.append({"type": "QuotaReserved", "status": "False", "reason": "Pending", "message": PENDING,
                           "lastTransitionTime": made})  # fmt: skip
        status["resourceRequests"] = [{"name": "head", "resources": requests or {}}]
    if finished:
        conditions.append({"type": "Finished", "status": "True", "reason": "Succeeded", "lastTransitionTime": made})
    return {
        "apiVersion": "kueue.x-k8s.io/v1beta2", "kind": "Workload",
        "metadata": {"name": name, "namespace": "rollout", "creationTimestamp": made,
                     "labels": {"kueue.x-k8s.io/job-uid": f"uid-{job}"},
                     "ownerReferences": [{"apiVersion": "ray.io/v1", "kind": "RayJob", "name": job,
                                          "uid": f"uid-{job}"}]},
        "spec": {"queueName": queue, "podSets": [{"name": "head", "count": 1, "template": {"spec": {"containers": [
            {"name": "ray-head", "resources": {"requests": {"cpu": "2", "memory": "4Gi"}}}]}}}]},
        "status": status,
    }  # fmt: skip


class FakeKubernetes:
    """The parts of an API server the monitor reads and submits through: RayJobs made, the LocalQueue `runs`, the
    ClusterQueue `rollout`, the namespace's Workloads, and (with `visibility`) the ClusterQueue's pending Workloads in
    Kueue's order; `refused` answers 403 for paths that start with it."""

    def __init__(self, visibility: bool = True) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.workloads: list[dict[str, Any]] = []
        self.order: list[str] = []
        self.visibility = visibility
        self.refused: str | None = None

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if self.refused is not None and path.startswith(self.refused):
            return httpx.Response(403, json={"kind": "Status", "reason": "Forbidden", "message": "forbidden"})
        if request.method == "POST" and path == "/apis/ray.io/v1/namespaces/rollout/rayjobs":
            made = json.loads(request.content)
            self.jobs[made["metadata"]["name"]] = made
            return httpx.Response(201, json=made)
        if path == f"{KUEUE}/namespaces/rollout/localqueues/runs":
            return httpx.Response(200, json={"kind": "LocalQueue", "spec": {"clusterQueue": "rollout"}})
        if path == f"{KUEUE}/clusterqueues/rollout":
            return httpx.Response(200, json=CLUSTER_QUEUE)
        if path == f"{KUEUE}/namespaces/rollout/workloads":
            return httpx.Response(200, json={"kind": "WorkloadList", "items": self.workloads})
        if path == VISIBILITY and self.visibility:
            items = [{"metadata": {"name": name, "namespace": "rollout"}, "localQueueName": "runs", "priority": 0,
                      "positionInClusterQueue": place, "positionInLocalQueue": place}
                     for place, name in enumerate(self.order)]  # fmt: skip
            return httpx.Response(200, json={"kind": "PendingWorkloadsSummary", "items": items})
        return httpx.Response(404, json={"kind": "Status", "reason": "NotFound"})


def a_cluster(root: Path) -> Cluster:
    (root / "rayjob.yaml").write_text(yaml.safe_dump(TEMPLATE))
    path = root / "cluster.toml"
    path.write_text(CLUSTER.format(root=root))
    return load(path)


async def kueue_system(tmp_path: Path, server: FakeKubernetes) -> tuple[System, dict[str, str]]:
    """A monitor's system over a cluster with Kueue, where four runs were asked for (their RayJobs made through
    `server`), and the Workloads Kueue made of them: `alpha` and `beta` admitted, `gamma` and `delta` pending (`delta`
    made first, but behind `gamma` in Kueue's order), and a finished one; with a Workload of another queue. Each run's
    id, by its name."""
    cluster = a_cluster(tmp_path)
    assert cluster.kubernetes is not None
    stores = Stores.open(cluster)
    api = KubernetesApi(cluster.kubernetes.api, token="t", transport=httpx.MockTransport(server.handle))
    backend = RayJobResources(cluster.kubernetes, api)
    runs: dict[str, str] = {}
    jobs: dict[str, str] = {}
    for name in ("alpha", "beta", "gamma", "delta", "done"):
        launch = await submit(RunSettings({"kind": TRAIN, "name": name, "environment": WORDS}), cluster,
                              stores.ledger, backend=backend)  # fmt: skip
        assert launch.run is not None and launch.job is not None
        runs[name], jobs[name] = launch.run, launch.job
    server.workloads = [
        workload("rayjob-alpha", jobs["alpha"], "2026-10-05T17:10:00Z", admitted="2026-10-05T17:10:01Z",
                 usage={"cpu": "5", "memory": "9Gi", "nvidia.com/gpu": "1"}),
        workload("rayjob-beta", jobs["beta"], "2026-10-05T17:11:00Z", admitted="2026-10-05T17:11:02Z",
                 usage={"cpu": "2500m", "memory": "2764Mi"}),
        workload("rayjob-delta", jobs["delta"], "2026-10-05T17:12:00Z",
                 requests={"cpu": "3", "memory": "6Gi", "nvidia.com/gpu": "1"}),
        workload("rayjob-gamma", jobs["gamma"], "2026-10-05T17:13:00Z",
                 requests={"cpu": "2", "memory": "4Gi", "nvidia.com/gpu": "1"}),
        workload("rayjob-done", jobs["done"], "2026-10-05T17:00:00Z", admitted="2026-10-05T17:00:01Z",
                 usage={"cpu": "1"}, finished=True),
        workload("rayjob-other", "someone-elses", "2026-10-05T17:00:00Z", queue="elsewhere", requests={"cpu": "1"}),
    ]  # fmt: skip
    server.order = ["rayjob-gamma", "rayjob-delta"]
    return System(ledger=stores.ledger, cluster=cluster, backends={"kubernetes": backend}), runs


def test_quantities_are_numbers_of_cpus_bytes_and_gpus() -> None:
    assert [quantity("500m"), quantity("12"), quantity("16Gi"), quantity("200M"), quantity("2e3")] == [
        0.5, 12.0, 16 * GIB, 2e8, 2000.0,
    ]  # fmt: skip
    pending = workload("w", "j", "2026-10-05T17:00:00Z", requests={"cpu": "3", "memory": "6Gi"})
    assert workload_requests(pending) == {"cpu": 3.0, "memory": 6 * GIB}
    pending["status"].pop("resourceRequests")  # (not counted yet: its pod sets' requests)
    assert workload_requests(pending) == {"cpu": 2.0, "memory": 4 * GIB}


async def test_with_kueue_the_queue_is_its_quota_its_admitted_workloads_and_its_pending_ones_in_its_order(
    tmp_path: Path,
) -> None:
    server = FakeKubernetes()
    system, runs = await kueue_system(tmp_path, server)
    shown = await system.queue()

    assert shown["source"] == "kueue" and shown["order"] == "kueue" and "error" not in shown
    assert (shown["queue"], shown["cluster_queue"]) == ("runs", "rollout")
    assert shown["capacity"] == {"cpu": 12.0, "memory": 16 * GIB, "gpu": 1.0}
    assert shown["used"] == {"cpu": 8.5, "memory": 12 * GIB, "gpu": 1.0}
    alpha, beta = shown["admitted"]
    assert alpha == alpha | {"run": runs["alpha"], "name": "alpha", "workload": "rayjob-alpha"}
    assert alpha["requests"] == {"cpu": 5.5, "memory": 9 * GIB + 200 * 2**20, "gpu": 1.0}  # (with the submitter)
    assert alpha["since"] == datetime.datetime(2026, 10, 5, 17, 10, 1, tzinfo=datetime.UTC).timestamp()
    assert beta["run"] == runs["beta"] and "gpu" not in beta["requests"]
    gamma, delta = shown["pending"]  # (Kueue's order, not the order they were made in)
    assert gamma == gamma | {"run": runs["gamma"], "name": "gamma", "position": 1, "reason": PENDING}
    assert gamma["requests"] == {"cpu": 2.0, "memory": 4 * GIB, "gpu": 1.0}
    assert gamma["lacks"] == {"gpu": 1.0}  # (its memory fits what is free)
    assert gamma["held_by"] == [runs["alpha"]]  # (the one admitted run that holds a GPU)
    assert delta == delta | {"run": runs["delta"], "position": 2}
    assert delta["lacks"] == {"gpu": 1.0, "memory": 2 * GIB} and delta["held_by"] == [runs["alpha"], runs["beta"]]
    names = {each["name"] for each in [*shown["admitted"], *shown["pending"]]}
    assert "done" not in names and "someone-elses" not in names  # (finished; of another queue)

    async with monitor_client(f"sqlite:///{tmp_path}/ledger.db", beat=0.0) as client:  # (with no cluster config)
        answer = await client.get("/api/queue")
        assert answer.status_code == 200 and answer.json()["source"] is None


async def test_without_kueues_visibility_api_pending_workloads_are_in_the_order_they_were_made(tmp_path: Path) -> None:
    server = FakeKubernetes(visibility=False)
    system, runs = await kueue_system(tmp_path, server)
    shown = await system.queue()
    assert shown["order"] == "created"
    assert [(each["run"], each["position"]) for each in shown["pending"]] == [(runs["delta"], 1), (runs["gamma"], 2)]
    server.visibility, server.refused = True, "/apis/visibility.kueue.x-k8s.io"  # (served, but not to this account)
    assert (await system.queue())["order"] == "created"


async def test_an_api_server_that_refuses_is_said_and_a_run_that_ends_frees_its_share(tmp_path: Path) -> None:
    server = FakeKubernetes()
    system, runs = await kueue_system(tmp_path, server)
    server.workloads[0]["status"]["conditions"].append({"type": "Finished", "status": "True"})
    shown = await system.queue()
    assert [each["run"] for each in shown["admitted"]] == [runs["beta"]]
    server.refused = f"{KUEUE}/clusterqueues"
    refused = await system.queue()
    assert refused["source"] == "kueue" and "403" in refused["error"] and refused["admitted"] == []


def beat(name: str, about: dict[str, Any], *, age: float = 1.0) -> Beat:
    return Beat(name, time.time() - age, about, age=age)


def test_without_kueue_the_runs_drivers_say_what_they_hold_and_wait_for() -> None:
    held: dict[str, Any] = {"cpus": 4.0, "memory_gib": 6.0, "gpus": 1.0, "custom": {}}
    beats = [
        beat("run/a", {"run": "a", "demand": held, "asked": 100.0, "reserved": 110.0, "places": 2}),
        beat("run/b", {"kind": "run", "run": "b", "demand": {"cpus": 2.0, "memory_gib": 2.0, "gpus": 1.0},
                       "asked": 120.0, "reserved": None,
                       "waiting": ["run/b/engine/policy/0 (1 GPU: pending creation)"]}),
        beat("run/old", {"run": "old", "demand": held, "reserved": 50.0}, age=STALE + 1),  # (gone: it holds nothing)
        beat("gpu-1/runner", {"run": "a", "places": 6}),  # (a runner of the cluster's: no demand of its own)
    ]  # fmt: skip
    names = {"a": "alpha", "b": "beta"}
    alone = from_ray(beats, names)
    assert alone["source"] == "ray" and alone["capacity"] == {}
    (alpha,) = alone["admitted"]
    assert alpha == {"run": "a", "name": "alpha", "requests": {"cpu": 4.0, "memory": 6.0 * GIB, "gpu": 1.0},
                     "since": 110.0}  # fmt: skip
    assert alone["used"] == alpha["requests"]  # (the totals are not known: what the runs hold)
    (waiting,) = alone["pending"]
    assert waiting == waiting | {"run": "b", "name": "beta", "position": 1, "since": 120.0, "lacks": {}}
    assert waiting["reason"] == "waits for run/b/engine/policy/0 (1 GPU: pending creation)"

    totals = ({"cpu": 8.0, "memory": 16.0 * GIB, "gpu": 1.0}, {"cpu": 4.0, "memory": 6.0 * GIB, "gpu": 1.0})
    known = from_ray(beats, names, totals)
    assert known["capacity"] == totals[0] and known["used"] == totals[1]
    (waiting,) = known["pending"]
    assert waiting["lacks"] == {"gpu": 1.0} and waiting["held_by"] == ["a"]
    assert from_ray([], names) == {"source": None, "capacity": {}, "used": {}, "admitted": [], "pending": []}


async def test_a_monitor_connected_to_ray_says_its_totals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rollout_train.ledger import FileLedger

    ledger = FileLedger(tmp_path / "ledger")
    presence = presence_of(ledger)
    assert presence is not None
    await presence.beat("run/a", {"run": "a", "demand": {"cpus": 3.0, "memory_gib": 1.0, "gpus": 0.0},
                                  "asked": 1.0, "reserved": 2.0})  # fmt: skip
    ray = SimpleNamespace(
        is_initialized=lambda: True,
        cluster_resources=lambda: {"CPU": 16.0, "GPU": 1.0, "memory": 32.0 * GIB, "object_store_memory": 1.0},
        available_resources=lambda: {"CPU": 13.0, "GPU": 1.0, "memory": 31.0 * GIB},
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    shown = await System(ledger=ledger).queue()
    assert shown["capacity"] == {"cpu": 16.0, "gpu": 1.0, "memory": 32.0 * GIB}
    assert shown["used"] == {"cpu": 3.0, "gpu": 0.0, "memory": 1.0 * GIB}
    assert [each["run"] for each in shown["admitted"]] == ["a"]
