"""How the runs share what the cluster gives them: the monitor's `queue` topic (`System.queue`, `/api/queue`).

One shape, whichever says it:

    {"source": "kueue" | "ray" | None, "capacity": {RESOURCE: TOTAL}, "used": {RESOURCE: AMOUNT},
     "admitted": [{"run", "name", "requests", "since", "pods"}],
     "pending": [{"run", "name", "requests", "since", "position", "reason", "lacks", "held_by", "pods"}],
     "pods": [{"pod", "provider", "gpu", "price", "state", "run", "since", "spent"}]}

Resources are `cpu` (CPUs), `memory` (bytes) and `gpu` (Kubernetes' `nvidia.com/gpu`, Ray's `GPU`). `run` is the run's
id and `name` what the registry calls it; `requests` what it asks for in all, `since` when it was admitted or began to
wait (seconds since the epoch). A pending run's `position` is its place in the queue (1 first), `reason` why it waits as
its source says it, `lacks` what of its requests the free capacity does not hold (by resource; none where the capacity
is not known), and `held_by` the admitted runs that hold any of what it lacks, most first.

- **Kueue** (`kueue`), with `[kubernetes] queue`: the LocalQueue's ClusterQueue, its quota (`nominalQuota`, summed over
  its flavors) as `capacity` and what its admitted Workloads reserve (`flavorsReservation`) as `used`; each Workload
  of the queue that has not finished, with its owner RayJob (`job`, whose launch names the run) and its Workload's name
  (`workload`): admitted while it holds quota (`since` its admission), else pending (`since` it was made, with its
  `QuotaReserved` condition's message as `reason`). Pending Workloads are in Kueue's order where its visibility API
  serves the ClusterQueue's pending workloads (`order` is `kueue`), else in the order they were made (`order` is
  `created`). Where the API server refuses, `error` says why.
- **Pods** (`pods`), beside either: every pod the platform rents on RunPod, from its lease
  (`rollout_train.pods.leases`): its provider, GPU type, hourly price, state (`starting`, `held`, `idle`: warm), the run
  that holds it, since when, and what its run has been charged for it. Each run lists the pods it holds (`pods`). Pods
  are outside the queue's capacity: Kueue's quota covers only what runs in the cluster.
- **Ray** (`ray`), otherwise: each run whose driver beats what it asked Ray for (`demand`, the driver's and its
  placement group's resources, and `reserved`, when Ray reserved the group): admitted once its group is reserved,
  pending while its driver waits for it (`reason`: what it waits for). `capacity` is the Ray cluster's, and `used` what
  of it is taken, where this monitor's process is connected to Ray; else neither is known, and `used` is what the
  admitted runs hold.
"""

import datetime
import sys
from collections.abc import Mapping, Sequence
from typing import Any, cast

from rollout_train.launches import Launch
from rollout_train.monitor.machines import WAITING, kind_of
from rollout_train.presence import Beat, alive
from rollout_train.submitting import GPU, KUEUE, KubernetesApi

RESOURCES = {"cpu": "cpu", "memory": "memory", GPU: "gpu"}
"""Kubernetes' resources, by the name the topic gives each."""
RAY_RESOURCES = {"CPU": "cpu", "memory": "memory", "GPU": "gpu"}
VISIBILITY = "visibility.kueue.x-k8s.io/v1beta2"
"""Kueue's visibility API, which says the order of a ClusterQueue's pending Workloads."""
KUEUE_ORDER, CREATED = "kueue", "created"
"""How pending Workloads are ordered: as Kueue's visibility API says, or by when each was made."""
_SUFFIXES = {
    "Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40, "Pi": 2**50, "Ei": 2**60,
    "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15, "E": 1e18,
}  # fmt: skip


def nothing(source: str | None = None, **more: Any) -> dict[str, Any]:
    """The topic with nothing in it, from `source`."""
    return {"source": source, "capacity": {}, "used": {}, "admitted": [], "pending": [], **more}


def with_pods(queue: dict[str, Any], leases: Sequence[Any], times: Sequence[Any]) -> dict[str, Any]:
    """The topic with every pod's lease (`pods`), and each run's own pods beside it (each run's `pods`)."""
    spent: dict[str, float] = {}
    for each in times:
        if not each.closed and each.run is not None:
            spent[f"{each.pod}/{each.run}"] = spent.get(f"{each.pod}/{each.run}", 0.0) + each.dollars
    listed = [
        {"pod": lease.pod, "provider": lease.provider, "gpu": lease.gpu, "price": lease.price, "state": lease.state,
         "run": lease.run, "since": lease.released if lease.state == "idle" else lease.held,
         "spent": round(spent.get(f"{lease.pod}/{lease.run}", 0.0), 4)}
        for lease in leases
    ]  # fmt: skip
    for entry in [*queue.get("admitted", []), *queue.get("pending", [])]:
        entry["pods"] = [each for each in listed if each["run"] is not None and each["run"] == entry.get("run")]
    return {**queue, "pods": listed}


def quantity(text: object) -> float:
    """A Kubernetes quantity as a number: `500m` is 0.5, `16Gi` is 17179869184, `2e3` is 2000."""
    said = str(text).strip()
    for suffix in sorted(_SUFFIXES, key=len, reverse=True):
        if said.endswith(suffix) and said != suffix:
            return float(said.removesuffix(suffix)) * _SUFFIXES[suffix]
    return float(said)


def amounts(resources: Mapping[str, Any], names: Mapping[str, str] = RESOURCES) -> dict[str, float]:
    """The resources the topic shows, by its names (`RESOURCES`), from a mapping of quantities; the rest left out."""
    found: dict[str, float] = {}
    for key, value in resources.items():
        if key in names:
            found[names[key]] = found.get(names[key], 0.0) + quantity(value)
    return {key: round(value, 6) for key, value in found.items()}


def _added(total: dict[str, float], more: Mapping[str, float], times: float = 1.0) -> dict[str, float]:
    for key, value in more.items():
        total[key] = round(total.get(key, 0.0) + value * times, 6)
    return total


def _time(text: object) -> float | None:
    if not isinstance(text, str) or not text:
        return None
    return datetime.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def _items(value: Any) -> list[dict[str, Any]]:
    listed = cast(list[Any], value) if isinstance(value, list) else []
    return [cast(dict[str, Any], each) for each in listed if isinstance(each, dict)]


def _mapping(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def lacking(
    requests: Mapping[str, float], capacity: Mapping[str, float], used: Mapping[str, float]
) -> dict[str, float]:
    """What of `requests` the free capacity does not hold, by resource (none where the capacity is not known)."""
    found: dict[str, float] = {}
    for key, asked in requests.items():
        if key in capacity and asked > 0:
            short = asked - max(capacity[key] - used.get(key, 0.0), 0.0)
            if short > 1e-9:
                found[key] = round(short, 6)
    return found


def holders(lacks: Mapping[str, float], admitted: Sequence[Mapping[str, Any]]) -> list[str | None]:
    """The admitted runs that hold any of what a pending run lacks, those that would free most of it first (each
    resource counted as the share of what it lacks that the run holds)."""
    held = [(sum(min(each["requests"].get(key, 0.0), short) / short for key, short in lacks.items()), each["run"])
            for each in admitted]  # fmt: skip
    held = [(amount, run) for amount, run in held if amount > 0]
    held.sort(key=lambda pair: -pair[0])
    return list(dict.fromkeys(run for _, run in held))


def _waiting(pending: list[dict[str, Any]], capacity: Mapping[str, float], used: Mapping[str, float],
             admitted: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:  # fmt: skip
    for position, each in enumerate(pending, start=1):
        each["position"] = position
        each["lacks"] = lacking(each["requests"], capacity, used)
        each["held_by"] = holders(each["lacks"], admitted)
    return pending


def _condition(workload: Mapping[str, Any], kind: str) -> dict[str, Any] | None:
    for each in _items(_mapping(workload.get("status")).get("conditions")):
        if each.get("type") == kind:
            return each
    return None


def _holds(condition: Mapping[str, Any] | None) -> bool:
    return condition is not None and condition.get("status") == "True"


def workload_requests(workload: Mapping[str, Any]) -> dict[str, float]:
    """What a Workload asks for in all: what its admission assigned to each pod set (`resourceUsage`), else what Kueue
    counted of each pod set while it waits (`status.resourceRequests`), else its pod sets' containers' requests times
    their counts."""
    status = _mapping(workload.get("status"))
    total: dict[str, float] = {}
    assigned = _items(_mapping(status.get("admission")).get("podSetAssignments"))
    if assigned:
        for each in assigned:
            _added(total, amounts(_mapping(each.get("resourceUsage"))))
        return total
    counted = _items(status.get("resourceRequests"))
    if counted:
        for each in counted:
            _added(total, amounts(_mapping(each.get("resources"))))
        return total
    for pod_set in _items(_mapping(workload.get("spec")).get("podSets")):
        count = pod_set.get("count", 1)
        spec = _mapping(_mapping(pod_set.get("template")).get("spec"))
        for container in _items(spec.get("containers")):
            requests = _mapping(_mapping(container.get("resources")).get("requests"))
            _added(total, amounts(requests), float(count if isinstance(count, int) else 1))
    return total


def _owner(workload: Mapping[str, Any]) -> str | None:
    for each in _items(_mapping(workload.get("metadata")).get("ownerReferences")):
        if each.get("kind") == "RayJob":
            return str(each.get("name"))
    return None


def _quota(cluster_queue: Mapping[str, Any]) -> dict[str, float]:
    total: dict[str, float] = {}
    for group in _items(_mapping(cluster_queue.get("spec")).get("resourceGroups")):
        for flavor in _items(group.get("flavors")):
            for each in _items(flavor.get("resources")):
                name = str(each.get("name"))
                if name in RESOURCES and "nominalQuota" in each:
                    _added(total, {RESOURCES[name]: quantity(each["nominalQuota"])})
    return total


def _reserved(cluster_queue: Mapping[str, Any]) -> dict[str, float]:
    total: dict[str, float] = {}
    for flavor in _items(_mapping(cluster_queue.get("status")).get("flavorsReservation")):
        for each in _items(flavor.get("resources")):
            name = str(each.get("name"))
            if name in RESOURCES and "total" in each:
                _added(total, {RESOURCES[name]: quantity(each["total"])})
    return total


async def from_kueue(
    api: KubernetesApi, namespace: str, queue: str, launches: Sequence[Launch], names: Mapping[str, str]
) -> dict[str, Any]:
    """The topic as Kueue says it, for the LocalQueue `queue` in `namespace` (the module's docstring): each Workload's
    run found by its RayJob's launch (`launches`), called as the registry calls it (`names`)."""
    try:
        local = await api.read(f"/apis/{KUEUE}/namespaces/{namespace}/localqueues/{queue}")
        if local is None:
            return nothing("kueue", error=f"there is no LocalQueue {queue} in {namespace}")
        cluster_queue = str(_mapping(local.get("spec")).get("clusterQueue") or "")
        found = await api.read(f"/apis/{KUEUE}/clusterqueues/{cluster_queue}") if cluster_queue else None
        if found is None:
            return nothing("kueue", error=f"there is no ClusterQueue {cluster_queue or '(none named)'}")
        listed = await api.read(f"/apis/{KUEUE}/namespaces/{namespace}/workloads")
    except RuntimeError as error:
        return nothing("kueue", error=str(error))
    order = await _pending_order(api, cluster_queue)
    by_job = {each.job: each for each in launches if each.job is not None and each.backend == "kubernetes"}
    admitted: list[dict[str, Any]] = []
    pending: list[tuple[tuple[float, float], dict[str, Any]]] = []
    for workload in _items(_mapping(listed).get("items")):
        metadata = _mapping(workload.get("metadata"))
        spec = _mapping(workload.get("spec"))
        status = _mapping(workload.get("status"))
        ours = spec.get("queueName") == queue or _mapping(status.get("admission")).get("clusterQueue") == cluster_queue
        if not ours or _holds(_condition(workload, "Finished")):
            continue
        job = _owner(workload)
        launch = by_job.get(job) if job is not None else None
        run = launch.run if launch is not None else None
        name = names.get(run or "") or (launch.asked.name if launch is not None else None)
        shown: dict[str, Any] = {"run": run, "name": name or job or metadata.get("name"),
                                 "requests": workload_requests(workload), "job": job,
                                 "workload": metadata.get("name")}  # fmt: skip
        reserved = _condition(workload, "QuotaReserved")
        if _holds(reserved):
            admission = _condition(workload, "Admitted")
            since = admission if _holds(admission) else reserved
            admitted.append(shown | {"since": _time(cast(dict[str, Any], since).get("lastTransitionTime"))})
            continue
        made = _time(metadata.get("creationTimestamp"))
        reason = str((reserved or {}).get("message") or (reserved or {}).get("reason") or "")
        if spec.get("active") is False:
            reason = "deactivated"
        place = order.get(str(metadata.get("name"))) if order is not None else None
        rank = (float(place) if place is not None else float("inf"), made or 0.0)
        pending.append((rank, shown | {"since": made, "reason": reason or None}))
    capacity, used = _quota(found), _reserved(found)
    admitted.sort(key=lambda each: each["since"] or 0.0)
    waits = _waiting([each for _, each in sorted(pending, key=lambda pair: pair[0])], capacity, used, admitted)
    return {"source": "kueue", "queue": queue, "cluster_queue": cluster_queue, "capacity": capacity, "used": used,
            "admitted": admitted, "pending": waits, "order": KUEUE_ORDER if order is not None else CREATED}  # fmt: skip


async def _pending_order(api: KubernetesApi, cluster_queue: str) -> dict[str, int] | None:
    """Each pending Workload's place in the ClusterQueue, by name, as Kueue's visibility API says (none where it is not
    served, or this account may not read it)."""
    try:
        said = await api.read(f"/apis/{VISIBILITY}/clusterqueues/{cluster_queue}/pendingworkloads")
    except RuntimeError:
        return None
    if said is None:
        return None
    found: dict[str, int] = {}
    for each in _items(said.get("items")):
        place = each.get("positionInClusterQueue")
        name = _mapping(each.get("metadata")).get("name")
        if isinstance(place, int) and isinstance(name, str):
            found[name] = place
    return found


def ray_totals() -> tuple[dict[str, float], dict[str, float]] | None:
    """The Ray cluster's resources and what of them is taken, where this process is connected to Ray (none where it is
    not: Ray is not imported here, or not initialized)."""
    ray: Any = sys.modules.get("ray")
    if ray is None or not ray.is_initialized():
        return None
    total = amounts(cast(dict[str, float], ray.cluster_resources()), RAY_RESOURCES)
    free = amounts(cast(dict[str, float], ray.available_resources()), RAY_RESOURCES)
    return total, {key: round(max(value - free.get(key, 0.0), 0.0), 6) for key, value in total.items()}


def _demanded(said: Any) -> dict[str, float]:
    """A run's recorded demand (`rollout_train.demand.Resources.to_json`) in the topic's resources."""
    demand = _mapping(said)
    found = {"cpu": demand.get("cpus"), "memory": demand.get("memory_gib"), "gpu": demand.get("gpus")}
    shown = {key: float(value) for key, value in found.items() if isinstance(value, int | float)}
    if "memory" in shown:
        shown["memory"] = float(int(shown["memory"] * 2**30))
    return shown


def from_ray(
    beats: Sequence[Beat], names: Mapping[str, str], totals: tuple[dict[str, float], dict[str, float]] | None = None
) -> dict[str, Any]:
    """The topic as the runs' drivers say it in their beats (the module's docstring), with the Ray cluster's `totals`
    (its resources and what is taken) where they are known."""
    admitted: dict[str, dict[str, Any]] = {}
    pending: dict[str, dict[str, Any]] = {}
    for beat in sorted(beats, key=lambda each: each.age, reverse=True):  # (newest last: its word stands)
        said = beat.about
        run = said.get("run")
        if not alive(beat) or not isinstance(run, str) or not isinstance(said.get("demand"), dict):
            continue
        shown: dict[str, Any] = {"run": run, "name": names.get(run) or run, "requests": _demanded(said["demand"])}
        reserved = said.get("reserved")
        if isinstance(reserved, int | float):
            admitted[run] = shown | {"since": float(reserved)}
            pending.pop(run, None)
        elif kind_of(beat) == WAITING and run not in admitted:
            waits = [str(each) for each in cast(list[Any], said.get("waiting") or [])]
            since = said.get("asked")
            pending[run] = shown | {"since": float(since) if isinstance(since, int | float) else None,
                                    "reason": ("waits for " + ", ".join(waits)) if waits else None}  # fmt: skip
    if not admitted and not pending and totals is None:
        return nothing()
    held = sorted(admitted.values(), key=lambda each: each["since"])
    capacity, used = totals if totals is not None else ({}, {})
    if totals is None:
        used = dict[str, float]()
        for each in held:
            _added(used, each["requests"])
    waits = sorted(pending.values(), key=lambda each: each["since"] or 0.0)
    return {"source": "ray", "capacity": capacity, "used": used, "admitted": held,
            "pending": _waiting(waits, capacity, used, held)}  # fmt: skip
