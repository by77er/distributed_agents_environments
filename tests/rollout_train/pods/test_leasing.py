"""Leasing RunPod's pods, against a fake of RunPod's API (nothing rented, nothing spent): a run claims pods (started, or
taken warm), renews and releases them; a released pod stays warm and is taken by the next run with its image and model,
reset to that run; the reaper deletes idle pods past their idle stop, stale leases' pods and pods no lease names; a pod
not ready in time is deleted and the run told why; and what each run held is charged at the pod's price. A lease
changed under its run (by hand, or between a read and a compare-and-set) is still renewed, and still reaped where it is
stale; a resumed run frees a slot its earlier start held whose pod is gone; the reaper never deletes a lease renewed
meanwhile."""

import asyncio
import json
import tomllib
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from rollout_train.cluster import Cluster, ClusterError, parsed
from rollout_train.database import DatabaseLedger
from rollout_train.pods.identity import POD, pod_identity
from rollout_train.pods.leases import HELD, IDLE, DatabasePodLeases, PodLease, pod_leases_of
from rollout_train.pods.leasing import LeaseLost, PodNeed, Pods, PodsDidNotStart, needs_of, reap, tag_of
from rollout_train.presence import presence_of
from rollout_train.providers import pod_table
from rollout_train.run_settings import RunSettings
from rollout_train.testing import LEDGER_TOKEN
from tests.rollout_runpod.fake import KEY, FakeRunPod

ENVIRON = {"ROLLOUT_LEDGER_TOKEN": LEDGER_TOKEN, "RUNPOD_API_KEY": KEY}


def cluster_of(
    directory: Path, options: str = 'max_lora_rank = 32, args = "--max-model-len 8192"', **more: Any
) -> Cluster:
    table = {"idle_stop": 600, "start_timeout": 20, "max_pods": 2, "price": 0.5, **more}
    said = "\n".join(f"{key} = {json.dumps(value)}" for key, value in table.items())
    return parsed(
        tomllib.loads(f"""
name = "test"
[ledger]
url = "sqlite:///{directory / "ledger.db"}"
token_env = "ROLLOUT_LEDGER_TOKEN"
public = "https://ledger.example.com"
[tls]
ca = "~/ca.pem"
certificate = "~/gateway.crt"
key = "~/gateway.key"
[inference.pods]
kind = "runpod-inference"
image = "ghcr.io/by77er/rollout-inference@sha256:0"
gpu_types = ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000"]
cloud = "community"
regions = ["EU-RO-1"]
{said}
[inference.pods.models."m"]
context = 8192
options = {{ {options} }}
""")
    )


class StandIns:
    """Processes standing in for what runs on each pod the fake starts: each beats as its pod, ready for the run its
    lease names (unless `ready` says it never is)."""

    def __init__(self, ledger: DatabaseLedger, *, ready: bool = True, says: dict[str, JsonValue] | None = None) -> None:
        self.ledger = ledger
        self.ready = ready
        self.says = says or {}
        """More that each beat says of its pod (where it claims to be reached, say)."""
        self.tasks: dict[str, asyncio.Task[None]] = {}

    def created(self, pod: dict[str, Any], body: dict[str, Any]) -> None:
        self.tasks[pod["id"]] = asyncio.get_running_loop().create_task(self._beating(body["name"]))

    def deleted(self, id: str) -> None:
        if (task := self.tasks.pop(id, None)) is not None:
            task.cancel()

    async def _beating(self, name: str) -> None:
        leases, presence = pod_leases_of(self.ledger), presence_of(self.ledger)
        assert leases is not None and presence is not None
        while True:
            lease = await leases.get(name)
            run = lease.run if lease is not None and lease.state != IDLE else None
            pod: dict[str, JsonValue] = {
                "name": name,
                "identity": pod_identity(name),
                "role": "inference",
                "ready": self.ready and run is not None,
                "run": run,
                "serial": f"s-{name}",
                **self.says,
            }
            await presence.beat(name, {POD: pod})
            await asyncio.sleep(0.05)


@pytest.fixture
async def world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[DatabaseLedger, FakeRunPod, StandIns]]:
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    ledger = DatabaseLedger(f"sqlite:///{tmp_path / 'ledger.db'}")
    stand_ins = StandIns(ledger)
    fake = FakeRunPod(key=KEY, cost=0.79, created=stand_ins.created, deleted=stand_ins.deleted)
    yield ledger, fake, stand_ins
    for task in stand_ins.tasks.values():
        task.cancel()


def pods_of(run: str, cluster: Cluster, ledger: DatabaseLedger, fake: FakeRunPod, **options: Any) -> Pods:
    return Pods(run, cluster, ledger, api=lambda provider: fake.client(), ca=lambda table: None, environ=ENVIRON,
                look=0.05, **options)  # fmt: skip


NEED = PodNeed("pods", "inference", 1, "m", "policy")


async def test_a_renewal_follows_a_pod_runpod_maps_to_another_port(
    tmp_path: Path, world: tuple[DatabaseLedger, FakeRunPod, StandIns]
) -> None:
    # (a pod given another image in place, or started again, may come back on another public port)
    ledger, fake, _ = world
    pods = pods_of("run_1", cluster_of(tmp_path), ledger, fake)
    (lease,) = await pods.claim([NEED])
    assert lease.address == "https://203.0.113.7:40123" and lease.id is not None
    fake.pods[lease.id]["portMappings"] = {"8443": 40999}
    await pods.renewed()
    store = pod_leases_of(ledger)
    assert store is not None
    assert (await store.get(lease.pod)).address == "https://203.0.113.7:40999"  # type: ignore[union-attr]
    await pods.release()


async def test_a_pod_is_asked_for_its_providers_least_cpus_and_memory_and_its_lease_says_what_it_got(
    tmp_path: Path, world: tuple[DatabaseLedger, FakeRunPod, StandIns]
) -> None:
    ledger, fake, _ = world
    pods = pods_of("run_1", cluster_of(tmp_path, min_vcpus_per_gpu=12, min_memory_gb_per_gpu=96), ledger, fake)
    (lease,) = await pods.claim([NEED])
    (body,) = fake.created_bodies()
    assert (body["minVCPUPerGPU"], body["minRAMPerGPU"]) == (12, 96)
    assert (lease.vcpus, lease.memory_gb) == (16, 188.0)  # (as RunPod's answer said)
    assert lease.id is not None
    fake.pods[lease.id]["vcpuCount"] = 24  # (RunPod says more of it now)
    await pods.renewed()
    store = pod_leases_of(ledger)
    assert store is not None
    renewed = await store.get(lease.pod)
    assert renewed is not None and (renewed.vcpus, renewed.memory_gb) == (24, 188.0)
    await pods.release()
    plain = cluster_of(tmp_path)
    table = pod_table("runpod-inference", plain.inference["pods"].settings)
    spec = pods_of("run_2", plain, ledger, fake)._spec(lease, table, NEED)  # pyright: ignore[reportPrivateUsage]
    assert "minVCPUPerGPU" not in spec.body()  # (unsaid: RunPod's defaults)


async def test_a_pod_whose_provider_says_its_thinking_starts_vllm_bounding_it(
    tmp_path: Path, world: tuple[DatabaseLedger, FakeRunPod, StandIns]
) -> None:
    ledger, fake, _ = world
    reasoning = 'reasoning = { parser = "qwen3", open = "<think>", close = "\\n</think>\\n\\n" }'
    cluster = cluster_of(tmp_path, f'max_lora_rank = 32, args = "--max-model-len 8192", {reasoning}')
    pods = pods_of("run_1", cluster, ledger, fake)
    await pods.claim([NEED])
    (body,) = fake.created_bodies()
    assert body["env"]["VLLM_REASONING_PARSER"] == "qwen3"
    config = json.loads(body["env"]["VLLM_REASONING_CONFIG"])
    assert config == {"reasoning_start_str": "<think>", "reasoning_end_str": "\n</think>\n\n"}
    await pods.release()
    with pytest.raises(ClusterError, match="reasoning is a table of parser"):
        cluster_of(tmp_path, 'reasoning = { parser = "qwen3", open = "<think>" }')


async def test_what_pods_cost_while_the_run_waited_for_them_is_told_with_the_first_renewal(
    tmp_path: Path, world: tuple[DatabaseLedger, FakeRunPod, StandIns]
) -> None:
    ledger, fake, _ = world
    pods = pods_of("run_1", cluster_of(tmp_path), ledger, fake, renew=0.05)
    await pods.claim([NEED])
    await asyncio.sleep(0.1)
    await pods.renewed()  # (as the wait for a pod renews its leases)
    waited = pods.spent
    told: list[float] = []

    async def spent(dollars: float) -> None:
        told.append(dollars)

    renewing = asyncio.create_task(pods.renewing(spent))
    await asyncio.sleep(0.3)
    renewing.cancel()
    await asyncio.gather(renewing, return_exceptions=True)
    assert waited > 0 and told and told[0] > waited  # (the first told has what the wait cost)
    assert sum(told) == pytest.approx(pods.spent)
    await pods.release()


async def test_a_pod_claimed_while_the_run_renews_its_others_is_renewed_by_them_alone(
    tmp_path: Path, world: tuple[DatabaseLedger, FakeRunPod, StandIns]
) -> None:
    ledger, fake, _ = world
    pods = pods_of("run_1", cluster_of(tmp_path), ledger, fake, renew=0.05)
    (first,) = await pods.claim([NEED])
    told: list[float] = []

    async def spent(dollars: float) -> None:
        told.append(dollars)

    renewing = asyncio.create_task(pods.renewing(spent))
    try:
        await asyncio.sleep(0.1)
        later = await pods.claim([NEED])  # (as a trainer's pod is claimed once a step is coming)
        assert len(later) == 2 and first.pod in {each.pod for each in later} and len(pods.ready) == 2
        await asyncio.sleep(0.2)
        assert sum(told) == pytest.approx(pods.spent)  # (the later pod's time told too)
    finally:
        renewing.cancel()
        await asyncio.gather(renewing, return_exceptions=True)
        await pods.release()


async def test_a_run_starts_a_pod_renews_it_and_releases_it_warm(
    tmp_path: Path, world: tuple[DatabaseLedger, FakeRunPod, StandIns]
) -> None:
    ledger, fake, _ = world
    cluster = cluster_of(tmp_path)
    pods = pods_of("run_1", cluster, ledger, fake)
    (lease,) = await pods.claim([NEED])
    assert lease.state == HELD and lease.run == "run_1" and lease.channel == "policy" and lease.id in fake.pods
    assert lease.pod.startswith(tag_of(cluster) + "pods-0-") and lease.price == 0.79  # (RunPod's price for it)
    assert lease.address == "https://203.0.113.7:40123"  # (where RunPod says it is reached)
    assert (lease.gpu, lease.cloud) == ("NVIDIA GeForce RTX 4090", "COMMUNITY")
    (body,) = fake.created_bodies()
    env = body["env"]
    assert body["allowedCudaVersions"] == ["13.0"]  # (a machine whose driver runs the images' CUDA 13)
    assert (body["cloudType"], body["dataCenterIds"], body["gpuTypeIds"]) == (
        "COMMUNITY", ["EU-RO-1"], ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000"])  # fmt: skip
    assert json.loads(env["ROLLOUT_LEDGER"]) == {
        "kind": "rollout_train.ledger_service:HttpLedger",
        "url": "https://ledger.example.com",
        "token_env": "ROLLOUT_LEDGER_TOKEN",
    }
    assert env["ROLLOUT_LEDGER_TOKEN"] == lease.token and lease.token != LEDGER_TOKEN
    assert env["VLLM_ARGS"] == "--max-model-len 8192 --max-logprobs 20" and env["VLLM_MAX_LORA_RANK"] == "32"
    assert env["ROLLOUT_MODEL"] == "m" and env["ROLLOUT_ROLE"] == "inference"
    store = pod_leases_of(ledger)
    assert store is not None
    there = await store.get(lease.pod)
    assert there is not None
    await store.put(replace(there, address=None), expect=there.version)  # (as a lease from before addresses were kept)
    before = (await store.get(lease.pod)).renewed  # type: ignore[union-attr]
    await asyncio.sleep(0.1)
    added = await pods.renewed()
    assert (await store.get(lease.pod)).address == "https://203.0.113.7:40123"  # type: ignore[union-attr]
    assert (await store.get(lease.pod)).renewed > before and added > 0  # type: ignore[union-attr,operator]
    await pods.release()
    released = await store.get(lease.pod)
    assert released is not None and (released.state, released.run, released.token) == (IDLE, None, None)
    assert lease.id in fake.pods  # (warm)
    (spent,) = await store.times("run_1")
    assert spent.released and not spent.closed and spent.dollars == pytest.approx(spent.seconds * 0.79 / 3600)


async def test_the_next_run_takes_the_warm_pod_reset_to_it_and_the_first_is_charged_its_warm_time(
    tmp_path: Path, world: tuple[DatabaseLedger, FakeRunPod, StandIns]
) -> None:
    ledger, fake, _ = world
    cluster = cluster_of(tmp_path)
    first = pods_of("run_1", cluster, ledger, fake)
    (held,) = await first.claim([NEED])
    await first.release()
    await asyncio.sleep(0.1)
    second = pods_of("run_2", cluster, ledger, fake)
    (taken,) = await second.claim([PodNeed("pods", "inference", 1, "m", "opponent")])
    assert taken.pod == held.pod and taken.id == held.id and len(fake.created_bodies()) == 1  # (no cold start)
    assert (taken.run, taken.channel, taken.state) == ("run_2", "opponent", HELD) and taken.token != held.token
    store = pod_leases_of(ledger)
    assert store is not None
    (charged,) = await store.times("run_1")
    assert charged.closed and charged.idle >= 0.1  # (its warm time, until run_2 took it)
    (holding,) = await store.times("run_2")
    assert not holding.closed and holding.since >= charged.until
    other = pods_of("run_3", cluster, ledger, fake)
    (started,) = await other.claim([PodNeed("pods", "inference", 1, "another-model", "policy")])
    assert started.pod != held.pod and started.slot == 1  # (another model: a pod of its own)
    await asyncio.gather(second.release(), other.release())


async def test_the_reaper_deletes_idle_pods_past_their_idle_stop_and_stale_leases_and_orphans(
    tmp_path: Path, world: tuple[DatabaseLedger, FakeRunPod, StandIns]
) -> None:
    ledger, fake, _ = world
    cluster = cluster_of(tmp_path, idle_stop=0.2)
    store = pod_leases_of(ledger)
    assert store is not None
    warm = pods_of("run_1", cluster, ledger, fake)
    (idle,) = await warm.claim([NEED])
    await warm.release()
    assert await reap(cluster, ledger, api=lambda provider: fake.client(), ca=lambda table: None) == []  # (warm yet)
    await asyncio.sleep(0.3)
    (said,) = await reap(cluster, ledger, api=lambda provider: fake.client(), ca=lambda table: None)
    assert "no run held it" in said and idle.id not in fake.pods and await store.get(idle.pod) is None
    (charged,) = await store.times("run_1")
    assert charged.closed and charged.idle >= 0.2
    crashed = pods_of("run_2", cluster, ledger, fake)
    (stale,) = await crashed.claim([NEED])
    await asyncio.sleep(0.1)
    (said,) = await reap(cluster, ledger, api=lambda provider: fake.client(), ca=lambda table: None, stale=0.05)
    assert "was not renewed" in said and stale.id not in fake.pods and await store.get(stale.pod) is None
    with pytest.raises(LeaseLost):
        await crashed.renewed()
    client = fake.client()
    from rollout_runpod import PodSpec

    orphan = await client.create(PodSpec(tag_of(cluster) + "pods-5-abcdef", "image", ["NVIDIA GeForce RTX 4090"]))
    stranger = await client.create(PodSpec("someone-elses-pod", "image", ["NVIDIA GeForce RTX 4090"]))
    (said,) = await reap(cluster, ledger, api=lambda provider: fake.client(), ca=lambda table: None)
    assert "no lease names it" in said and orphan.id not in fake.pods and stranger.id in fake.pods


async def test_a_pod_not_ready_in_time_is_deleted_and_the_run_told_why(
    tmp_path: Path, world: tuple[DatabaseLedger, FakeRunPod, StandIns]
) -> None:
    ledger, fake, stand_ins = world
    stand_ins.ready = False
    told: list[list[str]] = []

    async def heard(waits: Any) -> None:
        told.append(list(waits))

    pods = pods_of("run_1", cluster_of(tmp_path, start_timeout=1), ledger, fake, told=heard)
    with pytest.raises(PodsDidNotStart, match=r"did not say it was ready within 1 seconds; it was deleted"):
        await pods.claim([NEED])
    assert fake.pods == {} and await pod_leases_of(ledger).all() == []  # type: ignore[union-attr]
    assert any("up, not ready" in each[0] for each in told)


async def test_a_providers_max_pods_bounds_its_pods_and_a_run_waits_for_one(
    tmp_path: Path, world: tuple[DatabaseLedger, FakeRunPod, StandIns]
) -> None:
    ledger, fake, _ = world
    cluster = cluster_of(tmp_path, max_pods=1)
    told: list[str] = []

    async def heard(waits: Any) -> None:
        told.extend(waits)

    first = pods_of("run_1", cluster, ledger, fake)
    await first.claim([NEED])
    second = pods_of("run_2", cluster, ledger, fake, told=heard)
    waiting = asyncio.ensure_future(second.claim([NEED]))
    await asyncio.sleep(0.3)
    assert not waiting.done() and any("all 1 of its max_pods are taken" in each for each in told)
    await first.release()
    (taken,) = await asyncio.wait_for(waiting, 5)
    assert taken.run == "run_2" and len(fake.pods) == 1
    await second.release()


async def test_slots_hold_one_lease_each_whoever_writes_first(tmp_path: Path) -> None:
    from rollout_train.ledger_service import Conflict

    store = pod_leases_of(DatabaseLedger(f"sqlite:///{tmp_path / 'ledger.db'}"))
    assert store is not None
    one = PodLease("pod-a", "pods", 0, "inference", "image", "m", "gpu", 1.0)
    made = await store.put(one, expect=None)
    assert made.version == 1
    with pytest.raises(Conflict):
        await store.put(PodLease("pod-b", "pods", 0, "inference", "image", "m", "gpu", 1.0), expect=None)
    with pytest.raises(Conflict):
        await store.put(made, expect=0)  # (changed since it was read)
    assert (await store.put(made, expect=1)).version == 2
    with pytest.raises(Conflict):
        await store.delete("pod-a", expect=1)
    await store.delete("pod-a", expect=2)


def test_a_runs_settings_say_the_pods_it_needs(tmp_path: Path) -> None:
    cluster = cluster_of(tmp_path)
    settings = RunSettings({"kind": "eval", "channels.policy.provider": "pods", "channels.policy.model": "m",
                            "channels.policy.replicas": 2, "channels.policy.renderer": "r:r"})  # fmt: skip
    assert needs_of(settings, cluster) == [PodNeed("pods", "inference", 2, "m", "policy")]


async def test_a_pods_token_reads_only_while_its_lease_names_its_run(
    tmp_path: Path, world: tuple[DatabaseLedger, FakeRunPod, StandIns]
) -> None:
    from rollout_train.ledger_service import Forbidden
    from rollout_train.ledger_service.service import held_by_run
    from rollout_train.record import STARTS, table
    from rollout_train.testing import served_ledger

    ledger, fake, _ = world
    cluster = cluster_of(tmp_path)
    fence = await ledger.take("all")
    await ledger.append(table("run_1", STARTS), "1", {"run_settings": {}}, fence)
    pods = pods_of("run_1", cluster, ledger, fake)
    (lease,) = await pods.claim([NEED])
    assert lease.token is not None
    pod = served_ledger(ledger, token=lease.token, honoured=held_by_run)
    assert await pod.read(table("run_1", STARTS))
    own = await pod.pods.get(lease.pod)
    assert own is not None and own.token == lease.token
    with pytest.raises(Forbidden):
        await pod.pods.get("another-pod")
    with pytest.raises(Forbidden):
        await pod.pods.all()
    await pods.release()
    with pytest.raises(Forbidden):  # (released: its run's records are no longer its)
        await pod.read(table("run_1", STARTS))
    await pod.presence.beat(lease.pod, {})  # (it still beats, warm)


Factory = Callable[..., Any]


async def test_the_queue_shows_every_pod_and_each_runs_own(
    tmp_path: Path, world: tuple[DatabaseLedger, FakeRunPod, StandIns]
) -> None:
    from rollout_train.monitor.queue import with_pods

    ledger, fake, _ = world
    store = pod_leases_of(ledger)
    assert store is not None
    held = pods_of("run_1", cluster_of(tmp_path), ledger, fake)
    (lease,) = await held.claim([NEED])
    await held.renewed()
    queue = {"source": "kueue", "admitted": [{"run": "run_1"}, {"run": "run_2"}], "pending": []}
    shown = with_pods(queue, await store.all(), await store.times())
    (pod,) = shown["pods"]
    assert (pod["pod"], pod["run"], pod["state"], pod["price"]) == (lease.pod, "run_1", "held", 0.79)
    assert pod["spent"] >= 0 and "token" not in pod
    assert [each["pod"] for each in shown["admitted"][0]["pods"]] == [lease.pod] and shown["admitted"][1]["pods"] == []
    await held.release()


def edited_by_hand(ledger: DatabaseLedger, pod: str, **changed: Any) -> None:
    """Change a lease's row as by hand in the database: `changed` into its `lease` column, with the version it was
    read at in there too, and its `version` column bumped."""
    from rollout_train.sql import fetch_one, sql

    def edited(connection: Any) -> None:
        found = fetch_one(connection, "SELECT version, lease FROM pod_leases WHERE pod = :pod", {"pod": pod})
        assert found is not None
        lease = {**json.loads(found[1]), "version": found[0], **changed}
        sql(connection, "UPDATE pod_leases SET version = version + 1, lease = :lease WHERE pod = :pod",
            {"pod": pod, "lease": json.dumps(lease)})  # fmt: skip

    ledger.database.write(edited)


async def test_a_lease_whose_version_moved_under_the_run_is_still_renewed(
    tmp_path: Path,
    world: tuple[DatabaseLedger, FakeRunPod, StandIns],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ledger, fake, _ = world
    store = pod_leases_of(ledger)
    assert store is not None
    pods = pods_of("run_1", cluster_of(tmp_path), ledger, fake)
    (lease,) = await pods.claim([NEED])
    held = await store.get(lease.pod)
    assert held is not None and held.renewed is not None
    edited_by_hand(ledger, lease.pod, address="https://203.0.113.7:40999")  # (its row's version column moved alone)
    edited = await store.get(lease.pod)
    assert edited is not None and edited.version == held.version + 1  # (the column's: what a compare-and-set compares)
    await asyncio.sleep(0.05)
    await pods.renewed()
    renewed = await store.get(lease.pod)
    assert renewed is not None and renewed.renewed is not None
    assert renewed.version == edited.version + 1 and renewed.renewed > held.renewed
    assert renewed.address == lease.address  # (RunPod's, again)
    put = DatabasePodLeases.put
    raced: list[str] = []

    async def racing(self: DatabasePodLeases, lease: PodLease, *, expect: int | None) -> PodLease:
        if not raced:  # (another writer changes the lease between the run's read and its compare-and-set)
            raced.append(lease.pod)
            await put(self, lease, expect=expect)
        return await put(self, lease, expect=expect)

    monkeypatch.setattr(DatabasePodLeases, "put", racing)
    with caplog.at_level("WARNING", logger="rollout_train.pods.leasing"):
        await pods.renewed()
    assert raced == [lease.pod] and f"the lease of pod {lease.pod} changed since run run_1 read it" in caplog.text
    again = await store.get(lease.pod)
    assert again is not None and again.version == renewed.version + 2 and again.run == "run_1"
    await pods.release()


async def test_a_stale_lease_whose_pod_is_gone_frees_its_slot_after_a_version_change_and_the_run_resumes(
    tmp_path: Path, world: tuple[DatabaseLedger, FakeRunPod, StandIns], monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, fake, _ = world
    cluster = cluster_of(tmp_path, max_pods=1)
    store = pod_leases_of(ledger)
    assert store is not None
    crashed = pods_of("run_1", cluster, ledger, fake)
    (lease,) = await crashed.claim([NEED])  # (its driver dies: no more renewals)
    assert lease.id is not None
    await fake.client().terminate(lease.id)  # (its pod deleted at RunPod)
    edited_by_hand(ledger, lease.pod, address="https://203.0.113.7:40999")
    delete = DatabasePodLeases.delete
    raced: list[str] = []

    async def racing(self: DatabasePodLeases, pod: str, *, expect: int) -> None:
        if not raced:  # (the lease is changed again between the reaper's read and its compare-and-set)
            raced.append(pod)
            edited_by_hand(ledger, pod, address="https://203.0.113.7:41000")
        await delete(self, pod, expect=expect)

    monkeypatch.setattr(DatabasePodLeases, "delete", racing)
    await asyncio.sleep(0.1)
    (said,) = await reap(cluster, ledger, api=lambda provider: fake.client(), ca=lambda table: None, stale=0.05)
    assert raced == [lease.pod] and said.startswith(f"deleted {lease.pod} of pods: its lease was not renewed")
    assert await store.all() == []  # (its slot is free)
    resumed = pods_of("run_1", cluster, ledger, fake)
    (started,) = await asyncio.wait_for(resumed.claim([NEED]), 5)
    assert started.pod != lease.pod and started.slot == 0 and started.run == "run_1" and started.id in fake.pods
    await resumed.release()


async def test_a_resumed_run_frees_a_slot_its_earlier_start_held_whose_pod_is_gone(
    tmp_path: Path, world: tuple[DatabaseLedger, FakeRunPod, StandIns], caplog: pytest.LogCaptureFixture
) -> None:
    ledger, fake, _ = world
    cluster = cluster_of(tmp_path, max_pods=1)
    store = pod_leases_of(ledger)
    assert store is not None
    crashed = pods_of("run_1", cluster, ledger, fake)
    (lease,) = await crashed.claim([NEED])
    assert lease.id is not None
    await fake.client().terminate(lease.id)
    other = pods_of("run_2", cluster, ledger, fake)
    with pytest.raises(TimeoutError):  # (another run's lease, renewed not long ago: not its to free)
        await asyncio.wait_for(other.claim([NEED]), 0.3)
    resumed = pods_of("run_1", cluster, ledger, fake)
    with caplog.at_level("WARNING", logger="rollout_train.pods.leasing"):
        (started,) = await asyncio.wait_for(resumed.claim([NEED]), 5)
    assert f"run run_1 frees slot 0 of pods: an earlier start of run run_1 held pod {lease.pod}" in caplog.text
    assert started.pod != lease.pod and await store.get(lease.pod) is None and started.id in fake.pods
    await resumed.release()


async def test_the_reaper_never_frees_a_lease_renewed_meanwhile(
    tmp_path: Path, world: tuple[DatabaseLedger, FakeRunPod, StandIns], monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, fake, _ = world
    cluster = cluster_of(tmp_path)
    store = pod_leases_of(ledger)
    assert store is not None
    pods = pods_of("run_1", cluster, ledger, fake)
    (lease,) = await pods.claim([NEED])
    delete = DatabasePodLeases.delete
    raced: list[str] = []

    async def racing(self: DatabasePodLeases, pod: str, *, expect: int) -> None:
        if not raced:  # (the run renews between the reaper's read and its compare-and-set)
            raced.append(pod)
            await pods.renewed()
        await delete(self, pod, expect=expect)

    monkeypatch.setattr(DatabasePodLeases, "delete", racing)
    await asyncio.sleep(0.4)
    assert await reap(cluster, ledger, api=lambda provider: fake.client(), ca=lambda table: None, stale=0.3) == []
    there = await store.get(lease.pod)
    assert raced == [lease.pod] and there is not None and there.run == "run_1" and lease.id in fake.pods
    await pods.renewed()  # (its lease is the run's yet)
    await pods.release()


FOUR = """
[trainers.four]
kind = "runpod-trainer"
trainer = "full"
image = "ghcr.io/by77er/rollout-trainer@sha256:0"
gpu_types = ["NVIDIA H100 80GB HBM3"]
gpu_count = 4
models = ["m"]
"""


async def test_a_trainer_pod_of_several_gpus_is_rented_with_them_and_its_trainer_told_how_many(
    tmp_path: Path, world: tuple[DatabaseLedger, FakeRunPod, StandIns]
) -> None:
    ledger, fake, _ = world
    base = cluster_of(tmp_path)
    trainers = parsed(tomllib.loads(f'name = "test"\n[ledger]\nurl = "sqlite:///{tmp_path / "x.db"}"\n{FOUR}')).trainers
    cluster = replace(base, trainers=trainers)
    need = PodNeed("four", "trainer", 1, "m", None, {"implementation": "rollout_lora:FullTrainer", "model": "m"})
    pods = pods_of("run_1", cluster, ledger, fake)
    try:
        await pods.claim([need])
        (body,) = fake.created_bodies()
        assert body["gpuCount"] == 4 and body["gpuTypeIds"] == ["NVIDIA H100 80GB HBM3"]
        env = body["env"]
        assert (env["ROLLOUT_ROLE"], env["ROLLOUT_TRAINER"], env["ROLLOUT_TRAINER_GPUS"]) == (
            "trainer", "rollout_lora:FullTrainer", "4")  # fmt: skip
    finally:
        await pods.release()
