"""Leasing RunPod's pods for runs: a run claims the pods its RunPod providers give it when it starts, renews their
leases while it runs, and releases them when it ends; a released pod stays warm for a while for the next run, and a
reaper deletes what no run holds. No runs, no pods.

**Claiming** (`Pods.claim`). The run's driver claims what its settings need (`needs_of`): a pod for each replica of a
channel on a `runpod-inference` or `runpod-host` provider, and one for a `runpod-trainer` (none for one whose
`colocate_with` names a host: the host's pod takes its steps). For each it first takes a warm pod: an idle lease of the
same provider, image and model, taken by compare-and-set, so two runs never take one pod. The pod is reset to the run:
its lease names the run, the channel, the trainer's settings and a ledger token for the run, which the pod reads and
follows (`rollout_train.pods.inference`, `rollout_train.pods.training`). Else it starts one: a free slot of the
provider's `max_pods` (none free: the run waits, saying so), the lease written first, then the pod asked of RunPod
with its name, a one-time token for its certificate from step-ca, its ledger token, and the run's store and key.
Where the pod is reached is what RunPod's API says (its public IP and the public port 8443/tcp is mapped to), written
into its lease (`PodLease.address`) when RunPod answers the request with it, or when RunPod is next asked about the pod
while the run waits for it. The pod is ready for the run once its lease has that address and it beats that it is ready
(its beat names the run). A pod not ready within its provider's `start_timeout` is deleted, and the run fails saying
which and why (`PodsDidNotStart`).

**Renewing** (`Pods.renewing`). Every `RENEW` seconds the run stamps each lease (`renewed`, writing where RunPod says
its pod is reached into a lease that does not say it, or says another place: a pod started again, or given another
image in place, may be mapped to another public port) and its time on each pod; a lease another took (a reaper that
found it stale) ends the run (`LeaseLost`). What the pods cost the run since the last renewal is told to `spent`, so a
run's `limits.spend` counts its pods.

**Releasing** (`Pods.release`). When the run ends, however it ends, each lease becomes idle: no run, no token. The pod
stays up, warm, for its provider's `idle_stop` seconds (default 600), charged to the run that last held it.

**Reaping** (`reap`). Run by `rollout pods reap` (a CronJob of the chart, every minute): it deletes the pod of each
lease not renewed for `STALE` seconds (the run's driver died, or its cluster went away), of each idle lease past its
`idle_stop`, and every pod RunPod lists with the cluster's tag (`rollout-CLUSTER-`) that no lease names. Deleting a pod
revokes its certificate (the serial its beats said), closes its time, and deletes its lease.

Pod time is charged at the pod's hourly price: RunPod's `costPerHr` for it, else the provider's `price`.
"""

import asyncio
import contextlib
import json
import logging
import re
import secrets
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import JsonValue

from rollout_train.ledger import Ledger
from rollout_train.ledger_service.scopes import pod_token
from rollout_train.ledger_service.wire import Conflict
from rollout_train.pods.environment import PORT
from rollout_train.pods.identity import POD, pod_identity
from rollout_train.pods.leases import HELD, IDLE, STARTING, PodLease, PodLeases, PodTime, pod_leases_of, time_key
from rollout_train.presence import Beat, alive, presence_of
from rollout_train.providers import RUNPOD, InferenceProvider, PodTable, TrainerProvider, pod_table

if TYPE_CHECKING:
    from rollout_runpod import PodSpec, RunPod, StepCa
    from rollout_train.cluster import Cluster
    from rollout_train.run_settings import RunSettings

__all__ = [
    "LOOK",
    "RENEW",
    "STALE",
    "LeaseLost",
    "PodNeed",
    "Pods",
    "PodsDidNotStart",
    "needs_of",
    "pod_name",
    "pods_store",
    "reap",
    "tag_of",
]

log = logging.getLogger(__name__)

RENEW = 30.0
"""Seconds between a run's renewals of its leases."""
STALE = 300.0
"""Seconds after its newest renewal that a lease is stale: its run is taken to be gone, and its pod is deleted."""
LOOK = 2.0
"""Seconds between looks at whether a run's pods are ready."""
INFERENCE, TRAINER, HOST = "inference", "trainer", "host"
"""What a pod does: serves a channel, takes steps, or both."""


class PodsDidNotStart(RuntimeError):
    """A pod did not say it was ready for its run in time: it was deleted."""


class LeaseLost(RuntimeError):
    """A lease the run held was taken from it (a reaper found it stale)."""


@dataclass(frozen=True)
class PodNeed:
    """What a run needs of one RunPod provider: `count` pods doing `role`, serving `model` (on `channel`), with the
    trainer's `settings` (its implementation, model and settings) for one that takes steps."""

    provider: str
    role: str
    count: int
    model: str
    channel: str | None = None
    settings: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])


def needs_of(settings: "RunSettings", cluster: "Cluster") -> list[PodNeed]:
    """The pods a run's settings need of the cluster's RunPod providers: one per replica of each channel on a
    `runpod-inference` or `runpod-host` provider (a host's doing the trained channel's steps too, where the run's
    trainer is a `runpod-trainer` on it), and one for a `runpod-trainer` of its own."""
    from rollout_train.run_settings import TRAINING, is_trainers

    trainer = cluster.trainers.get(str(settings["trainer.provider"])) if settings.kind in TRAINING else None
    trains = trainer is not None and trainer.kind == "runpod-trainer"
    steps: dict[str, JsonValue] = {}
    if trainer is not None and trains:
        given = {key.removeprefix("trainer."): value for key, value in settings.values.items() if is_trainers(key)}
        steps = {"implementation": trainer.runs.implementation, "model": settings.trainer_model or "",
                 "trainer": cast(JsonValue, given)}  # fmt: skip
    needs: list[PodNeed] = []
    hosted = False
    for channel in settings.channels:
        model = settings.get(f"channels.{channel}.model")
        for name in settings.providers(channel):
            provider = cluster.inference.get(name)
            if provider is None or provider.kind not in RUNPOD or model is None:
                continue
            asked = settings[f"channels.{channel}.replicas"]
            count = asked if isinstance(asked, int) else provider.replicas
            on_host = trains and trainer is not None and trainer.colocate_with == name and channel == settings.trained
            hosted = hosted or on_host
            needs.append(PodNeed(name, HOST if on_host else INFERENCE, count, str(model), channel,
                                 steps if on_host else {}))  # fmt: skip
    if trainer is not None and trains and trainer.colocate_with is None:
        needs.append(PodNeed(trainer.name, TRAINER, 1, settings.trainer_model or "", None, steps))
    elif trainer is not None and trains and not hosted:
        raise ValueError(f"the trainer {trainer.name} takes its steps on {trainer.colocate_with}'s pods: the run's "
                         f"trained channel is served there")  # fmt: skip
    return needs


def pods_store(settings: "RunSettings", cluster: "Cluster") -> str | None:
    """The blob store a run's RunPod providers read and write (`[stores.NAME]`), where they name one: the store its
    blobs go to."""
    for need in needs_of(settings, cluster):
        table = _table(cluster, need.provider)
        if table is not None and table.store is not None:
            return table.store
    return None


def tag_of(cluster: "Cluster") -> str:
    """What every pod a cluster rents is named beginning with: `rollout-CLUSTER-`."""
    return f"rollout-{cluster.name}-"


def pod_name(cluster: "Cluster", provider: str, slot: int) -> str:
    """A new pod's name: the cluster's tag, its provider, its slot and a few random letters, at most 63 characters (a
    pod's identity is made from it)."""
    tail = f"-{slot}-{secrets.token_hex(3)}"
    stem = re.sub(r"[^a-z0-9-]+", "-", provider.lower()).strip("-") or "pod"
    room = 63 - len(tag_of(cluster)) - len(tail)
    return f"{tag_of(cluster)}{stem[: max(room, 1)].rstrip('-')}{tail}"


def _provider(cluster: "Cluster", name: str) -> InferenceProvider | TrainerProvider | None:
    return cluster.inference.get(name) or cluster.trainers.get(name)


def _table(cluster: "Cluster", name: str) -> PodTable | None:
    provider = _provider(cluster, name)
    if provider is None or provider.kind not in RUNPOD:
        return None
    if isinstance(provider, TrainerProvider) and provider.colocate_with is not None:
        return _table(cluster, provider.colocate_with)
    return pod_table(provider.kind, provider.settings)


def _runpod(cluster: "Cluster", name: str) -> "RunPod":
    from rollout_runpod import RunPod

    provider = _provider(cluster, name)
    key = provider.secrets.get("api_key") if provider is not None else None
    return RunPod(key_env=key.env if key is not None and key.env else "RUNPOD_API_KEY")


def _step_ca(table: PodTable) -> "StepCa | None":
    if not table.step_ca:
        return None
    from rollout_runpod import StepCa

    said = table.step_ca
    return StepCa.from_files(said["url"], provisioner=said["provisioner"], key=Path(said["key_file"]).expanduser(),
                             root=Path(said["root"]).expanduser(), system=said.get("trust") == "system")  # fmt: skip


def _beat_of(beats: Sequence[Beat], pod: str) -> Mapping[str, Any] | None:
    """What a pod's newest fresh beat says of the pod."""
    for beat in beats:
        said: Any = beat.about.get(POD)
        if beat.runner == pod and alive(beat) and isinstance(said, dict):
            return cast(dict[str, Any], said)
    return None


class Pods:
    """A run's pods on RunPod: claimed when it starts (`claim`), renewed while it runs (`renewing`), released when it
    ends (`release`). `api` gives the RunPod client of a provider (by default one with the provider's key), `ca` its
    step-ca (by default the one its table says, if any). `told` hears what the run waits for."""

    def __init__(
        self,
        run: str,
        cluster: "Cluster",
        ledger: Ledger,
        *,
        api: Callable[[str], "RunPod"] | None = None,
        ca: Callable[[PodTable], "StepCa | None"] | None = None,
        environ: Mapping[str, str] | None = None,
        told: Callable[[Sequence[str]], Awaitable[None]] | None = None,
        renew: float = RENEW,
        look: float = LOOK,
    ) -> None:
        store = pod_leases_of(ledger)
        if store is None:
            raise ValueError("this ledger keeps no pods' leases")
        self.run = run
        self.cluster = cluster
        self.ledger = ledger
        self.store: PodLeases = store
        self._api: Callable[[str], RunPod] = api or (lambda provider: _runpod(cluster, provider))
        self._clients: dict[str, RunPod] = {}
        self._ca = ca or _step_ca
        self.environ = environ
        self._told = told
        self.renew = renew
        self.look = look
        self.leases: dict[str, PodLease] = {}
        """The leases it holds, by pod."""
        self.times: dict[str, PodTime] = {}
        """Its time on each pod it holds, by pod."""
        self.ready: set[str] = set()
        self.spent = 0.0
        """Dollars its pods cost it so far, as its renewals counted."""

    def api(self, provider: str) -> "RunPod":
        if provider not in self._clients:
            self._clients[provider] = self._api(provider)
        return self._clients[provider]

    async def claim(self, needs: Sequence[PodNeed]) -> list[PodLease]:
        """Take or start the pods `needs` say, and wait until each is ready for the run (`PodsDidNotStart` where one
        is not in time: it is deleted). Raises `ValueError` where the cluster cannot give pods (no ledger service for
        them to reach)."""
        for need in needs:
            for _ in range(need.count):
                while (lease := await self._taken(need) or await self._started(need)) is None:
                    table = _table(self.cluster, need.provider)
                    most = table.max_pods if table is not None else 0
                    await self._say([f"a pod of {need.provider} (all {most} of its max_pods are taken)"])
                    await asyncio.sleep(self.look)
                self.leases[lease.pod] = lease
        await self._waited()
        return list(self.leases.values())

    def of(self, role: str | None = None, *, channel: str | None = None, provider: str | None = None) -> list[PodLease]:
        """The leases it holds, of `role` (`trainer` counts a host), `channel` and `provider` where said."""
        roles = {role, HOST} if role in (INFERENCE, TRAINER) else {role}
        return [each for each in self.leases.values() if (role is None or each.role in roles)
                and (channel is None or each.channel == channel)
                and (provider is None or each.provider == provider)]  # fmt: skip

    async def _taken(self, need: PodNeed) -> PodLease | None:
        """A warm pod of the provider, with the image and model the need asks for, taken for the run."""
        table = _table(self.cluster, need.provider)
        assert table is not None
        for lease in await self.store.all():
            wanted = (need.provider, IDLE, need.model, table.image)
            if (lease.provider, lease.state, lease.model, lease.image) != wanted:
                continue
            now = await self.store.now()
            try:
                taken = await self.store.put(replace(
                    lease, run=self.run, channel=need.channel, role=need.role, token=self._token(lease.pod),
                    settings=dict(need.settings), state=STARTING, held=now, renewed=now, released=None,
                ), expect=lease.version)  # fmt: skip
            except Conflict:  # (another run took it first)
                continue
            await self._closed(lease.pod, now)
            self.times[taken.pod] = PodTime(time_key(taken.pod, self.run, now), taken.pod, self.run, taken.provider,
                                            taken.gpu, taken.price, now, now)  # fmt: skip
            await self.store.charge(self.times[taken.pod])
            log.info("run %s took warm pod %s of %s", self.run, taken.pod, taken.provider)
            return taken
        return None

    async def _started(self, need: PodNeed) -> PodLease | None:
        """A new pod of the provider, in a free slot; none while every slot is taken. A slot a warm pod holds that the
        need cannot take (another model) is freed: its pod is deleted."""
        table = _table(self.cluster, need.provider)
        assert table is not None
        mine = [each for each in await self.store.all() if each.provider == need.provider]
        free = [slot for slot in range(table.max_pods) if slot not in {each.slot for each in mine}]
        if not free:
            for idle in [each for each in mine if each.state == IDLE]:
                with contextlib.suppress(Conflict):
                    await self.deleted(idle, f"its slot is wanted for a pod of {need.model}")
                    free = [idle.slot]
                    break
        if not free:
            return None
        now = await self.store.now()
        name = pod_name(self.cluster, need.provider, free[0])
        lease = PodLease(
            name, need.provider, free[0], need.role, table.image, need.model, table.gpu_types[0], table.price or 0.0,
            table.cloud, run=self.run, channel=need.channel, token=self._token(name), settings=dict(need.settings),
            state=STARTING, created=now, held=now, renewed=now,
        )  # fmt: skip
        try:
            lease = await self.store.put(lease, expect=None)
        except Conflict:  # (another run took the slot first)
            return None
        try:
            pod = await self.api(need.provider).create(self._spec(lease, table, need))
        except BaseException:
            with contextlib.suppress(Exception):
                await self.store.delete(lease.pod, expect=lease.version)
            raise
        given = replace(lease, id=pod.id, gpu=pod.gpu or lease.gpu, address=pod.address(PORT),
                        price=pod.cost_per_hour if pod.cost_per_hour is not None else lease.price)  # fmt: skip
        made = await self.store.put(given, expect=lease.version)
        self.times[made.pod] = PodTime(time_key(made.pod, self.run, made.created), made.pod, self.run, made.provider,
                                       made.gpu, made.price, made.created, made.created)  # fmt: skip
        await self.store.charge(self.times[made.pod])
        log.info("run %s started pod %s (%s) of %s", self.run, made.pod, pod.id, made.provider)
        return made

    def _token(self, pod: str) -> str:
        """The ledger service's token for `pod` serving the run (signed with the platform's token)."""
        secret = self.cluster.ledger.token.resolve(self.environ) if self.cluster.ledger.token is not None else None
        if not secret:
            raise ValueError("pods reach the ledger service with tokens signed with the platform's: name it in "
                             "[ledger] token_env, and set it here")  # fmt: skip
        return pod_token(secret, pod, self.run)

    def _spec(self, lease: PodLease, table: PodTable, need: PodNeed) -> "PodSpec":
        """The pod as RunPod is asked for it."""
        from rollout_runpod import PodSpec, fingerprint
        from rollout_train.stores import SERVICES, for_pods

        public = self.cluster.ledger.public or (
            self.cluster.ledger.url if (self.cluster.ledger.url or "").startswith(SERVICES) else None
        )
        if public is None:
            raise ValueError("pods reach the ledger service at [ledger] public, which the cluster config does not say")
        location, keys = for_pods(self.cluster, table.store, writes=lease.role != INFERENCE, environ=self.environ)
        ledger = {"kind": "rollout_train.ledger_service:HttpLedger", "url": public, "token_env": "ROLLOUT_LEDGER_TOKEN"}
        env = {
            "ROLLOUT_POD_NAME": lease.pod, "ROLLOUT_ROLE": lease.role, "ROLLOUT_MODEL": lease.model,
            "ROLLOUT_LEDGER": json.dumps(ledger), "ROLLOUT_BLOBS": json.dumps(location),
        }  # fmt: skip
        provider = _provider(self.cluster, need.provider)
        if lease.role in (INFERENCE, HOST) and isinstance(provider, InferenceProvider):
            offer = provider.models.get(lease.model)
            args = str(offer.options.get("args") or "") if offer is not None else ""
            if table.memory_fraction is not None and "--gpu-memory-utilization" not in args:
                args = f"{args} --gpu-memory-utilization {table.memory_fraction:g}".strip()
            if "--max-logprobs" not in args:
                args = f"{args} --max-logprobs {provider.capabilities.top_logprobs}".strip()
            env["VLLM_ARGS"] = args
            if offer is not None and offer.max_lora_rank is not None:
                env["VLLM_MAX_LORA_RANK"] = str(offer.max_lora_rank)
            if lease.role == HOST and table.sleep:
                env["ROLLOUT_SLEEP_VLLM"] = "1"
        if lease.role in (TRAINER, HOST):
            env["ROLLOUT_TRAINER"] = str(need.settings.get("implementation") or "rollout_lora:LoraTrainer")
            env["ROLLOUT_TRAINER_MODEL"] = str(need.settings.get("model") or lease.model)
        sensitive = {"ROLLOUT_LEDGER_TOKEN": lease.token or "", **keys}
        ca = self._ca(table)
        if ca is not None:
            root = Path(table.step_ca["root"]).expanduser().read_bytes()
            env["STEP_CA_URL"] = table.step_ca["url"]
            env["STEP_FINGERPRINT"] = fingerprint(root)
            env["STEP_ROOT"] = root.decode()  # (pinned: the root the pod's certificates chain to, which it checks by)
            env["STEP_CA_TRUST"] = table.step_ca.get("trust", "root")
            sensitive["STEP_TOKEN"] = ca.pod_token(pod_identity(lease.pod))
        return PodSpec(
            name=lease.pod, image=table.image, gpu_types=list(table.gpu_types), gpu_count=table.gpu_count, env=env,
            secrets=dict(table.secrets), sensitive=sensitive, ports=(f"{PORT}/tcp",), volume_gb=table.volume_gb,
            container_disk_gb=table.container_disk_gb, cloud=table.cloud, data_centers=list(table.regions),
        )  # fmt: skip

    async def _waited(self) -> None:
        """Wait until every pod it holds has the address RunPod says it is reached at in its lease (`_addressed`) and
        beats that it is ready for the run, renewing meanwhile; delete one that is not within its provider's
        `start_timeout`, and fail the run (`PodsDidNotStart`)."""
        presence = presence_of(self.ledger)
        if presence is None:
            raise ValueError("this ledger keeps no beats: pods say they are ready in theirs")
        renewed = time.monotonic()
        while True:
            beats = await presence.beats()
            waiting: list[str] = []
            now = await self.store.now()
            for lease in list(self.leases.values()):
                if lease.pod in self.ready:
                    continue
                if lease.address is None:
                    lease = self.leases[lease.pod] = await self._addressed(lease)
                said = _beat_of(beats, lease.pod)
                ready = said is not None and said.get("ready") is True and said.get("run") == self.run
                if ready and lease.address is not None:
                    self.leases[lease.pod] = await self.store.put(replace(lease, state=HELD), expect=lease.version)
                    self.ready.add(lease.pod)
                    continue
                table = _table(self.cluster, lease.provider)
                timeout = table.start_timeout if table is not None else 1200.0
                since = lease.held if lease.held is not None else lease.created
                if now - since > timeout:
                    why = f": it said {said.get('why')}" if said and said.get("why") else ""
                    await self.deleted(lease, f"not ready for run {self.run} within {timeout:.0f} s")
                    raise PodsDidNotStart(f"pod {lease.pod} of {lease.provider} did not say it was ready within "
                                          f"{timeout:.0f} seconds{why}; it was deleted")  # fmt: skip
                if said is None:
                    state = "starting"
                elif lease.address is None:
                    state = "up, RunPod has not said its public address yet"
                else:
                    state = f"up, not ready: {said.get('why') or 'loading'}"
                waiting.append(f"pod {lease.pod} of {lease.provider} ({state})")
            if not waiting:
                return
            await self._say(waiting)
            if time.monotonic() - renewed >= self.renew:
                await self.renewed()
                renewed = time.monotonic()
            await asyncio.sleep(self.look)

    async def _addressed(self, lease: PodLease) -> PodLease:
        """The lease with where RunPod says its pod is reached (`https://IP:PORT`), asked of RunPod's API now and
        written into the lease where it says another than the lease does; the lease as it was while it says none, says
        the same, or the pod has no id."""
        if lease.id is None:
            return lease
        try:
            pod = await self.api(lease.provider).pod(lease.id)
        except Exception as error:  # (asked again at the next look)
            log.info("RunPod did not say where pod %s is: %s", lease.pod, error)
            return lease
        address = pod.address(PORT)
        if address is None or address == lease.address:
            return lease
        if lease.address is not None:
            log.info("pod %s is reached at %s now, not %s", lease.pod, address, lease.address)
        return await self.store.put(replace(lease, address=address), expect=lease.version)

    async def _say(self, waits: Sequence[str]) -> None:
        if self._told is not None:
            await self._told(waits)

    async def renewed(self) -> float:
        """Stamp each lease it holds, and its time on each pod; the dollars that time cost since the last renewal.
        Raises `LeaseLost` where a lease is no longer the run's."""
        now = await self.store.now()
        added = 0.0
        for pod in list(self.leases):
            there = await self.store.get(pod)
            if there is None or there.run != self.run:
                raise LeaseLost(f"pod {pod} is no longer held by run {self.run} (its lease went stale and was reaped)")
            try:
                there = await self._addressed(there)  # (none yet, or another public port since)
            except Conflict:
                continue  # (changed meanwhile: the next renewal looks again)
            try:
                self.leases[pod] = await self.store.put(replace(there, renewed=now), expect=there.version)
            except Conflict:
                continue  # (changed meanwhile: the next renewal looks again)
            if (spent := self.times.get(pod)) is not None:
                updated = replace(spent, until=now)
                added += updated.dollars - spent.dollars
                self.times[pod] = updated
                await self.store.charge(updated)
        self.spent += added
        return added

    async def renewing(self, spent: Callable[[float], Awaitable[None]] | None = None) -> None:
        """Renew every `renew` seconds until cancelled, telling `spent` the dollars each renewal adds."""
        while True:
            await asyncio.sleep(self.renew)
            added = await self.renewed()
            if spent is not None and added:
                await spent(added)

    async def release(self) -> None:
        """Release every lease it holds: each pod stays warm for its provider's `idle_stop` (deleted at once where that
        is 0), its warm time charged to the run."""
        for pod in list(self.leases):
            with contextlib.suppress(Exception):  # (a lease that cannot be released goes stale, and is reaped)
                there = await self.store.get(pod)
                if there is None or there.run != self.run:
                    continue
                now = await self.store.now()
                table = _table(self.cluster, there.provider)
                if (spent := self.times.get(pod)) is not None:
                    self.times[pod] = replace(spent, until=now, released=True)
                    await self.store.charge(self.times[pod])
                if (table is not None and table.idle_stop <= 0) or pod not in self.ready:
                    await self.deleted(there, f"released by run {self.run}")
                    continue
                await self.store.put(replace(there, run=None, channel=None, token=None, state=IDLE, released=now,
                                             renewed=now), expect=there.version)  # fmt: skip
                log.info("run %s released pod %s: it stays warm for %.0f s", self.run, pod,
                         table.idle_stop if table else 0)  # fmt: skip
        self.leases.clear()
        for client in self._clients.values():
            with contextlib.suppress(Exception):
                await client.aclose()

    async def deleted(self, lease: PodLease, why: str) -> None:
        """Delete a lease's pod (`delete`)."""
        await delete(self.cluster, self.ledger, lease, why, api=self.api(lease.provider), ca=self._ca)
        self.leases.pop(lease.pod, None)

    async def _closed(self, pod: str, now: float) -> None:
        """Close the time another run was charged for a pod it released: its warm time ends now."""
        for entry in await self.store.times():
            if entry.pod == pod and entry.released and not entry.closed:
                await self.store.charge(replace(entry, idle=max(0.0, now - entry.until), closed=True))


async def delete(
    cluster: "Cluster",
    ledger: Ledger,
    lease: PodLease,
    why: str,
    *,
    api: "RunPod | None" = None,
    ca: Callable[[PodTable], "StepCa | None"] = _step_ca,
) -> None:
    """Delete a lease's pod at RunPod, revoke its certificate (the serial its beats said), close the time charged for
    it, and delete its lease."""
    store = pod_leases_of(ledger)
    assert store is not None
    client = api or _runpod(cluster, lease.provider)
    if lease.id is not None:
        try:
            await client.terminate(lease.id)
        except Exception as error:
            if getattr(error, "status", None) not in (400, 404):  # (already gone)
                raise
    table = _table(cluster, lease.provider)
    presence = presence_of(ledger)
    if table is not None and presence is not None and (authority := ca(table)) is not None:
        said = _beat_of(await presence.beats(), lease.pod)
        if said is not None and isinstance(said.get("serial"), str):
            with contextlib.suppress(Exception):  # (it lapses within a day in any case)
                await authority.revoke(str(said["serial"]), reason=why)
    now = await store.now()
    for entry in await store.times():
        if entry.pod == lease.pod and not entry.closed:
            idle = max(0.0, now - entry.until) if entry.released else 0.0
            await store.charge(replace(entry, idle=idle, closed=True))
    with contextlib.suppress(Conflict):
        await store.delete(lease.pod, expect=lease.version)
    log.info("deleted pod %s (%s): %s", lease.pod, lease.id, why)


async def reap(
    cluster: "Cluster",
    ledger: Ledger,
    *,
    api: Callable[[str], "RunPod"] | None = None,
    ca: Callable[[PodTable], "StepCa | None"] = _step_ca,
    stale: float = STALE,
) -> list[str]:
    """Delete the pods no run holds: those of leases not renewed for `stale` seconds, of idle leases past their
    provider's `idle_stop`, and those RunPod lists with the cluster's tag that no lease names. What it did, in words."""
    store = pod_leases_of(ledger)
    if store is None:
        return []
    clients: dict[str, RunPod] = {}

    def client(provider: str) -> "RunPod":
        if provider not in clients:
            clients[provider] = api(provider) if api is not None else _runpod(cluster, provider)
        return clients[provider]

    done: list[str] = []
    now = await store.now()
    leases = await store.all()
    for lease in leases:
        table = _table(cluster, lease.provider)
        idle_stop = table.idle_stop if table is not None else 0.0
        if lease.state in (STARTING, HELD) and now - (lease.renewed or lease.created) > stale:
            why = f"its lease was not renewed for {now - (lease.renewed or lease.created):.0f} s (run {lease.run})"
        elif lease.state == IDLE and now - (lease.released or lease.created) >= idle_stop:
            why = f"no run held it for {now - (lease.released or lease.created):.0f} s"
        else:
            continue
        await delete(cluster, ledger, lease, why, api=client(lease.provider), ca=ca)
        done.append(f"deleted {lease.pod} of {lease.provider}: {why}")
    named = {each.pod for each in await store.all()}
    providers = [name for name, each in [*cluster.inference.items(), *cluster.trainers.items()]
                 if each.kind in RUNPOD and _table(cluster, name) is not None]  # fmt: skip
    seen: set[str] = set()
    for provider in providers:
        key = _provider(cluster, provider)
        account = key.secrets.get("api_key") if key is not None else None
        if (said := str(account)) in seen:
            continue
        seen.add(said)
        for pod in await client(provider).pods():
            if pod.name.startswith(tag_of(cluster)) and pod.name not in named:
                with contextlib.suppress(Exception):
                    await client(provider).terminate(pod.id)
                    done.append(f"deleted {pod.name} ({pod.id}): no lease names it")
    for each in clients.values():
        with contextlib.suppress(Exception):
            await each.aclose()
    return done
