"""Sandbox pools on a host pod, beside its engine and trainer: a pool of each kind the pod's settings name
(`ROLLOUT_SANDBOXES`, from the cluster config's `[sandboxes.KIND] on_pods` for the kinds its provider lists), served on
the pod's loopback interface, each under its kind (`/KIND/acquire`), behind the pod's proxy at `/v1/sandboxes/KIND/...`
(docs/research/run-placement.md, "Sandbox pool on the pod"). The run that holds the pod reaches them through its
`PodPools` (`rollout_train.pods.pools`).

**How many.** Each pool holds as many sandboxes as the pod's spare CPUs and memory hold (`sized`): the pod's vCPUs less
`ROLLOUT_SANDBOX_RESERVED_CPUS` (4: vLLM, the trainer, the follower, Envoy), divided by what one sandbox takes
(`cpus`, 1 by default), and no more than its memory less `ROLLOUT_SANDBOX_RESERVED_GIB` (64) divided by what one takes
(`memory_gib`, 2.4 by default), and no more than the kind's `size` where it says one. A pod of 16 vCPUs and 188 GB holds
12 Minecraft worlds; one of 8 vCPUs and 125 GB, 4. The pod's vCPUs and memory are what RunPod says it gave it, as its
lease records them; where the lease does not say, the container's limits (its cgroup), else the machine's. Several
kinds share what is spare, in order of their names.

**Whose leases.** The pod's lease says which run holds it, and its pools follow it, as the follower does
(`rollout_train.pods.inference`):

- a key is admitted only while it is of the run that holds the pod (its first part is the run's id, or the id of one of
  the run's evals, `RUN-...`); any other key gets `LeaseRefused`;
- when the lease names another run, or none (the pod was released), every lease is released and every sandbox
  deleted;
- a lease that no acquire or operation has used for `ROLLOUT_SANDBOX_IDLE` seconds (1800) is released: its runner is
  gone. The run's driver releases the leases of claims that lapsed itself (its keeper, beside the ledger), so this
  only ends what a driver that stopped left behind. A run's episodes pause while its trainer takes a step, which this
  outlasts.

The pod's ledger token reads its own lease, which is all this needs: claims are checked by the run's driver, which
holds the platform's token, not here.

**What a restart keeps.** Each pool keeps its leases in `sandboxes.json` in its kind's directory on the pod's volume
(`ROLLOUT_SANDBOX_DIRECTORY/KIND`). A pool started again finds them and marks lost those whose sandboxes are gone, so
their keys get `SandboxLost` rather than a new sandbox: their episodes are played again. When the process is stopped,
its sandboxes are deleted and its leases kept, for the same reason. The entrypoint starts the process again whenever it
ends; it never takes vLLM or the trainer down with it.

    python -m rollout_train.pods.sandboxes

- `ROLLOUT_SANDBOXES`: the pools, as JSON: `{"KIND": {"provider": "module:name", "settings": {...}, "size": N or null,
  "cpus": 1.0, "memory_gib": 2.4}}`. Each provider is made as `provider(directory, size=SIZE, **settings)`;
- `ROLLOUT_SANDBOX_ADDRESS`: where the pools are served (default `127.0.0.1:8710`);
- `ROLLOUT_SANDBOX_DIRECTORY`: each kind's directory is in it (default `/workspace/sandboxes`);
- `ROLLOUT_SANDBOX_RESERVED_CPUS`, `ROLLOUT_SANDBOX_RESERVED_GIB`: what is kept for the pod's other processes;
- `ROLLOUT_SANDBOX_IDLE`: seconds a lease may go unused before it is released;

and `ROLLOUT_POD_NAME` and `ROLLOUT_LEDGER` (`rollout_train.pods.environment`).
"""

import asyncio
import contextlib
import json
import logging
import math
import os
import signal
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import JsonValue

from rollout.contracts import ToolResult
from rollout.harness.sandboxes import Lease, Leases, Provider, SandboxPool, SandboxSpec
from rollout.names import named
from rollout_train.ledger import opened
from rollout_train.pods.environment import listening, location, required, served
from rollout_train.pods.leases import IDLE, PodLease, PodLeases, pod_leases_of
from rollout_train.sandboxes import FileLeases

if TYPE_CHECKING:
    from starlette.applications import Starlette

__all__ = ["Machine", "PodSandboxes", "Served", "UsedPool", "machine_of", "sized"]

log = logging.getLogger(__name__)

ADDRESS = "127.0.0.1:8710"
DIRECTORY = "/workspace/sandboxes"
RESERVED_CPUS = 4.0
"""vCPUs kept for vLLM, the trainer, the follower and Envoy."""
RESERVED_GIB = 64.0
"""GiB of memory kept for them (the trainer and vLLM each load the model through the machine's memory)."""
IDLE_SECONDS = 1800.0
"""Seconds a lease may go unused before it is released."""
EVERY = 15.0
"""Seconds between looks at the pod's lease and the pools' leases."""
GIB = 2**30


@dataclass(frozen=True)
class Served:
    """A pool the pod serves, as `ROLLOUT_SANDBOXES` says it."""

    provider: str
    settings: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    size: int | None = None
    cpus: float = 1.0
    memory_gib: float = 2.4


@dataclass(frozen=True)
class Machine:
    """What the pod has: vCPUs and memory in GiB."""

    cpus: float
    memory_gib: float


def sized(machine: Machine, pools: Mapping[str, Served], *, cpus: float = RESERVED_CPUS,
          memory_gib: float = RESERVED_GIB) -> dict[str, int]:  # fmt: skip
    """How many sandboxes each pool holds: one per `Served.cpus` of the pod's vCPUs less `cpus`, and no more than its
    memory less `memory_gib` holds at `Served.memory_gib` each, nor than the pool's `size`; the kinds, in order of their
    names, share what is spare."""
    spare_cpus, spare_gib = machine.cpus - cpus, machine.memory_gib - memory_gib
    sizes: dict[str, int] = {}
    for kind in sorted(pools):
        pool = pools[kind]
        count = max(0, min(math.floor(spare_cpus / pool.cpus), math.floor(spare_gib / pool.memory_gib)))
        if pool.size is not None:
            count = min(count, pool.size)
        sizes[kind] = count
        spare_cpus, spare_gib = spare_cpus - count * pool.cpus, spare_gib - count * pool.memory_gib
    return sizes


def machine_of(lease: PodLease | None, cgroup: Path = Path("/sys/fs/cgroup")) -> Machine:
    """The pod's vCPUs and memory: as its lease says RunPod gave them; where it does not, the container's limits; where
    it has none, the machine's."""
    limited = _limits(cgroup)
    cpus = float(lease.vcpus) if lease is not None and lease.vcpus else limited[0] or float(_machine_cpus())
    gib = lease.memory_gb * 1e9 / GIB if lease is not None and lease.memory_gb else limited[1] or _machine_gib()
    return Machine(cpus, gib)


def _limits(cgroup: Path) -> tuple[float | None, float | None]:
    """The container's CPU and memory limits (in vCPUs and GiB), as its cgroup (v2, else v1) says; none where unset."""

    def read(path: Path) -> str | None:
        try:
            return path.read_text().strip()
        except OSError:
            return None

    cpus = gib = None
    if (quota := read(cgroup / "cpu.max")) is not None:
        allowed, _, period = quota.partition(" ")
        if allowed != "max" and period:
            cpus = int(allowed) / int(period)
    elif (allowed := read(cgroup / "cpu" / "cpu.cfs_quota_us")) is not None and int(allowed) > 0:
        period = read(cgroup / "cpu" / "cpu.cfs_period_us")
        cpus = int(allowed) / int(period or 100000)
    memory = read(cgroup / "memory.max") or read(cgroup / "memory" / "memory.limit_in_bytes")
    if memory is not None and memory != "max" and int(memory) < 2**60:  # (v1 says "unlimited" as a huge number)
        gib = int(memory) / GIB
    return cpus, gib


def _machine_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # (not on Linux)
        return os.cpu_count() or 1


def _machine_gib() -> float:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GIB


class UsedPool(SandboxPool):
    """A `SandboxPool` that notes when each lease was last used (acquired, or operated on)."""

    def __init__(
        self,
        provider: Provider,
        *,
        name: str | None = None,
        leases: Leases | None = None,
        admits: Callable[[str], Awaitable[bool]] | None = None,
    ) -> None:
        super().__init__(provider, name=name, leases=leases, admits=admits)
        self.used: dict[str, float] = {}
        """When each lease was last used, by key, on this process's monotonic clock."""
        self.started = time.monotonic()

    async def acquire(self, spec: SandboxSpec, key: str, environment: Mapping[str, str] | None = None) -> Lease:
        lease = await super().acquire(spec, key, environment)
        self.used[key] = time.monotonic()
        return lease

    async def call(
        self, key: str, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
        self.used[key] = time.monotonic()
        try:
            return await super().call(key, name, arguments, effect_id=effect_id, arguments_digest=arguments_digest)
        finally:
            self.used[key] = time.monotonic()

    def unused(self, key: str) -> float:
        """Seconds since the lease of `key` was last used (or since this process started, for one it has not)."""
        return time.monotonic() - self.used.get(key, self.started)


def of_run(key: str, run: str) -> bool:
    """Whether a lease's key is of `run`: its first part is the run's id, or one of its evals' (`RUN-...`)."""
    first = key.split("/", 1)[0]
    return first == run or first.startswith(f"{run}-")


class PodSandboxes:
    """The pools of pod `name` (by kind), following its lease in `leases`: a key is admitted while it is of the run
    that holds the pod (`admitted`); every lease is released when another run, or none, holds it; a lease unused for
    `idle` seconds is released."""

    def __init__(self, name: str, leases: PodLeases | None, *, idle: float = IDLE_SECONDS) -> None:
        self.name = name
        self.leases = leases
        self.idle = idle
        self.pools: dict[str, UsedPool] = {}
        self.run: str | None = None
        """The run that holds the pod, as its lease said at the last look."""
        self._looked = False
        self._following = asyncio.Lock()

    async def admitted(self, key: str) -> bool:
        """Whether `key` may hold a lease now: it is of the run that holds the pod (the lease is read again for a key
        of another run, which may have taken the pod since the last look)."""
        if self.run is not None and of_run(key, self.run):
            return True
        await self.followed()
        return self.run is not None and of_run(key, self.run)

    async def followed(self) -> list[str]:
        """Read the pod's lease; where it names another run than it did, or none (or at the first look), release every
        lease not of the run it names, mark lost those whose sandboxes are gone, and delete what no lease names.
        Returns the keys released or lost."""
        gone: list[str] = []
        async with self._following:
            lease = await self.leases.get(self.name) if self.leases is not None else None
            run = lease.run if lease is not None and lease.state != IDLE else None
            if run == self.run and self._looked:
                return gone
            for kind, pool in self.pools.items():
                ended = [each.key for each in await pool.held() if run is None or not of_run(each.key, run)]
                for key in ended:
                    await pool.release(key)
                gone += [*ended, *await pool.sweep()]
                if ended:
                    log.info("pod %s is held by %s: released %d %s leases of others", self.name, run or "no run",
                             len(ended), kind)  # fmt: skip
            self.run, self._looked = run, True
        return gone

    async def follow(self) -> list[str]:
        """One look: follow the lease, and sweep each pool, releasing the leases unused for `idle` seconds (marking
        lost those whose sandboxes are gone, deleting those no lease names). Returns the keys released or lost."""
        gone = await self.followed()
        for pool in self.pools.values():
            gone += await pool.sweep(lambda lease, pool=pool: pool.unused(lease.key) >= self.idle)
        return gone

    async def serve(self, every: float = EVERY) -> None:
        """Look every `every` seconds until cancelled."""
        while True:
            try:
                if gone := await self.follow():
                    log.info("released or lost: %s", ", ".join(gone))
            except Exception:  # (looked at again next time)
                log.exception("following pod %s's lease failed; trying again", self.name)
            await asyncio.sleep(every)

    def app(self) -> "Starlette":
        """Each pool, served under its kind (`/KIND/acquire`, as `rollout.harness.remote.serve_pool` serves it)."""
        from starlette.applications import Starlette
        from starlette.routing import Mount

        from rollout.harness.remote import serve_pool

        return Starlette(routes=[Mount(f"/{kind}", app=serve_pool(pool)) for kind, pool in sorted(self.pools.items())])

    async def stop(self) -> None:
        """Delete every sandbox, keeping the leases: started again, the pools mark them lost."""
        for pool in self.pools.values():
            closing = getattr(pool.provider, "close", None)
            if closing is not None:
                with contextlib.suppress(Exception):
                    await closing()


def served_pools(environ: Mapping[str, str]) -> dict[str, Served]:
    """The pools `ROLLOUT_SANDBOXES` names."""
    try:
        said: Any = json.loads(required(environ, "ROLLOUT_SANDBOXES"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"ROLLOUT_SANDBOXES is not JSON: {error.msg}") from None
    if not isinstance(said, dict):
        raise SystemExit("ROLLOUT_SANDBOXES is a JSON object of pools, by kind")
    pools: dict[str, Served] = {}
    for kind, each in cast(dict[str, Any], said).items():
        given = cast(dict[str, Any], each)
        pools[kind] = Served(
            str(given["provider"]), dict(given.get("settings") or {}), given.get("size"),
            float(given.get("cpus") or 1.0), float(given.get("memory_gib") or 2.4),
        )  # fmt: skip
    return pools


async def main(environ: Mapping[str, str]) -> None:
    """Serve the pod's pools, following its lease, until cancelled."""
    pools = served_pools(environ)
    name = required(environ, "ROLLOUT_POD_NAME")
    leases = pod_leases_of(opened(location(environ, "ROLLOUT_LEDGER")))
    try:
        lease = await leases.get(name) if leases is not None else None
    except Exception as error:  # (sized by the container's limits instead)
        log.warning("pod %s's lease could not be read: %s", name, error)
        lease = None
    machine = machine_of(lease)
    sizes = sized(machine, pools, cpus=float(environ.get("ROLLOUT_SANDBOX_RESERVED_CPUS") or RESERVED_CPUS),
                  memory_gib=float(environ.get("ROLLOUT_SANDBOX_RESERVED_GIB") or RESERVED_GIB))  # fmt: skip
    following = PodSandboxes(name, leases, idle=float(environ.get("ROLLOUT_SANDBOX_IDLE") or IDLE_SECONDS))
    root = Path(environ.get("ROLLOUT_SANDBOX_DIRECTORY") or DIRECTORY)
    for kind, size in sizes.items():
        log.info("pod %s (%g vCPUs, %.0f GiB): %d %s sandboxes", name, machine.cpus, machine.memory_gib, size, kind)
        if size < 1:
            continue
        directory = root / kind
        directory.mkdir(parents=True, exist_ok=True)
        provider = named(pools[kind].provider)(directory, size=size, **dict(pools[kind].settings))
        following.pools[kind] = UsedPool(provider, name=f"{kind}@{name}", leases=FileLeases(directory),
                                         admits=following.admitted)  # fmt: skip
    if not following.pools:
        log.warning("pod %s has room for no sandboxes: it serves none", name)
        await asyncio.Event().wait()  # (ended with the container: started again, it would find no more room)
    await following.follow()
    host, port = listening(environ, "ROLLOUT_SANDBOX_ADDRESS", ADDRESS)
    try:
        await asyncio.gather(following.serve(), served(following.app(), host, port))
    finally:
        await following.stop()


async def _until_stopped(environ: Mapping[str, str]) -> None:
    """`main`, until the process is told to stop (`SIGTERM`, `SIGINT`): its sandboxes deleted on the way out."""
    task = asyncio.current_task()
    assert task is not None
    loop = asyncio.get_running_loop()
    for each in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(each, task.cancel)
    with contextlib.suppress(asyncio.CancelledError):
        await main(environ)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(_until_stopped(os.environ))
