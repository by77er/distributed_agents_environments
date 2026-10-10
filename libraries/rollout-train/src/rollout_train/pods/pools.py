"""The sandboxes of a kind a run's pods serve, as one pool: each pod the run leases whose provider lists the kind
serves a pool of it behind its proxy (`rollout_train.pods.sandboxes`), and the run's runner acquires from all of them
through `PodPools`, which the runner holds as it holds any pool (docs/research/run-placement.md).

- **Which pods.** At each look (`discover`, `rollout_train.pods.routing.LeasedPools`), the pods whose leases the run
  holds, at the addresses their leases say: those that beat fresh are live and take new leases; the rest still answer
  for the leases they hold.
- **Acquiring.** A key with no lease goes to the live pod with the most room (its pool's `capacity`, less the acquires
  on their way to it), then the next while each answers full; when every pod is full, to the cluster's own pool of the
  kind (`fallback`, the section's `url`), where there is one; else `NoCapacity`. A key with a lease goes to where its
  lease is, and only there: its sandbox is never made again elsewhere.
- **Releasing and operations** go to where the key's lease is.
- **Capacity** is the sum of the live pods' pools' and the fallback's, so a runner asks one pool for room as before.
- **A lost pod.** A lease whose pod the run no longer holds (released, deleted, taken by another run) is lost: its key
  gets `SandboxLost`, and its episode is played again. A pod that only misses beats keeps its leases.
- **Where each lease is** is kept in `leases` (by key: the pod's lease, under this pool's name, its `handle` saying the
  pod and the pod's own handle), so a driver started again routes the leases of runs it adopts to where they are.
- **Claims.** As a pool beside the ledger: `admits` refuses a key whose claim has lapsed (releasing its lease), and a
  keeper (`rollout_train.sandboxes.keep`) sweeps this pool, releasing on its pod each lease whose claim lapsed. A pod's
  own pool also ends leases unused for long, and every lease when another run takes the pod.
"""

import asyncio
import contextlib
import logging
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import JsonValue

from rollout.contracts import ToolResult, ToolSpecification
from rollout.harness.imports import deduplicates
from rollout.harness.sandboxes import (
    Capacity,
    Lease,
    LeaseRefused,
    Leases,
    MemoryLeases,
    NoCapacity,
    Pool,
    SandboxLost,
    SandboxSpec,
)

__all__ = ["FALLBACK", "PodPools", "Reached"]

log = logging.getLogger(__name__)

FALLBACK = "_cluster"
"""Where a lease of the fallback pool is, in its handle (no pod's name: those are lowercase letters, digits and
hyphens)."""
LOOK = 2.0
"""Seconds a look at the run's pods is kept for."""


@dataclass(frozen=True)
class Reached:
    """A pod's pool, as a look found it: `live` while the pod beats fresh (only a live pod takes new leases)."""

    pool: Pool
    live: bool = True


class PodPools:
    """A `Pool` of `kind` over the pools of the pods `discover` finds (by pod name), with `fallback` for when every pod
    is full; where each lease is, kept in `leases` under `name`. `admits` says whether a key may hold a lease now
    (beside the ledger: whether its claim holds)."""

    def __init__(
        self,
        kind: str,
        discover: Callable[[], Awaitable[Mapping[str, Reached]]],
        *,
        name: str | None = None,
        leases: Leases | None = None,
        admits: Callable[[str], Awaitable[bool]] | None = None,
        fallback: Pool | None = None,
        look: float = LOOK,
    ) -> None:
        self._kind = kind
        self.discover = discover
        self.name = name or f"{kind}@pods"
        self.leases: Leases = leases or MemoryLeases()
        self.admits = admits
        self.fallback = fallback
        self.look = look
        self._held: dict[str, Lease] = {}
        self._loaded = False
        self._making: set[str] = set()
        self._pending: Counter[str] = Counter()
        """Acquires on their way to each pod: counted against its room, so acquires at once spread."""
        self._choosing = asyncio.Lock()
        self._locks: dict[str, asyncio.Lock] = {}
        self._found: tuple[float, Mapping[str, Reached]] | None = None
        self._operations: list[ToolSpecification] | None = None
        self._deduplicates = False

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def deduplicates(self) -> bool:
        return self._deduplicates

    def operations(self) -> Sequence[ToolSpecification]:
        """What can be done to its sandboxes, as the first pool it acquired from said (none before)."""
        return list(self._operations or [])

    async def acquire(self, spec: SandboxSpec, key: str, environment: Mapping[str, str] | None = None) -> Lease:
        if spec.kind != self.kind:
            raise ValueError(f"the pool {self.name} holds {self.kind} sandboxes, not {spec.kind}")
        await self._load()
        async with self._lock(key):
            record = self._held.get(key)
            if self.admits is not None and not await self.admits(key):
                if record is not None:
                    await self._end(record)
                raise LeaseRefused(f"{key} may hold no sandbox: the claim it names no longer holds")
            if record is not None:
                return await self._again(record, spec, environment)
            self._making.add(key)
            try:
                where, pool, lease = await self._placed(spec, key, environment)
            finally:
                self._making.discard(key)
            await self._describe(pool)
            kept = lease.model_copy(update={"pool": self.name, "handle": f"{where}/{lease.handle}"})
            await self.leases.put(kept)
            self._held[key] = kept
            return lease

    async def release(self, key: str) -> None:
        await self._load()
        async with self._lock(key):
            record = self._held.get(key)
            if record is not None:
                await self._end(record)

    async def capacity(self) -> Capacity:
        """The sum of the live pods' pools' room and the fallback's (a pool that does not answer: none)."""
        pools = [each.pool for each in (await self._reached()).values() if each.live]
        if self.fallback is not None:
            pools.append(self.fallback)
        size = leased = 0
        for found in await asyncio.gather(*(pool.capacity() for pool in pools), return_exceptions=True):
            if isinstance(found, Capacity):
                size, leased = size + found.size, leased + found.leased
        return Capacity(size=size, leased=leased)

    async def call(
        self, key: str, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
        await self._load()
        record = self._held.get(key)
        if record is None or record.lost:
            raise SandboxLost(f"the pool {self.name} holds no sandbox of {key}")
        pool = self._pool_at(_where(record), await self._reached())
        if pool is None:
            raise SandboxLost(f"the sandbox of {key} is gone with pod {_where(record)}")
        return await pool.call(key, name, arguments, effect_id=effect_id, arguments_digest=arguments_digest)

    async def held(self) -> list[Lease]:
        """Its leases, those whose pods are gone included, as it keeps them (`handle`: `POD/HANDLE`)."""
        await self._load()
        return list(self._held.values())

    async def sweep(self, ended: Callable[[Lease], bool] = lambda lease: False) -> list[str]:
        """Release the leases `ended` says have ended, on their pods; mark lost those whose pod the run no longer
        holds. Returns the keys released or marked lost."""
        await self._load()
        reached = await self._reached(fresh=True)
        gone: list[str] = []
        for record in list(self._held.values()):
            if record.key in self._making:
                continue
            if ended(record):
                await self.release(record.key)
                gone.append(record.key)
            elif not record.lost and self._pool_at(_where(record), reached) is None:
                async with self._lock(record.key):
                    lost = record.model_copy(update={"lost": True})
                    await self.leases.put(lost)
                    self._held[record.key] = lost
                gone.append(record.key)
        return gone

    async def close(self) -> None:
        """Release every lease it holds, on its pod (a pod's own pool ends what is left)."""
        for record in await self.held():
            with contextlib.suppress(Exception):
                await self.release(record.key)

    async def _placed(
        self, spec: SandboxSpec, key: str, environment: Mapping[str, str] | None
    ) -> tuple[str, Pool, Lease]:
        """A new lease of `key`: on the live pod with the most room, the next while each is full, then the fallback.
        A pod is chosen by one acquire at a time, its room asked then less the acquires on their way to it, so acquires
        at once spread over the pods."""
        tried: set[str] = set()
        while True:
            async with self._choosing:
                reached = await self._reached()
                live = {where: each.pool for where, each in reached.items() if each.live and where not in tried}
                asked = await asyncio.gather(*(pool.capacity() for pool in live.values()), return_exceptions=True)
                room = {
                    where: found.free - self._pending[where] for where, found in zip(live, asked, strict=True)
                    if isinstance(found, Capacity) and found.free > self._pending[where]
                }  # fmt: skip
                where = min(room, key=lambda each: (-room[each], each), default=None)
                if where is None:
                    break
                self._pending[where] += 1
            try:
                return where, live[where], await live[where].acquire(spec, key, environment)
            except NoCapacity:
                tried.add(where)
            finally:
                self._pending[where] -= 1
        if self.fallback is not None:
            return FALLBACK, self.fallback, await self.fallback.acquire(spec, key, environment)
        raise NoCapacity(f"the pool {self.name}: every pod's pool is full")

    async def _again(self, record: Lease, spec: SandboxSpec, environment: Mapping[str, str] | None) -> Lease:
        """The lease of a key that has one, from where it is; `SandboxLost` where that is gone. Hold its lock."""
        pool = None if record.lost else self._pool_at(_where(record), await self._reached())
        if pool is None:
            await self._forget(record)
            raise SandboxLost(f"the sandbox of {record.key} is gone with pod {_where(record)}")
        try:
            lease = await pool.acquire(spec, record.key, environment)
        except SandboxLost:
            await self._forget(record)
            raise
        await self._describe(pool)
        return lease

    async def _end(self, record: Lease) -> None:
        """Release a lease where it is, and forget it. Hold its key's lock."""
        pool = None if record.lost else self._pool_at(_where(record), await self._reached())
        if pool is not None:
            try:
                await pool.release(record.key)
            except Exception as error:  # (its pod ends it: unused for long, or another run takes the pod)
                log.warning("releasing %s on %s failed: %s", record.key, _where(record), error)
        await self._forget(record)

    async def _forget(self, record: Lease) -> None:
        self._held.pop(record.key, None)
        await self.leases.delete(record.key)

    def _pool_at(self, where: str, reached: Mapping[str, Reached]) -> Pool | None:
        if where == FALLBACK:
            return self.fallback
        found = reached.get(where)
        return found.pool if found is not None else None

    async def _reached(self, *, fresh: bool = False) -> Mapping[str, Reached]:
        """The run's pods' pools, as a look within `look` seconds found them (`fresh`: as a look now does)."""
        now = time.monotonic()
        if fresh or self._found is None or now - self._found[0] > self.look:
            self._found = (now, dict(await self.discover()))
        return self._found[1]

    async def _describe(self, pool: Pool) -> None:
        """Learn its sandboxes' operations from the first pool it acquired from (all of a kind's pools offer the
        same)."""
        if self._operations is not None:
            return
        describe: Any = getattr(pool, "describe", None)
        if describe is not None:
            await describe()
        self._operations = list(pool.operations())
        self._deduplicates = deduplicates(pool)

    async def _load(self) -> None:
        if not self._loaded:
            self._held = {lease.key: lease for lease in await self.leases.all() if lease.pool == self.name}
            self._loaded = True

    def _lock(self, key: str) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())


def _where(record: Lease) -> str:
    """The pod (or `FALLBACK`) a kept lease is on."""
    return record.handle.partition("/")[0]
