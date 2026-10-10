"""The sandboxes of a kind a run's pods serve, as one pool: each pod the run leases whose provider lists the kind
serves a pool of it behind its proxy (`rollout_train.pods.sandboxes`), and the run's runner acquires from all of them
through `PodPools`, which the runner holds as it holds any pool (docs/research/run-placement.md).

- **Which pods.** At each look (`discover`, `rollout_train.pods.routing.LeasedPools`), the pods whose leases the run
  holds, at the addresses their leases say: those that beat fresh are live and take new leases; the rest still answer
  for the leases they hold.
- **Acquiring.** A key with no lease goes to a live pod with room, the one with the most first (its pool's `capacity`,
  asked of every pod at once and never while another acquire chooses, less the acquires on their way to it or sent
  since, which only orders the pods); a pod that answers full (or does not answer) passes it to the next; when every pod
  is full, to the cluster's own pool of the kind (`fallback`, the section's `url`), where there is one; else
  `NoCapacity`. A pod that does not answer an acquire passes it on too, once it is asked to release what it may have
  made, and so does one that refuses the key (another run took it since the look). The driver's own refusal of a key
  whose claim lapsed (`admits`) is the acquire's. A key with a lease goes to where its lease is, and only there: its
  sandbox is never made again elsewhere.
- **Where each lease is** is kept (`placements`, in the run's directory) before the pod is asked, as `acquiring`, and
  as `held` once it answers, so a driver started again routes the leases of the runs it adopts. An acquire whose
  answer was lost (a timeout) is released on its pod at once, or, failing that, at the next sweep: no world is left
  behind unnamed.
- **Releasing and operations** go to where the key's lease is. A release that fails is kept (`releasing`) and tried
  again at each sweep; the key gets `SandboxLost` meanwhile.
- **Capacity** is the sum of the live pods' pools' and the fallback's, so a runner asks one pool for room as before.
- **A lost pod.** A lease whose pod the run no longer holds (released, deleted, taken by another run) is lost: its key
  gets `SandboxLost`, and its episode is played again. A pod that only misses beats keeps its leases. A pod's pool
  says `SandboxLost` itself for a lease it ended (its run's driver was gone, or it was started again).
- **Claims.** As a pool beside the ledger: `admits` refuses a key whose claim has lapsed (releasing its lease), and a
  keeper (`rollout_train.sandboxes.keep`) sweeps this pool, releasing on its pod each lease whose claim lapsed.
"""

import asyncio
import contextlib
import json
import logging
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import httpx
from pydantic import JsonValue

from rollout.contracts import ToolResult, ToolSpecification
from rollout.harness.imports import deduplicates
from rollout.harness.sandboxes import (
    Capacity,
    Lease,
    LeaseRefused,
    NoCapacity,
    Pool,
    SandboxLost,
    SandboxSpec,
)
from rollout_train.ledger import locked

__all__ = [
    "ACQUIRING",
    "FALLBACK",
    "HELD",
    "LOST",
    "RELEASING",
    "FilePlacements",
    "MemoryPlacements",
    "Placement",
    "Placements",
    "PodPools",
    "Reached",
]

log = logging.getLogger(__name__)

FALLBACK = "_cluster"
"""Where a lease of the fallback pool is (no pod's name: those are lowercase letters, digits and hyphens)."""
LOOK = 2.0
"""Seconds a look at the run's pods is kept for."""
ACQUIRING, HELD, RELEASING, LOST = "acquiring", "held", "releasing", "lost"
"""A placement's states: asked of its pod (no answer yet), held, being released (a release failed), lost."""


@dataclass(frozen=True)
class Reached:
    """A pod's pool, as a look found it: `live` while the pod beats fresh (only a live pod takes new leases)."""

    pool: Pool
    live: bool = True


@dataclass(frozen=True)
class Placement:
    """Where a key's lease is: the pod (or `FALLBACK`), the pod's handle for its sandbox, and its state."""

    key: str
    where: str
    handle: str = ""
    state: str = ACQUIRING
    at: float = 0.0


class Placements(Protocol):
    async def get(self, key: str) -> Placement | None: ...
    async def put(self, placement: Placement) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def all(self) -> list[Placement]: ...


class MemoryPlacements:
    def __init__(self) -> None:
        self.placements: dict[str, Placement] = {}

    async def get(self, key: str) -> Placement | None:
        return self.placements.get(key)

    async def put(self, placement: Placement) -> None:
        self.placements[placement.key] = placement

    async def delete(self, key: str) -> None:
        self.placements.pop(key, None)

    async def all(self) -> list[Placement]:
        return list(self.placements.values())


class FilePlacements:
    """`Placements` in a JSON file (in the run's directory), written whole under the directory's lock."""

    def __init__(self, path: Path) -> None:
        self.path = path

    async def get(self, key: str) -> Placement | None:
        return (await asyncio.to_thread(self._read_locked)).get(key)

    async def put(self, placement: Placement) -> None:
        await asyncio.to_thread(self._change, lambda found: {**found, placement.key: placement})

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._change, lambda found: {k: v for k, v in found.items() if k != key})

    async def all(self) -> list[Placement]:
        return list((await asyncio.to_thread(self._read_locked)).values())

    def _read_locked(self) -> dict[str, Placement]:
        with locked(self.path.parent):
            return self._read()

    def _change(self, change: Callable[[dict[str, Placement]], dict[str, Placement]]) -> None:
        with locked(self.path.parent):
            found = change(self._read())
            staged = self.path.with_suffix(".staged")
            staged.write_text(json.dumps([asdict(each) for each in found.values()]))
            staged.replace(self.path)

    def _read(self) -> dict[str, Placement]:
        if not self.path.exists():
            return {}
        return {each["key"]: Placement(**each) for each in json.loads(self.path.read_text())}


class PodPools:
    """A `Pool` of `kind` over the pools of the pods `discover` finds (by pod name), with `fallback` for when every pod
    is full; where each lease is, kept in `placements`. `admits` says whether a key may hold a lease now (beside the
    ledger: whether its claim holds)."""

    def __init__(
        self,
        kind: str,
        discover: Callable[[], Awaitable[Mapping[str, Reached]]],
        *,
        name: str | None = None,
        placements: Placements | None = None,
        admits: Callable[[str], Awaitable[bool]] | None = None,
        fallback: Pool | None = None,
        look: float = LOOK,
    ) -> None:
        self._kind = kind
        self.discover = discover
        self.name = name or f"{kind}@pods"
        self.placements: Placements = placements or MemoryPlacements()
        self.admits = admits
        self.fallback = fallback
        self.look = look
        self._held: dict[str, Placement] = {}
        self._loaded = False
        self._making: set[str] = set()
        self._sent: Counter[str] = Counter()
        """Acquires sent to each pod, ever."""
        self._pending: Counter[str] = Counter()
        """Acquires on their way to each pod now. With those sent since a pod's room was asked, they order the pods (a
        pod's room less the more of the two), so acquires at once spread; they exclude none."""
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
            placement = self._held.get(key)
            if self.admits is not None and not await self.admits(key):
                if placement is not None:
                    await self._end(placement)
                raise LeaseRefused(f"{key} may hold no sandbox: the claim it names no longer holds")
            self._making.add(key)
            try:
                if placement is not None:
                    return await self._again(placement, spec, environment)
                _, pool, lease = await self._placed(spec, key, environment)
            finally:
                self._making.discard(key)
            await self._describe(pool)
            return lease

    async def release(self, key: str) -> None:
        await self._load()
        async with self._lock(key):
            placement = self._held.get(key)
            if placement is not None:
                await self._end(placement)

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
        placement = self._held.get(key)
        if placement is None or placement.state != HELD:
            raise SandboxLost(f"the pool {self.name} holds no live sandbox of {key}")
        pool = self._pool_at(placement.where, await self._reached())
        if pool is None:
            raise SandboxLost(f"the sandbox of {key} is gone with pod {placement.where}")
        return await pool.call(key, name, arguments, effect_id=effect_id, arguments_digest=arguments_digest)

    async def held(self) -> list[Lease]:
        """Its leases, as leases (`handle`: `POD/HANDLE`; `lost` for those not held), for its keeper."""
        await self._load()
        return [
            Lease(key=each.key, kind=self.kind, pool=self.name, handle=f"{each.where}/{each.handle}", at=each.at,
                  lost=each.state != HELD)
            for each in self._held.values()
        ]  # fmt: skip

    async def sweep(self, ended: Callable[[Lease], bool] = lambda lease: False) -> list[str]:
        """Release, on their pods, the leases `ended` says have ended and those whose release or acquire did not finish;
        mark lost those whose pod the run no longer holds. Returns the keys released or marked lost."""
        await self._load()
        reached = await self._reached(fresh=True)
        gone: list[str] = []
        for lease in await self.held():
            placement = self._held.get(lease.key)
            if placement is None or lease.key in self._making:
                continue
            if ended(lease) or placement.state in (ACQUIRING, RELEASING):
                await self.release(lease.key)
                if lease.key not in self._held:
                    gone.append(lease.key)
            elif placement.state == HELD and self._pool_at(placement.where, reached) is None:
                async with self._lock(lease.key):
                    await self._put(replace(placement, state=LOST))
                gone.append(lease.key)
        return gone

    async def close(self) -> None:
        """Release every lease it holds, on its pod (a pod's own pool ends what is left)."""
        for lease in await self.held():
            with contextlib.suppress(Exception):
                await self.release(lease.key)

    async def _placed(
        self, spec: SandboxSpec, key: str, environment: Mapping[str, str] | None
    ) -> tuple[str, Pool, Lease]:
        """A new lease of `key` (hold its lock): on a live pod with room, the most first, the next while each answers
        full or does not answer, then the fallback."""
        live = {where: each.pool for where, each in (await self._reached()).items() if each.live}
        before = dict(self._sent)  # (what was sent to each pod before its room was asked: its answer counts those)
        asked = await asyncio.gather(*(pool.capacity() for pool in live.values()), return_exceptions=True)
        free = {where: found.free for where, found in zip(live, asked, strict=True) if isinstance(found, Capacity)}
        async with self._choosing:  # (no network here: only the order, from what was asked)

            def room(where: str) -> int:
                return free[where] - max(self._pending[where], self._sent[where] - before.get(where, 0))

            order = sorted((where for where in free if free[where] > 0), key=lambda each: (-room(each), each))
            if order:
                self._sent[order[0]] += 1
                self._pending[order[0]] += 1
        for number, where in enumerate(order):
            if number:
                self._sent[where] += 1
                self._pending[where] += 1
            try:
                lease = await self._asked(where, live[where], spec, key, environment)
            except NoCapacity:
                continue
            except LeaseRefused as error:  # (a pod another run took since the look: the key is not its run's)
                log.info("%s refused %s: %s; trying the next pod", where, key, error)
                continue
            except httpx.TransportError as error:  # (the pod did not answer: released there, as far as it can be)
                log.warning("acquiring %s on %s failed: %s; trying the next pod", key, where, error)
                continue
            finally:
                self._pending[where] -= 1
            return where, live[where], lease
        if self.fallback is not None:
            return FALLBACK, self.fallback, await self._asked(FALLBACK, self.fallback, spec, key, environment)
        raise NoCapacity(f"the pool {self.name}: every pod's pool is full")

    async def _asked(
        self, where: str, pool: Pool, spec: SandboxSpec, key: str, environment: Mapping[str, str] | None
    ) -> Lease:
        """Acquire `key` from the pool at `where`, its placement kept before it is asked: forgotten where the pool says
        no; released where its answer was lost (a timeout, a dropped connection), at once or at the next sweep."""
        await self._put(Placement(key, where, state=ACQUIRING, at=round(time.time(), 1)))
        try:
            lease = await pool.acquire(spec, key, environment)
        except (NoCapacity, LeaseRefused, SandboxLost):
            await self._forget(key)
            raise
        except Exception:
            try:
                await pool.release(key)
            except Exception as error:  # (its placement stays `acquiring`: the next sweep releases it)
                log.warning("releasing %s on %s after a lost acquire failed: %s", key, where, error)
            else:
                await self._forget(key)
            raise
        await self._put(Placement(key, where, lease.handle, HELD, round(time.time(), 1)))
        return lease

    async def _again(self, placement: Placement, spec: SandboxSpec, environment: Mapping[str, str] | None) -> Lease:
        """The lease of a key that has one, from where it is; `SandboxLost` where that is gone. Hold its lock."""
        if placement.state in (LOST, RELEASING):
            if placement.state == LOST:
                await self._forget(placement.key)
            raise SandboxLost(f"the sandbox of {placement.key} is gone")
        pool = self._pool_at(placement.where, await self._reached())
        if pool is None:
            await self._forget(placement.key)
            raise SandboxLost(f"the sandbox of {placement.key} is gone with pod {placement.where}")
        if placement.state == ACQUIRING:  # (an acquire whose answer was lost: asked again, where it was asked)
            return await self._asked(placement.where, pool, spec, placement.key, environment)
        try:
            lease = await pool.acquire(spec, placement.key, environment)
        except SandboxLost:
            await self._forget(placement.key)
            raise
        await self._describe(pool)
        return lease

    async def _end(self, placement: Placement) -> None:
        """Release a lease where it is and forget it; one whose release fails is kept as `releasing`, for the next
        sweep. Hold its key's lock."""
        pool = None if placement.state == LOST else self._pool_at(placement.where, await self._reached())
        if pool is not None:
            try:
                await pool.release(placement.key)
            except Exception as error:
                log.warning("releasing %s on %s failed: %s; tried again at the next sweep", placement.key,
                            placement.where, error)  # fmt: skip
                await self._put(replace(placement, state=RELEASING))
                return
        await self._forget(placement.key)

    async def _put(self, placement: Placement) -> None:
        await self.placements.put(placement)
        self._held[placement.key] = placement

    async def _forget(self, key: str) -> None:
        self._held.pop(key, None)
        await self.placements.delete(key)

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
            self._held = {each.key: each for each in await self.placements.all()}
            self._loaded = True

    def _lock(self, key: str) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())
