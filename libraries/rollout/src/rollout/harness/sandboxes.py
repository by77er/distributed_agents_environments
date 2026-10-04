"""Sandboxes: what a program runs against, leased for the run (docs/libraries/rollout/sandboxes.md).

A program declares the sandboxes it needs by name (`Program.sandboxes()`), each a `SandboxSpec` of a kind: a
Minecraft world, a container, a computer. The runner acquires each from the pool its binding names for the kind
before the program starts, and releases it when the program ends; the program reaches it as `run.sandbox(name)`: its
addresses, the environment variables a harness inside it is given, and its operations, each a recorded effect.

A `Pool` hands out sandboxes under leases. `acquire(spec, key)` returns the lease of `key` when there is one, so a
retried or replayed acquire gets the same sandbox. A pool says how many sandboxes it can hold and how many are free.
`SandboxPool` is a pool over a `Provider`, which only makes, deletes and operates sandboxes of one kind; it keeps its
leases in a `Leases` table: in memory, or beside a ledger (`rollout_train.sandboxes`), where a lease ends with the
claim it was acquired under and a key whose claim has lapsed is refused.
"""

import asyncio
import contextlib
import hashlib
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Protocol, Self

from pydantic import Field, JsonValue, model_validator

from rollout.contracts import (
    ContractModel,
    EffectKind,
    FrozenSequence,
    ModelAddress,
    RetryClass,
    ToolResult,
    ToolSpecification,
)
from rollout.harness.model import Effects


class Process(ContractModel):
    """A process the sandbox runs from its start (an environment's worker, a coding agent): the pool launches it."""

    command: FrozenSequence[str]
    environment: Mapping[str, str] = Field(default_factory=dict[str, str])
    """Its own variables; the lease's (its slots' model addresses) are added to them."""
    directory: str | None = None
    """Its working directory, inside the sandbox."""


class Mount(ContractModel):
    """Files the sandbox sees, read-only: an environment version's files, its virtual environment."""

    source: str
    """Where they are, as the pool finds them: a path on its machine, or a name it resolves."""
    target: str
    """Where they appear inside the sandbox."""


class Scratch(ContractModel):
    """A directory the sandbox may write, empty when it starts."""

    path: str = "/scratch"
    mib: int = 1024
    """The most it may hold, in MiB."""


class Network(ContractModel):
    """What the sandbox may reach besides its connection to the runner (its lease's addresses): nothing, unless hosts
    are allowed."""

    allow: FrozenSequence[str] = ()
    """Hosts it may reach (`pypi.org`, say)."""


class SandboxLimits(ContractModel):
    """What the sandbox may use. Unset: as much as the pool gives."""

    cpus: float | None = None
    memory_mib: int | None = None
    processes: int | None = None
    seconds: float | None = None
    """Wall time from its start: past it, its lease ends and the pool deletes it."""


class SandboxSpec(ContractModel):
    """A sandbox a program needs: its kind, what it is made from, and what it may do. A world for an episode needs
    only a kind and parameters; a worker for an environment names a process, mounts, scratch, network and limits."""

    kind: str
    """The kind of sandbox (`minecraft`, say): the binding names the pool that serves each kind."""
    parameters: Mapping[str, JsonValue] = Field(default_factory=dict[str, JsonValue])
    """What the pool makes it from: a task and its seeds, an image."""
    slots: FrozenSequence[str] = ()
    """Model slots a harness inside the sandbox samples. Each one's address is put in the sandbox's environment:
    `OPENAI_BASE_URL`, `OPENAI_API_KEY` and `OPENAI_MODEL`, suffixed with the slot's name in capitals (`_AGENT_1`),
    and unsuffixed too when there is one slot. A key names the run's session of its slot, and stops working once the
    recorder forgets the run: an episode runner has it forget the run as the episode ends."""
    process: Process | None = None
    mounts: FrozenSequence[Mount] = ()
    scratch: Scratch | None = None
    """Without it, the sandbox writes nowhere."""
    network: Network = Network()
    limits: SandboxLimits = SandboxLimits()


class Reach(ContractModel):
    """How a sandbox is reached, as its provider says once it has made it."""

    addresses: Mapping[str, str] = Field(default_factory=dict[str, str])
    """Where its services listen, by name (`{"game": "127.0.0.1:25565"}`, say)."""
    environment: Mapping[str, str] = Field(default_factory=dict[str, str])
    """Environment variables for what runs inside it: those it was given, and any of the provider's own."""


class Lease(ContractModel):
    """A sandbox held under a key: what a pool hands out, and what its `Leases` table keeps."""

    key: str
    """What it was acquired under: a run's lease and the sandbox's name (`RUN/GROUP/EPISODE/ATTEMPT/world` for an
    episode, whose claim it ends with)."""
    kind: str
    pool: str
    """The pool that holds it."""
    handle: str
    """The pool's name for the sandbox."""
    addresses: Mapping[str, str] = Field(default_factory=dict[str, str])
    environment: Mapping[str, str] = Field(default_factory=dict[str, str])
    at: float = 0.0
    """When it was made, in seconds since the epoch."""
    ends: float | None = None
    """When its wall time is over (`SandboxLimits.seconds`), in seconds since the epoch."""
    lost: bool = False
    """Its sandbox is gone (it ended with the pool's process, say): the key cannot have it back."""


class Capacity(ContractModel):
    size: int
    """How many sandboxes the pool can hold at once."""
    leased: int
    """How many it holds, or is making, now."""

    @property
    def free(self) -> int:
        return max(self.size - self.leased, 0)

    def to_json(self) -> dict[str, JsonValue]:
        return {"size": self.size, "leased": self.leased, "free": self.free}


class NoCapacity(Exception):
    """The pool holds as many sandboxes as it can; an acquire may succeed once one is released."""


class LeaseRefused(Exception):
    """The key may hold no lease now: the claim it was acquired under no longer holds."""


class SandboxLost(Exception):
    """The key's sandbox is gone, and a new one would not be the one its run was using."""


class PoolBinding(ContractModel):
    """How a kind of sandbox is served. Exactly one kind is set."""

    local: str | None = None
    """The name of a pool registered with the runner, in process."""
    url: str | None = None
    """A pool served over HTTP (`rollout.harness.remote.serve_pool`), wherever its sandboxes live."""

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        if (self.local is None) == (self.url is None):
            raise ValueError("a pool binding is exactly one of `local` or `url`")
        return self


class Provider(Protocol):
    """Makes, deletes and operates sandboxes of one kind: Paper servers, containers, a provider's API. A
    `SandboxPool` leases them out."""

    @property
    def kind(self) -> str: ...

    @property
    def size(self) -> int:
        """How many sandboxes it can hold at once."""
        ...

    def operations(self) -> Sequence[ToolSpecification]:
        """What can be done to one of its sandboxes (none: a harness inside reaches it by its addresses)."""
        ...

    async def create(self, handle: str, spec: SandboxSpec, environment: Mapping[str, str]) -> Reach:
        """Make the sandbox `handle`, giving what runs inside it `environment`, or say how to reach it if it is
        there already."""
        ...

    async def delete(self, handle: str) -> None:
        """Delete the sandbox, or do nothing if it is gone."""
        ...

    async def held(self) -> Sequence[str]:
        """The handles of the sandboxes it has now."""
        ...

    async def call(
        self, handle: str, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
        """Perform an operation on a sandbox. Errors the operation reports are results with `is_error`; exceptions
        are platform failures."""
        ...


class Pool(Protocol):
    """Hands out sandboxes of one kind under leases: `SandboxPool`, or one served over HTTP (`RemotePool`). Runners
    acquire and release; programs perform operations through `run.sandbox(name)`."""

    @property
    def deduplicates(self) -> bool:
        """Whether it performs each operation's `effect_id` at most once."""
        ...

    def operations(self) -> Sequence[ToolSpecification]: ...

    async def acquire(self, spec: SandboxSpec, key: str, environment: Mapping[str, str] | None = None) -> Lease:
        """The lease of `key`: the one there is, or a new sandbox. Raises `NoCapacity` when the pool is full,
        `LeaseRefused` for a key that may hold no lease now (its claim lapsed), and `SandboxLost` for a key whose
        sandbox is gone."""
        ...

    async def release(self, key: str) -> None:
        """End the lease of `key` and delete its sandbox; nothing if there is no such lease."""
        ...

    async def capacity(self) -> Capacity: ...

    async def call(
        self, key: str, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
        """Perform an operation on the sandbox leased under `key`."""
        ...


class Leases(Protocol):
    """Where a pool keeps its leases, by key: ordinary state, changed in place."""

    async def get(self, key: str) -> Lease | None: ...
    async def put(self, lease: Lease) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def all(self) -> list[Lease]: ...


class MemoryLeases:
    """`Leases` in this process: they end with it."""

    def __init__(self) -> None:
        self.leases: dict[str, Lease] = {}

    async def get(self, key: str) -> Lease | None:
        return self.leases.get(key)

    async def put(self, lease: Lease) -> None:
        self.leases[lease.key] = lease

    async def delete(self, key: str) -> None:
        self.leases.pop(key, None)

    async def all(self) -> list[Lease]:
        return list(self.leases.values())


def handle_of(key: str) -> str:
    """The handle a pool gives the sandbox of `key`: the same for every acquire of the key, so a creation that a
    crash interrupted is found again."""
    return "s-" + hashlib.sha256(key.encode()).hexdigest()[:16]


class SandboxPool:
    """A `Pool` over a `Provider`: at most `provider.size` leases at once, kept in `leases` under the pool's `name`
    (by default the provider's kind; several pools sharing a table need names of their own)."""

    def __init__(
        self,
        provider: Provider,
        *,
        name: str | None = None,
        leases: Leases | None = None,
        admits: Callable[[str], Awaitable[bool]] | None = None,
    ) -> None:
        """`admits` says whether a key may hold a lease now (beside a ledger: whether its claim holds,
        `rollout_train.sandboxes.admits`); without it, every key may."""
        self.provider = provider
        self.name = name or provider.kind
        self.leases: Leases = leases or MemoryLeases()
        self.admits = admits
        self._held: dict[str, Lease] = {}
        """This pool's leases, by key (read from `leases` once, then kept here)."""
        self._loaded = False
        self._making: set[str] = set()
        """Keys whose sandboxes are being made."""
        self._made: set[str] = set()
        """Handles this process made and has not deleted: never taken for a sandbox no lease names."""
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def deduplicates(self) -> bool:
        return bool(getattr(self.provider, "deduplicates", False))

    def operations(self) -> Sequence[ToolSpecification]:
        return self.provider.operations()

    async def acquire(self, spec: SandboxSpec, key: str, environment: Mapping[str, str] | None = None) -> Lease:
        """The lease of `key`, or a new sandbox. Raises `LeaseRefused` for a key `admits` refuses (releasing a lease
        it has), `SandboxLost` for a key whose sandbox is gone, and `NoCapacity` when the pool is full."""
        if spec.kind != self.provider.kind:
            raise ValueError(f"the pool {self.name} makes {self.provider.kind} sandboxes, not {spec.kind}")
        await self._load()
        async with self._lock(key):
            lease = self._held.get(key) or await self.leases.get(key)
            if lease is not None and lease.pool != self.name:
                raise RuntimeError(f"{key} is leased from the pool {lease.pool}, not {self.name}")
            if self.admits is not None and not await self.admits(key):
                if lease is not None:
                    await self._end(lease)
                raise LeaseRefused(f"{key} may hold no sandbox: the claim it names no longer holds")
            if lease is not None:
                if not lease.lost and (lease.handle in self._made or lease.handle in await self.provider.held()):
                    self._held[key] = lease
                    return lease
                await self._end(lease)
                raise SandboxLost(f"the sandbox of {key} is gone")
            if len(self._live() - {key}) + len(self._making) >= self.provider.size:
                raise NoCapacity(f"the pool {self.name} holds {self.provider.size} sandboxes, as many as it can")
            handle = handle_of(key)
            self._making.add(key)
            self._made.add(handle)
            try:
                reach = await self.provider.create(handle, spec, dict(environment or {}))
                at = time.time()
                lease = Lease(
                    key=key,
                    kind=spec.kind,
                    pool=self.name,
                    handle=handle,
                    addresses=reach.addresses,
                    environment={**(environment or {}), **reach.environment},
                    at=round(at, 1),
                    ends=at + spec.limits.seconds if spec.limits.seconds is not None else None,
                )
                await self.leases.put(lease)
            except BaseException:
                with contextlib.suppress(Exception):
                    await asyncio.shield(self.provider.delete(handle))
                self._made.discard(handle)
                raise
            finally:
                self._making.discard(key)
            self._held[key] = lease
            return lease

    async def release(self, key: str) -> None:
        await self._load()
        async with self._lock(key):
            lease = self._held.get(key) or await self.leases.get(key)
            if lease is not None and lease.pool == self.name:
                await self._end(lease)

    async def capacity(self) -> Capacity:
        await self._load()
        return Capacity(size=self.provider.size, leased=len(self._live() | self._making))

    async def call(
        self, key: str, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
        return await self.provider.call(
            handle_of(key), name, arguments, effect_id=effect_id, arguments_digest=arguments_digest
        )

    async def held(self) -> list[Lease]:
        """This pool's leases, those whose sandboxes are lost included."""
        await self._load()
        return list(self._held.values())

    async def sweep(self, ended: Callable[[Lease], bool] = lambda lease: False) -> list[str]:
        """Release the leases `ended` says have ended, and those past their wall time; mark lost those whose sandbox
        is gone (the pool's process was started again, say), which their keys cannot have back; and delete the
        sandboxes no lease names. Returns the keys released or marked lost."""
        await self._load()
        there = set(await self.provider.held())
        now = time.time()
        gone: list[str] = []
        for lease in list(self._held.values()):
            if lease.key in self._making:
                continue
            if ended(lease) or (lease.ends is not None and now >= lease.ends):
                await self.release(lease.key)
                gone.append(lease.key)
            elif not lease.lost and lease.handle not in there and lease.handle not in self._made:
                async with self._lock(lease.key):
                    lost = lease.model_copy(update={"lost": True})
                    await self.leases.put(lost)
                    self._held[lease.key] = lost
                gone.append(lease.key)
        named = {lease.handle for lease in self._held.values() if not lease.lost} | self._made
        for handle in there - named:
            await self.provider.delete(handle)
        return gone

    async def close(self, *, release: bool = True) -> None:
        """Release every lease the pool holds (deleting its sandboxes), and close the provider. With `release` False,
        the leases stay, for runs a durable runner resumes to acquire again: a sandbox that outlived the provider is
        theirs again, and one that did not is lost."""
        for lease in await self.held() if release else []:
            with contextlib.suppress(Exception):
                await self.release(lease.key)
        closing = getattr(self.provider, "close", None)
        if closing is not None:
            await closing()

    async def _end(self, lease: Lease) -> None:
        """Delete a lease's sandbox and forget the lease. Hold its key's lock."""
        if not lease.lost:
            await self.provider.delete(lease.handle)
        self._made.discard(lease.handle)
        self._held.pop(lease.key, None)
        await self.leases.delete(lease.key)

    def _live(self) -> set[str]:
        """The keys of the leases whose sandboxes are there."""
        return {key for key, lease in self._held.items() if not lease.lost}

    async def _load(self) -> None:
        if not self._loaded:
            self._held = {lease.key: lease for lease in await self.leases.all() if lease.pool == self.name}
            self._loaded = True

    def _lock(self, key: str) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())


def deduplicating(pool: Pool) -> bool:
    return bool(getattr(pool, "deduplicates", False))


class Sandbox:
    """A sandbox a run holds, as `run.sandbox(name)` gives it: how to reach it, and its operations, each a
    `tool.call` effect."""

    def __init__(self, name: str, lease: Lease, pool: Pool, effects: Effects) -> None:
        self.name = name
        self.lease = lease
        self._pool = pool
        self._effects = effects

    @property
    def addresses(self) -> Mapping[str, str]:
        return self.lease.addresses

    @property
    def environment(self) -> Mapping[str, str]:
        """For what runs inside it: those the runner gave it (its slots' model addresses) and the pool's own."""
        return self.lease.environment

    def specifications(self) -> list[ToolSpecification]:
        """Its operations."""
        return list(self._pool.operations())

    async def call(self, operation: str, arguments: Mapping[str, JsonValue] | None = None) -> ToolResult:
        """Perform an operation as a `tool.call` effect. A side-effecting one is guarded unless the pool
        deduplicates: after a crash it completes as `OUTCOME_UNKNOWN` rather than happen twice."""
        given = dict(arguments or {})

        async def execute(effect_id: str, arguments_digest: str) -> ToolResult:
            return await self._pool.call(
                self.lease.key, operation, given, effect_id=effect_id, arguments_digest=arguments_digest
            )

        specification = next((each for each in self._pool.operations() if each.name == operation), None)
        retry_class = specification.retry_class if specification is not None else RetryClass.UNKNOWN
        side_effecting = retry_class in (RetryClass.SIDE_EFFECTING, RetryClass.UNKNOWN)
        return await self._effects.perform(
            EffectKind.TOOL_CALL,
            {"sandbox": self.name, "tool": operation, "arguments": given},
            execute,
            completion=lambda result: result.model_dump(mode="json", exclude_none=True),
            guard=side_effecting and not deduplicating(self._pool),
        )


def harness_environment(slots: Sequence[str], addresses: Callable[[str], ModelAddress]) -> dict[str, str]:
    """The environment variables that point a harness at its slots' models (`SandboxSpec.slots`)."""
    environment: dict[str, str] = {}
    for slot in slots:
        address = addresses(slot)
        values = {"OPENAI_BASE_URL": address.base_url, "OPENAI_API_KEY": address.api_key, "OPENAI_MODEL": address.model}
        suffix = "_" + re.sub(r"[^A-Z0-9]", "_", slot.upper())
        environment |= {name + suffix: value for name, value in values.items()}
        if len(slots) == 1:
            environment |= values
    return environment


ACQUIRE_SECONDS = 300.0
"""How long a runner waits for room in a full pool before the run fails."""
RETRY_SECONDS = 1.0


async def acquire(
    pool: Pool, spec: SandboxSpec, key: str, environment: Mapping[str, str], *, seconds: float | None = None
) -> Lease:
    """Acquire from `pool`, waiting up to `seconds` (`ACQUIRE_SECONDS`) while it is full: runners that share a pool
    can claim more than it holds at once."""
    deadline = time.monotonic() + (ACQUIRE_SECONDS if seconds is None else seconds)
    while True:
        try:
            return await pool.acquire(spec, key, environment)
        except NoCapacity:
            if time.monotonic() >= deadline:
                raise
            await asyncio.sleep(RETRY_SECONDS)
