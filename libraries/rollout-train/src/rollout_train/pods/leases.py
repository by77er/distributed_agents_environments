"""Pods' leases and the time each run held a pod: ordinary state kept beside the ledger, changed by compare-and-set.

A **lease** (`PodLease`) is a pod the platform rents: its name (the key, and its certificate's identity), RunPod's id
for it, its provider, slot, GPU type, hourly price, image and model, and which run holds it now. A pod is held by one
run at a time. Its state is `starting` (asked for, not yet ready for its run), `held` (its run has it), or `idle`
(released: no run holds it, and it stays warm for the next run with the same image, model and GPU until its
provider's `idle_stop` passes). The run renews its lease on a heartbeat (`renewed`); a lease not renewed for long is
stale, and the reaper deletes its pod (`rollout_train.pods.leasing.reap`).

Every change is a compare-and-set on the lease's `version`: a run takes an idle pod only if no one changed the lease
since it read it, so two runs never take one pod (`Conflict` to the one that lost). A provider's pods have slots, 0 to
its `max_pods` less one, and a slot holds one lease at a time, so no provider ever has more pods than `max_pods`.

**Pod time** (`PodTime`) is what a run is charged for a pod: one entry per run and pod and when it took the pod, from
then (`since`) to its newest renewal or its release (`until`), and the warm time after its release (`idle`), until the
pod was taken by another run or deleted. A pod's warm time is charged to the run that last held it.

Kept as rows of two tables in a database ledger's database (`DatabasePodLeases`: `pod_leases`, `pod_time`), as a file
beside a ledger of files (`FilePodLeases`), or reached through the ledger service (`HttpPodLeases`). Times are the
store's clock (`now`), so every reader ages a lease by the same clock it was stamped with.
"""

import asyncio
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import JsonValue

from rollout_train.ledger import FileLedger, Ledger, locked
from rollout_train.ledger_service.wire import Conflict

if TYPE_CHECKING:
    from rollout_train.ledger_service import HttpLedger
    from rollout_train.sql import Connection, Database

__all__ = [
    "HELD",
    "IDLE",
    "STARTING",
    "DatabasePodLeases",
    "FilePodLeases",
    "HttpPodLeases",
    "PodLease",
    "PodLeases",
    "PodTime",
    "pod_leases_of",
]

STARTING = "starting"
"""A lease's state while its pod is asked for and not yet ready for its run."""
HELD = "held"
"""A lease's state while its run holds its pod."""
IDLE = "idle"
"""A lease's state once its run released its pod: no run holds it, and it is kept warm."""


@dataclass(frozen=True)
class PodLease:
    pod: str
    """The pod's name: the key, and its certificate's identity (`spiffe://rollout/pod/NAME`)."""
    provider: str
    slot: int
    """Its place among its provider's `max_pods`."""
    role: str
    """What it does: `inference`, `trainer`, or `host` (both)."""
    image: str
    model: str
    gpu: str
    """The GPU type, as RunPod says it gave it (the first asked for until it says)."""
    price: float
    """Dollars an hour: RunPod's `costPerHr` for it, else the provider's `price`."""
    cloud: str = "SECURE"
    id: str | None = None
    """RunPod's id for it, once asked for."""
    run: str | None = None
    """The run that holds it (none: idle)."""
    channel: str | None = None
    """The run's channel it serves (an inference or host pod)."""
    token: str | None = None
    """The ledger service's token for the pod and the run that holds it: read by the pod itself and the platform."""
    settings: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """What the run asks of the pod beside its channel (a trainer's implementation and settings)."""
    state: str = STARTING
    created: float = 0.0
    """When it was asked for."""
    held: float | None = None
    """Since when its run has held it."""
    renewed: float | None = None
    """When its run last renewed it."""
    released: float | None = None
    """When it was released (idle since)."""
    version: int = 0
    """Its number of changes: what a compare-and-set compares."""

    def to_json(self) -> dict[str, JsonValue]:
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_json(cls, said: Mapping[str, Any]) -> "PodLease":
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in said.items() if key in known})

    def shown(self) -> dict[str, JsonValue]:
        """What may be shown of it: everything but its token."""
        return {key: value for key, value in self.to_json().items() if key != "token"}


@dataclass(frozen=True)
class PodTime:
    """A run's time on a pod, charged at its price."""

    key: str
    """`POD/RUN/SINCE`."""
    pod: str
    run: str
    provider: str
    gpu: str
    price: float
    since: float
    """When the run took the pod (asked for it, for a pod started for the run)."""
    until: float
    """Its newest renewal, or its release."""
    idle: float = 0.0
    """Seconds it stayed warm after its release, charged to this run."""
    released: bool = False
    closed: bool = False
    """No more time is charged to it: the pod was taken by another run, or deleted."""

    @property
    def seconds(self) -> float:
        return max(0.0, self.until - self.since) + max(0.0, self.idle)

    @property
    def dollars(self) -> float:
        return self.seconds * self.price / 3600

    def to_json(self) -> dict[str, JsonValue]:
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_json(cls, said: Mapping[str, Any]) -> "PodTime":
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in said.items() if key in known})


def time_key(pod: str, run: str, since: float) -> str:
    return f"{pod}/{run}/{since:.1f}"


class PodLeases(Protocol):
    async def now(self) -> float:
        """The store's clock, in seconds since the epoch."""
        ...

    async def all(self) -> list[PodLease]:
        """Every lease, by pod."""
        ...

    async def get(self, pod: str) -> PodLease | None: ...

    async def put(self, lease: PodLease, *, expect: int | None) -> PodLease:
        """Write a lease if the one there has version `expect` (none: there is none, and no other lease has its
        provider's slot), as version `expect + 1` (1 for a new one); the lease written. Raises `Conflict` where it
        changed since."""
        ...

    async def delete(self, pod: str, *, expect: int) -> None:
        """Delete a lease that has version `expect`. Raises `Conflict` where it changed since (or is gone)."""
        ...

    async def times(self, run: str | None = None) -> list[PodTime]:
        """The pod time charged to `run` (every run's, by default), oldest first."""
        ...

    async def charge(self, entry: PodTime) -> None:
        """Write a run's time on a pod, in place of what was written under its key."""
        ...


def pod_leases_of(ledger: Ledger) -> PodLeases | None:
    """The pods' leases beside a ledger: a file beside a ledger of files, tables in a database ledger's database, the
    service's for a ledger reached through it."""
    if isinstance(ledger, FileLedger):
        return FilePodLeases(ledger.directory)
    return getattr(ledger, "pods", None)


def _written(lease: PodLease, there: PodLease | None, expect: int | None, others: list[PodLease]) -> PodLease:
    """The lease to write, checked against what is there (`there`, and the provider's other leases)."""
    if expect is None:
        if there is not None:
            raise Conflict(f"there is a lease of {lease.pod} already")
        if any(each.provider == lease.provider and each.slot == lease.slot for each in others):
            raise Conflict(f"slot {lease.slot} of {lease.provider} is taken")
        return replace(lease, version=1)
    if there is None or there.version != expect:
        raise Conflict(f"the lease of {lease.pod} changed since it was read")
    return replace(lease, version=expect + 1)


class FilePodLeases:
    """`PodLeases` in `pods.json` beside a ledger of files, under the lock its files are written under."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path = directory / "pods.json"

    async def now(self) -> float:
        return time.time()

    def _read(self) -> tuple[dict[str, PodLease], dict[str, PodTime]]:
        if not self.path.exists():
            return {}, {}
        said = json.loads(self.path.read_text())
        leases = {key: PodLease.from_json(each) for key, each in said.get("leases", {}).items()}
        return leases, {key: PodTime.from_json(each) for key, each in said.get("times", {}).items()}

    def _write(self, leases: dict[str, PodLease], times: dict[str, PodTime]) -> None:
        staged = self.path.with_suffix(".staged")
        staged.write_text(json.dumps({"leases": {key: each.to_json() for key, each in leases.items()},
                                      "times": {key: each.to_json() for key, each in times.items()}}))  # fmt: skip
        staged.replace(self.path)

    async def all(self) -> list[PodLease]:
        def read() -> list[PodLease]:
            with locked(self.directory):
                return sorted(self._read()[0].values(), key=lambda each: each.pod)

        return await asyncio.to_thread(read)

    async def get(self, pod: str) -> PodLease | None:
        return next((each for each in await self.all() if each.pod == pod), None)

    async def put(self, lease: PodLease, *, expect: int | None) -> PodLease:
        def written() -> PodLease:
            with locked(self.directory):
                leases, times = self._read()
                made = _written(lease, leases.get(lease.pod), expect, list(leases.values()))
                leases[made.pod] = made
                self._write(leases, times)
                return made

        return await asyncio.to_thread(written)

    async def delete(self, pod: str, *, expect: int) -> None:
        def deleted() -> None:
            with locked(self.directory):
                leases, times = self._read()
                there = leases.get(pod)
                if there is None or there.version != expect:
                    raise Conflict(f"the lease of {pod} changed since it was read")
                del leases[pod]
                self._write(leases, times)

        await asyncio.to_thread(deleted)

    async def times(self, run: str | None = None) -> list[PodTime]:
        def read() -> list[PodTime]:
            with locked(self.directory):
                found = self._read()[1].values()
            return sorted((each for each in found if run is None or each.run == run), key=lambda each: each.since)

        return await asyncio.to_thread(read)

    async def charge(self, entry: PodTime) -> None:
        def written() -> None:
            with locked(self.directory):
                leases, times = self._read()
                times[entry.key] = entry
                self._write(leases, times)

        await asyncio.to_thread(written)


class DatabasePodLeases:
    """`PodLeases` in the `pod_leases` and `pod_time` tables of a database: a row per lease (its provider and slot
    unique) and per run's time on a pod. Stamped by the database's clock."""

    def __init__(self, database: "Database") -> None:
        self.database = database
        self._now = (
            "EXTRACT(EPOCH FROM clock_timestamp())" if database.shared else "(julianday('now') - 2440587.5) * 86400.0"
        )

    async def now(self) -> float:
        from rollout_train.sql import sql

        def read(connection: "Connection") -> float:
            return float(sql(connection, f"SELECT {self._now}").scalar_one())

        return await asyncio.to_thread(self.database.read, read)

    async def all(self) -> list[PodLease]:
        from rollout_train.sql import fetch_all

        def rows(connection: "Connection") -> list[tuple[Any, ...]]:
            return fetch_all(connection, "SELECT lease FROM pod_leases ORDER BY pod")

        return [PodLease.from_json(json.loads(row[0])) for row in await asyncio.to_thread(self.database.read, rows)]

    async def get(self, pod: str) -> PodLease | None:
        from rollout_train.sql import fetch_one

        def row(connection: "Connection") -> tuple[Any, ...] | None:
            return fetch_one(connection, "SELECT lease FROM pod_leases WHERE pod = :pod", {"pod": pod})

        found = await asyncio.to_thread(self.database.read, row)
        return PodLease.from_json(json.loads(found[0])) if found else None

    async def put(self, lease: PodLease, *, expect: int | None) -> PodLease:
        from rollout_train.sql import fetch_all, fetch_one, sql

        def written(connection: "Connection") -> PodLease:
            found = fetch_one(connection, "SELECT lease FROM pod_leases WHERE pod = :pod", {"pod": lease.pod})
            there = PodLease.from_json(json.loads(found[0])) if found else None
            others = [
                PodLease.from_json(json.loads(row[0]))
                for row in fetch_all(
                    connection, "SELECT lease FROM pod_leases WHERE provider = :provider", {"provider": lease.provider}
                )
            ]
            made = _written(lease, there, expect, others)
            values = {"pod": made.pod, "provider": made.provider, "slot": made.slot, "version": made.version,
                      "lease": json.dumps(made.to_json())}  # fmt: skip
            if expect is None:
                sql(connection, "INSERT INTO pod_leases (pod, provider, slot, version, lease) "
                    "VALUES (:pod, :provider, :slot, :version, :lease)", values)  # fmt: skip
            else:
                changed = sql(connection, "UPDATE pod_leases SET provider = :provider, slot = :slot, "
                              "version = :version, lease = :lease WHERE pod = :pod AND version = :expect",
                              {**values, "expect": expect})  # fmt: skip
                if changed.rowcount != 1:
                    raise Conflict(f"the lease of {lease.pod} changed since it was read")
            return made

        try:
            return await asyncio.to_thread(self.database.write, written, exclusive=f"pods:{lease.provider}")
        except Conflict:
            raise
        except Exception as error:
            if "pod_leases_slot" in str(error) or "UNIQUE" in str(error) or "unique" in str(error):
                raise Conflict(f"slot {lease.slot} of {lease.provider} is taken") from None
            raise

    async def delete(self, pod: str, *, expect: int) -> None:
        from rollout_train.sql import sql

        def deleted(connection: "Connection") -> None:
            gone = sql(connection, "DELETE FROM pod_leases WHERE pod = :pod AND version = :expect",
                       {"pod": pod, "expect": expect})  # fmt: skip
            if gone.rowcount != 1:
                raise Conflict(f"the lease of {pod} changed since it was read")

        await asyncio.to_thread(self.database.write, deleted, exclusive=f"pods:{pod}")

    async def times(self, run: str | None = None) -> list[PodTime]:
        from rollout_train.sql import fetch_all

        def rows(connection: "Connection") -> list[tuple[Any, ...]]:
            if run is None:
                return fetch_all(connection, "SELECT entry FROM pod_time")
            return fetch_all(connection, "SELECT entry FROM pod_time WHERE run = :run", {"run": run})

        found = [PodTime.from_json(json.loads(row[0])) for row in await asyncio.to_thread(self.database.read, rows)]
        return sorted(found, key=lambda each: each.since)

    async def charge(self, entry: PodTime) -> None:
        from rollout_train.sql import sql

        def written(connection: "Connection") -> None:
            sql(connection, "INSERT INTO pod_time (key, run, entry) VALUES (:key, :run, :entry) ON CONFLICT (key) "
                "DO UPDATE SET entry = excluded.entry", {"key": entry.key, "run": entry.run,
                                                          "entry": json.dumps(entry.to_json())})  # fmt: skip

        await asyncio.to_thread(self.database.write, written, exclusive=f"pod_time:{entry.key}")


class HttpPodLeases:
    """`PodLeases` through the ledger service."""

    def __init__(self, ledger: "HttpLedger") -> None:
        self._ledger = ledger

    async def now(self) -> float:
        return await self._ledger.now()

    async def all(self) -> list[PodLease]:
        return [PodLease.from_json(each) for each in await self._ledger.result("pods/all")]

    async def get(self, pod: str) -> PodLease | None:
        said = await self._ledger.result("pods/get", {"pod": pod})
        return PodLease.from_json(said) if said is not None else None

    async def put(self, lease: PodLease, *, expect: int | None) -> PodLease:
        said = await self._ledger.result("pods/put", {"lease": lease.to_json(), "expect": expect})
        return PodLease.from_json(said)

    async def delete(self, pod: str, *, expect: int) -> None:
        await self._ledger.result("pods/delete", {"pod": pod, "expect": expect})

    async def times(self, run: str | None = None) -> list[PodTime]:
        return [PodTime.from_json(each) for each in await self._ledger.result("pods/times", {"run": run})]

    async def charge(self, entry: PodTime) -> None:
        await self._ledger.result("pods/charge", {"entry": entry.to_json()})
