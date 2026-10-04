"""Sandboxes' leases beside the ledger, each ending with the claim of the episode it was acquired for.

An episode runner starts each episode's run with its claim's key as the run's lease (`RUN/GROUP/EPISODE/ATTEMPT`), so
the run's sandboxes are leased under `RUN/GROUP/EPISODE/ATTEMPT/NAME` (`rollout.harness.sandboxes`). A pool keeps its
leases beside the ledger, as ordinary state changed in place: `sandboxes.json` beside a ledger of files
(`FileLeases`), the `sandboxes` table of a database ledger's database (`rollout_train.database.DatabaseLeases`).

A claim holds as the scheduler says (`rollout_train.rollouts.scheduler.holds`): it lapses when its runner takes its
fence anew without adopting it, notes the attempt cut short, or stops beating for `STALE` seconds, or when the episode
has its record. A pool beside the ledger (`admits`) refuses to acquire under a key whose claim does not hold, and
releases the lease the key has: a run whose claim lapsed cannot have a sandbox, even for a moment. Its keeper (`keep`)
looks every few seconds, and releases a lease whose claim it has found lapsed twice running (a runner started again
adopts its claims a moment after taking its fence anew), deleting its sandbox: a runner that dies leaves its sandboxes
to their pools. A lease of a run the ledger does not know (a run started by hand, say) is admitted, and ends only when
it is released. With a name to beat under, the keeper beats too, saying how full the pool is: a pool served from a
machine of its own.
"""

import asyncio
import contextlib
import fcntl
import json
import logging
import socket
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from pathlib import Path

from pydantic import JsonValue

from rollout.harness.sandboxes import Lease, Leases, SandboxPool
from rollout_train.ledger import FileLedger, Ledger
from rollout_train.presence import Presence
from rollout_train.record import runs_in
from rollout_train.rollouts.scheduler import holding

__all__ = ["FileLeases", "admits", "ended", "keep", "leases_of", "sweep"]

logger = logging.getLogger(__name__)

POOL = "pool"
"""The `kind` of a pool's beat."""


def leases_of(ledger: Ledger) -> Leases | None:
    """The leases beside a ledger: a file beside a ledger of files, a table in a database ledger's database."""
    if isinstance(ledger, FileLedger):
        return FileLeases(ledger.directory)
    return getattr(ledger, "sandboxes", None)


def _claim_of(key: str) -> tuple[str, str] | None:
    """The run and the claim's key (`GROUP/EPISODE/ATTEMPT`) a lease's key names, if it names one."""
    run, *rest = key.split("/", 4)
    return (run, "/".join(rest[:3])) if len(rest) >= 4 else None


async def ended(leases: list[Lease], ledger: Ledger, presence: Presence | None) -> Callable[[Lease], bool]:
    """Which of `leases` have ended, as the ledger says now: those whose run it knows and whose claim does not hold."""
    named = {claim[0] for lease in leases if (claim := _claim_of(lease.key)) is not None}
    known = named & set(await runs_in(ledger))
    fences = await ledger.fences()
    beats = {beat.runner: beat for beat in await presence.beats()} if presence is not None else None
    held = {run: await holding(ledger, run, fences, beats) for run in known}

    def over(lease: Lease) -> bool:
        claim = _claim_of(lease.key)
        return claim is not None and claim[0] in held and claim[1] not in held[claim[0]]

    return over


def admits(ledger: Ledger, presence: Presence | None) -> Callable[[str], Awaitable[bool]]:
    """For a pool beside a ledger (`SandboxPool(admits=...)`): whether a key may hold a lease now. A key whose run the
    ledger knows may while its claim holds; any other key may."""

    async def admitted(key: str) -> bool:
        claim = _claim_of(key)
        if claim is None or claim[0] not in await runs_in(ledger):
            return True
        beats = {beat.runner: beat for beat in await presence.beats()} if presence is not None else None
        return claim[1] in await holding(ledger, claim[0], await ledger.fences(), beats)

    return admitted


async def sweep(pool: SandboxPool, ledger: Ledger, presence: Presence | None) -> list[str]:
    """Release the pool's leases whose claims have ended (and delete what no lease names); the keys released."""
    held = await pool.held()  # (before the claims are read: a lease is acquired after its claim is made)
    return await pool.sweep(await ended(held, ledger, presence))


async def keep(
    pool: SandboxPool,
    ledger: Ledger,
    presence: Presence | None,
    *,
    beat_as: str | None = None,
    every: float = 15.0,
) -> None:
    """Sweep the pool every `every` seconds, until cancelled, releasing a lease once its claim was found lapsed at two
    looks running; with `beat_as`, beat under that name too."""
    lapsed: set[str] = set()
    while True:
        try:
            held = await pool.held()
            over = await ended(held, ledger, presence)
            now = {lease.key for lease in held if over(lease)}
            twice = now & lapsed
            gone = await pool.sweep(lambda lease, twice=twice: lease.key in twice)
            lapsed = now
            if gone:
                logger.info("released the leases of claims that ended: %s", ", ".join(gone))
        except Exception:
            logger.exception("sweeping the pool %s failed; trying again", pool.name)
        if beat_as is not None and presence is not None:
            with contextlib.suppress(Exception):  # (a beat missed is noticed only if many are)
                capacity = (await pool.capacity()).to_json()
                about: dict[str, JsonValue] = {"kind": POOL, "host": socket.gethostname(), "pool": pool.name}
                await presence.beat(beat_as, {**about, "sandboxes": pool.provider.kind, **capacity})
        await asyncio.sleep(every)


class FileLeases:
    """`Leases` in `sandboxes.json` in a ledger's directory, under the lock the ledger's files are written under."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path = directory / "sandboxes.json"

    async def get(self, key: str) -> Lease | None:
        return (await asyncio.to_thread(self._locked_read)).get(key)

    async def put(self, lease: Lease) -> None:
        await asyncio.to_thread(self._change, lambda leases: {**leases, lease.key: lease})

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._change, lambda leases: {k: v for k, v in leases.items() if k != key})

    async def all(self) -> list[Lease]:
        return list((await asyncio.to_thread(self._locked_read)).values())

    def _locked_read(self) -> dict[str, Lease]:
        with self._locked():
            return self._read()

    def _change(self, change: Callable[[dict[str, Lease]], dict[str, Lease]]) -> None:
        with self._locked():
            leases = change(self._read())
            staged = self.path.with_suffix(".staged")
            staged.write_text(json.dumps([lease.model_dump(mode="json") for lease in leases.values()]))
            staged.replace(self.path)

    def _read(self) -> dict[str, Lease]:
        if not self.path.exists():
            return {}
        return {each["key"]: Lease.model_validate(each) for each in json.loads(self.path.read_text())}

    @contextmanager
    def _locked(self) -> Generator[None]:
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
