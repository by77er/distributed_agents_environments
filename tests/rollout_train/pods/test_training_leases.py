"""A training pod follows its lease: the run that holds it has its trainer made with the run's settings, and a run
that takes the pod warm has it made anew with its own; a pod no run holds is ready for none."""

import asyncio
import contextlib
from dataclasses import replace
from pathlib import Path
from typing import Any

from rollout.harness.blobs import FileBlobStore
from rollout_train.checkpoints import Checkpoints
from rollout_train.database import DatabaseLedger
from rollout_train.pods.leases import HELD, IDLE, PodLease, pod_leases_of
from rollout_train.pods.training import TrainerService, following, made
from rollout_train.testing import ScriptedTrainer

TESTING = "rollout_train.testing:ScriptedTrainer"


async def until(check: Any, seconds: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + seconds
    while not check():
        assert asyncio.get_running_loop().time() < deadline, "not in time"
        await asyncio.sleep(0.02)


async def test_a_training_pods_trainer_is_made_for_the_run_that_holds_it(tmp_path: Path) -> None:
    ledger = DatabaseLedger(f"sqlite:///{tmp_path / 'ledger.db'}")
    store = pod_leases_of(ledger)
    assert store is not None
    lease = await store.put(PodLease(
        "trainer-a", "pods", 0, "trainer", "image", "m", "gpu", 1.0, run="run_1", state=HELD,
        settings={"implementation": TESTING, "model": "m", "trainer": {"rank": 8}},
    ), expect=None)  # fmt: skip
    service = TrainerService(ScriptedTrainer("m"), Checkpoints(ledger, FileBlobStore(tmp_path / "blobs")), tmp_path)

    def make(said: Any) -> Any:
        return made(str(said.get("implementation") or TESTING), str(said.get("model") or "m"),
                    dict(said.get("trainer") or {}))  # fmt: skip

    task = asyncio.ensure_future(following(service, "trainer-a", make, every=0.02))
    try:
        await until(lambda: service.run == "run_1")
        assert isinstance(service.trainer, ScriptedTrainer) and service.trainer.settings == {"rank": 8}
        lease = await store.put(replace(lease, run=None, state=IDLE), expect=lease.version)
        await until(lambda: service.run is None)  # (released: ready for no run)
        again = {"implementation": TESTING, "model": "m", "trainer": {"rank": 16}}
        await store.put(replace(lease, run="run_2", state=HELD, settings=again), expect=lease.version)
        await until(lambda: service.run == "run_2")
        assert service.trainer.settings == {"rank": 16}  # pyright: ignore[reportAttributeAccessIssue]
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


class Closing(ScriptedTrainer):
    """A trainer that holds something between steps (`Resident`), and counts the times it was closed."""

    holding: str | None = "held"

    def __init__(self, model: str, **settings: Any) -> None:
        super().__init__(model, **settings)
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


async def test_a_trainer_that_holds_its_policy_is_closed_when_the_pod_is_released_or_taken_by_another_run(
    tmp_path: Path,
) -> None:
    ledger = DatabaseLedger(f"sqlite:///{tmp_path / 'ledger.db'}")
    store = pod_leases_of(ledger)
    assert store is not None
    lease = await store.put(PodLease(
        "trainer-a", "pods", 0, "trainer", "image", "m", "gpu", 1.0, run="run_1", state=HELD, settings={},
    ), expect=None)  # fmt: skip
    first = Closing("m")
    service = TrainerService(first, Checkpoints(ledger, FileBlobStore(tmp_path / "blobs")), tmp_path)
    made_ones: list[Closing] = []

    def make(said: Any) -> Any:
        made_ones.append(Closing("m"))
        return made_ones[-1]

    task = asyncio.ensure_future(following(service, "trainer-a", make, every=0.02))
    try:
        await until(lambda: service.run == "run_1")
        assert first.closed == 1  # (the trainer the service started with: closed for the run's own)
        lease = await store.put(replace(lease, run="run_2"), expect=lease.version)
        await until(lambda: service.run == "run_2")
        assert made_ones[0].closed == 1 and made_ones[1].closed == 0  # (another run took the pod)
        await store.put(replace(lease, run=None, state=IDLE), expect=lease.version)
        await until(lambda: service.run is None)
        assert made_ones[1].closed == 1  # (released: what it held is freed)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
