"""A step's weights first, its state after: a trainer that keeps a step's full state itself after the step returns
(`Keeps`) has the step answered as soon as its weights are kept, and its checkpoint served while the state is still
being kept; the checkpoint's state is completed once kept, or left incomplete (and said) when it never is. The training
loop records, serves and thins a training pod's checkpoints from manifests, reading none of their files; engines in its
own process read them as they load them. A step from a checkpoint whose state is incomplete goes on from the trainer
that holds it, waits for the state its pod is keeping, and is refused, never started afresh, where the state was never
kept."""

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from rollout.contracts import BlobReference
from rollout.harness import Blobs, FileBlobStore
from rollout_train import train
from rollout_train.checkpoints import Checkpoint, Checkpoints, Manifest
from rollout_train.evals import Fetched
from rollout_train.ledger import FileLedger
from rollout_train.pods import RemoteTrainer
from rollout_train.pods.training import TrainerService, app
from rollout_train.trainer import HELD, STATE, Budget, Files, Item, StateLost, Step, StepFailed
from tests.rollout_train.pods.test_training import BATCH, Holding
from tests.rollout_train.rollouts.games import Words
from tests.rollout_train.support import answering, here, made_by

MOMENTS = "optimizer.txt"


class Keeping(Holding):
    """A fake that holds its weights between steps and keeps the rest of each step's state (its moments) in the blob
    store itself, after the step returns, once `release` is set (each step clears it first, with `gated`); with
    `losing`, keeping it fails. A step from a parent
    it does not hold whose state has no moments is refused, as `rollout_lora`'s trainers refuse it."""

    def __init__(self, blobs: Blobs) -> None:
        super().__init__()
        self.blobs = blobs
        self.release = asyncio.Event()
        self.release.set()
        self.losing = False
        self.gated = False
        self.location: dict[str, JsonValue] | None = None
        self._keeping: dict[str, asyncio.Task[Manifest]] = {}

    def keep_in(self, blobs: Mapping[str, JsonValue]) -> None:
        self.location = dict(blobs)

    async def step(self, batch: Sequence[Item], *, seed: int, parent: Files | None, into: Path) -> Step:
        held = parent is not None and parent.state is not None and (parent.state / HELD).exists()
        if held and parent is not None and parent.state is not None:
            held = (parent.state / HELD).read_text() == self.holding
        if parent is not None and not held and (parent.state is None or not (parent.state / MOMENTS).exists()):
            raise StepFailed("the parent's state has no full state, and no process holds the parent any longer")
        step = await Holding.step(self, BATCH, seed=seed, parent=parent, into=into)  # (its files, whatever the batch)
        moments = (into / STATE / MOMENTS).read_bytes()
        if self.gated:
            self.release.clear()
        (into / STATE / MOMENTS).unlink()  # (kept after the step returns)
        self._keeping[into.name] = asyncio.create_task(self._kept(moments))
        return Step(step.metrics, keeping=True)

    async def _kept(self, moments: bytes) -> Manifest:
        await self.release.wait()
        if self.losing:
            raise OSError("the bucket refused the upload")
        return Manifest({MOMENTS: await self.blobs.put(moments, "application/octet-stream")})

    async def kept(self, into: str) -> Manifest:
        try:
            return await self._keeping[into]
        except OSError as error:
            raise StateLost(str(error)) from error


class Counted:
    """A blob store that notes every blob read through it, by hash (what the driver read)."""

    def __init__(self, store: Blobs) -> None:
        self.store = store
        self.read_blobs: list[str] = []

    async def put(self, data: bytes, media_type: str) -> BlobReference:
        return await self.store.put(data, media_type)

    async def read(self, reference: BlobReference) -> bytes:
        self.read_blobs.append(reference.sha256)
        return await self.store.read(reference)

    async def delete(self, reference: BlobReference, *, unused_for: float = 0.0) -> None:
        await self.store.delete(reference, unused_for=unused_for)


@contextlib.asynccontextmanager
async def pod(service: TrainerService, checkpoints: Checkpoints) -> AsyncGenerator[RemoteTrainer]:
    """A `RemoteTrainer` the driver holds (over `checkpoints`, the driver's), reaching `service` over ASGI."""
    import httpx

    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app(service)), base_url="https://pod")
    trainer = RemoteTrainer("https://pod", checkpoints, budget=Budget(4096, 8), changeable={"learning_rate": 1e-4},
                            client=client, every=0.01)  # fmt: skip
    try:
        yield trainer
    finally:
        await client.aclose()


def a_pod(tmp_path: Path, name: str = "pod") -> tuple[Keeping, TrainerService, Checkpoints]:
    """A training pod's service over a `Keeping` fake, and the driver's checkpoints (reads through them noted)."""
    ledger, store = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    keeping = Keeping(store)
    service = TrainerService(keeping, Checkpoints(ledger, store), tmp_path / name)
    return keeping, service, Checkpoints(ledger, Counted(store))


def checkpoint_of(id: str, made: Any) -> Checkpoint:
    return Checkpoint(id, made.weights, state=made.state, state_complete=made.complete)


async def test_a_step_is_answered_once_its_weights_are_kept_and_its_state_completes_after(tmp_path: Path) -> None:
    keeping, service, checkpoints = a_pod(tmp_path)
    keeping.release.clear()
    async with pod(service, checkpoints) as trainer:
        made = await trainer.made(BATCH, seed=7, parent=None, into="kmnopqrstuvwxyzk")
        assert not made.complete and made.state is not None and sorted(made.state.files) == [HELD]
        assert sorted(made.weights.files) == ["adapter.txt"]
        with pytest.raises(TimeoutError):  # (still being kept)
            await asyncio.wait_for(trainer.state("kmnopqrstuvwxyzk"), 0.2)
        keeping.release.set()
        whole = await trainer.state("kmnopqrstuvwxyzk")
    assert sorted(whole.files) == [HELD, MOMENTS]
    read = {name: await service.checkpoints.blobs.read(blob) for name, blob in whole.files.items()}
    assert read == {HELD: b"kmnopqrstuvwxyzk:held", MOMENTS: b"moments after 7"}
    assert not (tmp_path / "pod" / "steps" / "kmnopqrstuvwxyzk").exists()  # (deleted once the state was kept)


async def test_a_state_never_kept_is_said_and_stays_incomplete(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    keeping, service, checkpoints = a_pod(tmp_path)
    keeping.losing = True
    async with pod(service, checkpoints) as trainer:
        made = await trainer.made(BATCH, seed=7, parent=None, into="kmnopqrstuvwxyzk")
        with caplog.at_level(logging.ERROR), pytest.raises(StateLost, match="the bucket refused the upload"):
            await trainer.state("kmnopqrstuvwxyzk")
        # The next step goes on from what the trainer holds; once it holds nothing, a step from the checkpoint whose
        # state was never kept is refused, not started afresh.
        parent = checkpoint_of("kmnopqrstuvwxyzk", made)
        keeping.losing = False
        await trainer.made(BATCH, seed=8, parent=parent, into="lmnopqrstuvwxyzl")
        keeping.close()
        with pytest.raises(StepFailed, match="no full state"):
            await trainer.made(BATCH, seed=9, parent=parent, into="mmnopqrstuvwxyzm")
    assert "was not kept" in caplog.text and "kmnopqrstuvwxyzk" in caplog.text
    assert [step["into"] for step in keeping.steps] == ["kmnopqrstuvwxyzk", "lmnopqrstuvwxyzl"]


async def test_a_parent_whose_state_its_pod_is_keeping_is_waited_for(tmp_path: Path) -> None:
    keeping, service, checkpoints = a_pod(tmp_path)
    keeping.release.clear()
    async with pod(service, checkpoints) as trainer:
        made = await trainer.made(BATCH, seed=7, parent=None, into="kmnopqrstuvwxyzk")
        keeping.close()  # (it holds the parent no longer: its full state is needed)
        stepping = asyncio.create_task(
            trainer.made(BATCH, seed=8, parent=checkpoint_of("kmnopqrstuvwxyzk", made), into="lmnopqrstuvwxyzl")
        )
        await asyncio.sleep(0.2)
        assert not stepping.done() and len(keeping.steps) == 1  # (waiting for the parent's state)
        keeping.release.set()
        await stepping
    assert keeping.steps[1]["state"] == "moments after 7"  # (from the whole state, as it was kept)


class Published:
    """A publisher for engines elsewhere (as a run's driver serves a pod's channel): it notes what it published and
    whether the checkpoint's state was complete then, reading no file; with `local`, it reads them as engines in the
    driver's process would."""

    def __init__(self, checkpoints: Checkpoints, keeping: Keeping, *, local: bool = False) -> None:
        self.checkpoints = checkpoints
        self.keeping = keeping
        self.local = local
        self.published: list[tuple[str, bool, frozenset[str]]] = []
        """Each checkpoint published, whether its state was complete then, and the blobs read before it was."""

    async def __call__(self, channel: str, adapter: str, files: Fetched, version: int | None = None, *,
                       full: bool = False) -> int:  # fmt: skip
        checkpoint = await self.checkpoints.checkpoint(adapter)
        read = frozenset(getattr(self.checkpoints.blobs, "read_blobs", []))
        if self.local:
            assert (Path(await files()) / "adapter.txt").exists()
        self.published.append((adapter, checkpoint.state_complete, read))
        self.keeping.release.set()  # (the state is kept once the weights are served)
        return version or 0


def read_by_driver(checkpoints: Checkpoints, made: Sequence[Checkpoint]) -> set[str]:
    """The blobs of the checkpoints' files that the driver read."""
    named = {blob.sha256 for each in made for manifest in (each.weights, each.state) if manifest
             for blob in manifest.files.values()}  # fmt: skip
    counted = checkpoints.blobs
    assert isinstance(counted, Counted)
    return named & set(counted.read_blobs)


async def test_the_loop_serves_a_pod_s_checkpoints_from_manifests_before_their_state_is_kept(tmp_path: Path) -> None:
    keeping, service, checkpoints = a_pod(tmp_path)
    keeping.gated = True  # (each step's state is kept once its weights are served)
    published = Published(checkpoints, keeping)
    notes: list[Mapping[str, JsonValue]] = []

    class Noting:
        def on_note(self, event: Mapping[str, JsonValue]) -> None:
            notes.append(event)

    async with (
        pod(service, checkpoints) as trainer,
        here(checkpoints.ledger, answering(), FileBlobStore(tmp_path / "blobs")),
    ):
        await train(
            Words(), trainer, checkpoints, base="words-base", channel="policy", directory=tmp_path / "driver",
            publish=published, groups=4, groups_per_step=1, seed=1, hooks=[Noting()],
        )  # fmt: skip
    made = await made_by(checkpoints)
    assert len(made) >= 2 and all(each.state_complete and each.state_seconds is not None for each in made)
    assert published.published and all(not complete for _, complete, _ in published.published)  # (served first)
    assert not read_by_driver(checkpoints, made)  # (no file of a checkpoint was read to the driver's machine)
    assert not [each for each in (tmp_path / "driver").rglob("*") if each.is_file()]
    steps = [each for each in notes if each["kind"] == "step"]
    metrics: Any = steps[-1]["metrics"]
    assert {"publish_seconds", "step_seconds", "upload_adapter_seconds"} <= set(metrics)
    states = [each for each in notes if each["kind"] == "state"]
    assert len(states) == len(made) and all("seconds" in each for each in states)
    # Each step after the first went on from what the trainer held: its parent's state, still being kept, was not
    # needed.
    assert [len(each) for each in keeping.given[1:]] == [1] * (len(made) - 1)


async def test_engines_in_the_driver_s_process_read_a_checkpoint_s_files_as_they_load_them(tmp_path: Path) -> None:
    keeping, service, checkpoints = a_pod(tmp_path)
    published = Published(checkpoints, keeping, local=True)
    async with (
        pod(service, checkpoints) as trainer,
        here(checkpoints.ledger, answering(), FileBlobStore(tmp_path / "blobs")),
    ):
        await train(
            Words(), trainer, checkpoints, base="words-base", channel="policy", directory=tmp_path / "driver",
            publish=published, groups=2, groups_per_step=1, seed=1,
        )  # fmt: skip
    made = await made_by(checkpoints)
    assert len(published.published) == len(made)
    for adapter, _, before in published.published:  # (nothing of a checkpoint read before the publisher asked)
        weights = (await checkpoints.checkpoint(adapter)).weights
        assert weights is not None and not {blob.sha256 for blob in weights.files.values()} & before
    weights = {blob.sha256 for each in made if each.weights for blob in each.weights.files.values()}
    assert weights <= read_by_driver(checkpoints, made)  # (the weights, read as they were loaded)
    states = {blob.sha256 for each in made if each.state for blob in each.state.files.values()}
    assert not states & read_by_driver(checkpoints, made)  # (and never the state)


async def test_a_loop_started_again_completes_a_state_still_being_kept_and_refuses_one_never_kept(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    keeping, service, checkpoints = a_pod(tmp_path)
    keeping.losing = True
    published = Published(checkpoints, keeping)
    blobs = FileBlobStore(tmp_path / "blobs")
    with caplog.at_level(logging.ERROR):
        async with pod(service, checkpoints) as trainer, here(checkpoints.ledger, answering(), blobs):
            await train(
                Words(), trainer, checkpoints, base="words-base", channel="policy", directory=tmp_path / "driver",
                publish=published, groups=1, groups_per_step=1, seed=1,
            )  # fmt: skip
    first = await made_by(checkpoints)
    assert len(first) == 1 and not first[0].state_complete  # (left incomplete, and said)
    assert "stays incomplete" in caplog.text and first[0].id in caplog.text

    # A pod started again holds nothing: a step from the checkpoint whose state was never kept is refused (as often as
    # the loop tries before it stops), never taken from the base model or the weights alone.
    again, service_again, _ = a_pod(tmp_path, "pod-again")
    async with pod(service_again, checkpoints) as trainer, here(checkpoints.ledger, answering(), blobs):
        with pytest.raises(StepFailed, match="no full state"):
            await train(
                Words(), trainer, checkpoints, base="words-base", channel="policy", directory=tmp_path / "driver",
                publish=published, groups=6, groups_per_step=1, seed=1,
            )  # fmt: skip
    assert again.steps == [] and len(await made_by(checkpoints)) == 1
