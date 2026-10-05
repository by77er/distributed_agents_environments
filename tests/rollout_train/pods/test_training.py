"""The training service on a training pod and `RemoteTrainer`, its client: a step's batch and parent go through the
blob store and its weights come back the same way; a step is idempotent by the checkpoint it makes; failures are
`StepFailed`, with the pod's part said. The trainer is a fake that writes small files."""

import asyncio
import contextlib
import math
from collections.abc import AsyncGenerator, Mapping, Sequence
from pathlib import Path
from typing import cast

import httpx
import pytest
from pydantic import JsonValue

from rollout.harness import Blobs, FileBlobStore
from rollout_train.checkpoints import Checkpoints, kept
from rollout_train.database import DatabaseLedger
from rollout_train.pods import RemoteTrainer, TrainerBusy, TrainerRefused, TrainerUnreachable
from rollout_train.pods.training import (
    FAILED,
    MADE,
    RUNNING,
    StepAsked,
    TrainerService,
    app,
    batch_bytes,
    batch_of,
)
from rollout_train.record import scope
from rollout_train.recorder import Segment, Span
from rollout_train.trainer import STATE, WEIGHTS, Budget, Files, Item, Step, StepFailed, Weighted

BATCH = [
    Weighted(Segment([1, 2, 3, 4], [Span(2, 4, 3, "e/1")], [-0.5, float("nan")], "policy"), 1.5, "r/1/0/a/0"),
    Weighted(Segment([5, 6], [Span(1, 2, 3)], [-0.25]), -0.5, "r/2/0/a/0"),
]


class Fake:
    """A trainer whose weights are a file saying the steps it was made by: `parent+seed`. `gate`, while unset, holds
    a step back; `failing` makes the next step fail."""

    weights = "lora"
    budget = Budget(4096, 8)

    def __init__(self) -> None:
        self.settings: dict[str, JsonValue] = {"learning_rate": 1e-4}
        self.steps: list[dict[str, JsonValue]] = []
        self.gate = asyncio.Event()
        self.gate.set()
        self.failing = False

    @property
    def changeable(self) -> Mapping[str, JsonValue]:
        return dict(self.settings)

    def change(self, settings: Mapping[str, JsonValue]) -> None:
        if unknown := set(settings) - set(self.settings):
            raise ValueError(f"{sorted(unknown)} cannot change")
        self.settings.update(settings)

    async def step(self, batch: Sequence[Item], *, seed: int, parent: Files | None, into: Path) -> Step:
        await self.gate.wait()
        before = (parent.weights / "adapter.txt").read_text() if parent else "base"
        state = (parent.state / "optimizer.txt").read_text() if parent and parent.state else None
        self.steps.append({"into": into.name, "seed": seed, "segments": len(batch), "parent": before, "state": state,
                           "learning_rate": self.settings["learning_rate"]})  # fmt: skip
        if self.failing:
            self.failing = False
            raise StepFailed("the trainer ran out of memory")
        assert batch == BATCH or len(batch) == len(BATCH)
        (into / WEIGHTS).mkdir(parents=True)
        (into / WEIGHTS / "adapter.txt").write_text(f"{before}+{seed}")
        (into / STATE).mkdir()
        (into / STATE / "optimizer.txt").write_text(f"moments after {seed}")
        return Step({"loss": 0.5, "segments": float(len(batch))})


def checkpoints_of(tmp_path: Path, blobs: Blobs | None = None) -> Checkpoints:
    ledger = DatabaseLedger(f"sqlite:///{tmp_path / 'ledger.db'}")
    return Checkpoints(ledger, blobs or FileBlobStore(tmp_path / "blobs"))


@contextlib.asynccontextmanager
async def pod(service: TrainerService, **options: object) -> AsyncGenerator[RemoteTrainer]:
    """A `RemoteTrainer` reaching `service` (over ASGI: the TLS in front is Envoy's, tested apart)."""
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app(service)), base_url="https://pod")
    trainer = RemoteTrainer(
        "https://pod", service.checkpoints, budget=Budget(4096, 8), changeable={"learning_rate": 1e-4}, client=client,
        every=0.01, **options,  # pyright: ignore[reportArgumentType]
    )  # fmt: skip
    try:
        yield trainer
    finally:
        await client.aclose()


def test_a_batch_is_one_blob_and_comes_back_whole() -> None:
    back = batch_of(batch_bytes(BATCH))
    assert all(isinstance(each, Weighted) for each in back)
    back = cast(list[Weighted], back)
    assert [(each.advantage, each.source, each.segment.tokens, each.segment.spans) for each in back] == [
        (each.advantage, each.source, each.segment.tokens, each.segment.spans) for each in BATCH
    ]
    assert back[0].segment.logprobs[0] == -0.5 and math.isnan(back[0].segment.logprobs[1])  # (NaN stays NaN)


async def test_a_remote_step_trains_on_the_batch_from_the_parent_and_brings_its_files_back(
    tmp_path: Path, blob_store: Blobs
) -> None:
    fake = Fake()
    checkpoints = checkpoints_of(tmp_path, blob_store)
    async with pod(TrainerService(fake, checkpoints, tmp_path / "pod")) as trainer:
        first = await trainer.step(BATCH, seed=7, parent=None, into=tmp_path / "making" / "kmnopqrstuvwxyzk")
        assert first.metrics == {"loss": 0.5, "segments": 2.0}
        made = tmp_path / "making" / "kmnopqrstuvwxyzk"
        assert (made / WEIGHTS / "adapter.txt").read_text() == "base+7"
        assert (made / STATE / "optimizer.txt").read_text() == "moments after 7"
        trainer.change({"learning_rate": 5e-5})
        second = tmp_path / "making" / "lmnopqrstuvwxyzl"
        await trainer.step(BATCH, seed=8, parent=Files(made / WEIGHTS, made / STATE), into=second)
        assert (second / WEIGHTS / "adapter.txt").read_text() == "base+7+8"
    assert fake.steps == [
        {"into": "kmnopqrstuvwxyzk", "seed": 7, "segments": 2, "parent": "base", "state": None, "learning_rate": 1e-4},
        {"into": "lmnopqrstuvwxyzl", "seed": 8, "segments": 2, "parent": "base+7", "state": "moments after 7",
         "learning_rate": 5e-5},
    ]  # fmt: skip
    assert not (tmp_path / "pod" / "steps" / "lmnopqrstuvwxyzl").exists()  # (the pod's working files are deleted)


async def test_a_step_asked_for_again_is_the_same_step(tmp_path: Path) -> None:
    fake = Fake()
    checkpoints = checkpoints_of(tmp_path)
    service = TrainerService(fake, checkpoints, tmp_path / "pod")
    batch = await checkpoints.blobs.put(batch_bytes(BATCH), "application/json")
    fake.gate.clear()
    asked = StepAsked("kmnopqrstuvwxyzk", 7, batch)
    assert (await service.ask(asked)).state == RUNNING
    assert (await service.ask(asked)).state == RUNNING  # (while it runs: the same step)
    fake.gate.set()
    async with pod(service) as trainer:
        await trainer.step(BATCH, seed=7, parent=None, into=tmp_path / "a" / "kmnopqrstuvwxyzk")
        await trainer.step(BATCH, seed=7, parent=None, into=tmp_path / "b" / "kmnopqrstuvwxyzk")  # (made: its answer)
    assert len(fake.steps) == 1 and (tmp_path / "b" / "kmnopqrstuvwxyzk" / WEIGHTS / "adapter.txt").exists()
    again = TrainerService(fake, checkpoints, tmp_path / "pod")  # (the pod started again: the answer is on its disk)
    state = await again.ask(asked)
    assert state.state == MADE and state.made is not None and state.made.metrics == {"loss": 0.5, "segments": 2.0}
    assert len(fake.steps) == 1


async def test_a_step_the_ledger_has_is_answered_with_the_checkpoint_s_own_files(tmp_path: Path) -> None:
    fake = Fake()
    checkpoints = checkpoints_of(tmp_path)
    weights = tmp_path / "trained" / "weights"
    weights.mkdir(parents=True)
    (weights / "adapter.txt").write_text("from the ledger")
    fence = await checkpoints.ledger.take(scope("r"))
    await checkpoints.add(fence, "kmnopqrstuvwxyzk", weights=weights, run="r", base="m", metrics={"loss": 0.25})
    async with pod(TrainerService(fake, checkpoints, tmp_path / "pod")) as trainer:
        step = await trainer.step(BATCH, seed=7, parent=None, into=tmp_path / "making" / "kmnopqrstuvwxyzk")
    assert step.metrics == {"loss": 0.25} and fake.steps == []
    assert (tmp_path / "making" / "kmnopqrstuvwxyzk" / WEIGHTS / "adapter.txt").read_text() == "from the ledger"


async def test_a_failed_step_is_step_failed_and_taken_again_when_asked_for_again(tmp_path: Path) -> None:
    fake = Fake()
    service = TrainerService(fake, checkpoints_of(tmp_path), tmp_path / "pod")
    fake.failing = True
    async with pod(service) as trainer:
        with pytest.raises(StepFailed, match="StepFailed: the trainer ran out of memory"):
            await trainer.step(BATCH, seed=7, parent=None, into=tmp_path / "making" / "kmnopqrstuvwxyzk")
        assert (state := await service.state("kmnopqrstuvwxyzk")) is not None and state.state == FAILED
        await trainer.step(BATCH, seed=7, parent=None, into=tmp_path / "again" / "kmnopqrstuvwxyzk")
    assert len(fake.steps) == 2


async def test_settings_a_trainer_does_not_take_are_refused(tmp_path: Path) -> None:
    fake = Fake()
    service = TrainerService(fake, checkpoints_of(tmp_path), tmp_path / "pod")
    async with pod(service) as trainer:
        with pytest.raises(ValueError, match="rank cannot change between steps"):
            trainer.change({"rank": 64})
        trainer._changeable["rank"] = 64  # pyright: ignore[reportPrivateUsage]  (as a client that thought it could)
        with pytest.raises(StepFailed, match="cannot change"):
            await trainer.step(BATCH, seed=7, parent=None, into=tmp_path / "making" / "kmnopqrstuvwxyzk")


async def test_a_pod_taking_another_step_is_busy(tmp_path: Path) -> None:
    fake = Fake()
    checkpoints = checkpoints_of(tmp_path)
    service = TrainerService(fake, checkpoints, tmp_path / "pod")
    fake.gate.clear()
    batch = await checkpoints.blobs.put(batch_bytes(BATCH), "application/json")
    await service.ask(StepAsked("zzzzzzzzzzzzzzzz", 1, batch))
    async with pod(service) as trainer:
        with pytest.raises(TrainerBusy, match="zzzzzzzzzzzzzzzz"):
            await trainer.step(BATCH, seed=7, parent=None, into=tmp_path / "making" / "kmnopqrstuvwxyzk")
    fake.gate.set()
    assert isinstance(TrainerBusy("busy"), StepFailed)


async def test_a_pod_that_refuses_or_does_not_answer_is_said_so(tmp_path: Path) -> None:
    checkpoints = checkpoints_of(tmp_path)
    async with pod(TrainerService(Fake(), checkpoints, tmp_path / "pod")) as trainer:
        with pytest.raises(TrainerRefused, match="400"):  # (not a checkpoint's id)
            await trainer.step(BATCH, seed=7, parent=None, into=tmp_path / "making" / "..")
    from starlette.applications import Starlette
    from starlette.responses import Response
    from starlette.routing import Route

    async def refused(request: object) -> Response:
        return Response(status_code=404)  # (the proxy's answer to a path it does not pass on)

    proxy = Starlette(routes=[Route("/{path:path}", refused, methods=["GET", "POST"])])
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy), base_url="https://pod")
    trainer = RemoteTrainer("https://pod", checkpoints, client=client)
    with pytest.raises(TrainerRefused, match="refused POST /v1/steps: 404"):
        await trainer.step(BATCH, seed=7, parent=None, into=tmp_path / "making" / "kmnopqrstuvwxyzk")
    await client.aclose()
    gone = RemoteTrainer("http://127.0.0.1:9", checkpoints, every=0.05, patience=0.2)
    with pytest.raises(TrainerUnreachable, match="did not answer"):
        await gone.step(BATCH, seed=7, parent=None, into=tmp_path / "making" / "kmnopqrstuvwxyzk")
    await gone.aclose()


async def test_a_pod_says_what_its_trainer_is(tmp_path: Path) -> None:
    checkpoints = checkpoints_of(tmp_path)
    async with pod(TrainerService(Fake(), checkpoints, tmp_path / "pod", model="Qwen/Qwen3-0.6B")) as trainer:
        said = await trainer.describe()
    assert said == {
        "kind": f"{Fake.__module__}:Fake", "model": "Qwen/Qwen3-0.6B", "weights": "lora",
        "budget": {"segment_tokens": 4096, "segments": 8}, "changeable": {"learning_rate": 1e-4}, "running": None,
        "run": None,
    }  # fmt: skip


async def test_parent_files_the_store_has_are_named_not_written_again(tmp_path: Path) -> None:
    checkpoints = checkpoints_of(tmp_path)
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "adapter.txt").write_text("base+1")
    first = await kept(parent, checkpoints.blobs)
    assert await kept(parent, checkpoints.blobs) == first  # (content-addressed: the same blobs)
