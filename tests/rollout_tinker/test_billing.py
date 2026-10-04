"""A call Tinker refuses for billing (402: the balance ran out, with no automatic top-up) is fatal: the engine refuses
that turn and every later one without calling Tinker again, the trainer stops the run instead of failing one step, and
the SDK's own pause-and-retry on 402 is off."""

import importlib
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from tinker import BillingError, TinkerError

from rollout.contracts import Message, ModelEndpointError
from rollout.harness import RecordedModel
from rollout.harness.blobs import FileBlobStore
from rollout_tinker import TinkerEngine, TinkerTrainer
from rollout_tinker.service import Unpaid, no_billing_pause, unpaid
from rollout_tinker.testing import FakeSampler, FakeService
from rollout_train.inference import Channel
from rollout_train.ledger import FileLedger
from rollout_train.recorder import Renderer
from rollout_train.testing import PlainRenderer, admitted, recording, sample_request
from rollout_train.trainer import StepFailed
from tests.rollout_tinker.support import segments


def refused() -> BillingError:
    """What the SDK raises for a 402."""
    request = httpx.Request("POST", "https://tinker.invalid/api/v1/asample")
    return BillingError("billing: insufficient balance", response=httpx.Response(402, request=request), body=None)


class Unbilled(FakeService):
    """A fake whose sampling is refused for billing, counting the calls that reach it."""

    def __init__(self) -> None:
        super().__init__(vocabulary=24)
        self.refusals = 0

    async def create_sampling_client_async(
        self, model_path: str | None = None, base_model: str | None = None
    ) -> FakeSampler:
        sampler = await super().create_sampling_client_async(model_path, base_model)
        service = self

        async def sample_async(*arguments: Any, **options: Any) -> Any:
            service.refusals += 1
            raise ValueError("Error retrieving result: billing with status 402") from refused()  # (as the SDK wraps it)

        sampler.sample_async = sample_async  # type: ignore[method-assign]
        return sampler


def test_a_402_is_told_from_other_errors_wrapped_or_not() -> None:
    assert unpaid(refused())
    try:
        raise ValueError("Error retrieving result") from refused()
    except ValueError as wrapped:
        assert unpaid(wrapped)
    assert not unpaid(TinkerError("the service is down")) and not unpaid(ValueError("no"))


def test_the_sdk_raises_a_402_at_once_instead_of_pausing() -> None:
    holder: Any = importlib.import_module("tinker.lib.internal_client_holder").InternalClientHolder

    no_billing_pause()
    pausing: Any = holder._should_pause_on_billing
    assert pausing(cast(Any, object()), 402, "insufficient balance") is False


async def test_the_engine_refuses_every_turn_after_a_402_without_calling_tinker(tmp_path: Path) -> None:
    service = Unbilled()
    engine = TinkerEngine("tiny", service=service)
    for _ in range(3):
        with pytest.raises(Unpaid, match="for billing"):
            await engine.generate([1], max_tokens=4, temperature=1.0, top_p=1.0, stop_token_ids=[], adapter=None)
    assert service.refusals == 1  # (the first reached Tinker; the others were refused here)

    channel = Channel("policy", [engine], cast(Renderer, PlainRenderer()))
    ledger = FileLedger(tmp_path / "ledger")
    recorder = recording(channel, ledger=ledger, blobs=FileBlobStore(tmp_path / "blobs"))
    await admitted(recorder, "r_1")
    with pytest.raises(ModelEndpointError, match="for billing"):  # (the gateway answers it as the endpoint failing)
        await recorder.endpoint(RecordedModel(channel="policy")).sample(sample_request([Message.user("Hi.")], "e1"))
    assert service.refusals == 1 and (await recorder.sessions("train", "r_1")) == {}  # (nothing recorded)


async def test_the_trainer_stops_the_run_on_a_402_instead_of_failing_the_step(tmp_path: Path) -> None:
    service = FakeService(vocabulary=24)

    async def unbilled(*arguments: Any, **options: Any) -> Any:
        raise refused()

    service.create_lora_training_client_async = unbilled  # type: ignore[method-assign]
    trainer = TinkerTrainer("tiny", service=service, learning_rate=0.05)
    with pytest.raises(Unpaid, match="refused the step for billing") as raised:
        await trainer.step(segments(service, 4), seed=1, parent=None, into=tmp_path / "v1")
    assert not isinstance(raised.value, StepFailed)  # (the loop goes on past a failed step; not past this)
