"""The engine's contract on the fake: ids and logprobs out, the stop token kept; a published version sampled at once
through the channel and the recorder, the one before kept for turns in flight."""

from pathlib import Path
from typing import cast

import pytest

from rollout.contracts import Message
from rollout.harness import RecordedModel
from rollout_tinker import TinkerEngine, TinkerTrainer
from rollout_tinker.testing import FakeService
from rollout_tinker.weights import pointer
from rollout_train.inference import Channel
from rollout_train.recorder import Recorder, Renderer
from rollout_train.testing import PlainRenderer, sample_request
from rollout_train.trainer import WEIGHTS
from tests.support import segments

NEWLINE = ord("\n")


def favoring() -> FakeService:
    return FakeService(favored=[ord("a"), ord("b"), NEWLINE], lift=6.0)


async def test_it_samples_ids_with_a_logprob_each_and_keeps_the_stop_token() -> None:
    service = favoring()
    engine = TinkerEngine("tiny", service=service)
    made = await engine.generate([ord("x")], max_tokens=64, temperature=1.0, top_p=1.0, stop_token_ids=[NEWLINE],
                                 adapter=None)  # fmt: skip
    assert len(made.logprobs) == len(made.tokens) and all(value <= 0 for value in made.logprobs)
    assert made.finish_reason == "stop" and made.tokens[-1] == NEWLINE and NEWLINE not in made.tokens[:-1]
    cut = await engine.generate([ord("x")], max_tokens=1, temperature=1.0, top_p=1.0, stop_token_ids=[], adapter=None)
    assert len(cut.tokens) == 1 and cut.finish_reason in ("length", "stop")
    assert service.sampled == [None, None]  # (the base model)
    assert service.calls.count("sampler tiny") == 1  # (one client for the base model, kept)
    await engine.sleep()
    await engine.wake()  # (nothing to free, nothing freed)
    assert engine.processes == ()


async def test_a_version_published_on_the_channel_is_sampled_at_once_and_the_one_before_kept(tmp_path: Path) -> None:
    service = favoring()
    trainer = TinkerTrainer("tiny", service=service, learning_rate=0.05)
    first, second = tmp_path / "v1", tmp_path / "v2"
    await trainer.step(segments(service, 4), seed=1, parent=None, into=first)
    await trainer.step(segments(service, 4, seed=2), seed=2, parent=None, into=second)
    engine = TinkerEngine("tiny", service=service)
    channel = Channel("policy", [engine], cast(Renderer, PlainRenderer()))
    recorder = Recorder({"policy": channel})
    endpoint = recorder.endpoint(RecordedModel(channel="policy"))

    assert await recorder.publish("policy", "v1", str(first / WEIGHTS), 1) == 1
    await endpoint.sample(sample_request([Message.user("Say a.")], "e1"))
    assert service.sampled[-1] == pointer(first / WEIGHTS, "sampler")
    (segment,) = recorder.export("r_1/ada")
    assert segment.spans[0].version == 1 and len(segment.logprobs) == segment.sampled  # (exact ids and logprobs)

    await recorder.publish("policy", "v2", str(second / WEIGHTS), 2)
    made = await engine.generate(
        [ord("x")], max_tokens=4, temperature=1.0, top_p=1.0, stop_token_ids=[NEWLINE], adapter="v1"
    )  # fmt: skip  (a turn begun under v1 finishes under it)
    assert made.tokens and service.sampled[-1] == pointer(first / WEIGHTS, "sampler")
    await endpoint.sample(sample_request([Message.user("Say b.")], "e2", session_id="r_1/bob"))
    assert service.sampled[-1] == pointer(second / WEIGHTS, "sampler")


async def test_weights_that_name_no_tinker_checkpoint_are_refused(tmp_path: Path) -> None:
    engine = TinkerEngine("tiny", service=FakeService(vocabulary=24))
    with pytest.raises(ValueError, match="names no Tinker checkpoint"):
        await engine.load_adapter("elsewhere", str(tmp_path))
    with pytest.raises(ValueError, match="adapters"):
        await engine.load_weights(str(tmp_path))
