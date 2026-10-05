"""The loop over the Tinker trainer and engine (over the fake service): it plays groups, steps, keeps each version's
pointers in the blob store and serves each version from its sampler checkpoint, going on with the live client."""

import random
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import JsonValue

from rollout.contracts import Message
from rollout.environment import Description, Row, Start, drawn
from rollout.harness import End, Observation, ProgramReference, RunContext, Task, agent_program
from rollout.harness.blobs import FileBlobStore
from rollout_tinker import TinkerEngine, TinkerTrainer
from rollout_tinker.testing import reset
from rollout_tinker.weights import POINTER, pointer
from rollout_train import Checkpoints, FileLedger, train
from rollout_train.inference import Channel, Limits
from rollout_train.testing import Policy, plain_renderer
from tests.rollout_train.support import here

FAKE = "rollout_tinker.testing:fake_service"


class SayA(Task):
    """Say as many a's as can be said: the reward is the share of them in the reply."""

    async def start(self, run: RunContext) -> Observation:
        return Observation("Say a.")

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        text = reply.text.strip()
        share = text.count("a") / len(text) if text else 0.0
        await run.emit("result", {"solved": share > 0.5, "duration": 1})
        return End(reward=share)


class Letters:
    program: ProgramReference = agent_program(SayA)
    version = "1"
    description = Description()

    def rows(self) -> Sequence[Row]:
        return [Row("say-a", "say a", {})]

    def start(self, row: Row, rng: random.Random) -> JsonValue:
        return {"seed": rng.randrange(1000)}

    def evals(self) -> Mapping[str, Sequence[Start]]:
        return {"letters-held-out": drawn(self, seeds=[1])}


letters = Letters()


async def test_the_loop_trains_and_serves_on_tinker(tmp_path: Path) -> None:
    service = reset()
    model = "Qwen/Qwen3.5-9B"
    trainer = TinkerTrainer(model, service=FAKE, learning_rate=0.05, segment_tokens=2000, tokens_per_step=16)
    engine = TinkerEngine(model, service=FAKE, max_model_len=4096)
    channel = Channel("policy", [engine], plain_renderer(model), Limits(16, 16, sequence=2000))
    recorder = Policy(channel)
    checkpoints = Checkpoints(FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs"))
    async with here(checkpoints.ledger, recorder, checkpoints.blobs):
        await train(
            letters, trainer, checkpoints, base=model, channel="policy", directory=tmp_path / "checkpoints",
            publish=recorder.publish, run="train", groups=8, groups_per_step=2, episodes_at_once=2,
        )  # fmt: skip
    made = [each for each in await checkpoints.all() if each.run == "train"]
    assert len(made) >= 2
    for checkpoint in (each for each in made if each.released is None):  # pointers, kept as blobs like any weights
        assert checkpoint.weights is not None and POINTER in checkpoint.weights.files
        assert checkpoint.state is not None and POINTER in checkpoint.state.files
    newest = made[-1]
    assert channel.adapter == newest.id
    files = await checkpoints.files(newest.weights, tmp_path / "newest") if newest.weights else tmp_path
    assert pointer(files, "sampler") in service.saved  # served from its own sampler checkpoint
    assert service.calls.count("lora 32") == 1  # one training run: each step went on with the live client
    assert not any(call.startswith("state") for call in service.calls)
    sampled_from = {each for each in service.sampled if each is not None}
    assert sampled_from and sampled_from <= {path for path in service.saved if "/sampler_weights/" in path}
