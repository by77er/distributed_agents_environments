"""A whole profile that names the Tinker trainer and engine (over the fake): the loop plays groups, steps, keeps each
version's pointers in the blob store and serves each version from its sampler checkpoint, going on with the live
client; and the Minecraft environment's Tinker profile loads."""

import random
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import JsonValue

from rollout.contracts import Message
from rollout.environment import Description, Row, Start, binding_for, drawn
from rollout.harness import End, Observation, ProgramReference, RunContext, Task, agent_program
from rollout_tinker import TinkerEngine, TinkerSettings, TinkerTrainer
from rollout_tinker.testing import reset
from rollout_tinker.weights import POINTER, pointer
from rollout_train import train
from rollout_train.profile import Profile

ROOT = Path(__file__).resolve().parents[3]


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

PROFILE = """
directory = "{directory}"

[channels.policy]
model = "Qwen/Qwen3.5-9B"
renderer = "rollout_train.testing:plain_renderer"
engine = "rollout_tinker:TinkerEngine"
engines = [{{ service = "rollout_tinker.testing:fake_service", max_model_len = 4096 }}]
thinking_tokens = 16
answer_tokens = 16

[trainer]
kind = "rollout_tinker:TinkerTrainer"
channel = "policy"
service = "rollout_tinker.testing:fake_service"
learning_rate = 0.05
segment_tokens = 2000
tokens_per_step = 16
"""


async def test_a_profile_trains_and_serves_on_tinker(tmp_path: Path) -> None:
    service = reset()
    path = tmp_path / "profile.toml"
    path.write_text(PROFILE.format(directory=tmp_path / "run"))
    async with Profile.load(path).open() as platform:
        assert isinstance(platform.trainer, TinkerTrainer)  # (not colocated: nothing to share)
        (engine,) = platform.channels["policy"].engines
        assert isinstance(engine, TinkerEngine) and platform.channels["policy"].limits.sequence == 2000
        await train(
            letters, platform.trainer, platform.checkpoints, base="Qwen/Qwen3.5-9B", channel="policy",
            directory=tmp_path / "run" / "checkpoints", publish=platform.publish, run=platform.run.id, groups=8,
            groups_per_step=2, episodes_at_once=2, binding=binding_for(letters, "policy", platform.tool_bindings),
        )  # fmt: skip
        made = [each for each in await platform.checkpoints.all() if each.run == platform.run.id]
        assert len(made) >= 2
        for checkpoint in (each for each in made if each.released is None):  # pointers, kept as blobs like any weights
            assert checkpoint.weights is not None and POINTER in checkpoint.weights.files
            assert checkpoint.state is not None and POINTER in checkpoint.state.files
        newest = made[-1]
        assert platform.channels["policy"].adapter == newest.id
        files = await platform.checkpoints.files(newest.weights, tmp_path / "newest") if newest.weights else tmp_path
        assert pointer(files, "sampler") in service.saved  # served from its own sampler checkpoint
        assert service.calls.count("lora 32") == 1  # one training run: each step went on with the live client
        assert not any(call.startswith("state") for call in service.calls)
        sampled_from = {each for each in service.sampled if each is not None}
        assert sampled_from and sampled_from <= {path for path in service.saved if "/sampler_weights/" in path}


def test_the_minecraft_profile_for_tinker_loads() -> None:
    profile = Profile.load(ROOT / "environments" / "minecraft" / "profiles" / "tinker.toml")
    trainer = profile.trainer
    assert trainer is not None and trainer.kind == "rollout_tinker:TinkerTrainer" and not trainer.colocated
    channel = profile.channels[trainer.channel]
    assert channel.engine == "rollout_tinker:TinkerEngine" and channel.model == "Qwen/Qwen3.5-9B"
    settings = dict(trainer.settings)
    assert settings["rank"] == 32 and settings.get("weights", "pointer") == "pointer"
    TinkerSettings(**settings)  # (every setting it gives is one the trainer takes)
