# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""The trainers' settings between steps: those a run may change, and the rest refused. And how far a step has got,
as the trainer's processes say it."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from rollout_lora.settings import CHANGEABLE
from rollout_lora.trainer import FullTrainer, LoraTrainer
from rollout_train.trainer import MINIBATCH, START, Changeable, Progress, Progressing


def test_a_trainer_takes_its_changeable_settings_for_its_next_step_and_refuses_the_rest() -> None:
    trainer = LoraTrainer("tiny", learning_rate=5e-5, rank=8)
    components = {key for key in trainer.changeable if key.startswith("objective.")}
    assert isinstance(trainer, Changeable) and set(trainer.changeable) == {*CHANGEABLE, *components}
    assert {"objective.clip.low", "objective.importance.cap", "objective.kl.coefficient"} <= components
    assert trainer.changeable["learning_rate"] == 5e-5
    trainer.change({"learning_rate": 3e-5, "max_kl": None, "objective.clip.high": 0.3})
    assert trainer.settings.learning_rate == 3e-5 and trainer.settings.max_kl is None
    assert trainer.objective.clip.high == 0.3 and trainer.objective.clip.low == 0.2
    assert trainer._process.settings is trainer.settings  # pyright: ignore[reportPrivateUsage]  (the next step's)
    with pytest.raises(ValueError, match="rank cannot change"):
        trainer.change({"rank": 16})
    with pytest.raises(ValueError, match=r"objective.ratio cannot change"):  # (what shapes the loss is fixed)
        trainer.change({"objective.ratio": "segment"})
    assert trainer.settings.rank == 8
    assert FullTrainer("tiny").changeable["learning_rate"] == 5e-5


@pytest.mark.parametrize("count", [1, 2])
def test_a_trainer_says_how_far_each_step_has_got_as_its_processes_say_it(
    tiny: str, tmp_path: Path, count: int
) -> None:
    import torch

    from rollout_lora.policy import Policy
    from rollout_lora.workers import SEED
    from tests.rollout_lora.test_sharded import SETTINGS, batch

    settings: dict[str, Any] = {**SETTINGS, "tokens_per_step": 12}  # (three minibatches)
    trainer = LoraTrainer(tiny, gpus=count, **settings)
    trainer._process.device = "cpu"  # pyright: ignore[reportPrivateUsage]  (no GPU here)
    torch.manual_seed(SEED)
    policy = Policy.load(tiny, rank=trainer.settings.rank, alpha=trainer.settings.alpha, device="cpu")
    torch.manual_seed(1)
    told: list[Progress] = []
    assert isinstance(trainer, Progressing)
    trainer.watch(told.append)
    try:
        step = asyncio.run(trainer.step(batch(policy.logprobs), seed=0, parent=None, into=tmp_path / "first"))
    finally:
        trainer.close()
    assert told and told[0].phase == START and told[-1].phase == MINIBATCH
    last = told[-1]  # (every process's packs, rank 0 said)
    assert last.packs == last.packs_total == step.metrics["packs"] and last.fraction == 1.0
    assert last.minibatch == last.minibatches == step.metrics["optimizer_steps"]
    assert trainer._process.progress is None  # pyright: ignore[reportPrivateUsage]  (none between steps)
