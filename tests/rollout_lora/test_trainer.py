"""The trainers' settings between steps: those a run may change, and the rest refused."""

import pytest

from rollout_lora.settings import CHANGEABLE
from rollout_lora.trainer import FullTrainer, LoraTrainer
from rollout_train.trainer import Changeable


def test_a_trainer_takes_its_changeable_settings_for_its_next_step_and_refuses_the_rest() -> None:
    trainer = LoraTrainer("tiny", learning_rate=5e-5, rank=8)
    assert isinstance(trainer, Changeable) and set(trainer.changeable) == set(CHANGEABLE)
    assert trainer.changeable["learning_rate"] == 5e-5
    trainer.change({"learning_rate": 3e-5, "max_kl": None})
    assert trainer.settings.learning_rate == 3e-5 and trainer.settings.max_kl is None
    assert trainer._process.settings is trainer.settings  # pyright: ignore[reportPrivateUsage]  (the next step's)
    with pytest.raises(ValueError, match="rank cannot change"):
        trainer.change({"rank": 16})
    assert trainer.settings.rank == 8
    assert FullTrainer("tiny").changeable["learning_rate"] == 5e-5
