"""The settings of the LoRA and full-weight trainers: a policy step's (`rollout_objectives.settings.StepSettings`,
which the Tinker trainer takes too), the adapter's scaling, and the full-weight trainer's reference."""

from dataclasses import dataclass

from rollout_objectives.settings import CHANGEABLE, StepSettings

__all__ = ["CHANGEABLE", "LoraSettings"]


@dataclass(frozen=True)
class LoraSettings(StepSettings):
    """The settings of `LoraTrainer` and `FullTrainer`: a step's, and the adapter's scaling, which is twice its rank
    (`alpha`)."""

    frozen_reference: bool = False
    """For the full-weight trainer: hold a frozen copy of the model trained over (in bfloat16, beside the policy), the
    reference an objective may read. An adapter's reference is the model with the adapter switched off."""

    @property
    def alpha(self) -> float:
        return 2.0 * self.rank
