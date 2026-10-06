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
    state_every: int | None = None
    """For a trainer on several GPUs, which keeps its model and optimizer between steps: write its full state (the
    optimizer's, and a full-weight trainer's float32 weights) every this many steps; the steps between leave the
    weights alone (an adapter in float32, full weights in bfloat16). None: every step for an adapter, every 10 for full
    weights. A trainer on one GPU writes it every step, which its next step starts from."""
    whole_base: bool | None = None
    """For an adapter on several GPUs: whether each GPU holds the whole frozen model, gathered once (true: no gathering
    for each segment, for a model that fits one GPU beside its activations), or a share of it, each layer gathered as
    it computes (false: a model too large for one GPU). None: whole where the model takes at most half of one GPU's
    memory."""

    def __post_init__(  # (StepSettings' own init variables, passed on)
        self,
        ratio: str,
        clip_low: float,
        clip_high: float,
        segment_clip_low: float,
        segment_clip_high: float,
        truncate: float | None,
    ) -> None:
        super().__post_init__(ratio, clip_low, clip_high, segment_clip_low, segment_clip_high, truncate)
        if self.state_every is not None and self.state_every < 1:
            raise ValueError("state_every is 1 at least")

    @property
    def alpha(self) -> float:
        return 2.0 * self.rank

    def state_every_for(self, weights: str) -> int:
        """How often a resident trainer of `weights` (`lora`, `full`) writes its full state."""
        return self.state_every if self.state_every is not None else (10 if weights == "full" else 1)
