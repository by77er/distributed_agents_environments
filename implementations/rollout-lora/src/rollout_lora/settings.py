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
    state_every: int = 1
    """How often a trainer whose processes are kept between steps (one with its GPUs to itself) writes its full
    state: the optimizer's, and full weights in float32. Every step writes the weights (an adapter in float32, full
    weights as a bfloat16 serving copy) whatever this says. 1: every step, so that any trainer goes on from any
    checkpoint as the trainer that made it would. More: the steps between leave the full state out, which saves writing
    about 12 bytes a weight for full weights, and a step from one of them goes on only from the processes that hold
    it. Once those are gone (after a failed step, a restart, or another run taking a training pod) such a step fails
    (`StepFailed`) rather than go on from less than its parent was: a run started from the newest checkpoint with its
    full state goes on. A trainer whose processes end after each step (one beside an engine) writes it every step."""
    whole_base: bool | None = None
    """For an adapter on several GPUs: whether each GPU holds the whole frozen model, gathered once (true: no gathering
    for each segment, for a model that fits one GPU beside its activations), or a share of it, each layer gathered as
    it computes (false: a model too large for one GPU). None: whole where the model's files take at most half of a
    GPU's memory (`rollout_train.memory.holds_whole_base`, which the memory estimate decides by too)."""

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
        if self.state_every < 1:
            raise ValueError("state_every is 1 at least")

    @property
    def alpha(self) -> float:
        return 2.0 * self.rank
