"""The Tinker trainer's settings: a policy step's (`rollout_objectives.settings.StepSettings`, the ones `LoraTrainer`
takes too, so a profile switches trainers by changing `kind`), with Tinker's defaults, and a project."""

from dataclasses import dataclass

from rollout_objectives.settings import CHANGEABLE, StepSettings

__all__ = ["CHANGEABLE", "TinkerSettings"]


@dataclass(frozen=True)
class TinkerSettings(StepSettings):
    """Tinker scales an adapter by its own `lora_alpha / rank`, not by our twice the rank."""

    learning_rate: float = 1e-4
    """Twice `LoraTrainer`'s default: Tinker's adapters have an alpha of 32 (its archives say so), half our scale at
    rank 32, and Adam moves a weight by about the rate whatever its scale."""
    tokens_per_step: int = 65_536
    """Sampled tokens per optimizer step. A step that is one optimizer step needs no pass for the logprobs it starts
    from; every optimizer step costs Tinker at least one of its clock cycles."""
    project: str | None = None
    """A Tinker project's id (not a secret); else `TINKER_PROJECT_ID`, if set."""
