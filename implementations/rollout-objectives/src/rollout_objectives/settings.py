"""A policy step's settings, which the LoRA, full-weight and Tinker trainers take alike: the one place their defaults
are written. Importing this does not load torch (`rollout_train.providers.settings_of` reads it)."""

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, replace
from typing import Any, Self, cast

from pydantic import JsonValue

from rollout_train.objectives import COMPONENTS, DEFAULT, Objective, objective_of

__all__ = ["CHANGEABLE", "OBJECTIVE", "StepSettings"]

CHANGEABLE = ("learning_rate", "tokens_per_step", "max_kl", "max_gradient_norm")
"""The settings a trainer takes between steps: each step reads them afresh, and none changes what its weights are or
what a step can hold. Beside them, the components of its objective that may change (`objective.kl.coefficient`)."""
OBJECTIVE = "objective."
"""The start of a changeable setting that is a component of the objective, as the run's settings name it."""
_UNSAID: Any = object()


@dataclass(frozen=True)
class StepSettings:
    """A policy step's settings (`rollout_objectives.step`), whichever trainer takes it."""

    rank: int = 32
    """Of the adapter."""
    learning_rate: float = 5e-5
    tokens_per_step: int = 4_096
    """Sampled tokens per optimizer step (gradients accumulate over segments until then). Adam moves a weight by at
    most the learning rate a step, so how far an update goes is set by how many steps its tokens make."""
    max_kl: float | None = 0.02
    """Stop the pass when a minibatch, before its step, finds the policy this far from where the step began (in nats
    per token, estimated on the sampled tokens). A likelihood step does not stop."""
    max_gradient_norm: float = 1.0
    segment_tokens: int | None = None
    """The longest segment a step can hold (None: any). Longer ones are left out and counted (`segments_too_long`):
    one too long would end or stall the whole step. Leaving segments out biases training, so whoever serves the
    policy takes this as the longest turn to sample; the count says whether that held."""
    segments_per_step: int | None = None
    """How many segments a step can afford (None: any number)."""
    pack_tokens: int | None = None
    """The most tokens one forward and backward pass runs: a step packs its segments into rows of up to this many
    (`rollout_objectives.packing`). None: `segment_tokens`, so that a pack takes no more memory than the longest
    segment would alone, or 8,192 where that is none (what the trainer's memory estimate allows for,
    `rollout_train.memory.SEGMENT_TOKENS`). A segment longer than it has a pack of its own."""
    share_prefixes: bool = True
    """Whether segments of a pack that start with the same tokens share them: the prefix is run once, and each
    segment's rest after it."""
    passes: int = 1
    """Passes a step takes over its segments, each shuffled anew and cut into minibatches of its own: a small batch
    makes more optimizer updates (a supervised step on a small dataset, say)."""
    warmup_updates: int = 0
    """When a step's optimizer starts afresh (no state to go on from), its rate rises linearly over its first this
    many updates, from `learning_rate / warmup_updates` to `learning_rate`: a fresh Adam's first update moves every
    weight by about the full rate. A step that goes on from an optimizer's state is not warmed up."""
    objective: Objective | str | Mapping[str, Any] = DEFAULT
    """The objective (`rollout_train.objectives`): an `Objective`, a preset's name, or a table of `preset` and
    component overrides (`rollout_train.objectives.objective_of`). The run's `objective.*` settings say it; `loss` is
    it, resolved."""
    ratio: InitVar[str] = _UNSAID
    clip_low: InitVar[float] = _UNSAID
    clip_high: InitVar[float] = _UNSAID
    segment_clip_low: InitVar[float] = _UNSAID
    segment_clip_high: InitVar[float] = _UNSAID
    truncate: InitVar[float | None] = _UNSAID
    """The objective's components by a trainer's own names (`rollout_train.objectives.from_trainer_settings`): `ratio`
    (`token`, `segment`), the clip of a token ratio (`clip_low`, `clip_high`) or a segment ratio (`segment_clip_low`,
    `segment_clip_high`), the importance weight's cap (`truncate`; none: the weight untruncated)."""

    def __post_init__(
        self,
        ratio: str,
        clip_low: float,
        clip_high: float,
        segment_clip_low: float,
        segment_clip_high: float,
        truncate: float | None,
    ) -> None:
        if self.passes < 1 or self.warmup_updates < 0:
            raise ValueError("passes is at least 1, and warmup_updates is not negative")
        named = {
            "ratio": ratio, "clip_low": clip_low, "clip_high": clip_high, "segment_clip_low": segment_clip_low,
            "segment_clip_high": segment_clip_high, "truncate": truncate,
        }  # fmt: skip
        legacy = {key: value for key, value in named.items() if value is not _UNSAID}
        object.__setattr__(self, "objective", objective_of(self.objective, legacy))

    @property
    def loss(self) -> Objective:
        """The objective a step takes, resolved."""
        return cast(Objective, self.objective)

    def rate(self, update: int, *, fresh: bool) -> float:
        """The learning rate of a step's `update`-th optimizer update (from 0): warmed up if its optimizer is
        `fresh`."""
        if not fresh or self.warmup_updates <= 0:
            return self.learning_rate
        return self.learning_rate * min(1.0, (update + 1) / self.warmup_updates)

    def changeable(self) -> dict[str, JsonValue]:
        """The settings of `CHANGEABLE` and the changeable components of its objective's family (by their run
        settings' keys, `objective.kl.coefficient`), with their values (what `rollout_train.trainer.Changeable`
        says)."""
        said: dict[str, JsonValue] = {name: getattr(self, name) for name in CHANGEABLE}
        objective = self.loss
        for each in COMPONENTS:
            if each.changeable and objective.family in each.families:
                said[f"{OBJECTIVE}{each.key}"] = objective.get(each.key)
        return said

    def changed(self, changes: Mapping[str, JsonValue]) -> Self:
        """These settings with `changes`, each one of `changeable` (else `ValueError`), checked as any settings are."""
        takes = self.changeable()
        if unknown := sorted(set(changes) - set(takes)):
            raise ValueError(f"{', '.join(unknown)} cannot change between steps (these can: {', '.join(takes)})")
        own = {key: value for key, value in changes.items() if not key.startswith(OBJECTIVE)}
        components = {key.removeprefix(OBJECTIVE): value for key, value in changes.items() if key.startswith(OBJECTIVE)}
        objective = self.loss.changed(components) if components else self.loss
        return replace(self, **own, objective=objective)
