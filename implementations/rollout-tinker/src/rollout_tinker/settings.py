"""The Tinker trainer's settings: `LoraSettings`' names where they mean the same, so a profile switches trainers by
changing `kind`; and what only a hosted trainer has."""

from dataclasses import dataclass

from rollout_lora.settings import Objective

WEIGHTS = ("pointer", "peft")


@dataclass(frozen=True)
class TinkerSettings:
    rank: int = 32
    """Of the adapter. Tinker scales it by its own `lora_alpha / rank`, not by our twice the rank."""
    learning_rate: float = 1e-4
    """Twice `LoraTrainer`'s default: Tinker's cookbook takes its adapters' alpha to be 32, half our scale at rank 32,
    and Adam moves a weight by about the rate whatever its scale."""
    clip_low: float = 0.2
    clip_high: float = 0.28
    """A token's ratio to its logprob at the step's start is clipped to 1 - `clip_low` .. 1 + `clip_high`."""
    segment_clip_low: float = 3e-4
    segment_clip_high: float = 4e-4
    """With `ratio = "segment"`, the segment's ratio is clipped to 1 - `segment_clip_low` .. 1 + `segment_clip_high`."""
    truncate: float | None = 2.0
    """The most a token's importance weight (its logprob at the step's start against the one it was sampled at) may
    be (None: not truncated)."""
    tokens_per_step: int = 65_536
    """Sampled tokens per optimizer step. A step that is one optimizer step needs no pass for the logprobs it starts
    from; every optimizer step costs Tinker at least one of its clock cycles."""
    max_kl: float | None = 0.02
    """Stop the pass when a minibatch finds the policy this far from where the step began (nats per token, on the
    sampled tokens)."""
    strict_kl: bool = True
    """Read each minibatch's distance before its update is sent (two clock cycles a minibatch); else send both at
    once, and a stop comes one minibatch late."""
    max_gradient_norm: float = 1.0
    segment_tokens: int | None = None
    """The longest segment a step trains on (None: any up to the model's context). Longer ones are left out and
    counted (`segments_too_long`); whoever serves the policy takes it as the longest turn."""
    segments_per_step: int | None = None
    """How many segments a step can afford (None: any number)."""
    passes: int = 1
    """Passes a step takes over its segments, each shuffled anew and cut into minibatches of its own."""
    warmup_updates: int = 0
    """When a step's optimizer starts afresh, its rate rises linearly over its first this many updates."""
    objective: str = "policy_gradient"
    """`policy_gradient` or `likelihood`, as `LoraSettings` says."""
    ratio: str = "token"
    """`token` (PPO) or `segment` (GSPO), as `LoraSettings` says."""
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    """Adam's, as torch's AdamW has them (Tinker's own defaults are 0.95 and 1e-12)."""
    train_unembed: bool = False
    """Also adapt the output layer. Off: an adapter of attention and MLP layers is what local engines and the merge
    take most simply."""
    weights: str = "pointer"
    """What a step leaves in its weights: `pointer`, a file naming the Tinker checkpoint to sample from; `peft`, that
    and the adapter itself, downloaded and in PEFT's layout, for local engines, the merge and the blob store."""
    project: str | None = None
    """A Tinker project's id (not a secret); else `TINKER_PROJECT_ID`, if set."""

    def __post_init__(self) -> None:
        if self.passes < 1 or self.warmup_updates < 0:
            raise ValueError("passes is at least 1, and warmup_updates is not negative")
        if self.weights not in WEIGHTS:
            raise ValueError(f"weights is one of {', '.join(WEIGHTS)}, not {self.weights!r}")
        self.loss  # noqa: B018 (an objective or ratio it does not know is an error now, not at the first step)

    @property
    def loss(self) -> Objective:
        """The objective a step takes, by these settings."""
        segment = self.ratio == "segment"
        return Objective(
            kind=self.objective,
            ratio=self.ratio,
            clip_low=self.segment_clip_low if segment else self.clip_low,
            clip_high=self.segment_clip_high if segment else self.clip_high,
            truncate=self.truncate,
        )

    def rate(self, update: int, *, fresh: bool) -> float:
        """The learning rate of a step's `update`-th optimizer update (from 0): warmed up if its optimizer is
        `fresh`."""
        if not fresh or self.warmup_updates <= 0:
            return self.learning_rate
        return self.learning_rate * min(1.0, (update + 1) / self.warmup_updates)
