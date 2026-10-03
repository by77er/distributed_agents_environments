"""The LoRA trainer's settings: the one place their defaults are written."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LoraSettings:
    rank: int = 32
    """Of the adapter. Its scaling is twice the rank (`alpha`)."""
    learning_rate: float = 5e-5
    clip_low: float = 0.2
    clip_high: float = 0.28
    """The probability ratio is clipped to 1 - `clip_low` .. 1 + `clip_high` (DAPO's clip-higher)."""
    tokens_per_step: int = 4_096
    """Sampled tokens per optimizer step (gradients accumulate over segments until then). Adam moves a weight by at
    most the learning rate a step, so how far an update goes is set by how many steps its tokens make."""
    max_kl: float | None = 0.02
    """Stop the pass when a minibatch, before its step, finds the policy this far (in nats per token, estimated on
    the sampled tokens) beyond where the first minibatch found it. The first minibatch's value is the floor: the
    engine's and the trainer's numerical difference, and how stale the segments are."""
    max_gradient_norm: float = 1.0
    segment_tokens: int | None = None
    """The longest segment a step can hold on its accelerator (None: any). Longer ones are left out and counted
    (`segments_too_long`): one too long would end or stall the whole step. Leaving segments out biases training,
    so whoever serves the policy takes this as the longest turn to sample; the count says whether that held."""
    segments_per_step: int | None = None
    """How many segments a step can afford (None: any number)."""
    objective: str = "policy_gradient"
    """`policy_gradient`: the clipped policy gradient over the sampled tokens, each weighted by its segment's
    advantage, against the logprobs they were sampled at. `likelihood`: raise the log-likelihood of the sampled tokens,
    each weighted by its segment's advantage (imitation: what was sampled is what to do), with no ratio, clip or
    stop at `max_kl`."""

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective is one of {', '.join(OBJECTIVES)}, not {self.objective!r}")

    @property
    def alpha(self) -> float:
        return 2.0 * self.rank


OBJECTIVES = ("policy_gradient", "likelihood")
