"""The LoRA trainer's settings: the one place their defaults are written."""

from dataclasses import dataclass

KINDS = ("policy_gradient", "likelihood")
RATIOS = ("token", "segment")
CHANGEABLE = (
    "learning_rate",
    "clip_low",
    "clip_high",
    "segment_clip_low",
    "segment_clip_high",
    "truncate",
    "tokens_per_step",
    "max_kl",
    "max_gradient_norm",
)
"""The settings a trainer takes between steps: each step reads them afresh, and none changes what its weights are or
what a step can hold."""


@dataclass(frozen=True)
class Objective:
    """Which loss a policy step takes, and its numbers (`rollout_lora.objectives` evaluates it)."""

    kind: str = "policy_gradient"
    ratio: str = "token"
    clip_low: float = 0.2
    clip_high: float = 0.28
    truncate: float | None = 2.0
    """The most the importance weight `old / behavior` may be (None: not truncated)."""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"objective is one of {', '.join(KINDS)}, not {self.kind!r}")
        if self.ratio not in RATIOS:
            raise ValueError(f"ratio is one of {', '.join(RATIOS)}, not {self.ratio!r}")

    @property
    def reads_old(self) -> bool:
        """Whether the step must give each token's logprob on the weights it starts from (`old`)."""
        return self.kind == "policy_gradient"

    def units(self, tokens: int) -> float:
        """What a segment of `tokens` sampled tokens counts for in its minibatch's mean."""
        return 1.0 if self.kind == "policy_gradient" and self.ratio == "segment" else float(tokens)


@dataclass(frozen=True)
class LoraSettings:
    rank: int = 32
    """Of the adapter. Its scaling is twice the rank (`alpha`)."""
    learning_rate: float = 5e-5
    clip_low: float = 0.2
    clip_high: float = 0.28
    """A token's ratio to its logprob at the step's start is clipped to 1 - `clip_low` .. 1 + `clip_high` (DAPO's
    clip-higher)."""
    segment_clip_low: float = 3e-4
    segment_clip_high: float = 4e-4
    """With `ratio = "segment"`, the segment's ratio is clipped to 1 - `segment_clip_low` .. 1 + `segment_clip_high`
    (GSPO's)."""
    truncate: float | None = 2.0
    """The most a token's importance weight (its logprob at the step's start against the one it was sampled at) may
    be (None: not truncated)."""
    tokens_per_step: int = 4_096
    """Sampled tokens per optimizer step (gradients accumulate over segments until then). Adam moves a weight by at
    most the learning rate a step, so how far an update goes is set by how many steps its tokens make."""
    max_kl: float | None = 0.02
    """Stop the pass when a minibatch, before its step, finds the policy this far from where the step began (in nats
    per token, estimated on the sampled tokens)."""
    max_gradient_norm: float = 1.0
    segment_tokens: int | None = None
    """The longest segment a step can hold on its accelerator (None: any). Longer ones are left out and counted
    (`segments_too_long`): one too long would end or stall the whole step. Leaving segments out biases training,
    so whoever serves the policy takes this as the longest turn to sample; the count says whether that held."""
    layer_inputs_on_host: bool = False
    """Keep each layer's input in pinned system memory between the forward and backward passes, instead of on the
    GPU (`rollout_lora.activations`): a quarter of a megabyte a token, for Qwen3.5-9B."""
    mlp_rows: int | None = None
    """Run each layer's MLP over this many tokens at a time when it is computed again for the backward pass, and in
    passes without a gradient (None: the whole segment at once). The same numbers, at a lower peak."""
    segments_per_step: int | None = None
    """How many segments a step can afford (None: any number)."""
    passes: int = 1
    """Passes a step takes over its segments, each shuffled anew and cut into minibatches of its own: a small batch
    makes more optimizer updates (a supervised step on a small dataset, say)."""
    warmup_updates: int = 0
    """When a step's optimizer starts afresh (no state to go on from), its rate rises linearly over its first this
    many updates, from `learning_rate / warmup_updates` to `learning_rate`: a fresh Adam's first update moves every
    weight by about the full rate. A step that goes on from an optimizer's state is not warmed up."""
    objective: str = "policy_gradient"
    """`policy_gradient`: the clipped policy gradient over the sampled tokens, each weighted by its segment's
    advantage, with an importance weight for where they were sampled. `likelihood`: raise the log-likelihood of the
    sampled tokens, each weighted by its segment's advantage (imitation: what was sampled is what to do), with no
    ratio, weight or stop at `max_kl` (`rollout_lora.objectives`)."""
    ratio: str = "token"
    """`token`: a ratio for each token (PPO). `segment`: one for each segment, the geometric mean of its tokens'
    (GSPO)."""

    def __post_init__(self) -> None:
        if self.passes < 1 or self.warmup_updates < 0:
            raise ValueError("passes is at least 1, and warmup_updates is not negative")
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

    @property
    def alpha(self) -> float:
        return 2.0 * self.rank
