# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave parts of autograd untyped.)
"""The losses a policy step takes, by name: pure functions of one segment's logprobs, for any trainer in torch.

Three logprobs of each sampled token meet here:

- `behavior`: what the engine recorded while sampling it, under whichever checkpoint was served then;
- `old`: what the trainer gives it at the start of the step, on the weights the step starts from (no gradient);
- `logprobs`: what the trainer gives it now, as the step updates the weights (with gradient).

They differ for two separate reasons, and each has its own term. `old` against `behavior` is where the data came
from: an older checkpoint, and the engine computing differently from the trainer. It is corrected by an importance
weight `old / behavior`, truncated at `truncate` (a constant: no gradient flows through it). `logprobs` against `old` is
how far the step has moved the policy: the ratio PPO clips, which is exactly 1 when the step begins.

`policy_gradient` with `ratio = "token"`: each token's ratio clipped to 1 - `clip_low` .. 1 + `clip_high`
(DAPO's clip-higher), the loss a mean over the minibatch's tokens. With `ratio = "segment"` (GSPO): one ratio for the
segment, the geometric mean of its tokens' (and one weight, likewise), clipped as a whole, the loss a mean over each
segment's tokens and then over the minibatch's segments. `likelihood`: raise the log-likelihood of the sampled tokens,
each by its segment's advantage (imitation), reading neither `old` nor `behavior`.
"""

from dataclasses import dataclass

import torch

from rollout_lora.settings import Objective

__all__ = ["Objective", "Terms", "terms"]


@dataclass
class Terms:
    """One segment's part of a minibatch's loss: `loss` is summed over its units (the trainer divides by the
    minibatch's units); the rest are counts and sums over its tokens, for the step's statistics."""

    loss: torch.Tensor
    tokens: float
    clipped: float = 0.0
    """Tokens whose ratio was clipped."""
    truncated: float = 0.0
    """Tokens whose importance weight was truncated."""
    ratio: float = 0.0
    weight: float = 0.0
    moved: float = 0.0
    """The sum of `old - logprobs`: an estimate of KL(old || now) on the sampled tokens, times their number."""


def terms(
    objective: Objective,
    logprobs: torch.Tensor,
    advantage: float,
    old: torch.Tensor | None = None,
    behavior: torch.Tensor | None = None,
) -> Terms:
    """One segment's loss under `objective`, from its sampled tokens' logprobs now (with gradient) and, for a policy
    gradient, at the step's start (`old`) and when they were sampled (`behavior`)."""
    tokens = float(logprobs.numel())
    if objective.kind == "likelihood":
        return Terms(loss=-(logprobs * advantage).sum(), tokens=tokens)
    assert old is not None and behavior is not None
    old, behavior = old.detach(), behavior.detach()
    if objective.ratio == "token":
        weight = torch.exp(old - behavior)
        ratio = torch.exp(logprobs - old)
    else:  # one for the segment, and its gradient spread over its tokens (GSPO's token form)
        weight = torch.exp((old - behavior).mean()).expand_as(logprobs)
        mean = (logprobs - old).mean()
        ratio = torch.exp(mean.detach() + logprobs - logprobs.detach())
    capped = weight if objective.truncate is None else weight.clamp(max=objective.truncate)
    clipped = ratio.clamp(1 - objective.clip_low, 1 + objective.clip_high)
    per_token = -capped * torch.minimum(ratio * advantage, clipped * advantage)
    loss = per_token.sum() if objective.ratio == "token" else per_token.mean()
    with torch.no_grad():
        return Terms(
            loss=loss,
            tokens=tokens,
            clipped=float((ratio != clipped).sum()),
            truncated=float((capped != weight).sum()),
            ratio=float(ratio.sum()),
            weight=float(capped.sum()),
            moved=float((old - logprobs).sum()),
        )
