# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave parts of autograd untyped.)
"""An objective's loss, composed from its components (`rollout_train.objectives`): pure functions of logprobs, for any
trainer in torch.

Four logprobs of each sampled token meet here:

- `behavior`: what the engine recorded while sampling it, under whichever checkpoint was served then;
- `old`: what the trainer gives it at the start of the step, on the weights the step starts from (no gradient);
- `logprobs`: what the trainer gives it now, as the step updates the weights (with gradient);
- `reference`: what the reference model gives it (the base model; no gradient).

`old` against `behavior` is where the data came from: an older checkpoint, and the engine computing differently from
the trainer. The importance correction weighs it (`importance`: a constant, with no gradient). `logprobs` against
`old` is how far the step has moved the policy: the ratio that clipping bounds, exactly 1 when the step begins.
`logprobs` against `reference` (or `old`) is what a KL penalty measures.

A **policy gradient** (`policy_gradient`) of one segment: each token's surrogate is the advantage times its ratio
(`ratio = token`), its segment's ratio (`segment`, GSPO's, its gradient spread over the tokens) or its logprob
(`none`), clipped as `clip.kind` says, times the importance weight; less the KL penalty in the loss, or with it taken
from the advantage; less the entropy bonus. Its loss is the negative surrogate, reduced by `aggregate` (`reduced`)
and divided by the minibatch's `units`. A **likelihood** (`likelihood`) is the advantage times the logprob, reduced
likewise. A **preference** loss (`pair`, `labelled`) is a function of each side's sums (or means) of logprobs, a
mean over the minibatch's items.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional

from rollout_train.objectives import LIKELIHOOD, PREFERENCE, Objective

__all__ = [
    "SUMS",
    "Scored",
    "Terms",
    "kl_estimate",
    "labelled",
    "likelihood",
    "pair",
    "policy_gradient",
    "reduced",
    "tally",
    "terms",
    "units",
]


@dataclass
class Terms:
    """One segment's part of a minibatch's loss (`loss`, which the trainer divides by the minibatch's units), or one
    preference item's; the rest are counts and sums, for the step's statistics."""

    loss: torch.Tensor
    tokens: float
    clipped: float = 0.0
    """Tokens whose ratio was clipped."""
    truncated: float = 0.0
    """Tokens whose importance weight was truncated, or that a mask dropped."""
    ratio: float = 0.0
    weight: float = 0.0
    moved: float = 0.0
    """The sum of `old - logprobs`: an estimate of KL(old || now) on the sampled tokens, times their number."""
    kl: float = 0.0
    """The sum of the KL penalty's estimate over the tokens."""
    entropy: float = 0.0
    items: float = 0.0
    """Preference items: pairs or examples."""
    pairs: float = 0.0
    accurate: float = 0.0
    """Of the items, those whose chosen side's log ratio is above the rejected's (an example's above the reference
    point if desirable, below it if not)."""
    margin: float = 0.0
    """The sum of each pair's `rho_chosen - rho_rejected`, and of each example's distance from the reference point on
    its label's side (`rho - z` if desirable, `z - rho` if not)."""
    chosen: float = 0.0
    rejected: float = 0.0
    """Sums of each pair's `rho_chosen` and `rho_rejected`."""


def units(objective: Objective, tokens: int) -> float:
    """What a segment of `tokens` sampled tokens counts for in its minibatch's mean (a preference item counts 1)."""
    if objective.family == PREFERENCE:
        return 1.0
    return float(tokens) if objective.aggregate == "token_mean" else 1.0


def reduced(objective: Objective, per_token: torch.Tensor) -> torch.Tensor:
    """A segment's per-token losses as its part of the minibatch's (`aggregate`)."""
    if objective.aggregate == "segment_mean":
        return per_token.mean()
    if objective.aggregate == "constant":
        return per_token.sum() / objective.constant_tokens
    return per_token.sum()  # (token_mean: the minibatch's tokens are its units; segment_sum)


def kl_estimate(estimator: str, logprobs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Each sampled token's estimate of KL(policy || target), from `log_r = target - logprobs` (Schulman's k1, k2,
    k3)."""
    log_r = target - logprobs
    if estimator == "k1":
        return -log_r
    if estimator == "k2":
        return 0.5 * log_r.square()
    return torch.exp(log_r) - 1 - log_r


def policy_gradient(
    objective: Objective,
    logprobs: torch.Tensor,
    advantage: float,
    old: torch.Tensor,
    behavior: torch.Tensor | None = None,
    reference: torch.Tensor | None = None,
    entropy: torch.Tensor | None = None,
) -> Terms:
    """One segment's policy-gradient loss, from its sampled tokens' logprobs now (with gradient), at the step's start
    (`old`), when they were sampled (`behavior`, for an importance correction), under the reference (for a KL to it)
    and each position's entropy (for an entropy bonus)."""
    tokens = float(logprobs.numel())
    old = old.detach()
    importance, clip, kl = objective.importance, objective.clip, objective.kl

    weight: torch.Tensor | None = None
    raw: torch.Tensor | None = None
    if importance.correction != "none":
        if behavior is None:
            raise ValueError("an importance correction needs each sampled token's behaviour logprob")
        behavior = behavior.detach()
        if importance.level == "token":
            raw = torch.exp(old - behavior)
        else:  # (one for the segment: the geometric mean of its tokens')
            raw = torch.exp((old - behavior).mean()).expand_as(logprobs)
        if importance.correction == "untruncated":
            weight = raw
        elif importance.correction == "truncate":
            weight = raw.clamp(max=importance.cap)
        else:
            kept = (raw >= importance.floor) & (raw <= importance.cap)
            weight = torch.where(kept, raw, torch.zeros_like(raw))

    ratio: torch.Tensor | None = None
    if objective.ratio == "token":
        ratio = torch.exp(logprobs - old)
    elif (
        objective.ratio == "segment"
    ):  # one for the segment, and its gradient spread over its tokens (GSPO's token form)
        mean = (logprobs - old).mean()
        ratio = torch.exp(mean.detach() + logprobs - logprobs.detach())

    penalty: torch.Tensor | None = None
    if kl.target != "none":
        target = reference if kl.target == "reference" else old
        if target is None:
            raise ValueError("a KL to the reference needs each sampled token's reference logprob")
        penalty = kl_estimate(kl.estimator, logprobs, target.detach())
    advantages: float | torch.Tensor = advantage
    if penalty is not None and kl.placement == "reward":
        advantages = advantage - kl.coefficient * penalty.detach()

    bounded: torch.Tensor | None = None
    if ratio is None:
        surrogate = logprobs * advantages
    elif clip.kind == "none":
        surrogate = ratio * advantages
    elif clip.kind == "weight":  # the clipped ratio as a weight, with no gradient, times the logprob (CISPO)
        bounded = ratio.clamp(1 - clip.low, 1 + clip.high)
        surrogate = bounded.detach() * logprobs * advantages
    else:
        bounded = ratio.clamp(1 - clip.low, 1 + clip.high)
        surrogate = torch.minimum(ratio * advantages, bounded * advantages)
        if clip.kind == "dual":
            floor = clip.dual * torch.as_tensor(advantages, dtype=surrogate.dtype, device=surrogate.device)
            surrogate = torch.where(floor < 0, torch.maximum(surrogate, floor), surrogate)
    per_token = -surrogate if weight is None else -weight * surrogate
    if penalty is not None and kl.placement == "loss":
        per_token = per_token + kl.coefficient * penalty
    if objective.entropy.coefficient != 0.0:
        if entropy is None:
            raise ValueError("an entropy bonus needs each sampled position's entropy")
        per_token = per_token - objective.entropy.coefficient * entropy
    loss = reduced(objective, per_token)
    with torch.no_grad():
        shown = ratio if ratio is not None else torch.exp(logprobs - old)
        return Terms(
            loss=loss,
            tokens=tokens,
            clipped=float((ratio != bounded).sum()) if ratio is not None and bounded is not None else 0.0,
            truncated=float((weight != raw).sum()) if weight is not None and raw is not None else 0.0,
            ratio=float(shown.sum()),
            weight=float(weight.sum()) if weight is not None else tokens,
            moved=float((old - logprobs).sum()),
            kl=float(penalty.sum()) if penalty is not None else 0.0,
            entropy=float(entropy.sum()) if entropy is not None else 0.0,
        )


def likelihood(objective: Objective, logprobs: torch.Tensor, advantage: float) -> Terms:
    """One segment's likelihood loss: its sampled tokens' log-likelihood, weighted by its advantage (imitation: what
    was sampled is what to do), reading neither `old` nor `behavior`."""
    return Terms(loss=reduced(objective, -(logprobs * advantage)), tokens=float(logprobs.numel()))


def terms(
    objective: Objective,
    logprobs: torch.Tensor,
    advantage: float,
    old: torch.Tensor | None = None,
    behavior: torch.Tensor | None = None,
    reference: torch.Tensor | None = None,
    entropy: torch.Tensor | None = None,
) -> Terms:
    """One weighted segment's loss under `objective`: a policy gradient's or a likelihood's."""
    if objective.family == LIKELIHOOD:
        return likelihood(objective, logprobs, advantage)
    if objective.family == PREFERENCE:
        raise ValueError("a preference loss is of pairs or labelled examples, not of weighted segments")
    if old is None:
        raise ValueError("a policy gradient needs each sampled token's logprob at the step's start")
    return policy_gradient(objective, logprobs, advantage, old, behavior, reference, entropy)


@dataclass
class Scored:
    """One side of a preference item: each of its segments' sampled tokens' logprobs now, and the reference's."""

    logprobs: Sequence[torch.Tensor]
    reference: Sequence[torch.Tensor] | None = None

    def tokens(self) -> int:
        return sum(int(each.numel()) for each in self.logprobs)

    def likelihood(self, objective: Objective) -> torch.Tensor:
        """Its log-likelihood: the sum of its tokens' logprobs, or their mean (`length_normalized`)."""
        total = torch.stack([each.sum() for each in self.logprobs]).sum()
        return total / self.tokens() if objective.preference.length_normalized else total

    def rho(self, objective: Objective) -> torch.Tensor:
        """Its log-likelihood ratio to the reference, or its log-likelihood where the loss has none."""
        mine = self.likelihood(objective)
        if objective.reference == "none":
            return mine
        if self.reference is None:
            raise ValueError("the preference loss compares with the reference, and no reference logprobs were given")
        theirs = torch.stack([each.sum() for each in self.reference]).sum()
        if objective.preference.length_normalized:
            theirs = theirs / self.tokens()
        return mine - theirs


def _log_odds(mean: torch.Tensor) -> torch.Tensor:
    """log(p / (1 - p)) of `p = exp(mean)`, a length-normalized likelihood (ORPO's odds)."""
    return mean - torch.log1p(-torch.exp(mean.clamp(max=-1e-6)))


def pair(objective: Objective, chosen: Scored, rejected: Scored) -> Terms:
    """A pair's preference loss (`preference.loss`), and a likelihood term on its chosen side
    (`likelihood.coefficient`, ORPO's)."""
    said = objective.preference
    rho_chosen, rho_rejected = chosen.rho(objective), rejected.rho(objective)
    h = rho_chosen - rho_rejected
    if said.loss == "sigmoid":
        loss = -functional.logsigmoid(said.beta * h)
    elif said.loss == "hinge":
        loss = torch.relu(1 - said.beta * h)
    elif said.loss == "square":
        loss = (h - 1 / (2 * said.beta)).square()
    elif said.loss == "margin":
        loss = -functional.logsigmoid(said.beta * h - said.margin)
    elif said.loss == "odds_ratio":
        loss = -said.beta * functional.logsigmoid(_log_odds(rho_chosen) - _log_odds(rho_rejected))
    else:
        raise ValueError(f"the {said.loss} loss is of labelled examples, not pairs")
    if objective.likelihood.coefficient != 0.0:
        mean = torch.stack([each.sum() for each in chosen.logprobs]).sum() / chosen.tokens()
        loss = loss - objective.likelihood.coefficient * mean
    with torch.no_grad():
        return Terms(
            loss=loss,
            tokens=float(chosen.tokens() + rejected.tokens()),
            items=1.0,
            pairs=1.0,
            accurate=float(h > 0),
            margin=float(h),
            chosen=float(rho_chosen),
            rejected=float(rho_rejected),
        )


def labelled(objective: Objective, examples: Sequence[tuple[Scored, bool]]) -> list[Terms]:
    """KTO's loss of each labelled example (desirable or not), against the reference point `z`: the mean of the
    examples' log ratios, no less than 0, with no gradient (an estimate of the policy's KL to the reference)."""
    said = objective.preference
    rhos = [scored.rho(objective) for scored, _ in examples]
    if not rhos:
        return []
    z = torch.stack([each.detach() for each in rhos]).mean().clamp(min=0)
    found: list[Terms] = []
    for (scored, desirable), rho in zip(examples, rhos, strict=True):
        if desirable:
            loss = said.desirable * (1 - torch.sigmoid(said.beta * (rho - z)))
        else:
            loss = said.undesirable * (1 - torch.sigmoid(said.beta * (z - rho)))
        with torch.no_grad():
            found.append(
                Terms(
                    loss=loss,
                    tokens=float(scored.tokens()),
                    items=1.0,
                    accurate=float((rho > z) if desirable else (rho < z)),
                    margin=float(rho - z) if desirable else float(z - rho),
                )
            )
    return found


SUMS: tuple[str, ...] = (
    "loss", "units", "clipped", "truncated", "tokens", "ratio", "weight", "moved", "segments", "kl", "entropy",
    "items", "pairs", "accurate", "margin", "chosen", "rejected",
)  # fmt: skip
"""What a minibatch's `Terms` add up to, for its statistics and the step's."""


def tally(sums: dict[str, float], found: Terms, objective: Objective, *, segments: float = 1.0) -> None:
    """Add one segment's `found` terms (or one preference item's, of `segments` segments) to `sums` (keyed by
    `SUMS`)."""
    sums["loss"] += float(found.loss.detach())
    sums["units"] += units(objective, int(found.tokens))
    sums["segments"] += segments
    for key in ("clipped", "truncated", "tokens", "ratio", "weight", "moved", "kl", "entropy", "items", "pairs",
                "accurate", "margin", "chosen", "rejected"):  # fmt: skip
        sums[key] += getattr(found, key)
