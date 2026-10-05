# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave parts of autograd untyped.)
"""Distillation in torch: a divergence between a teacher's next-token distribution and the policy's at each sampled
token (`rollout_train.objectives.Distillation`), from the teacher's scores a distilled segment carries
(`rollout_train.trainer.Distilled`).

Written per sampled token `y_t` of a segment, with `pi` the policy now (with gradient), `pi_old` the policy at the
step's start (no gradient), `T` the teacher, and `w_t` the importance weight (`rollout_objectives.terms
.importance_weight`; 1 without a correction):

- **The policy-gradient form** (`form = policy_gradient`, `reverse_kl`): the advantage `A_t = clip(log T(y_t) - log
  pi_old(y_t), -A_max, A_max)` (no gradient; `A_max = advantage_clip`, unclipped at 0), and the loss `-w_t A_t log
  pi(y_t)`. Its gradient at the step's start is the reverse KL's, estimated from the sampled tokens alone (on-policy
  distillation; MOPD's Eq. 3 and 4, its student's logprob the one the step starts from, as NeMo-RL's
  `prev_logprobs`).
- **The top-k form** (`form = top_k`), over the teacher's top-k tokens `V_t` at the position:
  - `reverse_kl`: `sum over v in V_t of [pi(v) log(pi(v) / T(v)) - pi(v) + T(v)]`, the probabilities as they are
    (MOPD's Eq. 5: each term is at least 0, and 0 where the two agree; the tail outside `V_t` is left out);
  - `forward_kl`: `tau² sum over v in V_t of q_T(v) log(q_T(v) / q_pi(v))`, both distributions renormalized over
    `V_t` at temperature `tau` (`q(v) = p(v)^(1/tau) / sum over u in V_t of p(u)^(1/tau)`): the teacher's mass outside
    its top k is dropped rather than spread, so the student is fitted to the teacher's top-k distribution, which is the
    teacher's own where its top k hold all of its mass (Hinton et al.'s soft targets, the full distribution's KL at
    k = the vocabulary);
  - `jsd`: `tau² [beta KL(q_T || m) + (1 - beta) KL(q_pi || m)]`, `m = beta q_T + (1 - beta) q_pi`, of the same
    renormalized distributions (GKD's generalized JSD, Agarwal et al., 2024, Eq. 3).
  The term is weighed by `w_t`.

A token the teacher did not score (none: beyond the teacher's context) adds nothing to the loss and counts in
its mean. A KL penalty is added (`kl.placement = loss`) or, in the policy-gradient form, taken from the advantage
(`reward`), as a policy gradient's; the per-token losses are reduced by `aggregate` (`rollout_objectives.terms
.reduced`). A policy gradient with a distillation term (`distillation.coefficient`) is the policy gradient's loss plus
the coefficient times the distillation term (no KL penalty of its own: the policy gradient's applies).
"""

import math
from dataclasses import dataclass

import torch

from rollout_objectives.terms import Terms, importance_weight, kl_estimate, moved_kl, policy_gradient, reduced
from rollout_train.objectives import DISTILLATION, POLICY_GRADIENT, Objective
from rollout_train.recorder import TeacherScores
from rollout_train.trainer import Distilled

__all__ = ["Taught", "distillation", "distilled", "term", "top_k_divergence"]


@dataclass
class Taught:
    """A segment's teacher scores as tensors: its logprob of each sampled token (NaN where it scored none), and, for the
    top-k form, its top tokens at each (padded with token 0) with their logprobs, and which are real (`kept`)."""

    logprobs: torch.Tensor
    top_tokens: torch.Tensor | None = None
    top_logprobs: torch.Tensor | None = None
    kept: torch.Tensor | None = None

    @classmethod
    def of(
        cls,
        scores: TeacherScores,
        top_k: int = 0,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "Taught":
        """The tensors of `scores`, with at most `top_k` top tokens at each position (0: none)."""
        logprobs = torch.tensor(
            [math.nan if each is None else each for each in scores.logprobs], dtype=dtype, device=device
        )
        if not top_k:
            return cls(logprobs)
        count = len(scores.logprobs)
        tokens = torch.zeros((count, top_k), dtype=torch.long)
        values = torch.zeros((count, top_k), dtype=dtype)
        kept = torch.zeros((count, top_k), dtype=torch.bool)
        for row, (ids, logged) in enumerate(zip(scores.top_tokens, scores.top_logprobs, strict=True)):
            width = min(len(ids), top_k)
            if width:
                tokens[row, :width] = torch.tensor(ids[:width])
                values[row, :width] = torch.tensor(logged[:width], dtype=dtype)
                kept[row, :width] = True
        return cls(logprobs, tokens.to(device), values.to(device), kept.to(device))


def _renormalized(logged: torch.Tensor, kept: torch.Tensor, temperature: float) -> torch.Tensor:
    """Logprobs over the kept entries of each row, at `temperature` and renormalized there (0 elsewhere, to be
    masked)."""
    scaled = logged / temperature
    rows = kept.any(-1, keepdim=True)
    held = kept | ~rows  # (a row with nothing kept is normalized over all, harmlessly, and masked)
    total = torch.logsumexp(torch.where(held, scaled, torch.full_like(scaled, -math.inf)), dim=-1, keepdim=True)
    return torch.where(kept, scaled - total, torch.zeros_like(scaled))


def top_k_divergence(
    objective: Objective, among: torch.Tensor, teacher: torch.Tensor, kept: torch.Tensor
) -> torch.Tensor:
    """Each position's divergence over the teacher's top-k tokens there (`objective.distillation.divergence`), from the
    policy's logprobs of those tokens (`among`, with gradient) and the teacher's, `kept` saying which are real."""
    said = objective.distillation
    zero = torch.zeros_like(among)
    teacher = torch.where(kept, teacher.to(among.dtype), zero)  # (padding as 0, so nothing below is infinite)
    if said.divergence == "reverse_kl":  # MOPD's Eq. 5, of the probabilities as they are
        mine, theirs = torch.exp(among), torch.exp(teacher)
        each = mine * (among - teacher) - mine + theirs
        return torch.where(kept, each, zero).sum(-1)
    tau = said.temperature
    log_mine, log_theirs = _renormalized(among, kept, tau), _renormalized(teacher, kept, tau)
    mine, theirs = torch.exp(log_mine), torch.exp(log_theirs)
    if said.divergence == "forward_kl":
        each = theirs * (log_theirs - log_mine)
    else:  # jsd
        beta = said.beta
        log_mixed = torch.logaddexp(math.log(beta) + log_theirs, math.log(1 - beta) + log_mine)
        each = beta * theirs * (log_theirs - log_mixed) + (1 - beta) * mine * (log_mine - log_mixed)
    return tau**2 * torch.where(kept, each, zero).sum(-1)


def term(
    objective: Objective,
    logprobs: torch.Tensor,
    old: torch.Tensor,
    taught: Taught,
    *,
    weight: torch.Tensor | None = None,
    among: torch.Tensor | None = None,
    shift: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Terms]:
    """The distillation term of each sampled token (0 where the teacher scored none), weighed by `weight`, and its
    statistics (with a zero loss): `among` is the policy's logprobs of the teacher's top-k tokens (the top-k form),
    and `shift` is taken from the policy-gradient form's advantage (a KL penalty in the reward)."""
    said = objective.distillation
    teacher = taught.logprobs.to(logprobs.dtype)
    scored = torch.isfinite(teacher)
    zero = torch.zeros_like(logprobs)
    clipped = 0.0
    divergence = 0.0
    if said.form == "policy_gradient":
        advantage = torch.where(scored, teacher - old.detach(), zero)
        if said.advantage_clip > 0:
            bounded = advantage.clamp(-said.advantage_clip, said.advantage_clip)
            clipped = float((bounded != advantage).sum())
            advantage = bounded
        if shift is not None:
            advantage = torch.where(scored, advantage - shift.detach(), zero)
        each = -advantage * logprobs
    else:
        if among is None or taught.top_logprobs is None or taught.kept is None:
            raise ValueError("the top_k form needs the policy's logprobs of the teacher's top-k tokens")
        kept = taught.kept & scored.unsqueeze(-1)
        each = top_k_divergence(objective, among, taught.top_logprobs, kept)
        divergence = float(each.detach().sum())
    if weight is not None:
        each = weight * each
    each = torch.where(scored, each, zero)
    with torch.no_grad():
        found = Terms(
            loss=torch.zeros((), dtype=logprobs.dtype, device=logprobs.device),
            tokens=float(logprobs.numel()),
            distilled=float(logprobs.numel()),
            scored=float(scored.sum()),
            gap=float(torch.where(scored, logprobs.detach() - teacher, zero).sum()),
            divergence=divergence,
            advantage_clipped=clipped,
        )
    return each, found


def distillation(
    objective: Objective,
    logprobs: torch.Tensor,
    old: torch.Tensor,
    taught: Taught,
    behavior: torch.Tensor | None = None,
    reference: torch.Tensor | None = None,
    among: torch.Tensor | None = None,
) -> Terms:
    """One distilled segment's loss (the distillation family), from its sampled tokens' logprobs now (with gradient),
    at the step's start (`old`), when they were sampled (`behavior`, for an importance correction), under the reference
    (for a KL to it), the teacher's scores, and for the top-k form the policy's logprobs of the teacher's top-k tokens
    (`among`, with gradient)."""
    old = old.detach()
    weight, raw = importance_weight(objective, old, behavior)
    kl = objective.kl
    penalty: torch.Tensor | None = None
    if kl.target != "none":
        target = reference if kl.target == "reference" else old
        if target is None:
            raise ValueError("a KL to the reference needs each sampled token's reference logprob")
        penalty = kl_estimate(kl.estimator, logprobs, target.detach())
    shift = kl.coefficient * penalty if penalty is not None and kl.placement == "reward" else None
    per_token, found = term(objective, logprobs, old, taught, weight=weight, among=among, shift=shift)
    if penalty is not None and kl.placement == "loss":
        per_token = per_token + kl.coefficient * penalty
    found.loss = reduced(objective, per_token)
    with torch.no_grad():
        found.ratio = float(torch.exp(logprobs - old).sum())
        found.weight = float(weight.sum()) if weight is not None else found.tokens
        found.truncated = float((weight != raw).sum()) if weight is not None and raw is not None else 0.0
        found.moved = float(moved_kl(old, logprobs).sum())
        found.kl = float(penalty.sum()) if penalty is not None else 0.0
    return found


def distilled(
    objective: Objective,
    item: Distilled,
    logprobs: torch.Tensor,
    old: torch.Tensor,
    behavior: torch.Tensor | None = None,
    reference: torch.Tensor | None = None,
    entropy: torch.Tensor | None = None,
    among: torch.Tensor | None = None,
) -> Terms:
    """A distilled segment's loss under `objective`: a distillation's, or a policy gradient's with its distillation
    term, the coefficient times the term reduced as the policy gradient is."""
    taught = Taught.of(item.scores, objective.needs_top, device=logprobs.device, dtype=logprobs.dtype)
    if objective.family == DISTILLATION:
        return distillation(objective, logprobs, old, taught, behavior, reference, among)
    if objective.family != POLICY_GRADIENT or not objective.distills:
        raise ValueError(f"a {objective.family} objective without a distillation term takes no distilled segments")
    found = policy_gradient(objective, logprobs, item.advantage, old, behavior, reference, entropy)
    weight, _ = importance_weight(objective, old, behavior)
    per_token, extra = term(objective, logprobs, old, taught, weight=weight, among=among)
    found.loss = found.loss + objective.distillation.coefficient * reduced(objective, per_token)
    for key in ("distilled", "scored", "gap", "divergence", "advantage_clipped"):
        setattr(found, key, getattr(extra, key))
    return found
