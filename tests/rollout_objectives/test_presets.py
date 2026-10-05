# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Each preset against its paper: its values pinned to the source, and its composed loss (`rollout_objectives.terms`,
with the advantages of `rollout_train.algorithm`) equal, in value and gradient, to a direct transcription of the
paper's formula on a fixed batch."""

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import pytest
import torch
from torch.nn import functional

from rollout_objectives.step import preference_terms
from rollout_objectives.terms import terms, units
from rollout_train.algorithm import advantages_of
from rollout_train.objectives import PRESETS, Objective
from rollout_train.recorder import Segment, Span
from rollout_train.trainer import Labelled, Pair

REWARDS = [1.0, 0.0, 0.0, 1.0, 0.5]
LENGTHS = [3, 5, 2, 4, 6]


@dataclass
class Sampled:
    """One segment of a group: its sampled tokens' logprobs now (a leaf), at the step's start, when sampled, and under
    the reference."""

    now: torch.Tensor
    old: torch.Tensor
    behavior: torch.Tensor
    reference: torch.Tensor


def group(seed: int = 0) -> list[Sampled]:
    """A group whose logprobs now have moved from the step's start, some far enough to be clipped."""
    generator = torch.Generator().manual_seed(seed)
    made: list[Sampled] = []
    for length in LENGTHS:
        old = -torch.rand(length, generator=generator, dtype=torch.float64) * 3 - 0.1
        moved = (torch.rand(length, generator=generator, dtype=torch.float64) - 0.5) * 0.6
        behavior = old + (torch.rand(length, generator=generator, dtype=torch.float64) - 0.5) * 1.6
        reference = old + (torch.rand(length, generator=generator, dtype=torch.float64) - 0.5) * 0.8
        made.append(Sampled((old + moved).clamp(max=-1e-3).requires_grad_(True), old, behavior, reference))
    return made


def composed(objective: Objective, sampled: Sequence[Sampled], rewards: Sequence[float]) -> torch.Tensor:
    """The minibatch's loss as a trainer takes it: each segment's terms over the minibatch's units."""
    advantages = advantages_of(rewards, objective.advantage)
    assert advantages is not None
    total = sum(units(objective, int(each.now.numel())) for each in sampled)
    loss = torch.zeros((), dtype=torch.float64)
    for each, advantage in zip(sampled, advantages, strict=True):
        loss = loss + terms(objective, each.now, advantage, each.old, each.behavior, each.reference).loss
    return loss / total


def same(objective: Objective, transcribed: Callable[[list[Sampled]], torch.Tensor]) -> None:
    """The composed loss and the transcription agree in value and in the gradient of every logprob."""
    ours, theirs = group(), group()
    mine = composed(objective, ours, REWARDS)
    paper = transcribed(theirs)
    mine.backward()
    paper.backward()
    torch.testing.assert_close(mine, paper, rtol=1e-12, atol=1e-12)
    for a, b in zip(ours, theirs, strict=True):
        assert a.now.grad is not None and b.now.grad is not None
        torch.testing.assert_close(a.now.grad, b.now.grad, rtol=1e-10, atol=1e-12)


def total(values: Iterable[torch.Tensor]) -> torch.Tensor:
    """The sum of tensors, as a tensor."""
    return torch.stack(list(values)).sum()


def standardized(rewards: Sequence[float]) -> list[float]:
    """(r - mean) / std, the sample standard deviation (GRPO's implementations)."""
    tensor = torch.tensor(rewards, dtype=torch.float64)
    return ((tensor - tensor.mean()) / tensor.std()).tolist()


def centred(rewards: Sequence[float]) -> list[float]:
    mean = sum(rewards) / len(rewards)
    return [reward - mean for reward in rewards]


def ratio_of(each: Sampled) -> torch.Tensor:
    return torch.exp(each.now - each.old)


def ppo_term(each: Sampled, advantage: float, low: float, high: float) -> torch.Tensor:
    ratio = ratio_of(each)
    return torch.minimum(ratio * advantage, ratio.clamp(1 - low, 1 + high) * advantage)


def test_the_presets_are_the_papers() -> None:
    p = {name: preset.objective for name, preset in PRESETS.items()}
    assert set(p) == {"default", "reinforce", "rloo", "ppo_clip", "grpo", "dr_grpo", "dapo", "gspo", "cispo", "sft",
                      "dpo", "ipo", "simpo", "kto", "orpo"}  # fmt: skip
    # Schulman et al. 2017, Sec. 3 and Table 1: epsilon 0.2.
    assert (p["ppo_clip"].clip.kind, p["ppo_clip"].clip.low, p["ppo_clip"].clip.high) == ("ratio", 0.2, 0.2)
    # DeepSeekMath, Eq. 3 and 4, Sec. 4.2: beta 0.04, the k3 estimator in the loss, 1/G sum 1/|o| sum_t. (Epsilon is
    # not stated; with one update a step the clip never acts. 0.2, as its implementations.)
    grpo = p["grpo"]
    assert (grpo.kl.target, grpo.kl.estimator, grpo.kl.placement, grpo.kl.coefficient) == (
        "reference",
        "k3",
        "loss",
        0.04,
    )
    assert (grpo.aggregate, grpo.advantage.scale, grpo.reference, grpo.clip.low, grpo.clip.high) == (
        "segment_mean", "group_std", "base", 0.2, 0.2)  # fmt: skip
    # Dr. GRPO, Sec. 3.2 and Listing 1: no std, no 1/|o|, a constant (their code's generation budget, 3000); beta 0.
    dr = p["dr_grpo"]
    assert (dr.advantage.scale, dr.aggregate, dr.constant_tokens, dr.kl.target, dr.clip.low) == (
        "none", "constant", 3000, "none", 0.2)  # fmt: skip
    # DAPO, Sec. 4.1 and Eq. 8, 11: (0.2, 0.28), the token mean, dynamic sampling, no KL; Eq. 9 normalizes by std.
    dapo = p["dapo"]
    assert (dapo.clip.low, dapo.clip.high, dapo.aggregate, dapo.advantage.filter, dapo.advantage.scale) == (
        0.2, 0.28, "token_mean", "equal_scores", "group_std")  # fmt: skip
    # GSPO, Eq. 5-7 and Sec. 5.1: the length-normalized sequence ratio clipped to (3e-4, 4e-4), 1/G over sequences.
    gspo = p["gspo"]
    assert (gspo.ratio, gspo.clip.low, gspo.clip.high, gspo.aggregate) == ("segment", 3e-4, 4e-4, "segment_mean")
    # MiniMax-M1, Eq. 4-5: sg(clip(r)) A log pi over all tokens, no lower bound; the upper is tuned and not given in the
    # paper: 4, Tinker's default (and among ScaleRL's equals).
    cispo = p["cispo"]
    assert (cispo.clip.kind, 1 - cispo.clip.low, 1 + cispo.clip.high, cispo.aggregate) == (
        "weight", 0.0, 4.0, "token_mean")  # fmt: skip
    # Ahmadian et al. 2024, Sec. 2.3: the leave-one-out baseline, the whole completion one action.
    assert (p["rloo"].advantage.baseline, p["rloo"].ratio, p["rloo"].aggregate) == (
        "leave_one_out",
        "none",
        "segment_sum",
    )
    # Williams 1992, Eq. 11: the reward times the episode's summed eligibilities.
    assert (p["reinforce"].advantage.baseline, p["reinforce"].ratio, p["reinforce"].aggregate) == (
        "none", "none", "segment_sum")  # fmt: skip
    # Rafailov et al. 2023, Appendix B: beta 0.1, sums. Azar et al. 2023 give no tau: 0.1 and token means, as TRL.
    assert (p["dpo"].preference.loss, p["dpo"].preference.beta, p["dpo"].preference.length_normalized) == (
        "sigmoid", 0.1, False)  # fmt: skip
    assert (p["ipo"].preference.loss, p["ipo"].preference.beta, p["ipo"].preference.length_normalized) == (
        "square", 0.1, True)  # fmt: skip
    # Meng et al. 2024, Table 8 (Llama-3-Base): beta 2.0, gamma 1.0; no reference.
    simpo = p["simpo"].preference
    assert (simpo.loss, simpo.beta, simpo.margin, simpo.length_normalized, p["simpo"].reference) == (
        "margin", 2.0, 1.0, True, "none")  # fmt: skip
    # Ethayarajh et al. 2024, Sec. 4.2: beta 0.1, lambda_D = lambda_U = 1.
    kto = p["kto"].preference
    assert (kto.loss, kto.beta, kto.desirable, kto.undesirable, p["kto"].reference) == ("kto", 0.1, 1.0, 1.0, "base")
    # Hong et al. 2024, Eq. 3, 6, 7 and Sec. 6.1 (Mistral-ORPO): lambda 0.1, odds of the mean token logprob.
    orpo = p["orpo"]
    assert (orpo.preference.loss, orpo.preference.beta, orpo.likelihood.coefficient, orpo.reference) == (
        "odds_ratio", 0.1, 1.0, "none")  # fmt: skip
    # This platform's default: Dr. GRPO's advantages, DAPO's clip-higher and token mean, TIS at 2.
    default = p["default"]
    assert (default.advantage.scale, default.advantage.filter, default.clip.low, default.clip.high) == (
        "none", "equal_scores", 0.2, 0.28)  # fmt: skip
    assert (default.importance.correction, default.importance.cap, default.aggregate) == ("truncate", 2.0, "token_mean")


def test_reinforce() -> None:  # -(1/G) sum_i r_i sum_t log pi
    same(PRESETS["reinforce"].objective,
         lambda g: -total(r * each.now.sum() for each, r in zip(g, REWARDS, strict=True)) / len(g))  # fmt: skip


def test_rloo() -> None:  # -(1/k) sum_i (R_i - 1/(k-1) sum_{j != i} R_j) log pi(y_i)
    def paper(g: list[Sampled]) -> torch.Tensor:
        k = len(g)
        return (
            -total((r - (sum(REWARDS) - r) / (k - 1)) * each.now.sum() for each, r in zip(g, REWARDS, strict=True)) / k
        )

    same(PRESETS["rloo"].objective, paper)


def test_ppo_clip() -> None:  # -(1/T) sum min(r A, clip(r, 1 - 0.2, 1 + 0.2) A)
    def paper(g: list[Sampled]) -> torch.Tensor:
        tokens = sum(each.now.numel() for each in g)
        return -total(ppo_term(e, a, 0.2, 0.2).sum() for e, a in zip(g, standardized(REWARDS), strict=True)) / tokens

    same(PRESETS["ppo_clip"].objective, paper)


def test_grpo() -> None:  # -(1/G) sum_i (1/|o_i|) sum_t [min(r A, clip(r) A) - beta (ref/pi - log(ref/pi) - 1)]
    def paper(g: list[Sampled]) -> torch.Tensor:
        total = torch.zeros((), dtype=torch.float64)
        for each, advantage in zip(g, standardized(REWARDS), strict=True):
            quotient = torch.exp(each.reference - each.now)
            kl = quotient - torch.log(quotient) - 1
            total = total + (ppo_term(each, advantage, 0.2, 0.2) - 0.04 * kl).mean()
        return -total / len(g)

    same(PRESETS["grpo"].objective, paper)


def test_dr_grpo() -> None:  # -(1/G) sum_i sum_t min(r A~, clip(r) A~) / MAX_TOKENS, A~ = R - mean
    def paper(g: list[Sampled]) -> torch.Tensor:
        return -total(ppo_term(e, a, 0.2, 0.2).sum() / 3000 for e, a in zip(g, centred(REWARDS), strict=True)) / len(g)

    same(PRESETS["dr_grpo"].objective, paper)


def test_dapo() -> None:  # -(1/sum |o_i|) sum_i sum_t min(r A, clip(r, 1 - 0.2, 1 + 0.28) A)
    def paper(g: list[Sampled]) -> torch.Tensor:
        tokens = sum(each.now.numel() for each in g)
        return -total(ppo_term(e, a, 0.2, 0.28).sum() for e, a in zip(g, standardized(REWARDS), strict=True)) / tokens

    same(PRESETS["dapo"].objective, paper)
    assert advantages_of([1.0, 1.0, 1.0], PRESETS["dapo"].objective.advantage) is None  # (dynamic sampling)


def test_gspo() -> None:  # -(1/G) sum_i min(s_i A_i, clip(s_i, 1 - 3e-4, 1 + 4e-4) A_i), s_i = (pi/pi_old)^(1/|y|)
    def paper(g: list[Sampled]) -> torch.Tensor:
        total = torch.zeros((), dtype=torch.float64)
        for each, advantage in zip(g, standardized(REWARDS), strict=True):
            s = torch.exp((each.now - each.old).mean())
            total = total + torch.minimum(s * advantage, s.clamp(1 - 3e-4, 1 + 4e-4) * advantage)
        return -total / len(g)

    same(PRESETS["gspo"].objective, paper)


def test_cispo() -> None:  # -(1/sum |o_i|) sum_i sum_t sg(clip(r, 0, 4)) A log pi
    def paper(g: list[Sampled]) -> torch.Tensor:
        tokens = sum(each.now.numel() for each in g)
        return (
            -total(
                (ratio_of(e).clamp(0.0, 4.0).detach() * a * e.now).sum()
                for e, a in zip(g, standardized(REWARDS), strict=True)
            )
            / tokens
        )

    same(PRESETS["cispo"].objective, paper)


def test_default() -> None:  # -(1/T) sum min(old/behavior, 2) min(r A, clip(r, 0.8, 1.28) A), A = R - mean
    def paper(g: list[Sampled]) -> torch.Tensor:
        tokens = sum(each.now.numel() for each in g)
        return (
            -total(
                (torch.exp(e.old - e.behavior).clamp(max=2.0) * ppo_term(e, a, 0.2, 0.28)).sum()
                for e, a in zip(g, centred(REWARDS), strict=True)
            )
            / tokens
        )

    same(PRESETS["default"].objective, paper)


def test_sft() -> None:  # -(1/T) sum A log pi
    def paper(g: list[Sampled]) -> torch.Tensor:
        tokens = sum(each.now.numel() for each in g)
        return -total((a * e.now).sum() for e, a in zip(g, centred(REWARDS), strict=True)) / tokens

    same(PRESETS["sft"].objective, paper)


# Preferences: pairs of the group's segments, (0 over 1), (3 over 2), (4 over 1).

PAIRS = [(0, 1), (3, 2), (4, 1)]


SEGMENTS = [Segment(list(range(length + 1)), [Span(1, length + 1, 0)], [0.0] * length) for length in LENGTHS]
"""Segments standing for the group's: the step finds each one's logprobs by the segment."""
NOW: dict[int, torch.Tensor] = {}
REFERENCE: dict[int, torch.Tensor] = {}


def preference(objective: Objective, items: Sequence[Pair | Labelled]) -> torch.Tensor:
    """The composed preference loss of `items`, a mean over them, as the step takes it."""
    found = preference_terms(objective, items, NOW, REFERENCE)
    return torch.stack([each.loss for _, each in found]).sum() / len(items)


def same_preference(
    objective: Objective, items: Sequence[Pair | Labelled], transcribed: Callable[[list[Sampled]], torch.Tensor]
) -> None:
    ours, theirs = group(), group()
    NOW.clear()
    REFERENCE.clear()
    for segment, each in zip(SEGMENTS, ours, strict=True):
        NOW[id(segment)], REFERENCE[id(segment)] = each.now, each.reference
    mine = preference(objective, items)
    paper = transcribed(theirs)
    mine.backward()
    paper.backward()
    torch.testing.assert_close(mine, paper, rtol=1e-12, atol=1e-12)
    for a, b in zip(ours, theirs, strict=True):
        if b.now.grad is None:
            assert a.now.grad is None or not bool(a.now.grad.any())
            continue
        assert a.now.grad is not None
        torch.testing.assert_close(a.now.grad, b.now.grad, rtol=1e-10, atol=1e-12)


def pairs() -> list[Pair]:
    return [Pair((SEGMENTS[winner],), (SEGMENTS[loser],)) for winner, loser in PAIRS]


def test_dpo() -> None:  # -log sigmoid(beta log(pi_w/ref_w) - beta log(pi_l/ref_l)), sums over tokens
    def paper(g: list[Sampled]) -> torch.Tensor:
        def rho(e: Sampled) -> torch.Tensor:
            return e.now.sum() - e.reference.sum()

        return -total(functional.logsigmoid(0.1 * (rho(g[winner]) - rho(g[loser]))) for winner, loser in PAIRS) / len(
            PAIRS
        )

    same_preference(PRESETS["dpo"].objective, pairs(), paper)


def test_ipo() -> None:  # (h - 1/(2 tau))^2, h of the token-mean log ratios, tau 0.1
    def paper(g: list[Sampled]) -> torch.Tensor:
        def rho(e: Sampled) -> torch.Tensor:
            return (e.now.sum() - e.reference.sum()) / e.now.numel()

        return total((rho(g[winner]) - rho(g[loser]) - 1 / (2 * 0.1)) ** 2 for winner, loser in PAIRS) / len(PAIRS)

    same_preference(PRESETS["ipo"].objective, pairs(), paper)


def test_simpo() -> None:  # -log sigmoid(beta/|y_w| log pi(y_w) - beta/|y_l| log pi(y_l) - gamma)
    def paper(g: list[Sampled]) -> torch.Tensor:
        def reward(e: Sampled) -> torch.Tensor:
            return 2.0 / e.now.numel() * e.now.sum()

        return -total(
            functional.logsigmoid(reward(g[winner]) - reward(g[loser]) - 1.0) for winner, loser in PAIRS
        ) / len(PAIRS)

    same_preference(PRESETS["simpo"].objective, pairs(), paper)


def test_orpo() -> None:  # L_SFT + lambda L_OR, odds = P/(1 - P), P = exp(mean log pi)
    def paper(g: list[Sampled]) -> torch.Tensor:
        def log_odds(e: Sampled) -> torch.Tensor:
            mean = e.now.mean()
            return torch.log(torch.exp(mean) / (1 - torch.exp(mean)))

        total = torch.zeros((), dtype=torch.float64)
        for winner, loser in PAIRS:
            total = total - g[winner].now.mean() - 0.1 * functional.logsigmoid(log_odds(g[winner]) - log_odds(g[loser]))
        return total / len(PAIRS)

    same_preference(PRESETS["orpo"].objective, pairs(), paper)


def test_kto() -> None:  # lambda_y - v(x, y): v = lambda_D sigmoid(beta (r - z0)) or lambda_U sigmoid(beta (z0 - r))
    desirable = [True, False, False, True, True]

    def paper(g: list[Sampled]) -> torch.Tensor:
        r = [e.now.sum() - e.reference.sum() for e in g]
        z0 = max(0.0, float(torch.stack(r).mean()))  # (no gradient through it; the minibatch's own examples)
        losses = [
            1.0 - torch.sigmoid(0.1 * (each - z0)) if good else 1.0 - torch.sigmoid(0.1 * (z0 - each))
            for each, good in zip(r, desirable, strict=True)
        ]
        return total(losses) / len(g)

    items = [Labelled((segment,), good) for segment, good in zip(SEGMENTS, desirable, strict=True)]
    same_preference(PRESETS["kto"].objective, items, paper)


@pytest.mark.parametrize("name", ["dpo", "ipo", "simpo", "kto", "orpo"])
def test_a_preference_preset_is_of_the_preference_family(name: str) -> None:
    objective = PRESETS[name].objective
    assert objective.family == "preference" and objective.labelled == (name == "kto")
    assert objective.needs_reference == (name in ("dpo", "ipo", "kto")) and not objective.needs_behaviour
    assert math.isfinite(objective.preference.beta)
