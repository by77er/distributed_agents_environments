# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Each component on its own: what it does to one segment's loss and gradient, the default as the step has always
computed it, and the step over pairs and labelled examples on a toy policy."""

import copy
from collections.abc import Sequence

import pytest
import torch
from torch import nn

from rollout_objectives.settings import StepSettings
from rollout_objectives.step import PolicyStep, positions
from rollout_objectives.terms import kl_estimate, reduced, terms, units
from rollout_train.objectives import DEFAULT, Objective, resolved
from rollout_train.recorder import Segment, Span
from rollout_train.trainer import Labelled, Pair, Weighted


def objective(preset: str = "default", **overrides: object) -> Objective:
    return resolved(preset, {key.replace("__", "."): value for key, value in overrides.items()})  # type: ignore[misc]


LOGPROBS = [-1.0, -2.0, -0.5, -1.5]
OLD = [-1.0, -2.6, -0.5, -1.0]  # ratios 1, e^0.6, 1, e^-0.5
BEHAVIOR = [-1.0, -2.6, -3.0, -1.0]  # weights 1, 1, e^2.5, 1


def evaluated(made: Objective, advantage: float = 1.0, **given: torch.Tensor) -> tuple[float, list[float], object]:
    logprobs = torch.tensor(LOGPROBS, dtype=torch.float64, requires_grad=True)
    found = terms(
        made,
        logprobs,
        advantage,
        torch.tensor(OLD, dtype=torch.float64),
        torch.tensor(BEHAVIOR, dtype=torch.float64),
        given.get("reference"),
        given.get("entropy"),
    )
    found.loss.backward()
    assert logprobs.grad is not None
    return float(found.loss), logprobs.grad.tolist(), found


def test_the_default_is_the_step_as_it_always_was() -> None:
    """The default preset against the step's loss as it was written before the objective was composed (kept here
    word for word): the same numbers, to the last bit."""

    def before(logprobs: torch.Tensor, advantage: float, old: torch.Tensor, behavior: torch.Tensor, segment: bool):
        if not segment:
            weight = torch.exp(old - behavior)
            ratio = torch.exp(logprobs - old)
            low, high = 0.2, 0.28
        else:
            weight = torch.exp((old - behavior).mean()).expand_as(logprobs)
            mean = (logprobs - old).mean()
            ratio = torch.exp(mean.detach() + logprobs - logprobs.detach())
            low, high = 3e-4, 4e-4
        capped = weight.clamp(max=2.0)
        clipped = ratio.clamp(1 - low, 1 + high)
        per_token = -capped * torch.minimum(ratio * advantage, clipped * advantage)
        return per_token.sum() if not segment else per_token.mean()

    generator = torch.Generator().manual_seed(1)
    for segment, made in ((False, DEFAULT), (True, StepSettings(ratio="segment").loss)):
        for advantage in (0.7, -1.3):
            old = -torch.rand(9, generator=generator) * 2
            behavior = old + torch.randn(9, generator=generator) * 0.7
            a = (old + torch.randn(9, generator=generator) * 0.3).requires_grad_(True)
            b = a.detach().clone().requires_grad_(True)
            ours = terms(made, a, advantage, old, behavior).loss
            theirs = before(b, advantage, old, behavior, segment)
            ours.backward()
            theirs.backward()
            assert torch.equal(ours, theirs) and a.grad is not None and b.grad is not None
            assert torch.equal(a.grad, b.grad)


def test_clipping_by_kind() -> None:
    _, plain, _ = evaluated(objective(clip__kind="none", importance__correction="none"))
    assert plain == pytest.approx([-1.0, -(2.6 - 2.0 + 0) * 0 - torch.exp(torch.tensor(0.6)).item(), -1.0,
                                   -torch.exp(torch.tensor(-0.5)).item()])  # fmt: skip
    _, ratio, found = evaluated(objective(importance__correction="none"))  # (0.2, 0.28): e^0.6 clipped, e^-0.5 too
    shrunk = torch.exp(torch.tensor(-0.5)).item()  # (below the bound, a positive advantage's ratio is not clipped)
    assert ratio == pytest.approx([-1.0, 0.0, -1.0, -shrunk]) and found.clipped == 2.0  # type: ignore[attr-defined]
    _, negative, _ = evaluated(objective(importance__correction="none"), advantage=-1.0)
    assert negative == pytest.approx([1.0, torch.exp(torch.tensor(0.6)).item(), 1.0, 0.0])  # (min: the larger loss)
    _, weight, _ = evaluated(objective(clip__kind="weight", importance__correction="none"))  # CISPO: sg(clip(r)) log pi
    assert weight == pytest.approx([-1.0, -1.28, -1.0, -0.8])  # (every token keeps a gradient)
    _, dual, _ = evaluated(objective(clip__kind="dual", clip__dual=1.5, importance__correction="none"), advantage=-1.0)
    assert dual == pytest.approx([1.0, 0.0, 1.0, 0.0])  # (e^0.6 > 1.5: bounded at 1.5 times the advantage)
    with pytest.raises(ValueError, match="clips a ratio, and ratio is none"):
        objective(ratio="none")


def test_importance_corrections() -> None:
    made = objective(clip__kind="none", ratio="none")
    _, untruncated, _ = evaluated(objective(clip__kind="none", ratio="none", importance__correction="untruncated"))
    _, truncated, found = evaluated(made)
    _, masked, masking = evaluated(objective(clip__kind="none", ratio="none", importance__correction="mask",
                                             importance__cap=2.0))  # fmt: skip
    e = torch.exp(torch.tensor(2.5)).item()
    assert untruncated == pytest.approx([-1.0, -1.0, -e, -1.0])
    assert truncated == pytest.approx([-1.0, -1.0, -2.0, -1.0]) and found.truncated == 1.0  # type: ignore[attr-defined]
    assert masked == pytest.approx([-1.0, -1.0, 0.0, -1.0]) and masking.truncated == 1.0  # type: ignore[attr-defined]
    _, none, _ = evaluated(objective(clip__kind="none", ratio="none", importance__correction="none"))
    assert none == pytest.approx([-1.0] * 4)
    segment = objective(
        clip__kind="none", ratio="none", importance__correction="untruncated", importance__level="segment"
    )
    _, spread, _ = evaluated(segment)
    assert spread == pytest.approx([-torch.exp(torch.tensor(2.5 / 4)).item()] * 4)  # (the geometric mean's)


def test_the_kl_estimators_and_where_the_penalty_goes() -> None:
    logprobs, target = torch.tensor([-1.0, -2.0]), torch.tensor([-1.5, -1.0])
    log_r = target - logprobs
    assert kl_estimate("k1", logprobs, target).tolist() == pytest.approx((-log_r).tolist())
    assert kl_estimate("k2", logprobs, target).tolist() == pytest.approx((log_r**2 / 2).tolist())
    assert kl_estimate("k3", logprobs, target).tolist() == pytest.approx((log_r.exp() - 1 - log_r).tolist())
    reference = torch.tensor([-1.2, -2.0, -0.4, -1.5], dtype=torch.float64)
    base = objective(importance__correction="none", clip__kind="none", ratio="none")
    _, plain, _ = evaluated(base)
    loss = objective(importance__correction="none", clip__kind="none", ratio="none", kl__target="reference",
                     kl__estimator="k1", kl__coefficient=0.5)  # fmt: skip
    assert loss.reference == "base"  # (a KL to the reference reads the base model)
    _, penalized, found = evaluated(loss, reference=reference)
    assert penalized == pytest.approx([each + 0.5 for each in plain])  # (k1's gradient is the logprob's)
    assert found.kl == pytest.approx(float((torch.tensor(LOGPROBS, dtype=torch.float64) - reference).sum()))  # type: ignore[attr-defined]
    reward = objective(importance__correction="none", clip__kind="none", ratio="none", kl__target="reference",
                       kl__estimator="k1", kl__placement="reward", kl__coefficient=0.5)  # fmt: skip
    _, shaped, _ = evaluated(reward, reference=reference)
    k1 = torch.tensor(LOGPROBS, dtype=torch.float64) - reference
    assert shaped == pytest.approx((-(1.0 - 0.5 * k1)).tolist())  # (from each token's advantage, with no gradient)
    old = objective(importance__correction="none", kl__target="old", kl__estimator="k2", kl__coefficient=1.0)
    assert old.reference == "none"
    _, toward_old, _ = evaluated(old)
    assert toward_old[0] == pytest.approx(-1.0)  # (where the policy has not moved, k2's gradient is nothing)
    with pytest.raises(ValueError, match=r"kl.target is none"):
        objective(kl__coefficient=0.1)


def test_the_entropy_bonus() -> None:
    entropy = torch.tensor([1.0, 2.0, 0.5, 0.0], dtype=torch.float64)
    loss, _, found = evaluated(objective(importance__correction="none", entropy__coefficient=0.1), entropy=entropy)
    unbonused, _, _ = evaluated(objective(importance__correction="none"))
    assert loss == pytest.approx(unbonused - 0.1 * 3.5) and found.entropy == 3.5  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="entropy"):
        evaluated(objective(importance__correction="none", entropy__coefficient=0.1))


def test_aggregation() -> None:
    per_token = torch.tensor([1.0, 2.0, 3.0, 6.0])
    assert float(reduced(objective(aggregate="token_mean"), per_token)) == 12.0  # (over the minibatch's tokens)
    assert units(objective(aggregate="token_mean"), 4) == 4.0
    assert float(reduced(objective(aggregate="segment_mean"), per_token)) == 3.0
    assert float(reduced(objective(aggregate="segment_sum"), per_token)) == 12.0
    assert float(reduced(objective(aggregate="constant", constant_tokens=6), per_token)) == 2.0
    assert {units(objective(aggregate=each), 4) for each in ("segment_mean", "segment_sum", "constant")} == {1.0}
    assert units(resolved("dpo"), 100) == 1.0  # (a preference item counts 1)


class ToyPolicy:
    """A bigram model over a tiny vocabulary, with a frozen copy of where it started as its reference."""

    def __init__(self) -> None:
        torch.manual_seed(0)
        self.model = nn.Sequential(nn.Embedding(8, 16), nn.Linear(16, 8)).double()
        self.frozen = copy.deepcopy(self.model)

    def parameters(self) -> list[nn.Parameter]:
        return list(self.model.parameters())

    def _scored(self, model: nn.Module, tokens: Sequence[int], at: Sequence[int]) -> torch.Tensor:
        ids = torch.tensor(list(tokens))
        logits = model(ids[[p - 1 for p in at]])
        return torch.log_softmax(logits, -1).gather(-1, ids[list(at)].unsqueeze(-1)).squeeze(-1)

    def logprobs(self, tokens: Sequence[int], at: Sequence[int]) -> torch.Tensor:
        return self._scored(self.model, tokens, at)

    def reference(self, tokens: Sequence[int], at: Sequence[int]) -> torch.Tensor:
        with torch.no_grad():
            return self._scored(self.frozen, tokens, at)

    def logprobs_and_entropy(self, tokens: Sequence[int], at: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.tensor(list(tokens))
        logged = torch.log_softmax(self.model(ids[[p - 1 for p in at]]), -1)
        found = logged.gather(-1, ids[list(at)].unsqueeze(-1)).squeeze(-1)
        return found, -(logged.exp() * logged).sum(-1)


def sampled(tokens: list[int]) -> Segment:
    """Every token after the first sampled (behaviour logprobs are not read by a preference loss: none here)."""
    return Segment(tokens, [Span(1, len(tokens), 0)], [float("nan")] * (len(tokens) - 1))


def test_a_preference_step_takes_the_gradient_of_the_whole_loss_one_segment_at_a_time() -> None:
    """The step's gradient, taken in two parts (the loss's gradient of each logprob, then each segment's logprobs moved
    by it), is the gradient of the loss computed at once."""
    chosen, rejected = sampled([1, 2, 3, 4]), sampled([1, 5, 6])
    other = sampled([2, 7, 7, 1, 3])
    items = [Pair((chosen,), (rejected,)), Pair((other, chosen), (rejected,))]  # (a side of two segments)
    for preset in ("dpo", "ipo", "simpo", "orpo"):
        policy = ToyPolicy()
        settings = StepSettings(
            objective=preset, learning_rate=0.0, tokens_per_step=10**6, max_kl=None, max_gradient_norm=1e9
        )
        stepping = PolicyStep(policy, settings)  # type: ignore[arg-type]
        stepping.step(items)
        gradients = [parameter.grad for parameter in policy.parameters()]
        assert all(each is None for each in gradients)  # (stepped on: cleared)

        # At once: the loss of every side's logprobs, with the graph of each.
        direct = ToyPolicy()
        made = settings.loss
        from rollout_objectives.step import preference_terms

        now = {id(each): direct.logprobs(each.tokens, positions(each)) for each in (chosen, rejected, other)}
        reference = {id(each): direct.reference(each.tokens, positions(each)) for each in (chosen, rejected, other)}
        total = torch.stack([found.loss for _, found in preference_terms(made, items, now, reference)]).sum() / 2
        total.backward()

        twice = ToyPolicy()
        captured: list[torch.Tensor] = []
        stepper = PolicyStep(twice, settings)  # type: ignore[arg-type]
        original = stepper.optimizer.step

        def keep(
            *arguments: object,
            captured: list[torch.Tensor] = captured,
            twice: ToyPolicy = twice,
            original: object = original,
            **options: object,
        ) -> None:
            captured.extend(parameter.grad.clone() for parameter in twice.parameters() if parameter.grad is not None)
            original()  # type: ignore[operator]

        stepper.optimizer.step = keep  # type: ignore[method-assign]
        stepper.step(items)
        for found, expected in zip(captured, [p.grad for p in direct.parameters()], strict=True):
            assert expected is not None
            torch.testing.assert_close(found, expected, rtol=1e-9, atol=1e-12)


def test_a_dpo_step_prefers_the_chosen_and_kto_its_desirable() -> None:
    chosen, rejected = sampled([1, 2, 3, 4]), sampled([1, 5, 6, 7])
    for preset, items in (
        ("dpo", [Pair((chosen,), (rejected,))]),
        ("simpo", [Pair((chosen,), (rejected,))]),
        ("kto", [Labelled((chosen,), True), Labelled((rejected,), False)]),
    ):
        policy = ToyPolicy()
        before = [float(policy.logprobs(each.tokens, positions(each)).sum()) for each in (chosen, rejected)]
        stepping = PolicyStep(policy, StepSettings(objective=preset, learning_rate=0.05, max_kl=None))  # type: ignore[arg-type]
        metrics = [stepping.step(items, seed=seed) for seed in range(3)]
        after = [float(policy.logprobs(each.tokens, positions(each)).sum()) for each in (chosen, rejected)]
        assert after[0] - after[1] > before[0] - before[1], preset
        assert metrics[0]["items"] == len(items) and metrics[0]["segments"] == 2.0
        assert metrics[-1]["preference_accuracy"] == 1.0 and metrics[-1]["loss"] < metrics[0]["loss"]
        assert ("chosen_log_ratio" in metrics[0]) == (preset != "kto")


def test_a_preference_step_stops_at_max_kl_and_a_weighted_segment_is_refused() -> None:
    chosen, rejected = sampled([1, 2, 3, 4]), sampled([1, 5, 6, 7])
    policy = ToyPolicy()
    settings = StepSettings(objective="dpo", learning_rate=0.5, tokens_per_step=3, max_kl=1e-4)
    metrics = PolicyStep(policy, settings).step([Pair((chosen,), (rejected,))] * 6)  # type: ignore[arg-type]
    assert metrics["stopped_at_max_kl"] == 1.0 and 0 < metrics["optimizer_steps"] < 6
    with pytest.raises(ValueError, match="pairs or labelled examples"):
        PolicyStep(ToyPolicy(), StepSettings(objective="dpo")).step([Weighted(chosen, 1.0)])  # type: ignore[arg-type]
    weighed = Segment(chosen.tokens, chosen.spans, [-1.0] * 3)
    with pytest.raises(ValueError, match="weighted segments"):
        PolicyStep(ToyPolicy(), StepSettings()).step([Pair((weighed,), (weighed,))])  # type: ignore[arg-type]


def test_a_policy_gradient_with_a_kl_and_an_entropy_bonus_reads_the_reference_and_the_entropy() -> None:
    policy = ToyPolicy()
    segment = Segment([1, 2, 3, 4], [Span(1, 4, 0)], policy.logprobs([1, 2, 3, 4], [1, 2, 3]).tolist())
    settings = StepSettings(
        objective={"preset": "grpo", "entropy": {"coefficient": 0.01}}, learning_rate=0.05, max_kl=None
    )
    metrics = PolicyStep(policy, settings).step([Weighted(segment, 1.0)] * 2)  # type: ignore[arg-type]
    assert metrics["kl_penalty"] == pytest.approx(0.0, abs=1e-12)  # (the first minibatch: where the reference is)
    assert metrics["entropy"] > 0


def test_turns_without_behaviour_logprobs_are_trained_on_without_an_importance_correction_and_refused_with_one() -> (
    None
):
    policy = ToyPolicy()
    segment = sampled([1, 2, 3, 4])
    reinforce = StepSettings(objective="reinforce", learning_rate=0.05)
    metrics = PolicyStep(policy, reinforce).step([Weighted(segment, 1.0)])  # type: ignore[arg-type]
    assert metrics["optimizer_steps"] == 1.0 and metrics["mean_mismatch"] == 0.0  # (no finite behaviour: none)
    with pytest.raises(ValueError, match="no behavior logprob"):
        PolicyStep(policy, StepSettings()).step([Weighted(segment, 1.0)])  # type: ignore[arg-type]
