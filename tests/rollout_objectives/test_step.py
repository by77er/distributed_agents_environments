# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""The policy step, on a toy policy: the direction of an update, what is never trained on, when a pass stops, and
how its objectives weigh tokens."""

import copy
import itertools
import math
from collections.abc import Sequence

import pytest
import torch
from torch import nn

from rollout_objectives.settings import StepSettings
from rollout_objectives.step import GRADIENT_WORK, PolicyStep, minibatches
from rollout_objectives.terms import terms
from rollout_train import Weighted
from rollout_train.recorder import Segment, Span, TeacherScores
from rollout_train.trainer import MINIBATCH, START, Distilled, Progress


class ToyPolicy:
    """A bigram model over a tiny vocabulary: enough to check an update's direction."""

    def __init__(self) -> None:
        torch.manual_seed(0)
        self.model = nn.Sequential(nn.Embedding(8, 16), nn.Linear(16, 8))

    def parameters(self) -> list[nn.Parameter]:
        return list(self.model.parameters())

    def distribution(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        """The logprobs of every token of the vocabulary at each position."""
        ids = torch.tensor(list(tokens))
        return torch.log_softmax(self.model(ids[[p - 1 for p in positions]]), -1)

    def logprobs(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        ids = torch.tensor(list(tokens))
        return self.distribution(tokens, positions).gather(-1, ids[list(positions)].unsqueeze(-1)).squeeze(-1)


def segment(policy: ToyPolicy, tokens: list[int], advantage: float) -> Weighted:
    """Every token after the first was sampled, at the logprobs the policy gives them now."""
    with torch.no_grad():
        behavior = policy.logprobs(tokens, range(1, len(tokens))).tolist()
    return Weighted(Segment(tokens, [Span(1, len(tokens), 0)], behavior), advantage)


def test_a_positive_advantage_makes_its_tokens_likelier_and_a_negative_one_rarer() -> None:
    policy = ToyPolicy()
    good, bad = [1, 2, 3, 4], [1, 5, 6, 7]
    before = {name: float(policy.logprobs(t, range(1, 4)).sum()) for name, t in (("good", good), ("bad", bad))}
    trainer = PolicyStep(policy, StepSettings(learning_rate=0.05, tokens_per_step=100))  # type: ignore[arg-type]
    for step in range(3):
        trainer.step([segment(policy, good, 1.0), segment(policy, bad, -1.0)], seed=step)
    after = {name: float(policy.logprobs(t, range(1, 4)).sum()) for name, t in (("good", good), ("bad", bad))}
    assert after["good"] > before["good"] and after["bad"] < before["bad"]


def test_a_pass_keeps_what_each_minibatch_did() -> None:
    policy = ToyPolicy()
    settings = StepSettings(learning_rate=0.05, tokens_per_step=3, max_kl=None)
    trainer = PolicyStep(policy, settings)  # type: ignore[arg-type]
    metrics = trainer.step([segment(policy, [1, 2, 3, 4], 1.0), segment(policy, [1, 5, 6, 7], -1.0)])
    assert len(trainer.minibatches) == metrics["optimizer_steps"] == 2
    assert [each["tokens"] for each in trainer.minibatches] == [3.0, 3.0]
    assert set(trainer.minibatches[0]) == {"segments", "tokens", "loss", "clip_fraction", "kl", "gradient_norm",
                                          "learning_rate"}  # fmt: skip


def test_a_fresh_optimizers_rate_rises_over_its_warmup_and_one_that_goes_on_is_not_warmed_up() -> None:
    settings = StepSettings(learning_rate=1e-5, warmup_updates=4)
    assert [settings.rate(update, fresh=True) for update in range(6)] == pytest.approx(
        [2.5e-6, 5e-6, 7.5e-6, 1e-5, 1e-5, 1e-5]
    )
    assert [settings.rate(update, fresh=False) for update in range(3)] == [1e-5] * 3
    assert StepSettings(learning_rate=1e-5).rate(0, fresh=True) == 1e-5  # (no warmup unless asked)
    with pytest.raises(ValueError):
        StepSettings(passes=0)


def test_passes_make_more_updates_each_at_its_warmed_up_rate() -> None:
    policy = ToyPolicy()
    settings = StepSettings(learning_rate=0.04, tokens_per_step=100, max_kl=None, passes=4, warmup_updates=2,
                            objective="likelihood")  # fmt: skip
    trainer = PolicyStep(policy, settings)  # type: ignore[arg-type]
    metrics = trainer.step([segment(policy, [1, 2, 3, 4], 1.0), segment(policy, [1, 5, 6, 7], 1.0)])
    assert metrics["optimizer_steps"] == 4 and metrics["passes"] == 4  # (a pass of 6 tokens is one update)
    assert [each["learning_rate"] for each in trainer.minibatches] == pytest.approx([0.02, 0.04, 0.04, 0.04])
    assert [each["segments"] for each in trainer.minibatches] == [2.0] * 4
    goes_on = PolicyStep(policy, settings, fresh=False)  # type: ignore[arg-type]
    goes_on.step([segment(policy, [1, 2, 3, 4], 1.0)])
    assert {each["learning_rate"] for each in goes_on.minibatches} == {0.04}


def test_forced_tokens_are_never_trained_on() -> None:
    policy = ToyPolicy()
    tokens = [1, 2, 3]
    forced = Weighted(Segment(tokens, [], []), 1.0)
    trainer = PolicyStep(policy, StepSettings(learning_rate=0.05))  # type: ignore[arg-type]
    weights = [parameter.detach().clone() for parameter in policy.parameters()]
    metrics = trainer.step([forced])
    assert metrics["tokens"] == 0
    assert all(torch.equal(a, b) for a, b in zip(weights, policy.parameters(), strict=True))


def test_segments_too_long_for_the_gpu_are_left_out_and_counted() -> None:
    policy = ToyPolicy()
    trainer = PolicyStep(policy, StepSettings(learning_rate=0.05, segment_tokens=4))  # type: ignore[arg-type]
    short = segment(policy, [1, 2, 3], 1.0)
    long = segment(policy, [1, 2, 3, 4, 5], -1.0)
    metrics = trainer.step([short, long])
    assert metrics["segments"] == 1 and metrics["segments_too_long"] == 1
    assert metrics["longest_segment_tokens"] == 3
    assert metrics["segments_given"] == 2 and metrics["gradient_norm"] >= 0


def test_a_small_last_minibatch_joins_the_one_before() -> None:
    policy = ToyPolicy()
    four = [segment(policy, [1, 2, 3, 4, 5], 1.0) for _ in range(5)]  # four sampled tokens each
    assert [len(batch) for batch in minibatches(four, 10)] == [3, 2]
    assert [len(batch) for batch in minibatches(four[:4], 10)] == [4]  # not [3, 1]: Adam would step as far for one
    assert [len(batch) for batch in minibatches(four[:1], 10)] == [1] and minibatches([], 10) == []


def test_a_step_says_how_far_it_has_got_after_each_pack_and_each_minibatch() -> None:
    policy = ToyPolicy()
    told: list[Progress] = []
    settings = StepSettings(learning_rate=0.05, tokens_per_step=3, max_kl=5.0, passes=2)
    trainer = PolicyStep(policy, settings, progress=told.append)  # type: ignore[arg-type]
    given = [segment(policy, tokens, advantage) for tokens, advantage in (([1, 2, 3, 4], 1.0), ([1, 5, 6, 7], -1.0),
                                                                         ([2, 3, 4, 5], 0.5))]  # fmt: skip
    metrics = trainer.step(given)
    # Three minibatches of a segment each, twice over: the start computes the second's and the third's (the first
    # minibatch computes its own), then each minibatch its segment with a gradient.
    starting = [each for each in told if each.phase == START]
    assert [each.packs for each in starting] == [1, 2] and all(each.minibatches == 6 for each in starting)
    assert starting[-1].fraction == pytest.approx(8 / (8 + 6 * 4 * GRADIENT_WORK))  # (a token with a gradient: 3)
    assert {each.minibatch for each in told if each.phase == MINIBATCH} == set(range(1, 7))
    last = told[-1]
    assert last.packs == last.packs_total == metrics["packs"] == 8 and last.fraction == 1.0 and last.eta_seconds == 0
    assert all(one.fraction <= two.fraction for one, two in itertools.pairwise(told))
    assert last.loss is not None and last.clip_fraction is not None and last.max_kl == 5.0
    assert last.kl == pytest.approx(metrics["kl_moved"]) and last.tokens_per_second > 0
    assert last.gpu_gib is None and last.gpu_utilization == ()  # (on the CPU)
    assert len([each for each in told if each.loss is not None and each.packs == each.packs_total]) >= 1

    told.clear()
    PolicyStep(policy, StepSettings(objective="sft", tokens_per_step=3), progress=told.append).step(given)  # type: ignore[arg-type]
    assert told[0].phase == MINIBATCH and told[-1].fraction == 1.0  # (a likelihood reads no start, and never stops)
    assert told[-1].kl is None and told[-1].max_kl is None


def test_the_pass_stops_once_the_policy_has_moved_as_far_as_allowed() -> None:
    def run(max_kl: float | None) -> tuple[dict[str, float], ToyPolicy]:
        torch.manual_seed(0)
        policy = ToyPolicy()
        good, bad = segment(policy, [1, 2, 3, 4], 1.0), segment(policy, [1, 5, 6, 7], -1.0)
        trainer = PolicyStep(policy, StepSettings(learning_rate=0.5, tokens_per_step=6, max_kl=max_kl))  # type: ignore[arg-type]
        return trainer.step([good, bad] * 10), policy

    free, _ = run(None)
    assert free["optimizer_steps"] == 10 and not free["stopped_at_max_kl"] and free["kl_moved"] > 0.05
    assert abs(free["kl_floor"]) < 1e-6  # the first minibatch finds the policy where it was sampled
    held, _ = run(0.05)
    assert held["stopped_at_max_kl"] and held["optimizer_steps"] < 10 and held["segments"] < 20
    assert held["kl_moved"] <= 0.05  # the minibatch that found it further was not stepped on


SAME_WAY = [[1, 2, 3, 4], [1, 5, 6, 7], [2, 3, 4, 5], [6, 1, 2, 0]]


def pushed_the_same_way(policy: ToyPolicy, preset: str) -> list[Weighted | Distilled]:
    """Items whose every token's advantage is positive: REINFORCE's of a reward of 1 with no baseline, or a
    policy-gradient distillation's from a teacher surer of each sampled token than the student."""
    made: list[Weighted | Distilled] = []
    for tokens in SAME_WAY:
        sampled = segment(policy, tokens, 1.0)
        if preset == "reinforce":
            made.append(sampled)
        else:
            made.append(Distilled(sampled.segment, TeacherScores("teacher", [-0.05] * (len(tokens) - 1))))
    return made * 6


def moved(start: ToyPolicy, now: ToyPolicy) -> tuple[float, float]:
    """Over the sampled positions: the mean of `old - now` of the sampled tokens (the k1 estimate of KL(start || now)),
    and the mean KL(start || now) over the whole vocabulary."""
    k1, exact, count = 0.0, 0.0, 0
    with torch.no_grad():
        for tokens in SAME_WAY:
            at = range(1, len(tokens))
            k1 += float((start.logprobs(tokens, at) - now.logprobs(tokens, at)).sum())
            before, after = start.distribution(tokens, at), now.distribution(tokens, at)
            exact += float((before.exp() * (before - after)).sum())
            count += len(at)
    return k1 / count, exact / count


@pytest.mark.parametrize("preset", ["reinforce", "on_policy_distillation"])
def test_the_stop_reads_how_far_the_policy_moved_when_every_advantage_pushes_the_same_way(preset: str) -> None:
    """Each update makes every sampled token likelier, so the k1 estimate of how far the policy moved falls below 0
    while the policy moves away; the stop reads the k3 estimate, which is never below 0, and stops the pass."""

    def run(max_kl: float | None) -> tuple[dict[str, float], ToyPolicy, ToyPolicy]:
        policy = ToyPolicy()
        start = copy.deepcopy(policy)
        settings = StepSettings(objective=preset, learning_rate=0.05, tokens_per_step=12, max_kl=max_kl)
        metrics = PolicyStep(policy, settings).step(pushed_the_same_way(policy, preset))  # type: ignore[arg-type]
        return metrics, start, policy

    free, start, after = run(None)
    k1, exact = moved(start, after)
    assert free["optimizer_steps"] == 6 and k1 < -1.0 and exact > 1.0  # (k1: -1.7; the KL over the vocabulary: 1.8)
    held, start, stopped = run(0.05)
    assert held["stopped_at_max_kl"] and held["optimizer_steps"] == 1
    k1, exact = moved(start, stopped)  # where the second minibatch found it: past 0.05, which k1 does not say
    assert k1 < 0 and exact > 0.05  # (k1: -0.62; the KL over the vocabulary: 0.07; k3 on the sampled tokens: 0.33)


def test_a_sampled_token_without_a_logprob_is_refused() -> None:
    policy = ToyPolicy()
    broken = Weighted(Segment([1, 2, 3], [Span(1, 3, 0)], [-0.5, float("nan")]), 1.0)
    weights = [parameter.detach().clone() for parameter in policy.parameters()]
    with pytest.raises(ValueError, match="no behavior logprob"):
        PolicyStep(policy).step([broken])  # type: ignore[arg-type]
    assert all(torch.equal(a, b) for a, b in zip(weights, policy.parameters(), strict=True))


def test_the_likelihood_objective_makes_what_was_sampled_likelier_whatever_it_was_sampled_at() -> None:
    policy = ToyPolicy()
    shown = [1, 2, 3, 4]
    before = float(policy.logprobs(shown, range(1, 4)).sum())
    # Recorded logprobs from another prompt (the guidance taken out): the objective does not read them.
    taught = Weighted(Segment(shown, [Span(1, 4, 0)], [-9.0, -9.0, -9.0]), 1.0)
    settings = StepSettings(learning_rate=0.05, tokens_per_step=100, objective="likelihood")
    trainer = PolicyStep(policy, settings)  # type: ignore[arg-type]
    metrics = [trainer.step([taught], seed=step) for step in range(5)]
    assert float(policy.logprobs(shown, range(1, 4)).sum()) > before
    assert metrics[-1]["loss"] < metrics[0]["loss"] and metrics[0]["clip_fraction"] == 0.0
    with pytest.raises(ValueError, match="objective"):
        StepSettings(objective="something else")


def test_where_a_token_was_sampled_is_weighed_and_how_far_the_step_moves_it_is_clipped() -> None:
    policy = ToyPolicy()
    good = segment(policy, [1, 2, 3, 4], 1.0)
    # The engine gave the same tokens other logprobs (another kernel, an older version): far off for one of them.
    off = Weighted(
        Segment(
            good.segment.tokens,
            good.segment.spans,
            [value + shift for value, shift in zip(good.segment.logprobs, [0.05, -0.05, -3.0], strict=True)],
        ),
        1.0,
    )
    trainer = PolicyStep(policy, StepSettings(learning_rate=0.05, tokens_per_step=100))  # type: ignore[arg-type]
    metrics = trainer.step([off])
    # The step starts at its own logprobs: nothing is clipped for the difference, which is weighed instead (the
    # third token's weight, e^3, truncated at 2).
    assert metrics["clip_fraction"] == 0.0 and metrics["mean_ratio"] == pytest.approx(1.0)
    assert metrics["truncated_fraction"] == pytest.approx(1 / 3)
    assert metrics["mean_weight"] == pytest.approx((math.exp(-0.05) + math.exp(0.05) + 2.0) / 3, rel=1e-4)
    assert metrics["mean_mismatch"] == pytest.approx((0.05 + 0.05 + 3.0) / 3, rel=1e-4)
    assert metrics["kl_floor"] == pytest.approx((0.05 - 0.05 - 3.0) / 3, rel=1e-4)


def test_a_token_ratio_is_clipped_once_the_step_has_moved_it_far_enough() -> None:
    logprobs = torch.tensor([-1.0, -1.0], requires_grad=True)
    old = torch.tensor([-1.0, -1.5])  # the second token's ratio is e^0.5: past 1.28
    found = terms(StepSettings(truncate=None).loss, logprobs, 1.0, old, old.clone())
    found.loss.backward()
    assert found.clipped == 1.0
    assert logprobs.grad is not None and logprobs.grad.tolist() == pytest.approx([-1.0, 0.0])  # the clipped one: none


def test_a_segment_ratio_is_the_geometric_mean_of_its_tokens_and_its_gradient_is_spread_over_them() -> None:
    logprobs = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
    old = torch.tensor([-1.0001, -2.0, -2.9999])  # the segment's log ratio: (0.0001 + 0 - 0.0001) / 3 = 0
    behavior = torch.tensor([-1.3, -2.0, -2.9])
    objective = StepSettings(ratio="segment", segment_clip_low=3e-4, segment_clip_high=4e-4, truncate=None).loss
    found = terms(objective, logprobs, 2.0, old, behavior)
    found.loss.backward()
    weight = math.exp(float((old - behavior).mean()))  # one for the segment, likewise
    assert found.clipped == 0.0 and found.ratio == pytest.approx(3.0)
    assert logprobs.grad is not None and logprobs.grad.tolist() == pytest.approx([-weight * 2.0 / 3] * 3)
    far = terms(objective, logprobs.detach().requires_grad_(True), 2.0, old - 0.01, behavior)  # ratio e^0.01: clipped
    assert far.clipped == 3.0
