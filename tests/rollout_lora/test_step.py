# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""The clipped policy-gradient step, on a toy policy: the direction of an update, what is never trained on, and
when a pass stops."""

from collections.abc import Sequence

import pytest
import torch
from torch import nn

from rollout_lora.settings import LoraSettings
from rollout_lora.step import ClippedPolicyGradient, minibatches
from rollout_train import Weighted
from rollout_train.recorder import Epoch, Span


class ToyPolicy:
    """A bigram model over a tiny vocabulary: enough to check an update's direction."""

    def __init__(self) -> None:
        torch.manual_seed(0)
        self.model = nn.Sequential(nn.Embedding(8, 16), nn.Linear(16, 8))

    def parameters(self) -> list[nn.Parameter]:
        return list(self.model.parameters())

    def logprobs(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        ids = torch.tensor(list(tokens))
        logits = self.model(ids[[p - 1 for p in positions]])
        return torch.log_softmax(logits, -1).gather(-1, ids[list(positions)].unsqueeze(-1)).squeeze(-1)


def sequence(policy: ToyPolicy, tokens: list[int], advantage: float) -> Weighted:
    """Every token after the first was sampled, at the logprobs the policy gives them now."""
    with torch.no_grad():
        behavior = policy.logprobs(tokens, range(1, len(tokens))).tolist()
    return Weighted(Epoch(tokens, [Span(1, len(tokens), 0)], behavior), advantage)


def test_a_positive_advantage_makes_its_tokens_likelier_and_a_negative_one_rarer() -> None:
    policy = ToyPolicy()
    good, bad = [1, 2, 3, 4], [1, 5, 6, 7]
    before = {name: float(policy.logprobs(t, range(1, 4)).sum()) for name, t in (("good", good), ("bad", bad))}
    trainer = ClippedPolicyGradient(policy, LoraSettings(learning_rate=0.05, tokens_per_step=100))  # type: ignore[arg-type]
    for step in range(3):
        trainer.step([sequence(policy, good, 1.0), sequence(policy, bad, -1.0)], seed=step)
    after = {name: float(policy.logprobs(t, range(1, 4)).sum()) for name, t in (("good", good), ("bad", bad))}
    assert after["good"] > before["good"] and after["bad"] < before["bad"]


def test_forced_tokens_are_never_trained_on() -> None:
    policy = ToyPolicy()
    tokens = [1, 2, 3]
    forced = Weighted(Epoch(tokens, [], []), 1.0)
    trainer = ClippedPolicyGradient(policy, LoraSettings(learning_rate=0.05))  # type: ignore[arg-type]
    weights = [parameter.detach().clone() for parameter in policy.parameters()]
    metrics = trainer.step([forced])
    assert metrics["tokens"] == 0
    assert all(torch.equal(a, b) for a, b in zip(weights, policy.parameters(), strict=True))


def test_sequences_too_long_for_the_gpu_are_left_out_and_counted() -> None:
    policy = ToyPolicy()
    trainer = ClippedPolicyGradient(policy, LoraSettings(learning_rate=0.05, sequence_tokens=4))  # type: ignore[arg-type]
    short = sequence(policy, [1, 2, 3], 1.0)
    long = sequence(policy, [1, 2, 3, 4, 5], -1.0)
    metrics = trainer.step([short, long])
    assert metrics["sequences"] == 1 and metrics["sequences_too_long"] == 1
    assert metrics["longest_sequence_tokens"] == 3
    assert metrics["sequences_given"] == 2 and metrics["gradient_norm"] >= 0


def test_a_small_last_minibatch_joins_the_one_before() -> None:
    policy = ToyPolicy()
    four = [sequence(policy, [1, 2, 3, 4, 5], 1.0) for _ in range(5)]  # four sampled tokens each
    assert [len(batch) for batch in minibatches(four, 10)] == [3, 2]
    assert [len(batch) for batch in minibatches(four[:4], 10)] == [4]  # not [3, 1]: Adam would step as far for one
    assert [len(batch) for batch in minibatches(four[:1], 10)] == [1] and minibatches([], 10) == []


def test_the_pass_stops_once_the_policy_has_moved_as_far_as_allowed() -> None:
    def run(max_kl: float | None) -> tuple[dict[str, float], ToyPolicy]:
        torch.manual_seed(0)
        policy = ToyPolicy()
        good, bad = sequence(policy, [1, 2, 3, 4], 1.0), sequence(policy, [1, 5, 6, 7], -1.0)
        trainer = ClippedPolicyGradient(policy, LoraSettings(learning_rate=0.5, tokens_per_step=6, max_kl=max_kl))  # type: ignore[arg-type]
        return trainer.step([good, bad] * 10), policy

    free, _ = run(None)
    assert free["optimizer_steps"] == 10 and not free["stopped_at_max_kl"] and free["kl_moved"] > 0.05
    assert abs(free["kl_floor"]) < 1e-6  # the first minibatch finds the policy where it was sampled
    held, _ = run(0.05)
    assert held["stopped_at_max_kl"] and held["optimizer_steps"] < 10 and held["sequences"] < 20
    assert held["kl_moved"] <= 0.05  # the minibatch that found it further was not stepped on


def test_a_sampled_token_without_a_logprob_is_refused() -> None:
    policy = ToyPolicy()
    broken = Weighted(Epoch([1, 2, 3], [Span(1, 3, 0)], [-0.5, float("nan")]), 1.0)
    weights = [parameter.detach().clone() for parameter in policy.parameters()]
    with pytest.raises(ValueError, match="no behavior logprob"):
        ClippedPolicyGradient(policy).step([broken])  # type: ignore[arg-type]
    assert all(torch.equal(a, b) for a, b in zip(weights, policy.parameters(), strict=True))
