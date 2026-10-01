# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Group-relative updates: advantages, skipped groups, and the direction of an update (on a toy policy)."""

from collections.abc import Sequence

import torch
from torch import nn

from rollout.training.grpo import GroupRelativeTrainer, TrainingSequence, group_advantages


def test_advantages_are_centered_but_not_scaled() -> None:
    assert group_advantages([0, 0, 3, 1]) == [-1.0, -1.0, 2.0, 0.0]
    assert group_advantages([2, 2, 2, 2]) is None  # no signal: skipped
    assert group_advantages([5]) is None


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


def sequence(policy: ToyPolicy, tokens: list[int], advantage: float) -> TrainingSequence:
    mask = [False, *([True] * (len(tokens) - 1))]
    with torch.no_grad():
        behavior = policy.logprobs(tokens, range(1, len(tokens))).tolist()
    return TrainingSequence(tokens, mask, [float("nan"), *behavior], advantage)


def test_a_positive_advantage_makes_its_tokens_likelier_and_a_negative_one_rarer() -> None:
    policy = ToyPolicy()
    good, bad = [1, 2, 3, 4], [1, 5, 6, 7]
    before = {name: float(policy.logprobs(t, range(1, 4)).sum()) for name, t in (("good", good), ("bad", bad))}
    trainer = GroupRelativeTrainer(policy, learning_rate=0.05, tokens_per_step=100)  # type: ignore[arg-type]
    for step in range(3):
        trainer.step([sequence(policy, good, 1.0), sequence(policy, bad, -1.0)], seed=step)
    after = {name: float(policy.logprobs(t, range(1, 4)).sum()) for name, t in (("good", good), ("bad", bad))}
    assert after["good"] > before["good"] and after["bad"] < before["bad"]


def test_forced_tokens_are_never_trained_on() -> None:
    policy = ToyPolicy()
    tokens = [1, 2, 3]
    forced = TrainingSequence(tokens, [False, False, False], [float("nan")] * 3, 1.0)
    trainer = GroupRelativeTrainer(policy, learning_rate=0.05)  # type: ignore[arg-type]
    weights = [parameter.detach().clone() for parameter in policy.parameters()]
    metrics = trainer.step([forced])
    assert metrics["tokens"] == 0
    assert all(torch.equal(a, b) for a, b in zip(weights, policy.parameters(), strict=True))


def test_sequences_too_long_for_the_gpu_are_left_out_and_counted() -> None:
    policy = ToyPolicy()
    trainer = GroupRelativeTrainer(policy, learning_rate=0.05, max_sequence_tokens=4)  # type: ignore[arg-type]
    short = sequence(policy, [1, 2, 3], 1.0)
    long = sequence(policy, [1, 2, 3, 4, 5], -1.0)
    metrics = trainer.step([short, long])
    assert metrics["sequences"] == 1 and metrics["sequences_too_long"] == 1
    assert metrics["longest_sequence_tokens"] == 3
    assert "approx_kl" in metrics and metrics["gradient_norm"] >= 0
