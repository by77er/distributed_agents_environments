# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave parts of autograd untyped.)
"""Group-relative policy optimization for recorded turns.

- **Advantages** (Dr. GRPO): an episode's reward minus its group's mean, without dividing by the group's standard
  deviation (which favors groups that are nearly solved or nearly hopeless). Every sampled turn of the episode gets
  it: in a swarm, every agent's turns, so the swarm is rewarded equally.
- **Dynamic sampling** (DAPO): a group whose rewards are all equal carries no signal and is skipped.
- **Update**: PPO's clipped objective against the behavior policy, the logprobs the engine recorded while sampling
  (as asynchronous RL does): one pass both corrects the engine/trainer mismatch and bounds each update. The clip is
  asymmetric (DAPO's clip-higher: 1 - 0.2 to 1 + 0.28) and the loss is a token-level mean over the whole batch.
  No KL penalty. Tokens the recorder forced (closing an over-budget thought) are never trained on.
"""

import random
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn


class TrainablePolicy(Protocol):
    """What the trainer needs of a policy (`rollout.training.policy.Policy` is one)."""

    model: nn.Module

    def parameters(self) -> list[nn.Parameter]: ...

    def logprobs(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor: ...


@dataclass(frozen=True)
class TrainingSequence:
    tokens: list[int]
    loss_mask: list[bool]
    """True for tokens the policy sampled (the prompt and forced tokens are False)."""
    behavior_logprobs: list[float]
    """Per token (NaN where `loss_mask` is False)."""
    advantage: float


def group_advantages(rewards: Sequence[float]) -> list[float] | None:
    """Each reward minus the group's mean; None when all are equal (no signal)."""
    if len(rewards) < 2 or max(rewards) == min(rewards):
        return None
    mean = statistics.fmean(rewards)
    return [reward - mean for reward in rewards]


@dataclass
class GroupRelativeTrainer:
    policy: TrainablePolicy
    learning_rate: float = 2e-5
    clip_low: float = 0.2
    clip_high: float = 0.28
    tokens_per_step: int = 16_384
    """Sampled tokens per optimizer step (gradients accumulate over sequences until then)."""
    max_gradient_norm: float = 1.0

    def __post_init__(self) -> None:
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.learning_rate, weight_decay=0.0)

    def to(self, device: str) -> None:
        """Move the policy and the optimizer's state (between steps, to share the GPU with an engine)."""
        self.policy.model.to(device)
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        if device != "cpu":
            torch.cuda.synchronize()
        else:
            torch.cuda.empty_cache()

    def step(self, sequences: Sequence[TrainingSequence], *, seed: int = 0) -> dict[str, float]:
        """One pass over the sequences, in shuffled minibatches of about `tokens_per_step` sampled tokens."""
        started = time.monotonic()
        order = list(sequences)
        random.Random(seed).shuffle(order)
        batches: list[list[TrainingSequence]] = [[]]
        counted = 0
        for sequence in order:
            if counted >= self.tokens_per_step:
                batches.append([])
                counted = 0
            batches[-1].append(sequence)
            counted += sum(sequence.loss_mask)
        self.policy.model.train()
        totals = {"loss": 0.0, "clipped": 0.0, "tokens": 0.0, "ratio": 0.0, "mismatch": 0.0, "kl": 0.0}
        gradient_norms: list[float] = []
        for batch in batches:
            batch_tokens = sum(sum(sequence.loss_mask) for sequence in batch)
            if batch_tokens == 0:
                continue
            for sequence in batch:
                positions = [index for index, sampled in enumerate(sequence.loss_mask) if sampled and index > 0]
                if not positions:
                    continue
                logprobs = self.policy.logprobs(sequence.tokens, positions)
                behavior = torch.tensor([sequence.behavior_logprobs[i] for i in positions], device=logprobs.device)
                ratio = torch.exp(logprobs - behavior)
                advantage = torch.full_like(ratio, sequence.advantage)
                clipped = torch.clamp(ratio, 1 - self.clip_low, 1 + self.clip_high)
                per_token = -torch.minimum(ratio * advantage, clipped * advantage)
                (per_token.sum() / batch_tokens).backward()  # a token-level mean over the minibatch
                with torch.no_grad():
                    totals["loss"] += float(per_token.sum())
                    totals["clipped"] += float((ratio != clipped).sum())
                    totals["ratio"] += float(ratio.sum())
                    totals["mismatch"] += float((logprobs - behavior).abs().sum())
                    totals["kl"] += float((behavior - logprobs).sum())
                    totals["tokens"] += len(positions)
            norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_gradient_norm)
            gradient_norms.append(float(norm))
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        tokens = max(totals["tokens"], 1.0)
        return {
            "loss": totals["loss"] / tokens,
            "clip_fraction": totals["clipped"] / tokens,
            "mean_ratio": totals["ratio"] / tokens,
            "mean_mismatch": totals["mismatch"] / tokens,
            # KL(behavior || policy) estimated on the sampled tokens, as each minibatch saw the policy: the first
            # minibatch measures only the engine's and the trainer's numerical difference, later ones the drift.
            "approx_kl": totals["kl"] / tokens,
            "gradient_norm": sum(gradient_norms) / max(len(gradient_norms), 1),  # before clipping, mean over steps
            "tokens": totals["tokens"],
            "sequences": float(len(sequences)),
            "optimizer_steps": float(len(batches)),
            "seconds": time.monotonic() - started,
        }
