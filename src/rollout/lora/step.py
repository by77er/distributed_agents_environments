# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave parts of autograd untyped.)
"""The clipped policy-gradient step over weighted sequences.

PPO's clipped objective against the behavior policy, the logprobs the engine recorded while sampling (as asynchronous
RL does): one pass both corrects the engine/trainer mismatch and bounds each update. The clip is asymmetric (DAPO's
clip-higher) and the loss is a token-level mean over each minibatch. No KL penalty; the pass stops early if the
policy has moved further from the behavior policy than `max_kl`. Only tokens the policy sampled are trained on.
The numbers are `LoraSettings`'; which sequences, and with what advantages, is the algorithm's business.
"""

import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import nn

from rollout.lora.settings import LoraSettings
from rollout.training.trainer import Weighted


class TrainablePolicy(Protocol):
    """What the step needs of a policy (`rollout.lora.policy.Policy` is one)."""

    model: nn.Module

    def parameters(self) -> list[nn.Parameter]: ...

    def logprobs(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor: ...


@dataclass
class ClippedPolicyGradient:
    policy: TrainablePolicy
    settings: LoraSettings = field(default_factory=LoraSettings)

    def __post_init__(self) -> None:
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.settings.learning_rate, weight_decay=0.0)

    def step(self, sequences: Sequence[Weighted], *, seed: int = 0) -> dict[str, float]:
        """One pass over the sequences, in shuffled minibatches of about `tokens_per_step` sampled tokens."""
        started = time.monotonic()
        settings = self.settings
        longest = settings.sequence_tokens
        order = [sequence for sequence in sequences if longest is None or len(sequence.epoch.tokens) <= longest]
        too_long = len(sequences) - len(order)
        random.Random(seed).shuffle(order)
        batches = minibatches(order, settings.tokens_per_step)
        self.policy.model.train()
        totals = {"loss": 0.0, "clipped": 0.0, "tokens": 0.0, "ratio": 0.0, "mismatch": 0.0, "sequences": 0.0}
        gradient_norms: list[float] = []
        divergences: list[float] = []  # of each minibatch that was stepped on, as it found the policy
        out_of_memory = 0
        stopped = False
        for batch in batches:
            batch_tokens = sum(sequence.epoch.sampled for sequence in batch)
            sums = dict.fromkeys(totals, 0.0)
            divergence = 0.0
            try:
                for sequence in batch:
                    epoch = sequence.epoch
                    positions = [position for span in epoch.spans for position in range(span.start, span.end)]
                    if not positions:
                        continue
                    logprobs = self.policy.logprobs(epoch.tokens, positions)
                    behavior = torch.tensor(epoch.logprobs, device=logprobs.device)
                    if not bool(torch.isfinite(behavior).all()):  # (one NaN would make every weight NaN)
                        raise ValueError("a sampled token has no behavior logprob")
                    ratio = torch.exp(logprobs - behavior)
                    advantage = torch.full_like(ratio, sequence.advantage)
                    clipped = torch.clamp(ratio, 1 - settings.clip_low, 1 + settings.clip_high)
                    per_token = -torch.minimum(ratio * advantage, clipped * advantage)
                    (per_token.sum() / batch_tokens).backward()  # a token-level mean over the minibatch
                    with torch.no_grad():
                        sums["loss"] += float(per_token.sum())
                        sums["clipped"] += float((ratio != clipped).sum())
                        sums["ratio"] += float(ratio.sum())
                        sums["mismatch"] += float((logprobs - behavior).abs().sum())
                        sums["tokens"] += len(positions)
                        sums["sequences"] += 1
                        divergence += float((behavior - logprobs).sum())
            except torch.OutOfMemoryError:  # a gradient with a sequence missing is not this minibatch's: drop it
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                out_of_memory += 1
                continue
            if sums["tokens"] == 0:
                continue
            divergence /= sums["tokens"]
            if settings.max_kl is not None and divergences and divergence - divergences[0] > settings.max_kl:
                self.optimizer.zero_grad(set_to_none=True)
                stopped = True
                break
            divergences.append(divergence)
            for key, value in sums.items():
                totals[key] += value
            norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), settings.max_gradient_norm)
            gradient_norms.append(float(norm))
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        tokens = max(totals["tokens"], 1.0)
        return {
            "loss": totals["loss"] / tokens,
            "clip_fraction": totals["clipped"] / tokens,
            "mean_ratio": totals["ratio"] / tokens,
            "mean_mismatch": totals["mismatch"] / tokens,
            # KL(behavior || policy) estimated on the sampled tokens, as each minibatch found the policy before its
            # step. The first minibatch's is the floor (numerical difference between engine and trainer, and how
            # stale the turns are); the last one's, less the floor, is how far this update moved the policy.
            "kl_floor": divergences[0] if divergences else 0.0,
            "kl_moved": divergences[-1] - divergences[0] if divergences else 0.0,
            "gradient_norm": sum(gradient_norms) / max(len(gradient_norms), 1),  # before clipping, mean over steps
            "tokens": totals["tokens"],
            "sequences": totals["sequences"],
            "sequences_given": float(len(sequences)),
            "sequences_too_long": float(too_long),
            "longest_sequence_tokens": float(max((len(sequence.epoch.tokens) for sequence in order), default=0)),
            "optimizer_steps": float(len(gradient_norms)),
            "stopped_at_max_kl": float(stopped),
            "minibatches_out_of_memory": float(out_of_memory),
            "seconds": time.monotonic() - started,
        }


def minibatches(sequences: Sequence[Weighted], tokens_per_step: int) -> list[list[Weighted]]:
    """The sequences in order, cut where a minibatch has reached `tokens_per_step` sampled tokens. A last minibatch
    of less than half that joins the one before: Adam's step is as large for a handful of tokens as for a full
    minibatch."""
    batches: list[list[Weighted]] = [[]]
    counted = 0
    for sequence in sequences:
        if counted >= tokens_per_step:
            batches.append([])
            counted = 0
        batches[-1].append(sequence)
        counted += sequence.epoch.sampled
    if len(batches) > 1 and counted < tokens_per_step / 2:
        batches[-2].extend(batches.pop())
    return [batch for batch in batches if batch]
