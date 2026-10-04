# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave parts of autograd untyped.)
"""A policy step over weighted segments: the logprobs the step starts from, then one pass of updates.

First every sampled token's logprob is computed on the weights the step starts from, without a gradient (`old`).
Then the segments are taken in shuffled minibatches of about `tokens_per_step` sampled tokens, an optimizer step
each, under the objective the settings name (`rollout_lora.objectives`): a ratio to `old` that bounds how far the
step moves the policy, and an importance weight `old / behavior` for where each token was sampled (an older checkpoint,
and the engine computing differently from the trainer). No KL penalty; the pass stops early if a minibatch finds the
policy further than `max_kl` from where the step began. Only tokens the policy sampled are trained on. The numbers
are `LoraSettings`'; which segments, and with what advantages, is the algorithm's business.
"""

import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import nn

from rollout_lora.objectives import terms
from rollout_lora.settings import LoraSettings
from rollout_train.trainer import Weighted


class TrainablePolicy(Protocol):
    """What the step needs of a policy (`rollout_lora.policy.Policy` is one)."""

    model: nn.Module

    def parameters(self) -> list[nn.Parameter]: ...

    def logprobs(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor: ...


def sampled(weighted: Weighted) -> list[int]:
    """The positions of the tokens the policy sampled in a segment."""
    return [position for span in weighted.segment.spans for position in range(span.start, span.end)]


@dataclass
class PolicyStep:
    policy: TrainablePolicy
    settings: LoraSettings = field(default_factory=LoraSettings)

    def __post_init__(self) -> None:
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.settings.learning_rate, weight_decay=0.0)
        self.minibatches: list[dict[str, float]] = []
        """What each minibatch of the last pass did, in order (`step` returns their totals)."""

    def step(self, segments: Sequence[Weighted], *, seed: int = 0) -> dict[str, float]:
        """The logprobs the step starts from, then one pass over the segments in shuffled minibatches."""
        started = time.monotonic()
        settings, objective = self.settings, self.settings.loss
        longest = settings.segment_tokens
        order = [
            weighted
            for weighted in segments
            if (longest is None or len(weighted.segment.tokens) <= longest) and sampled(weighted)
        ]
        too_long = sum(1 for weighted in segments if longest is not None and len(weighted.segment.tokens) > longest)
        random.Random(seed).shuffle(order)
        self.policy.model.train()

        # Where the step starts: each sampled token's logprob on these weights, and where it was sampled.
        old: dict[int, torch.Tensor] = {}
        behaviors: dict[int, torch.Tensor] = {}
        start_out_of_memory = 0
        if objective.reads_old:
            with torch.no_grad():
                for weighted in list(order):
                    behavior = torch.tensor(weighted.segment.logprobs)
                    if not bool(torch.isfinite(behavior).all()):  # (one NaN would make every weight NaN)
                        raise ValueError("a sampled token has no behavior logprob")
                    try:
                        found = self.policy.logprobs(weighted.segment.tokens, sampled(weighted))
                    except torch.OutOfMemoryError:  # (left out of the step, and counted)
                        torch.cuda.empty_cache()
                        start_out_of_memory += 1
                        order.remove(weighted)
                        continue
                    old[id(weighted)] = found.detach()
                    behaviors[id(weighted)] = behavior.to(found.device)
        started_pass = time.monotonic()

        totals = dict.fromkeys(("loss", "units", "clipped", "truncated", "tokens", "ratio", "weight", "segments"), 0.0)
        gradient_norms: list[float] = []
        moved: list[float] = []  # how far each minibatch that was stepped on found the policy from the step's start
        out_of_memory = 0
        stopped = False
        self.minibatches = []
        for batch in minibatches(order, settings.tokens_per_step):
            units = sum(objective.units(weighted.segment.sampled) for weighted in batch)
            sums = dict.fromkeys(totals, 0.0)
            distance = 0.0
            try:
                for weighted in batch:
                    logprobs = self.policy.logprobs(weighted.segment.tokens, sampled(weighted))
                    found = terms(
                        objective, logprobs, weighted.advantage, old.get(id(weighted)), behaviors.get(id(weighted))
                    )
                    (found.loss / units).backward()
                    sums["loss"] += float(found.loss.detach())
                    sums["units"] += objective.units(int(found.tokens))
                    sums["clipped"] += found.clipped
                    sums["truncated"] += found.truncated
                    sums["tokens"] += found.tokens
                    sums["ratio"] += found.ratio
                    sums["weight"] += found.weight
                    sums["segments"] += 1
                    distance += found.moved
            except torch.OutOfMemoryError:  # a gradient with a segment missing is not this minibatch's: drop it
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                out_of_memory += 1
                continue
            if sums["tokens"] == 0:
                continue
            distance /= sums["tokens"]
            if objective.reads_old and settings.max_kl is not None and distance > settings.max_kl:
                self.optimizer.zero_grad(set_to_none=True)
                stopped = True
                break
            moved.append(distance)
            for key, value in sums.items():
                totals[key] += value
            norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), settings.max_gradient_norm)
            gradient_norms.append(float(norm))
            self.minibatches.append(
                {
                    "segments": sums["segments"],
                    "tokens": sums["tokens"],
                    "loss": sums["loss"] / units,
                    "clip_fraction": sums["clipped"] / sums["tokens"],
                    "kl": distance,
                    "gradient_norm": float(norm),
                }
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        tokens = max(totals["tokens"], 1.0)
        start_tokens = max(sum(float(each.numel()) for each in old.values()), 1.0)
        return {
            "loss": totals["loss"] / max(totals["units"], 1.0),
            "clip_fraction": totals["clipped"] / tokens,
            "mean_ratio": totals["ratio"] / tokens,
            # Where the tokens were sampled, against where the step starts: the engine's and the trainer's numerical
            # difference, and how stale the turns are. `kl_floor` estimates KL(behavior || start) on the sampled
            # tokens; `mean_mismatch` is the mean absolute difference of their logprobs.
            "kl_floor": sum(float((behaviors[key] - old[key]).sum()) for key in old) / start_tokens,
            "mean_mismatch": sum(float((behaviors[key] - old[key]).abs().sum()) for key in old) / start_tokens,
            "mean_weight": totals["weight"] / tokens,
            "truncated_fraction": totals["truncated"] / tokens,
            # How far the update moved the policy: KL(start || now) on the sampled tokens, as the last minibatch
            # stepped on found it before its step.
            "kl_moved": moved[-1] if moved else 0.0,
            "gradient_norm": sum(gradient_norms) / max(len(gradient_norms), 1),  # before clipping, mean over steps
            "tokens": totals["tokens"],
            "segments": totals["segments"],
            "segments_given": float(len(segments)),
            "segments_too_long": float(too_long),
            "longest_segment_tokens": float(max((len(weighted.segment.tokens) for weighted in order), default=0)),
            "optimizer_steps": float(len(gradient_norms)),
            "stopped_at_max_kl": float(stopped),
            "minibatches_out_of_memory": float(out_of_memory),
            "start_out_of_memory": float(start_out_of_memory),
            "start_seconds": started_pass - started,
            "seconds": time.monotonic() - started,
        }


def minibatches(segments: Sequence[Weighted], tokens_per_step: int) -> list[list[Weighted]]:
    """The segments in order, cut where a minibatch has reached `tokens_per_step` sampled tokens. A last minibatch
    of less than half that joins the one before: Adam's step is as large for a handful of tokens as for a full
    minibatch."""
    batches: list[list[Weighted]] = [[]]
    counted = 0
    for weighted in segments:
        if counted >= tokens_per_step:
            batches.append([])
            counted = 0
        batches[-1].append(weighted)
        counted += weighted.segment.sampled
    if len(batches) > 1 and counted < tokens_per_step / 2:
        batches[-2].extend(batches.pop())
    return [batch for batch in batches if batch]
