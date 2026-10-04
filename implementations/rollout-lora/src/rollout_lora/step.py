# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave parts of autograd untyped.)
"""A policy step over weighted segments: the logprobs the step starts from, then passes of updates.

First every sampled token's logprob is computed on the weights the step starts from, without a gradient (`old`).
Then the segments are taken in shuffled minibatches of about `tokens_per_step` sampled tokens, an optimizer step
each, under the objective the settings name (`rollout_lora.objectives`): a ratio to `old` that bounds how far the
step moves the policy, and an importance weight `old / behavior` for where each token was sampled (an older checkpoint,
and the engine computing differently from the trainer). No KL penalty; the pass stops early if a minibatch finds the
policy further than `max_kl` from where the step began. A step takes `passes` passes, each shuffled anew; a fresh
optimizer's rate is warmed up over its first `warmup_updates` updates. Only tokens the policy sampled are trained on.
The numbers are `LoraSettings`'; which segments, and with what advantages, is the algorithm's business.

`Plan` (which segments, in which minibatches), `metrics` and `line` are what any trainer of this step shares
(`rollout_tinker`'s takes it on Tinker).
"""

import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import nn

from rollout_lora.objectives import SUMS, tally, terms
from rollout_lora.settings import LoraSettings, StepSettings
from rollout_train.trainer import Weighted

MINIBATCHES = "minibatches.jsonl"
"""In a step's state: what each of its minibatches did, one line each (`line`)."""


class TrainablePolicy(Protocol):
    """What the step needs of a policy (`rollout_lora.policy.Policy` is one)."""

    model: nn.Module

    def parameters(self) -> list[nn.Parameter]: ...

    def logprobs(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor: ...


def sampled(weighted: Weighted) -> list[int]:
    """The positions of the tokens the policy sampled in a segment."""
    return [position for span in weighted.segment.spans for position in range(span.start, span.end)]


@dataclass
class Plan:
    """The segments a step trains on, in the order it takes them (those no longer than `segment_tokens`, with
    tokens sampled, shuffled by the step's seed), and how many it left out for their length."""

    segments: list[Weighted]
    too_long: int
    shuffled: random.Random
    """What shuffles each pass after the first."""

    @classmethod
    def of(cls, segments: Sequence[Weighted], settings: StepSettings, seed: int) -> "Plan":
        longest = settings.segment_tokens
        fits = [longest is None or len(weighted.segment.tokens) <= longest for weighted in segments]
        kept = [weighted for weighted, fit in zip(segments, fits, strict=True) if fit and sampled(weighted)]
        shuffled = random.Random(seed)
        shuffled.shuffle(kept)
        return cls(kept, fits.count(False), shuffled)

    def minibatches(self, settings: StepSettings) -> list[list[Weighted]]:
        """Every pass's minibatches, in order: the first pass takes the segments in the plan's order, and each
        further one shuffles them anew."""
        passes = [
            self.segments,
            *(self.shuffled.sample(self.segments, len(self.segments)) for _ in range(settings.passes - 1)),
        ]
        return [batch for each in passes for batch in minibatches(each, settings.tokens_per_step)]


@dataclass
class PolicyStep:
    policy: TrainablePolicy
    settings: LoraSettings = field(default_factory=LoraSettings)
    fresh: bool = True
    """Whether the optimizer starts afresh (warmed up), or goes on from a state loaded into it."""

    def __post_init__(self) -> None:
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.settings.learning_rate, weight_decay=0.0)
        self.minibatches: list[dict[str, float]] = []
        """What each minibatch of the last pass did, in order (`step` returns their totals)."""

    def step(self, segments: Sequence[Weighted], *, seed: int = 0) -> dict[str, float]:
        """The logprobs the step starts from, then `passes` passes over the segments in shuffled minibatches."""
        started = time.monotonic()
        settings, objective = self.settings, self.settings.loss
        plan = Plan.of(segments, settings, seed)
        self.policy.model.train()

        # Where the step starts: each sampled token's logprob on these weights, and where it was sampled.
        old: dict[int, torch.Tensor] = {}
        behaviors: dict[int, torch.Tensor] = {}
        start_out_of_memory = 0
        if objective.reads_old:
            with torch.no_grad():
                for weighted in list(plan.segments):
                    behavior = torch.tensor(weighted.segment.logprobs)
                    if not bool(torch.isfinite(behavior).all()):  # (one NaN would make every weight NaN)
                        raise ValueError("a sampled token has no behavior logprob")
                    try:
                        found = self.policy.logprobs(weighted.segment.tokens, sampled(weighted))
                    except torch.OutOfMemoryError:  # (left out of the step, and counted)
                        torch.cuda.empty_cache()
                        start_out_of_memory += 1
                        plan.segments.remove(weighted)
                        continue
                    old[id(weighted)] = found.detach()
                    behaviors[id(weighted)] = behavior.to(found.device)
        started_pass = time.monotonic()

        totals = dict.fromkeys(SUMS, 0.0)
        gradient_norms: list[float] = []
        moved = 0.0  # how far the last minibatch stepped on found the policy from the step's start
        out_of_memory = 0
        stopped = False
        self.minibatches = []
        for batch in plan.minibatches(settings):
            units = sum(objective.units(weighted.segment.sampled) for weighted in batch)
            sums = dict.fromkeys(SUMS, 0.0)
            try:
                for weighted in batch:
                    logprobs = self.policy.logprobs(weighted.segment.tokens, sampled(weighted))
                    found = terms(
                        objective, logprobs, weighted.advantage, old.get(id(weighted)), behaviors.get(id(weighted))
                    )
                    (found.loss / units).backward()
                    tally(sums, found, objective)
            except torch.OutOfMemoryError:  # a gradient with a segment missing is not this minibatch's: drop it
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                out_of_memory += 1
                continue
            if sums["tokens"] == 0:
                continue
            distance = sums["moved"] / sums["tokens"]
            if objective.reads_old and settings.max_kl is not None and distance > settings.max_kl:
                self.optimizer.zero_grad(set_to_none=True)
                stopped = True
                break
            moved = distance
            for key, value in sums.items():
                totals[key] += value
            norm = float(torch.nn.utils.clip_grad_norm_(self.policy.parameters(), settings.max_gradient_norm))
            rate = settings.rate(len(gradient_norms), fresh=self.fresh)
            for group in self.optimizer.param_groups:
                group["lr"] = rate
            gradient_norms.append(norm)
            self.minibatches.append({**line(sums, units, rate), "gradient_norm": norm})
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        return {
            **metrics(
                totals,
                [(behaviors[key], old[key]) for key in old],
                plan=plan,
                given=len(segments),
                moved=moved,
                updates=len(gradient_norms),
                settings=settings,
                fresh=self.fresh,
                stopped=stopped,
                start_seconds=started_pass - started,
            ),
            "gradient_norm": sum(gradient_norms) / max(len(gradient_norms), 1),  # before clipping, mean over steps
            "minibatches_out_of_memory": float(out_of_memory),
            "start_out_of_memory": float(start_out_of_memory),
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


def line(sums: Mapping[str, float], units: float, rate: float) -> dict[str, float]:
    """What a minibatch that was stepped on did (its `SUMS`, over `units`, stepped at `rate`), as `MINIBATCHES` keeps
    it."""
    tokens = max(sums["tokens"], 1.0)
    return {
        "segments": sums["segments"],
        "tokens": sums["tokens"],
        "loss": sums["loss"] / units,
        "clip_fraction": sums["clipped"] / tokens,
        "kl": sums["moved"] / tokens,
        "learning_rate": rate,
    }


def metrics(
    totals: Mapping[str, float],
    starts: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    plan: Plan,
    given: int,
    moved: float,
    updates: int,
    settings: StepSettings,
    fresh: bool,
    stopped: bool,
    start_seconds: float,
) -> dict[str, float]:
    """A step's metrics: from the `SUMS` of the minibatches it stepped on (`totals`), each segment's behaviour and
    start logprobs (`starts`), the plan of `given` segments, how far the last minibatch stepped on found the policy
    from the step's start (`moved`), and how many `updates` it made."""
    tokens = max(totals["tokens"], 1.0)
    start_tokens = max(sum(float(old.numel()) for _, old in starts), 1.0)
    return {
        "loss": totals["loss"] / max(totals["units"], 1.0),
        "clip_fraction": totals["clipped"] / tokens,
        "mean_ratio": totals["ratio"] / tokens,
        # Where the tokens were sampled, against where the step starts: the engine's and the trainer's numerical
        # difference, and how stale the turns are. `kl_floor` estimates KL(behavior || start) on the sampled
        # tokens; `mean_mismatch` is the mean absolute difference of their logprobs.
        "kl_floor": sum(float((behavior - old).sum()) for behavior, old in starts) / start_tokens,
        "mean_mismatch": sum(float((behavior - old).abs().sum()) for behavior, old in starts) / start_tokens,
        "mean_weight": totals["weight"] / tokens,
        "truncated_fraction": totals["truncated"] / tokens,
        # How far the update moved the policy: KL(start || now) on the sampled tokens, as the last minibatch
        # stepped on found it before its step.
        "kl_moved": moved,
        "tokens": totals["tokens"],
        "segments": totals["segments"],
        "segments_given": float(given),
        "segments_too_long": float(plan.too_long),
        "longest_segment_tokens": float(max((len(weighted.segment.tokens) for weighted in plan.segments), default=0)),
        "optimizer_steps": float(updates),
        "passes": float(settings.passes),
        "learning_rate": settings.learning_rate,
        "warmup_updates": float(settings.warmup_updates if fresh else 0),
        "stopped_at_max_kl": float(stopped),
        "start_seconds": start_seconds,
    }
