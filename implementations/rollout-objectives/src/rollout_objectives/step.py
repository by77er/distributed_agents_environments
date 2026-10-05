# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave parts of autograd untyped.)
"""A step over a batch: the logprobs the step starts from, then passes of updates, under the objective the settings
name (`rollout_objectives.terms`).

First every sampled token's logprob is computed on the weights the step starts from, without a gradient (`old`), and
the reference's where the objective reads it (`reference`: an adapter switched off, or a frozen copy). Then the batch's
items (weighted segments, pairs, labelled examples or distilled segments) are taken in shuffled minibatches of about
`tokens_per_step` sampled tokens, an optimizer step each. The pass stops early if a minibatch finds the policy further
than `max_kl` from where the step began, by the k3 estimate of KL(old || now) on its sampled tokens
(`rollout_objectives.terms.moved_kl`; a likelihood step reads no `old`, and does not stop). A step takes `passes`
passes, each shuffled anew; a fresh optimizer's rate is warmed up over its first `warmup_updates` updates. Only tokens
the policy sampled are trained on. The numbers are `StepSettings`'; which items, and with what advantages, is the
algorithm's business (`rollout_train.algorithm`).

A preference loss is a function of each side's whole log-likelihood, so a minibatch's gradient is taken in two parts,
which hold one segment's activations at a time: the loss of the logprobs computed without a gradient (on the weights
the minibatch steps from) gives each token's gradient, and each segment's logprobs, computed again with a gradient,
are moved by it (`d loss / d logprobs · logprobs`, whose gradient is the loss's).

`Plan` (which items, in which minibatches), `metrics` and `line` are what any trainer of this step shares
(`rollout_tinker`'s takes it on Tinker).
"""

import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import nn

from rollout_objectives.distillation import Taught, distilled
from rollout_objectives.settings import StepSettings
from rollout_objectives.terms import SUMS, Scored, Terms, labelled, moved_kl, pair, tally, terms, units
from rollout_train.objectives import LIKELIHOOD, PREFERENCE, Objective
from rollout_train.recorder import Segment
from rollout_train.trainer import Distilled, Item, Labelled, Pair, Weighted, segments_of

__all__ = [
    "MINIBATCHES",
    "Plan",
    "PolicyStep",
    "TrainablePolicy",
    "line",
    "metrics",
    "minibatches",
    "positions",
    "preference_terms",
    "sampled",
]

MINIBATCHES = "minibatches.jsonl"
"""In a step's state: what each of its minibatches did, one line each (`line`)."""


class TrainablePolicy(Protocol):
    """What the step needs of a policy (`rollout_lora.policy.Policy` is one). An objective with a KL to the reference
    or a preference loss against it needs `reference` too, one with an entropy bonus `logprobs_and_entropy`, and a
    distillation over the teacher's top-k tokens `logprobs_among` (the logprobs of given tokens at each position,
    beside the sampled ones')."""

    model: nn.Module

    def parameters(self) -> list[nn.Parameter]: ...

    def logprobs(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor: ...


def positions(segment: Segment) -> list[int]:
    """The positions of the tokens the policy sampled in a segment."""
    return [position for span in segment.spans for position in range(span.start, span.end)]


def sampled(weighted: Weighted) -> list[int]:
    """The positions of the tokens the policy sampled in a weighted segment."""
    return positions(weighted.segment)


def sampled_tokens(item: Item) -> int:
    """How many tokens the policy sampled in an item's segments."""
    return sum(segment.sampled for segment in segments_of(item))


def _trainable(item: Item) -> bool:
    """Whether an item has tokens to train on: a segment with sampled tokens, each side of a pair with some."""
    if isinstance(item, Pair):
        return any(positions(each) for each in item.chosen) and any(positions(each) for each in item.rejected)
    return any(positions(each) for each in segments_of(item))


@dataclass
class Plan:
    """The items a step trains on, in the order it takes them (those whose segments are all no longer than
    `segment_tokens`, with tokens sampled, shuffled by the step's seed), and how many it left out for their length."""

    items: list[Item]
    too_long: int
    shuffled: random.Random
    """What shuffles each pass after the first."""

    @classmethod
    def of(cls, items: Sequence[Item], settings: StepSettings, seed: int) -> "Plan":
        longest = settings.segment_tokens
        fits = [longest is None or all(len(each.tokens) <= longest for each in segments_of(item)) for item in items]
        kept = [item for item, fit in zip(items, fits, strict=True) if fit and _trainable(item)]
        shuffled = random.Random(seed)
        shuffled.shuffle(kept)
        return cls(kept, fits.count(False), shuffled)

    @property
    def segments(self) -> list[Segment]:
        """Every segment of its items, in order (a segment two items share, once)."""
        seen: dict[int, Segment] = {}
        for item in self.items:
            for each in segments_of(item):
                seen.setdefault(id(each), each)
        return list(seen.values())

    def minibatches(self, settings: StepSettings) -> list[list[Item]]:
        """Every pass's minibatches, in order: the first pass takes the items in the plan's order, and each further
        one shuffles them anew."""
        passes = [
            self.items,
            *(self.shuffled.sample(self.items, len(self.items)) for _ in range(settings.passes - 1)),
        ]
        return [batch for each in passes for batch in minibatches(each, settings.tokens_per_step)]


def minibatches[Each: Item](items: Sequence[Each], tokens_per_step: int) -> list[list[Each]]:
    """The items in order, cut where a minibatch has reached `tokens_per_step` sampled tokens. A last minibatch of less
    than half that joins the one before: Adam's step is as large for a handful of tokens as for a full minibatch."""
    batches: list[list[Each]] = [[]]
    counted = 0
    for item in items:
        if counted >= tokens_per_step:
            batches.append([])
            counted = 0
        batches[-1].append(item)
        counted += sampled_tokens(item)
    if len(batches) > 1 and counted < tokens_per_step / 2:
        batches[-2].extend(batches.pop())
    return [batch for batch in batches if batch]


def preference_terms(
    objective: Objective,
    batch: Sequence[Item],
    now: Mapping[int, torch.Tensor],
    reference: Mapping[int, torch.Tensor],
) -> list[tuple[Item, Terms]]:
    """The preference loss of each item of a minibatch, from each segment's logprobs (`now`, by the segment's `id`)
    and the reference's."""

    def scored(side: Sequence[Segment]) -> Scored:
        found = [reference[id(each)] for each in side] if objective.needs_reference else None
        return Scored([now[id(each)] for each in side], found)

    found: list[tuple[Item, Terms]] = []
    examples: list[tuple[Item, Scored, bool]] = []
    for item in batch:
        if isinstance(item, Pair):
            found.append((item, pair(objective, scored(item.chosen), scored(item.rejected))))
        elif isinstance(item, Labelled):
            examples.append((item, scored(item.side), item.desirable))
        else:
            raise ValueError("a preference loss is of pairs or labelled examples, not of weighted segments")
    if examples:
        each_terms = labelled(objective, [(side, desirable) for _, side, desirable in examples])
        found += [(item, each) for (item, _, _), each in zip(examples, each_terms, strict=True)]
    return found


@dataclass
class PolicyStep:
    policy: TrainablePolicy
    settings: StepSettings = field(default_factory=StepSettings)
    fresh: bool = True
    """Whether the optimizer starts afresh (warmed up), or goes on from a state loaded into it."""

    def __post_init__(self) -> None:
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.settings.learning_rate, weight_decay=0.0)
        self.minibatches: list[dict[str, float]] = []
        """What each minibatch of the last pass did, in order (`step` returns their totals)."""

    def _scored(self, segment: Segment, *, entropy: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        """A segment's sampled tokens' logprobs now, and their positions' entropies if asked for."""
        if not entropy:
            return self.policy.logprobs(segment.tokens, positions(segment)), None
        scoring = getattr(self.policy, "logprobs_and_entropy", None)
        if scoring is None:
            raise ValueError("an entropy bonus needs a policy that gives each position's entropy")
        found, entropies = scoring(segment.tokens, positions(segment))
        return found, entropies

    def _among(self, segment: Segment, candidates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A segment's sampled tokens' logprobs now, and those of `candidates` (one row of token ids for each sampled
        position)."""
        scoring = getattr(self.policy, "logprobs_among", None)
        if scoring is None:
            raise ValueError("the top_k form of distillation needs a policy that gives the logprobs of given tokens")
        found, among = scoring(segment.tokens, positions(segment), candidates)
        return found, among

    def _reference(self, segment: Segment) -> torch.Tensor:
        scoring = getattr(self.policy, "reference", None)
        if scoring is None:
            raise ValueError("the objective reads the reference model, and this policy has none")
        return scoring(segment.tokens, positions(segment)).detach()

    def step(self, items: Sequence[Item], *, seed: int = 0) -> dict[str, float]:
        """The logprobs the step starts from, then `passes` passes over the items in shuffled minibatches."""
        started = time.monotonic()
        settings, objective = self.settings, self.settings.loss
        plan = Plan.of(items, settings, seed)
        self.policy.model.train()

        # Where the step starts: each sampled token's logprob on these weights, where it was sampled, and the
        # reference's.
        old: dict[int, torch.Tensor] = {}
        behaviors: dict[int, torch.Tensor] = {}
        references: dict[int, torch.Tensor] = {}
        start_out_of_memory = 0
        if objective.family != LIKELIHOOD:
            with torch.no_grad():
                for item in list(plan.items):
                    try:
                        for segment in segments_of(item):
                            if id(segment) in old:
                                continue
                            behavior = torch.tensor(segment.logprobs)
                            if objective.needs_behaviour and not bool(torch.isfinite(behavior).all()):
                                raise ValueError("a sampled token has no behavior logprob")  # (one NaN: every weight)
                            found = self.policy.logprobs(segment.tokens, positions(segment))
                            old[id(segment)] = found.detach()
                            behaviors[id(segment)] = behavior.to(found.device)
                            if objective.needs_reference:
                                references[id(segment)] = self._reference(segment)
                    except torch.OutOfMemoryError:  # (left out of the step, and counted)
                        torch.cuda.empty_cache()
                        start_out_of_memory += 1
                        plan.items.remove(item)
        started_pass = time.monotonic()

        totals = dict.fromkeys(SUMS, 0.0)
        gradient_norms: list[float] = []
        moved = 0.0  # how far the last minibatch stepped on found the policy from the step's start
        out_of_memory = 0
        stopped = False
        self.minibatches = []
        for batch in plan.minibatches(settings):
            if objective.family == PREFERENCE:
                units = float(len(batch))
            else:
                units = sum(_units(objective, item) for item in batch)
            sums = dict.fromkeys(SUMS, 0.0)
            try:
                if objective.family == PREFERENCE:
                    distance = self._preference(
                        batch, units, old, references, sums, settings, updated=bool(gradient_norms)
                    )
                else:
                    distance = self._weighted(batch, units, old, behaviors, references, sums)
            except torch.OutOfMemoryError:  # a gradient with a segment missing is not this minibatch's: drop it
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                out_of_memory += 1
                continue
            if sums["tokens"] == 0:
                continue
            if objective.family != LIKELIHOOD and settings.max_kl is not None and distance > settings.max_kl:
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
                given=len(items),
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

    def _weighted(
        self,
        batch: Sequence[Item],
        units: float,
        old: Mapping[int, torch.Tensor],
        behaviors: Mapping[int, torch.Tensor],
        references: Mapping[int, torch.Tensor],
        sums: dict[str, float],
    ) -> float:
        """A minibatch of weighted segments' gradient, accumulated; how far it found the policy from the step's start
        (per token)."""
        objective = self.settings.loss
        for item in batch:
            if isinstance(item, Distilled):
                found = self._distilled(item, old, behaviors, references)
            elif isinstance(item, Weighted) and not objective.distills:
                segment = item.segment
                logprobs, entropy = self._scored(segment, entropy=objective.needs_entropy)
                key = id(segment)
                found = terms(
                    objective, logprobs, item.advantage, old.get(key), behaviors.get(key), references.get(key), entropy
                )
            else:
                takes = "distilled segments" if objective.distills else "weighted segments"
                raise ValueError(f"a {objective.family} loss is of {takes}, not {type(item).__name__} items")
            (found.loss / units).backward()
            tally(sums, found, objective)
        return sums["moved"] / max(sums["tokens"], 1.0)

    def _distilled(
        self,
        item: Distilled,
        old: Mapping[int, torch.Tensor],
        behaviors: Mapping[int, torch.Tensor],
        references: Mapping[int, torch.Tensor],
    ) -> Terms:
        """A distilled segment's terms, its logprobs computed now (and those of the teacher's top-k tokens, for the
        top-k form)."""
        objective = self.settings.loss
        segment, key = item.segment, id(item.segment)
        among: torch.Tensor | None = None
        entropy: torch.Tensor | None = None
        if objective.needs_distribution:
            if objective.needs_entropy:
                raise ValueError("an entropy bonus beside the top_k form of distillation is not computed")
            candidates = Taught.of(item.scores, objective.needs_top).top_tokens
            assert candidates is not None
            logprobs, among = self._among(segment, candidates)
        else:
            logprobs, entropy = self._scored(segment, entropy=objective.needs_entropy)
        start = old.get(key)
        if start is None:
            raise ValueError("a distillation needs each sampled token's logprob at the step's start")
        return distilled(objective, item, logprobs, start, behaviors.get(key), references.get(key), entropy, among)

    def _preference(
        self,
        batch: Sequence[Item],
        units: float,
        old: Mapping[int, torch.Tensor],
        references: Mapping[int, torch.Tensor],
        sums: dict[str, float],
        settings: StepSettings,
        *,
        updated: bool,
    ) -> float:
        """A minibatch of preference items' gradient, accumulated (unless it found the policy past `max_kl`); how far
        it found the policy from the step's start (per token)."""
        objective = settings.loss
        segments = list({id(each): each for item in batch for each in segments_of(item)}.values())
        now: dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for segment in segments:  # (before any update, where the step starts is what the policy gives now)
                found = old[id(segment)] if not updated else self.policy.logprobs(segment.tokens, positions(segment))
                now[id(segment)] = found.detach().clone().requires_grad_(True)
        distance = sum(float(moved_kl(old[key], now[key]).sum()) for key in now)
        tokens = sum(float(now[key].numel()) for key in now)
        found_terms = preference_terms(objective, batch, now, references)
        for item, found in found_terms:
            tally(sums, found, objective, segments=float(len(segments_of(item))))
        sums["moved"], sums["tokens"] = distance, tokens  # (each segment once, however many items hold it)
        distance /= max(tokens, 1.0)
        if settings.max_kl is not None and distance > settings.max_kl:
            return distance
        total = torch.stack([found.loss for _, found in found_terms]).sum()
        (total / units).backward()
        for segment in segments:
            gradient = now[id(segment)].grad
            if gradient is None or not bool(gradient.any()):
                continue
            logprobs = self.policy.logprobs(segment.tokens, positions(segment))
            (logprobs * gradient.to(logprobs.dtype)).sum().backward()
        return distance


def _units(objective: Objective, item: Item) -> float:
    return units(objective, item.segment.sampled) if isinstance(item, Weighted | Distilled) else 1.0


def line(sums: Mapping[str, float], units: float, rate: float) -> dict[str, float]:
    """What a minibatch that was stepped on did (its `SUMS`, over `units`, stepped at `rate`), as `MINIBATCHES` keeps
    it."""
    tokens = max(sums["tokens"], 1.0)
    said = {
        "segments": sums["segments"],
        "tokens": sums["tokens"],
        "loss": sums["loss"] / units,
        "clip_fraction": sums["clipped"] / tokens,
        "kl": sums["moved"] / tokens,
        "learning_rate": rate,
    }
    if sums["distilled"]:
        said["teacher_gap"] = sums["gap"] / max(sums["scored"], 1.0)
    return said


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
    start logprobs (`starts`; a behaviour logprob that is not finite, from a provider without them, is left out of
    theirs), the plan of `given` items, how far the last minibatch stepped on found the policy from the step's start
    (`moved`), and how many `updates` it made."""
    tokens = max(totals["tokens"], 1.0)
    finite = [(behavior, old) for behavior, old in starts if bool(torch.isfinite(behavior).all())]
    start_tokens = max(sum(float(old.numel()) for _, old in finite), 1.0)
    said = {
        "loss": totals["loss"] / max(totals["units"], 1.0),
        "clip_fraction": totals["clipped"] / tokens,
        "mean_ratio": totals["ratio"] / tokens,
        # Where the tokens were sampled, against where the step starts: the engine's and the trainer's numerical
        # difference, and how stale the turns are. `kl_floor` estimates KL(behavior || start) on the sampled
        # tokens; `mean_mismatch` is the mean absolute difference of their logprobs.
        "kl_floor": sum(float((behavior - old).sum()) for behavior, old in finite) / start_tokens,
        "mean_mismatch": sum(float((behavior - old).abs().sum()) for behavior, old in finite) / start_tokens,
        "mean_weight": totals["weight"] / tokens,
        "truncated_fraction": totals["truncated"] / tokens,
        # How far the update moved the policy: KL(start || now) on the sampled tokens by the k3 estimate
        # (`moved_kl`), as the last minibatch stepped on found it before its step: what the stop at `max_kl` reads.
        "kl_moved": moved,
        "kl_penalty": totals["kl"] / tokens,
        "entropy": totals["entropy"] / tokens,
        "tokens": totals["tokens"],
        "segments": totals["segments"],
        "segments_given": float(given),
        "segments_too_long": float(plan.too_long),
        "longest_segment_tokens": float(max((len(each.tokens) for each in plan.segments), default=0)),
        "optimizer_steps": float(updates),
        "passes": float(settings.passes),
        "learning_rate": settings.learning_rate,
        "warmup_updates": float(settings.warmup_updates if fresh else 0),
        "stopped_at_max_kl": float(stopped),
        "start_seconds": start_seconds,
    }
    if totals["items"]:
        items = totals["items"]
        said |= {
            "items": items,
            "preference_accuracy": totals["accurate"] / items,
            "preference_margin": totals["margin"] / items,
        }
    if totals["pairs"]:
        said |= {"chosen_log_ratio": totals["chosen"] / totals["pairs"]}
        said |= {"rejected_log_ratio": totals["rejected"] / totals["pairs"]}
    if totals["distilled"]:
        # The policy's logprob of each sampled token less the teacher's, as each minibatch found it before its update:
        # on the policy's own samples, an estimate of KL(policy || teacher) per token.
        scored = max(totals["scored"], 1.0)
        said |= {
            "teacher_gap": totals["gap"] / scored,
            "teacher_divergence": totals["divergence"] / scored,
            "advantage_clip_fraction": totals["advantage_clipped"] / scored,
            "unscored_fraction": 1.0 - totals["scored"] / totals["distilled"],
        }
    return said
