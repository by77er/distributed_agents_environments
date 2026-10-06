# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave parts of autograd untyped.)
"""A step over a batch: the logprobs the step starts from, then passes of updates, under the objective the settings
name (`rollout_objectives.terms`).

First every sampled token's logprob is computed on the weights the step starts from, without a gradient (`old`), and
the reference's where the objective reads it (`reference`: an adapter switched off, or a frozen copy). Then the batch's
items (weighted segments, pairs, labelled examples or distilled segments) are taken in shuffled minibatches of about
`tokens_per_step` sampled tokens, an optimizer step each. The first minibatch runs on the weights the step starts
from, so its segments' `old` is what it computes (with a gradient, detached), not a pass of its own (but a preference
loss's, which computes its logprobs without a gradient first either way). The pass stops early if a minibatch finds
the policy further than `max_kl` from where the step began, by the k3 estimate of KL(old || now) on its sampled tokens
(`rollout_objectives.terms.moved_kl`; a likelihood step reads no `old`, and does not stop). A step takes `passes`
passes, each shuffled anew; a fresh optimizer's rate is warmed up over its first `warmup_updates` updates. Only tokens
the policy sampled are trained on. The numbers are `StepSettings`'; which items, and with what advantages, is the
algorithm's business (`rollout_train.algorithm`).

Every pass over segments runs them in packs (`rollout_objectives.packing`): rows of up to `pack_tokens` tokens, each
segment seeing only itself, segments that start alike sharing their prefix (`share_prefixes`). A minibatch's gradient
is accumulated a pack at a time. A policy that runs packs says so (`PackingPolicy`: `packing`, `packed`); any other
runs one segment at a time, and the step's metrics say which (`packed`).

A preference loss is a function of each side's whole log-likelihood, so a minibatch's gradient is taken in two parts,
which hold one pack's activations at a time: the loss of the logprobs computed without a gradient (on the weights
the minibatch steps from) gives each token's gradient, and each segment's logprobs, computed again with a gradient,
are moved by it (`d loss / d logprobs · logprobs`, whose gradient is the loss's).

`Plan` (which items, in which minibatches), `metrics` and `line` are what any trainer of this step shares
(`rollout_tinker`'s takes it on Tinker, which batches a minibatch's segments itself).

A step may be shared among processes, one per GPU (`ranks`, `rollout_objectives.ranks`): each takes the same plan,
computes its share of every minibatch's segments and of the logprobs the step starts from, and the processes gather
those logprobs and add up each minibatch's sums before reading them. The segments are grouped by the prefixes they
share first, and the groups shared out, balanced by their tokens (`shares`); each process packs its own. Every item's
loss is divided by its minibatch's units counted over the whole minibatch, and the gradients are added up across
processes (a sharded model's reduction is a sum), so the update is the one a single process makes of the same
minibatch, whatever the number of processes. A process with fewer packs than the most takes idle passes (a two-token
sequence, its loss times zero) as many times as it lacks: a sharded model's layers are gathered by every process at
once, so each takes as many passes as the others. A step shared this way does not leave out a minibatch or a segment
that runs out of memory: the step fails.
"""

import functools
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from typing import Protocol, cast

import torch
from torch import nn

from rollout_objectives.distillation import Taught, distilled
from rollout_objectives.packing import Pack, Scores, grouped, packed
from rollout_objectives.ranks import Ranks, shares
from rollout_objectives.settings import StepSettings
from rollout_objectives.terms import SUMS, Scored, Terms, labelled, moved_kl, pair, tally, terms, units
from rollout_train.memory import SEGMENT_TOKENS
from rollout_train.objectives import LIKELIHOOD, PREFERENCE, Objective
from rollout_train.recorder import Segment
from rollout_train.trainer import Distilled, Item, Labelled, Pair, Weighted, segments_of

__all__ = [
    "MINIBATCHES",
    "PackingPolicy",
    "Plan",
    "PolicyStep",
    "SharedPolicy",
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
    beside the sampled ones'). A policy that runs packs is a `PackingPolicy`; a step shared among processes takes a
    `SharedPolicy`."""

    model: nn.Module

    def parameters(self) -> list[nn.Parameter]: ...

    def logprobs(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor: ...


class PackingPolicy(TrainablePolicy, Protocol):
    """A policy that runs a pack of segments in one pass (where `packing`): each segment's sampled tokens' logprobs
    as it gives them for the segment alone, with the entropies (`entropy`) or the logprobs of given tokens
    (`candidates`, a tensor for each segment) where asked; and under the reference, for an objective that reads it."""

    packing: bool

    def packed(
        self, pack: Pack, *, entropy: bool = False, candidates: Sequence[torch.Tensor] | None = None
    ) -> list[Scores]: ...

    def packed_reference(self, pack: Pack) -> list[torch.Tensor]: ...


class SharedPolicy(TrainablePolicy, Protocol):
    """A policy sharded among the processes a step is shared among, whose gradients are added up across them (not
    their mean: each item's loss is divided by its whole minibatch's units). It takes an idle pass (`idle`: one that
    learns nothing, under the reference with `reference`, with a backward pass with `gradient`) where a process has
    fewer passes than the others, since the processes gather a sharded model's layers together; and clips its
    gradient by the norm over every process's shard (`clip_gradients`, which returns the norm before)."""

    def idle(self, *, gradient: bool = False, reference: bool = False) -> None: ...

    def clip_gradients(self, maximum: float, ranks: Ranks) -> float: ...


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
    ranks: Ranks = field(default_factory=Ranks)
    """The processes the step is shared among (one by default: none)."""
    optimizer_given: InitVar[torch.optim.Optimizer | None] = None
    """An optimizer to go on with (a resident trainer's, from its last step); else a new AdamW."""

    def __post_init__(self, optimizer_given: torch.optim.Optimizer | None) -> None:
        self.optimizer: torch.optim.Optimizer = optimizer_given or torch.optim.AdamW(
            self.policy.parameters(), lr=self.settings.learning_rate, weight_decay=0.0
        )
        self.minibatches: list[dict[str, float]] = []
        """What each minibatch of the last pass did, in order (`step` returns their totals)."""
        self.packing = bool(getattr(self.policy, "packing", False))
        """Whether the policy runs packs (else one segment at a time)."""
        self._capacity = 1
        self._ran = dict.fromkeys(("packs", "rows", "tokens"), 0)
        """Of the last step, in this process: the packs it ran, their rows' tokens, and their segments' tokens."""

    def _turns(self, segments: Sequence[Segment]) -> list[Pack | None]:
        """This process's turns over `segments`: its packs (each segment by its index in `segments`), then an idle
        turn (none) for each pack it has fewer than the most any process has. Packing, segments that share a prefix
        are grouped, and the groups packed; else each segment is a pack of its own. The packs are the same however
        many processes share the step (so each segment is computed alike), and shared out balanced by their count,
        then their tokens (`shares`): no process takes more than its even share of the passes, rounded up."""
        if self.packing:
            groups = grouped(segments, self._capacity, share=self.settings.share_prefixes)
            units = packed(segments, groups, self._capacity)
        else:
            units = [Pack.single(index, each) for index, each in enumerate(segments)]
        if not self.ranks.shared:
            return list(units)
        parts = shares([unit.length for unit in units], self.ranks.size)
        mine: list[Pack | None] = [units[index] for index in parts[self.ranks.rank]]
        return mine + [None] * (max(len(each) for each in parts) - len(mine))

    def _score(
        self,
        pack: Pack,
        *,
        entropy: bool = False,
        candidates: Sequence[torch.Tensor] | None = None,
    ) -> list[Scores]:
        """Each of a pack's segments' sampled tokens' logprobs now (with their entropies, or the logprobs of
        `candidates`, where asked)."""
        if self.packing:
            found = cast(PackingPolicy, self.policy).packed(pack, entropy=entropy, candidates=candidates)
        else:
            found: list[Scores] = []
            for index, segment in enumerate(pack.segments):
                if candidates is not None:
                    logprobs, among = self._among(segment, candidates[index])
                    found.append(Scores(logprobs, among=among))
                else:
                    logprobs, entropies = self._scored(segment, entropy=entropy)
                    found.append(Scores(logprobs, entropy=entropies))
        self._ran["packs"] += 1
        self._ran["rows"] += pack.length
        self._ran["tokens"] += pack.segment_tokens
        return found

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

    def _logprobs(self, pack: Pack) -> list[torch.Tensor]:
        return [each.logprobs for each in self._score(pack)]

    def _reference(self, pack: Pack) -> list[torch.Tensor]:
        """Each of a pack's segments' sampled tokens' logprobs under the reference."""
        if self.packing:
            found = [each.detach() for each in cast(PackingPolicy, self.policy).packed_reference(pack)]
        else:
            scoring = getattr(self.policy, "reference", None)
            if scoring is None:
                raise ValueError("the objective reads the reference model, and this policy has none")
            found = [scoring(segment.tokens, positions(segment)).detach() for segment in pack.segments]
        self._ran["packs"] += 1
        self._ran["rows"] += pack.length
        self._ran["tokens"] += pack.segment_tokens
        return found

    def _idle(self, *, gradient: bool = False, reference: bool = False) -> None:
        """A pass that keeps this process in step with the others, learning nothing (`SharedPolicy.idle`)."""
        cast(SharedPolicy, self.policy).idle(gradient=gradient, reference=reference)

    def step(self, items: Sequence[Item], *, seed: int = 0) -> dict[str, float]:
        """The logprobs the step starts from, then `passes` passes over the items in shuffled minibatches."""
        started = time.monotonic()
        settings, objective = self.settings, self.settings.loss
        plan = Plan.of(items, settings, seed)
        self._validate(plan)
        self.policy.model.train()
        longest = max((len(each.tokens) for each in plan.segments), default=1)
        self._capacity = max(settings.pack_tokens or settings.segment_tokens or SEGMENT_TOKENS, longest)
        self._ran = dict.fromkeys(self._ran, 0)

        # Where the step starts: each sampled token's logprob on these weights, where it was sampled, and the
        # reference's.
        old: dict[int, torch.Tensor] = {}
        behaviors: dict[int, torch.Tensor] = {}
        references: dict[int, torch.Tensor] = {}
        reads_old = objective.family != LIKELIHOOD
        start_out_of_memory = self._start(plan, old, behaviors, references) if reads_old else 0
        started_pass = time.monotonic()

        totals = dict.fromkeys(SUMS, 0.0)
        gradient_norms: list[float] = []
        moved = 0.0  # how far the last minibatch stepped on found the policy from the step's start
        out_of_memory = 0
        stopped = False
        self.minibatches = []
        for batch in plan.minibatches(settings):
            if reads_old and gradient_norms:  # (an item whose first minibatch ran out of memory has no start)
                batch = [item for item in batch if all(id(each) in old for each in segments_of(item))]
            if not batch:
                continue
            if objective.family == PREFERENCE:
                units = float(len(batch))
            else:
                units = sum(_units(objective, item) for item in batch)
            sums = dict.fromkeys(SUMS, 0.0)
            try:
                if objective.family == PREFERENCE:
                    distance = self._preference(
                        batch, units, old, behaviors, references, sums, settings, updated=bool(gradient_norms)
                    )
                else:
                    distance = self._weighted(batch, units, old, behaviors, references, sums)
            except torch.OutOfMemoryError:  # a gradient with a segment missing is not this minibatch's: drop it
                if self.ranks.shared:  # (the other processes wait on this one's passes: the step fails)
                    raise
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
            norm = self._clipped(settings.max_gradient_norm)
            rate = settings.rate(len(gradient_norms), fresh=self.fresh)
            for group in self.optimizer.param_groups:
                group["lr"] = rate
            gradient_norms.append(norm)
            self.minibatches.append({**line(sums, units, rate), "gradient_norm": norm})
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        seconds = time.monotonic() - started
        packs_run, rows, tokens = self.ranks.summed([float(self._ran[key]) for key in ("packs", "rows", "tokens")])
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
            # How the step ran its segments (every process's, where it is shared): in packs or one at a time, how many
            # passes over packs it made (every pass over the batch counted), how full their rows were, the share of
            # the segments' tokens shared prefixes spared, and the segments' tokens it ran through the model a second.
            "packed": float(self.packing),
            "packs": packs_run,
            "pack_fill": rows / max(packs_run * self._capacity, 1.0),
            "prefix_shared_fraction": 1.0 - rows / max(tokens, 1.0),
            "segment_tokens_per_second": tokens / max(seconds, 1e-9),
            "seconds": seconds,
        }

    def _validate(self, plan: Plan) -> None:
        """What would fail the step, raised before it computes anything: every process holds the same plan, so all of
        them raise alike, before any of them waits on another."""
        objective = self.settings.loss
        for item in plan.items:
            if objective.family == PREFERENCE:
                if not isinstance(item, Pair | Labelled):
                    raise ValueError("a preference loss is of pairs or labelled examples, not of weighted segments")
            elif not (isinstance(item, Distilled) or (isinstance(item, Weighted) and not objective.distills)):
                takes = "distilled segments" if objective.distills else "weighted segments"
                raise ValueError(f"a {objective.family} loss is of {takes}, not {type(item).__name__} items")
        if objective.needs_distribution and objective.needs_entropy:
            raise ValueError("an entropy bonus beside the top_k form of distillation is not computed")
        if objective.needs_behaviour and objective.family != LIKELIHOOD:
            for segment in plan.segments:
                if not all(math.isfinite(each) for each in segment.logprobs):
                    raise ValueError("a sampled token has no behavior logprob")  # (one NaN: every weight)
        if self.ranks.shared:
            for needed in ("idle", "clip_gradients"):
                if getattr(self.policy, needed, None) is None:
                    raise ValueError(f"a step shared among processes needs a sharded policy's `{needed}`")

    def _start(
        self,
        plan: Plan,
        old: dict[int, torch.Tensor],
        behaviors: dict[int, torch.Tensor],
        references: dict[int, torch.Tensor],
    ) -> int:
        """Each sampled token's logprob on the weights the step starts from (`old`), but those of the first
        minibatch's segments, which it computes itself; and the reference's where the objective reads it. An item a
        pack of which runs out of memory, and that runs out again a segment at a time, is left out of the plan; how
        many were. Shared, each process computes its packs, and every process gathers every segment's."""
        segments = plan.segments
        folded = {id(each) for item in _first_minibatch(plan, self.settings) for each in segments_of(item)}
        failed: set[int] = set()
        with torch.no_grad():
            starting = [each for each in segments if id(each) not in folded]
            for index, found in self._computed(starting, self._logprobs, self._idle, failed).items():
                _started(starting[index], found, old, behaviors)
            if self.settings.loss.needs_reference:
                kept = [each for each in segments if id(each) not in failed]
                idle = functools.partial(self._idle, reference=True)
                for index, found in self._computed(kept, self._reference, idle, failed).items():
                    references[id(kept[index])] = found
        left_out = [item for item in plan.items if any(id(each) in failed for each in segments_of(item))]
        plan.items = [item for item in plan.items if not any(id(each) in failed for each in segments_of(item))]
        return len(left_out)

    def _computed(
        self,
        segments: Sequence[Segment],
        compute: Callable[[Pack], list[torch.Tensor]],
        idle: Callable[[], None],
        failed: set[int] | None = None,
    ) -> dict[int, torch.Tensor]:
        """`compute` (without a gradient, each pack's segments' tensors) over this process's turns of `segments` (and
        `idle` for an idle one), each segment's tensor by its index in `segments`; gathered from every process, where
        the step is shared. With `failed`, a pack that runs out of memory is run again a segment at a time, and a
        segment that runs out alone is added to it (by its `id`); else running out of memory fails the call."""
        found: dict[int, torch.Tensor] = {}
        for pack in self._turns(segments):
            if pack is None:
                idle()
                continue
            try:
                found.update(zip(pack.members, compute(pack), strict=True))
                continue
            except torch.OutOfMemoryError:  # (left out of the step, and counted)
                if failed is None or self.ranks.shared:
                    raise
                torch.cuda.empty_cache()
            for member, segment in zip(pack.members, pack.segments, strict=True):
                try:
                    found[member] = compute(Pack.single(member, segment))[0]
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    failed.add(id(segment))
        return self._gathered(found) if self.ranks.shared else found

    def _gathered(self, found: Mapping[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Every process's `found` tensors (by a segment's index), on this process's device."""
        merged: dict[int, torch.Tensor] = {}
        for part in self.ranks.gathered({index: each.detach().cpu() for index, each in found.items()}):
            merged.update(part)
        device = next(iter(self.policy.parameters())).device
        return {index: each.to(device) for index, each in merged.items()}

    def _weighted(
        self,
        batch: Sequence[Item],
        units: float,
        old: dict[int, torch.Tensor],
        behaviors: dict[int, torch.Tensor],
        references: Mapping[int, torch.Tensor],
        sums: dict[str, float],
    ) -> float:
        """A minibatch of weighted segments' gradient, accumulated a pack at a time (this process's packs of it, where
        the step is shared, and its sums added up across processes); how far it found the policy from the step's
        start (per token). A segment with no `old` yet is on the weights the step starts from (the first
        minibatch's): what it computes is its `old`."""
        objective = self.settings.loss
        segmented = [cast(Weighted | Distilled, item) for item in batch]  # (`_validate` said so)
        candidates: list[torch.Tensor] | None = None
        if objective.needs_distribution:
            candidates = [cast(torch.Tensor, Taught.of(cast(Distilled, item).scores, objective.needs_top).top_tokens)
                          for item in segmented]  # fmt: skip
        reads_old = objective.family != LIKELIHOOD
        lacking = reads_old and any(id(item.segment) not in old for item in segmented)
        folded: dict[int, torch.Tensor] = {}
        for pack in self._turns([item.segment for item in segmented]):
            if pack is None:
                self._idle(gradient=True)
                continue
            asked = None if candidates is None else [candidates[member] for member in pack.members]
            found = self._score(pack, entropy=objective.needs_entropy, candidates=asked)
            loss: torch.Tensor | None = None
            for member, scores in zip(pack.members, found, strict=True):
                item = segmented[member]
                segment, key = item.segment, id(item.segment)
                if reads_old and key not in old:  # (on the weights the step starts from)
                    _started(segment, scores.logprobs, old, behaviors)
                    folded[member] = scores.logprobs.detach()
                if isinstance(item, Distilled):
                    each = distilled(
                        objective, item, scores.logprobs, old[key], behaviors.get(key), references.get(key),
                        scores.entropy, scores.among,
                    )  # fmt: skip
                else:
                    each = terms(
                        objective, scores.logprobs, item.advantage, old.get(key), behaviors.get(key),
                        references.get(key), scores.entropy,
                    )  # fmt: skip
                loss = each.loss if loss is None else loss + each.loss
                tally(sums, each, objective)
            if loss is not None:
                (loss / units).backward()
        if self.ranks.shared:
            if lacking:  # (every process keeps every segment's start, as the others computed theirs)
                for member, found_old in self._gathered(folded).items():
                    _started(segmented[member].segment, found_old, old, behaviors)
            keys = list(sums)
            sums.update(zip(keys, self.ranks.summed([sums[key] for key in keys]), strict=True))
        return sums["moved"] / max(sums["tokens"], 1.0)

    def _preference(
        self,
        batch: Sequence[Item],
        units: float,
        old: dict[int, torch.Tensor],
        behaviors: dict[int, torch.Tensor],
        references: Mapping[int, torch.Tensor],
        sums: dict[str, float],
        settings: StepSettings,
        *,
        updated: bool,
    ) -> float:
        """A minibatch of preference items' gradient, accumulated (unless it found the policy past `max_kl`) a pack
        at a time; how far it found the policy from the step's start (per token). Its logprobs without a gradient are
        each process's packs', gathered, where the step is shared: the loss is of every item."""
        objective = settings.loss
        segments = list({id(each): each for item in batch for each in segments_of(item)}.values())
        computing = [each for each in segments if updated or id(each) not in old]
        with torch.no_grad():  # (before any update, where the step starts is what the policy gives now)
            found = self._computed(computing, self._logprobs, self._idle)
        computed = {id(computing[index]): each for index, each in found.items()}
        if not updated:
            for segment in computing:
                _started(segment, computed[id(segment)], old, behaviors)
        now = {
            id(each): computed.get(id(each), old[id(each)]).detach().clone().requires_grad_(True) for each in segments
        }
        distance = sum(float(moved_kl(old[key], now[key]).sum()) for key in now)
        tokens = sum(float(now[key].numel()) for key in now)
        found_terms = preference_terms(objective, batch, now, references)
        for item, each_terms in found_terms:
            tally(sums, each_terms, objective, segments=float(len(segments_of(item))))
        sums["moved"], sums["tokens"] = distance, tokens  # (each segment once, however many items hold it)
        distance /= max(tokens, 1.0)
        if settings.max_kl is not None and distance > settings.max_kl:
            return distance
        total = torch.stack([each.loss for _, each in found_terms]).sum()
        (total / units).backward()
        moving: list[Segment] = []
        for segment in segments:
            gradient = now[id(segment)].grad
            if gradient is not None and bool(gradient.any()):
                moving.append(segment)
        for pack in self._turns(moving):
            if pack is None:
                self._idle(gradient=True)
                continue
            moved: torch.Tensor | None = None
            for segment, logprobs in zip(pack.segments, self._logprobs(pack), strict=True):
                gradient = cast(torch.Tensor, now[id(segment)].grad)
                part = (logprobs * gradient.to(logprobs.dtype)).sum()
                moved = part if moved is None else moved + part
            if moved is not None:
                moved.backward()
        return distance

    def _clipped(self, maximum: float) -> float:
        """The gradient's norm before clipping, after scaling it to at most `maximum` (as `clip_grad_norm_` does):
        the policy's own clip where it has one (a sharded policy's, over every process's shard)."""
        clipping = getattr(self.policy, "clip_gradients", None)
        if clipping is not None:
            return float(cast(SharedPolicy, self.policy).clip_gradients(maximum, self.ranks))
        return float(torch.nn.utils.clip_grad_norm_(self.policy.parameters(), maximum))


def _first_minibatch(plan: Plan, settings: StepSettings) -> list[Item]:
    """The items of a plan's first minibatch, which runs on the weights the step starts from: none for a preference
    loss, whose minibatch computes its logprobs without a gradient before it computes them with one, so that its
    start is a pass however it is computed."""
    if settings.loss.family == PREFERENCE:
        return []
    return (minibatches(plan.items, settings.tokens_per_step) or [[]])[0]


def _started(
    segment: Segment, logprobs: torch.Tensor, old: dict[int, torch.Tensor], behaviors: dict[int, torch.Tensor]
) -> None:
    """Keep a segment's logprobs on the weights the step starts from, and those it was sampled at."""
    old[id(segment)] = logprobs.detach()
    behaviors[id(segment)] = torch.tensor(segment.logprobs).to(logprobs.device)


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
