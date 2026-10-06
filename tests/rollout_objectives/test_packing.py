# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Packs: segments laid out in rows of at most a pack's tokens, a prefix several share once; and the step over them, on
a toy policy that runs packs: a pack that runs out of memory runs again a segment at a time, and only the segment that
runs out alone is left out."""

import random
from collections.abc import Callable, Sequence

import pytest
import torch
from torch import nn

from rollout_objectives import step
from rollout_objectives.packing import Pack, Scores, grouped, packs, sampled_positions
from rollout_objectives.ranks import Ranks
from rollout_objectives.settings import StepSettings
from rollout_objectives.step import PolicyStep
from rollout_train import Weighted
from rollout_train.recorder import Segment, Span
from rollout_train.trainer import Item, Labelled
from tests.rollout_objectives.shared import nothing_folded


def segment(tokens: list[int], sampled: int = 3) -> Segment:
    return Segment(tokens, [Span(len(tokens) - sampled, len(tokens), 0)], [-1.0] * sampled)


def turns(count: int, seed: int = 0) -> list[Segment]:
    """Turns over one system prompt of 40 tokens, each its own observation and reply."""
    rng = random.Random(seed)
    system = [rng.randrange(100) for _ in range(40)]
    return [segment(system + [rng.randrange(100) for _ in range(rng.randrange(5, 30))]) for _ in range(count)]


def laid_out(pack: Pack) -> None:
    """Each segment's tokens are where its places say, at its own positions; runs follow each other, roots first."""
    for each, places in zip(pack.segments, pack.places, strict=True):
        assert [pack.tokens[place] for place in places] == each.tokens
        assert [pack.positions[place] for place in places] == list(range(len(each.tokens)))
    starts = [run.start for run in pack.runs]
    assert starts == sorted(starts) and starts[0] == 0
    assert all(run.end == after.start for run, after in zip(pack.runs, pack.runs[1:], strict=False))
    assert pack.runs[-1].end == pack.length
    roots = [run.parent is None for run in pack.runs]
    assert roots == sorted(roots, reverse=True)  # (every root, then every branch)
    assert all(pack.runs[run.parent].parent is None for run in pack.runs if run.parent is not None)


def test_segments_are_packed_first_fit_decreasing_into_rows_of_at_most_a_packs_tokens() -> None:
    segments = [segment(list(range(length))) for length in (60, 50, 40, 30, 20, 10, 90)]
    made = packs(segments, 100, share=False)
    assert [[len(each.tokens) for each in pack.segments] for pack in made] == [[90, 10], [60, 40], [50, 30, 20]]
    assert sorted(member for pack in made for member in pack.members) == list(range(7))
    for pack in made:
        laid_out(pack)
        assert pack.length == pack.segment_tokens <= 100
    (alone,) = packs([segment(list(range(150)))], 100)  # (longer than a pack: a pack of its own)
    assert alone.length == 150


def test_segments_that_start_alike_share_their_prefix_once() -> None:
    segments = turns(8)
    made = packs(segments, 10_000)
    (pack,) = made
    laid_out(pack)
    assert pack.length == 40 + sum(len(each.tokens) - 40 for each in segments)
    assert sum(run.parent is None for run in pack.runs) == 1 and len(pack.runs) == 9
    for index, each in enumerate(pack.segments):  # (scored from the hidden state before each sampled token)
        rows, targets = pack.scored(index)
        assert targets == [each.tokens[position] for position in sampled_positions(each)]
        assert rows == [pack.places[index][position - 1] for position in sampled_positions(each)]
    assert all(len(group.members) == 1 for group in grouped(segments, 10_000, least=41))  # (shorter than asked)
    small = packs(segments, 120)  # (a group no larger than a pack)
    assert len(small) > 1 and all(pack.length <= 120 for pack in small)
    for pack in small:
        laid_out(pack)


class PackingToy:
    """A bigram policy that says it runs packs (it scores each segment of a pack alone), and runs out of memory on any
    pack that holds the token `POISON` beside another segment, or alone where `alone_too`."""

    POISON = 7

    def __init__(self, *, alone_too: bool) -> None:
        torch.manual_seed(0)
        self.model = nn.Sequential(nn.Embedding(100, 8), nn.Linear(8, 100))
        self.packing = True
        self.alone_too = alone_too
        self.packs: list[int] = []

    def parameters(self) -> list[nn.Parameter]:
        return list(self.model.parameters())

    def logprobs(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        ids = torch.tensor(list(tokens))
        found = torch.log_softmax(self.model(ids[[p - 1 for p in positions]]), -1)
        return found.gather(-1, ids[list(positions)].unsqueeze(-1)).squeeze(-1)

    def packed(self, pack: Pack, *, entropy: bool = False, candidates: object = None) -> list[Scores]:
        poisoned = any(self.POISON in each.tokens for each in pack.segments)
        if poisoned and not torch.is_grad_enabled() and (len(pack.segments) > 1 or self.alone_too):  # (the start's)
            raise torch.OutOfMemoryError("too long")
        self.packs.append(len(pack.segments))
        return [Scores(self.logprobs(each.tokens, sampled_positions(each))) for each in pack.segments]

    def packed_reference(self, pack: Pack) -> list[torch.Tensor]:
        return [each.logprobs.detach() for each in self.packed(pack)]


def poisoned_batch() -> list[Weighted]:
    segments = [segment([10 + (index * 3 + place) % 50 for place in range(12)]) for index in range(6)]
    segments.append(segment([1, 2, PackingToy.POISON, 3, 4, 5, 6, 8, 9]))
    return [Weighted(each, 1.0 if index % 2 else -1.0) for index, each in enumerate(segments)]


@pytest.mark.parametrize("alone_too", [False, True])
def test_a_pack_out_of_memory_runs_again_a_segment_at_a_time(alone_too: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(step, "_first_minibatch", nothing_folded)  # (every start in the first pass)
    policy = PackingToy(alone_too=alone_too)
    settings = StepSettings(learning_rate=0.05, tokens_per_step=6, max_kl=None, pack_tokens=60)
    metrics = PolicyStep(policy, settings).step(poisoned_batch())  # type: ignore[arg-type]
    assert metrics["start_out_of_memory"] == float(alone_too)  # (left out only where it runs out alone)
    assert metrics["segments"] == 7.0 - alone_too
    assert metrics["packed"] == 1.0 and metrics["packs"] == float(len(policy.packs))
    assert max(policy.packs) > 1 and 0 < metrics["pack_fill"] <= 1.0


def test_a_step_says_how_it_ran_its_segments() -> None:
    policy = PackingToy(alone_too=False)
    batch = [Weighted(each, 1.0) for each in turns(6)]
    settings = StepSettings(
        learning_rate=0.05, tokens_per_step=100, max_kl=None, share_prefixes=False, pack_tokens=1000
    )
    metrics = PolicyStep(policy, settings).step(batch)  # type: ignore[arg-type]
    assert metrics["prefix_shared_fraction"] == 0.0 and metrics["segment_tokens_per_second"] > 0
    # (one minibatch: its start is the minibatch's own pass, folded into it)
    assert metrics["packs"] == 1.0 and policy.packs == [6]


class Recording(PackingToy):
    """A toy that records the segments of every pack it runs, and whether with a gradient (and never runs out of
    memory)."""

    POISON = -1

    def __init__(self) -> None:
        super().__init__(alone_too=False)
        self.ran: list[tuple[bool, tuple[int, ...]]] = []

    def packed(self, pack: Pack, *, entropy: bool = False, candidates: object = None) -> list[Scores]:
        self.ran.append((torch.is_grad_enabled(), tuple(id(each) for each in pack.segments)))
        return super().packed(pack, entropy=entropy, candidates=candidates)


def test_the_start_is_computed_in_the_packs_each_minibatch_makes() -> None:
    """Each minibatch of the first pass but the first (whose start is its own pass) computes its segments in the
    packs the step's start computed them in: on unchanged weights, bfloat16 rounds them alike, and each minibatch's
    ratios are exactly 1."""
    policy = Recording()
    batch: list[Item] = [Weighted(each, 1.0) for each in turns(24, seed=4)]
    settings = StepSettings(learning_rate=0.05, tokens_per_step=12, max_kl=None, pack_tokens=150)
    stepping = PolicyStep(policy, settings)  # type: ignore[arg-type]
    stepping.step(batch)
    assert len(stepping.minibatches) > 2
    starts = [packed for grad, packed in policy.ran if not grad]
    passes = [packed for grad, packed in policy.ran if grad]
    assert starts == passes[len(passes) - len(starts) :] and len(passes) > len(starts)
    assert any(len(packed) > 1 for packed in starts)


class Untouchable(PackingToy):
    """A policy that fails any pass: a step that calls it has computed something."""

    def logprobs(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        raise AssertionError("computed")


@pytest.mark.parametrize(
    ("objective", "batch", "said"),
    [
        ("default", lambda: [Weighted(segment([1, 2, 3, 4]), 1.0), Labelled((segment([5, 6, 7, 8]),), True)],
         "weighted segments, not Labelled"),
        ("dpo", lambda: [Weighted(segment([1, 2, 3, 4]), 1.0)], "pairs or labelled examples"),
        ("default", lambda: [Weighted(Segment([1, 2, 3, 4], [Span(2, 4, 0)], [-0.5, float("nan")]), 1.0)],
         "no behavior logprob"),
    ],
)  # fmt: skip
def test_a_step_says_what_is_wrong_before_it_computes_anything(
    objective: str, batch: Callable[[], list[Item]], said: str
) -> None:
    stepping = PolicyStep(Untouchable(alone_too=False), StepSettings(objective=objective))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=said):
        stepping.step(batch())


def test_a_shared_step_needs_a_sharded_policy() -> None:
    """Before any process waits on another (no process group here: none is reached)."""
    stepping = PolicyStep(BigramOnly(), StepSettings(), ranks=Ranks(0, 2))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sharded policy's `idle`"):
        stepping.step([Weighted(segment([1, 2, 3, 4]), 1.0)])


class BigramOnly:
    def __init__(self) -> None:
        self.model = nn.Sequential(nn.Embedding(100, 8), nn.Linear(8, 100))

    def parameters(self) -> list[nn.Parameter]:
        return list(self.model.parameters())

    def logprobs(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        raise AssertionError("computed")
