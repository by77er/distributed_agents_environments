"""Packs: many segments in one row of a model's input, so that one forward and backward pass runs them all.

A pack's row holds runs of tokens, laid out roots first, then branches. A root run is one segment's tokens, or a
prefix that several segments of the pack share token for token. A branch is the rest of one of those segments after
their shared prefix. Each token sees the tokens of its own segment before it and no others: a root's own earlier
tokens; a branch's, and all of its prefix. Positions restart at 0 at each root and go on from the prefix's end in a
branch, so each token has the position it has in its own segment. A policy that runs a pack (`Policy.packed` in
`rollout_lora.policy`) gives each segment's logprobs as one segment alone would have them.

`packs` cuts segments into packs of at most `capacity` tokens: segments sorted by their tokens, neighbours sharing a
prefix of at least `SHARED_PREFIX` tokens grouped under it while the group fits and sharing spares tokens (`grouped`),
then the groups placed first-fit-decreasing by the tokens each puts in the row (`packed`). A step shared among
processes makes the same packs whatever their number, and shares them out (a prefix and its branches are always one
pack's).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from rollout_train.recorder import Segment

__all__ = ["SHARED_PREFIX", "Group", "Pack", "Run", "Scores", "binned", "grouped", "packed", "packs",
           "sampled_positions"]  # fmt: skip

SHARED_PREFIX = 32
"""The fewest tokens a prefix holds for segments to share it in a pack."""


def sampled_positions(segment: Segment) -> list[int]:
    """The positions of the tokens the policy sampled in a segment."""
    return [position for span in segment.spans for position in range(span.start, span.end)]


@dataclass
class Scores:
    """What a policy gives of one segment of a pack."""

    logprobs: torch.Tensor
    """Of its sampled tokens."""
    entropy: torch.Tensor | None = None
    """Of the policy's distribution at each sampled position, where asked for."""
    among: torch.Tensor | None = None
    """The logprobs of given tokens at each sampled position (a row of them for each), where asked for."""


@dataclass(frozen=True)
class Run:
    """Tokens that are consecutive in a pack's row, and in their segment."""

    start: int
    """Where in the row it starts."""
    length: int
    parent: int | None = None
    """For a branch, the index (in `Pack.runs`) of the root whose tokens come before its own; none for a root."""

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclass
class Group:
    """Segments (by index) under a prefix of `shared` tokens they all start with (0 for a segment alone), and the
    tokens the group puts in a row: its prefix once, then each segment's rest."""

    members: list[int]
    shared: int
    tokens: int


@dataclass
class Pack:
    """Some of the segments a step was given (`members`: their indices in what `packs` was given), in one row."""

    members: list[int]
    segments: list[Segment]
    tokens: list[int] = field(default_factory=list[int])
    """The row."""
    positions: list[int] = field(default_factory=list[int])
    """Each row token's position in its segment."""
    runs: list[Run] = field(default_factory=list[Run])
    """The runs that make up the row, in its order: every root, then every branch."""
    places: list[list[int]] = field(default_factory=list[list[int]])
    """For each segment, where in the row each of its tokens is."""

    @property
    def length(self) -> int:
        """Tokens in the row."""
        return len(self.tokens)

    @property
    def segment_tokens(self) -> int:
        """Tokens of its segments, each counted in full (more than `length` where they share prefixes)."""
        return sum(len(each.tokens) for each in self.segments)

    def scored(self, index: int, positions: Sequence[int] | None = None) -> tuple[list[int], list[int]]:
        """For the `index`-th segment: the row of the hidden state before each of `positions` (the sampled tokens'
        by default), and the token at each."""
        segment, places = self.segments[index], self.places[index]
        at = sampled_positions(segment) if positions is None else positions
        if any(position < 1 for position in at):
            raise ValueError("a sampled token is scored from the tokens before it: none is at position 0")
        return [places[position - 1] for position in at], [segment.tokens[position] for position in at]

    @classmethod
    def single(cls, member: int, segment: Segment) -> "Pack":
        """One segment alone."""
        return cls.laid_out([segment], [Group([0], 0, len(segment.tokens))], [member])

    @classmethod
    def laid_out(
        cls, segments: Sequence[Segment], groups: Sequence[Group], numbers: Sequence[int] | None = None
    ) -> "Pack":
        """`groups` of `segments` in a row: each group's root (its prefix, or its one segment), then each shared
        prefix's branches. Its members are the segments' indices in `segments`, or their `numbers`."""
        indices = [index for group in groups for index in group.members]
        pack = cls([index if numbers is None else numbers[index] for index in indices], [segments[i] for i in indices])
        places: dict[int, list[int]] = {}
        roots: list[int] = []
        for group in groups:
            if group.shared == 0:
                (index,) = group.members
                start = pack.length
                roots.append(pack._root(segments[index].tokens))
                places[index] = list(range(start, pack.length))
            else:
                roots.append(pack._root(segments[group.members[0]].tokens[: group.shared]))
        for group, root in zip(groups, roots, strict=True):
            if group.shared == 0:
                continue
            prefix = pack.runs[root]
            for index in group.members:
                rest = segments[index].tokens[group.shared :]
                start = pack.length
                if rest:
                    pack.runs.append(Run(start, len(rest), root))
                    pack.tokens.extend(rest)
                    pack.positions.extend(range(group.shared, group.shared + len(rest)))
                places[index] = [*range(prefix.start, prefix.end), *range(start, start + len(rest))]
        pack.places = [places[index] for index in indices]
        return pack

    def _root(self, tokens: Sequence[int]) -> int:
        self.runs.append(Run(len(self.tokens), len(tokens)))
        self.tokens.extend(tokens)
        self.positions.extend(range(len(tokens)))
        return len(self.runs) - 1


def _common(first: Sequence[int], second: Sequence[int], most: int) -> int:
    """How many leading tokens two sequences share, up to `most`."""
    low, high = 0, min(most, len(first), len(second))  # (by halves: each comparison of slices runs in C)
    while low < high:
        middle = (low + high + 1) // 2
        if first[low:middle] == second[low:middle]:
            low = middle
        else:
            high = middle - 1
    return low


def grouped(
    segments: Sequence[Segment], capacity: int, *, share: bool = True, least: int = SHARED_PREFIX
) -> list[Group]:
    """The segments in groups, each of which a pack holds whole. Sharing (`share`): the segments sorted by their
    tokens, and each joins the group before it, under the prefix they all share, where that prefix is at least `least`
    tokens, the group takes at most `capacity` tokens, and the group with it puts fewer tokens in a row than the group
    and the segment apart (a segment that would cut a long prefix short starts a group of its own); else each segment
    alone."""
    lengths = [len(each.tokens) for each in segments]
    if not share:
        return [Group([index], 0, lengths[index]) for index in range(len(segments))]

    def cost(members: Sequence[int], shared: int) -> int:
        return shared + sum(lengths[each] - shared for each in members)

    order = sorted(range(len(segments)), key=lambda index: segments[index].tokens)
    found: list[Group] = []
    current: Group | None = None
    for index in order:
        if current is not None:
            head = segments[current.members[0]].tokens
            shared = _common(head, segments[index].tokens, current.shared)
            members = [*current.members, index]
            joined = cost(members, shared)
            if shared >= least and joined <= capacity and joined < current.tokens + lengths[index]:
                current = Group(members, shared, joined)
                continue
            found.append(current)
        current = Group([index], lengths[index], lengths[index])
    if current is not None:
        found.append(current)
    return [group if len(group.members) > 1 else Group(group.members, 0, group.tokens) for group in found]


def packed(segments: Sequence[Segment], groups: Sequence[Group], capacity: int) -> list[Pack]:
    """`groups` of `segments` in packs of at most `capacity` tokens (a group longer than that in a pack of its own),
    placed first-fit-decreasing by their tokens (`binned`)."""
    return [Pack.laid_out(segments, each) for each in binned(groups, capacity)]


def binned(groups: Sequence[Group], capacity: int) -> list[list[Group]]:
    """The groups each of `packed`'s packs holds, in its order, without laying out their rows."""
    bins: list[list[Group]] = []
    room: list[int] = []
    for group in sorted(groups, key=lambda each: -each.tokens):  # (stable: ties keep their order)
        for place, left in enumerate(room):
            if group.tokens <= left:
                bins[place].append(group)
                room[place] -= group.tokens
                break
        else:
            bins.append([group])
            room.append(capacity - group.tokens)
    return bins


def packs(segments: Sequence[Segment], capacity: int, *, share: bool = True, least: int = SHARED_PREFIX) -> list[Pack]:
    """`segments` in packs of at most `capacity` tokens, sharing prefixes of at least `least` tokens if `share`
    (`grouped`, then `packed`)."""
    return packed(segments, grouped(segments, capacity, share=share, least=least), capacity)
