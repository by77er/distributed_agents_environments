"""What a session's turns export: segments, joined by prefix-continuation.

A `Segment` is a token sequence with the spans the policy sampled (`Span`), their behaviour logprobs and the version of
the weights that sampled them (the served checkpoint's depth). A turn whose prompt begins with everything an earlier
turn held (its prompt and what it sampled) continues that turn's segment: an append-only conversation is one segment,
however many turns it has, and is trained in one pass. A turn whose context was edited (a compaction, a chat template
that drops earlier thinking, an observation replaced by a shorter form) begins a new segment. A turn that repeats an
earlier prompt exactly (a client's retry) replaces it.

A segment also says what its turns were sampled with (`sampled_with`: the capabilities every one of them had, of
`TOKEN_LEVEL`). A segment is trained on with an importance weight only if its tokens are the exact ones sampled and
their behaviour logprobs are known (`BEHAVIOUR`).

`segments_of` applies the rule to turns in the order they were sampled. It is a pure function of the turns: the
gateway's turn store reads a session's turns back and exports them with it.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Protocol

TOKEN_LEVEL = ("token_exact", "sampled_logprobs", "honours_sampling")
"""What a turn can be sampled with: its tokens are the exact ones sampled, each sampled token's behaviour logprob is
known, and temperature and top-p were applied (`rollout_train.providers.Capabilities`). vLLM and Tinker sample with
all three, and a turn that does not say what it was sampled with was sampled by one of them."""
BEHAVIOUR = ("token_exact", "sampled_logprobs")
"""What a segment's turns must have been sampled with for it to be trained on with an importance weight."""


@dataclass(frozen=True)
class Span:
    """Tokens `start` to `end` (exclusive) of a segment were sampled by the policy, at weights `version` (the depth
    of the checkpoint served then)."""

    start: int
    end: int
    version: int
    effect_id: str = ""
    """The sample that produced them: the `effect_id` its run's events know it by."""


@dataclass(frozen=True)
class Segment:
    """A piece of a session's trajectory: tokens that only grew, as the policy saw and continued them."""

    tokens: list[int]
    spans: list[Span]
    logprobs: list[float]
    """Behavior logprobs of the tokens inside the spans, in order."""
    channel: str = ""
    """The channel that sampled them. The spans' `version`s are the depths of the checkpoints it served."""
    sampled_with: tuple[str, ...] = TOKEN_LEVEL
    """What every one of its turns was sampled with, of `TOKEN_LEVEL`."""

    @property
    def sampled(self) -> int:
        return sum(span.end - span.start for span in self.spans)

    @property
    def lacks(self) -> tuple[str, ...]:
        """What its turns were sampled without, of `BEHAVIOUR` (empty: it can be trained on with an importance
        weight)."""
        return tuple(each for each in BEHAVIOUR if each not in self.sampled_with)


class Turn(Protocol):
    """One sample of a session, as segments are built from it."""

    @property
    def effect_id(self) -> str: ...
    @property
    def prompt(self) -> Sequence[int]: ...
    @property
    def completion(self) -> Sequence[int]: ...
    @property
    def mask(self) -> Sequence[bool]:
        """True where the policy sampled the token; False where it was forced."""
        ...

    @property
    def logprobs(self) -> Sequence[float]:
        """Of every completion token (forced ones: NaN)."""
        ...

    @property
    def version(self) -> int:
        """The depth of the checkpoint that sampled it."""
        ...

    @property
    def channel(self) -> str: ...

    @property
    def sampled_with(self) -> Sequence[str]:
        """What it was sampled with, of `TOKEN_LEVEL`."""
        ...


def segments_of(turns: Sequence[Turn], *, untrained: Collection[str] = ()) -> list[Segment]:
    """The segments of a session's turns, oldest first, each with at least one sampled span. What the turns named in
    `untrained` (by effect id) sampled is kept as context, with no span: it is not trained on."""
    segments: list[Segment] = []
    dropped: set[int] = set()  # continued by a later turn, or retried
    for index, turn in enumerate(turns):
        parent: Segment | None = None
        for earlier in range(index - 1, -1, -1):
            if turns[earlier].prompt == turn.prompt:
                dropped.add(earlier)
            held = segments[earlier].tokens
            size = len(held)
            if size <= len(turn.prompt) and turn.prompt[size - 1] == held[-1] and list(turn.prompt[:size]) == held:
                parent = segments[earlier]
                dropped.add(earlier)
                break
        spans = list(parent.spans) if parent else []
        mask = [False] * len(turn.mask) if turn.effect_id in untrained else turn.mask
        start: int | None = None
        for offset, sampled in enumerate([*mask, False]):  # (the False closes a span that runs to the end)
            position = len(turn.prompt) + offset
            if sampled and start is None:
                start = position
            elif not sampled and start is not None:
                spans.append(Span(start, position, turn.version, turn.effect_id))
                start = None
        logprobs = list(parent.logprobs) if parent else []
        logprobs += [value for value, sampled in zip(turn.logprobs, mask, strict=True) if sampled]
        held_with = parent.sampled_with if parent else TOKEN_LEVEL
        sampled_with = tuple(each for each in held_with if each in turn.sampled_with)
        segments.append(Segment([*turn.prompt, *turn.completion], spans, logprobs, turn.channel, sampled_with))
    return [segment for index, segment in enumerate(segments) if index not in dropped and segment.spans]
