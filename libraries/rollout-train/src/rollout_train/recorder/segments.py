"""What a session's turns export: segments, joined by prefix-continuation.

A turn whose prompt begins with everything an earlier turn held (its prompt and what it sampled) continues that turn's
segment; a turn that repeats an earlier prompt exactly replaces it. `segments_of` applies the rule to turns in the
order they were sampled. It is a pure function of the turns, so whoever has them (the recorder in memory, the gateway's
turn store read back) exports the same segments.
"""

from collections.abc import Collection, Sequence
from typing import Protocol

from rollout_train.recorder.recorder import Segment, Span


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
        segments.append(Segment([*turn.prompt, *turn.completion], spans, logprobs, turn.channel))
    return [segment for index, segment in enumerate(segments) if index not in dropped and segment.spans]
