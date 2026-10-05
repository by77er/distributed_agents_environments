"""The turn store: every turn the gateway samples, kept before its reply is returned.

A turn is two things:

- **a blob** holding all of it: who sampled it (the run, the episode and attempt, the program's run, the slot, the
  channel), the checkpoint that served it and its depth, the prompt's tokens, the completion's tokens, which of them
  were sampled and which forced, the behaviour logprobs, what it was sampled with (`sampled_with`), whether its slot is
  trained (`trained`), the reply (the parsed message, how it finished, usage), what it cost where its provider is
  metered (`spend`), the links its harness declared to earlier requests, and timings;
- **a ledger record** naming the blob, appended under the turn's effect id to the table of the program's run
  (`runs/RUN/turns/RUN_ID`) and under the fence its key names. The first append wins: a turn sampled twice (a retry
  that reached another replica while the first was still sampling) is recorded once, and both are answered with the
  turn that was recorded.

The ledger record is what makes a turn count: a blob no record names is never read. A turn whose fence was taken again
since its key was minted (a newer attempt of its episode, a runner started again) is refused (`Fenced`).

A turn has a use (`use`): `sample`, a reply sampled from the policy, or `score`, the logprobs a channel gave tokens it
was handed (a teacher scoring a student's tokens). A scoring turn samples nothing: its tokens are its prompt, it has no
completion, and its blob holds the scores. Only samples are trained on.

A turn a hosted API sampled (`sampled_with` empty) has no tokens: its prompt and completion are empty, and what it
keeps is its reply (its text, tool calls and usage) and what it cost (`spend`, also in its ledger record). It exports
no segment, and is never trained on.

**A turn stores only the tokens it adds.** A session's prompts repeat each other: each turn's prompt begins with most
of an earlier turn's tokens (its prompt and what it sampled), all of them where the context only grew, up to the first
edit where it was edited (an observation shortened once it is no longer the current one). A turn names that earlier
turn (its `parent`) and how many of its tokens it begins with (`shared`), and holds only the rest of its prompt. The
turns of a session are a tree, and a segment is a path from a root along turns that share all of their parent's tokens.

The parent is found without reading any tokens back: each turn's blob holds `marks`, digests of its tokens' prefixes
at every `chunk_tokens` tokens and at its end, and a new prompt is compared with the marks of the session's latest
turns. A turn read back is checked against its own last mark. Every blob is compressed, and logprobs are kept as 32-bit
floats where that loses nothing (as engines produce them), else as 64-bit ones.
"""

import asyncio
import hashlib
import json
import lzma
import math
import re
import sys
import time
from array import array
from collections import OrderedDict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import JsonValue, TypeAdapter

from rollout.contracts import BlobReference, SampleResult, SessionIdentity
from rollout.harness.blobs import Blobs
from rollout_train.inference.channel import Scores
from rollout_train.ledger import Fence, Ledger
from rollout_train.recorder.segments import TOKEN_LEVEL, Segment, segments_of

TURNS = "turns"
"""A run's tables of turns: `runs/RUN/turns/RUN_ID`, one per program's run."""
CHUNK_TOKENS = 512
"""Tokens between a turn's marks."""
LOOKED_AT = 3
"""How many of a session's latest turns a new turn is compared with, to find its parent."""
TURN = "application/x-rollout-turn"
"""A turn's blob: a header (JSON), then its arrays (little-endian), compressed with LZMA2 (`COMPRESSION`, raw)."""
COMPRESSION = [{"id": lzma.FILTER_LZMA2, "preset": 6}]
FORMAT = 1
MARK = 16
SAMPLE, SCORE = "sample", "score"
"""A turn's uses: a reply sampled from the policy, and the logprobs of given tokens."""
"""Bytes of a mark: a BLAKE2b digest."""
COMPACTION_ATTEMPT, COMPACTION = "compaction_attempt", "compaction"
"""Links a harness declares when it compacts: from the request it compacts to the request that asks for the summary
(an attempt), and from that attempt to the request that goes on from the summary (the attempt was accepted)."""
LINK_TYPE = re.compile(r"[A-Za-z][A-Za-z0-9._:-]{0,127}")
_BLOB = TypeAdapter(BlobReference)


def turns_table(run: str, run_id: str) -> str:
    """The table of a program's run's turns."""
    return f"runs/{run}/{TURNS}/{run_id}"


@dataclass(frozen=True)
class Link:
    """What a harness said of a request: that it follows from an earlier one (`source`, by its request id: its
    effect id), and how (`type`: `compaction_attempt`, `compaction`, `subagent_call`, `subagent_return`, or any other
    label, which is kept as it is)."""

    type: str
    source: str

    def __post_init__(self) -> None:
        if not LINK_TYPE.fullmatch(self.type) or not self.source:
            raise ValueError(f"a link is a type (a letter, then letters, digits, `._:-`) and a source: {self}")


@dataclass(frozen=True)
class TurnRecord:
    """One turn, as the gateway sampled and recorded it."""

    effect_id: str
    """Its request id."""
    run: str
    run_id: str
    slot: str
    channel: str
    checkpoint: str
    """What served it, by the name its endpoint answered with (a checkpoint's id, or the base model's name)."""
    depth: int
    """The checkpoint's depth: the version every sampled token is stamped with."""
    prompt: "array[int]"
    completion: list[int]
    mask: list[bool]
    """True where the policy sampled the token; False where it was forced."""
    logprobs: list[float]
    """Behaviour logprobs of every completion token (forced ones: NaN)."""
    result: SampleResult
    """The reply: the parsed message, how it finished, usage."""
    episode: str = ""
    """`GROUP/EPISODE`, for an attempt of an episode a run asked for."""
    attempt: int = 0
    links: tuple[Link, ...] = ()
    timings: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """`started` (seconds since the epoch), `phases` (seconds each generation took) and `seconds` (the whole turn)."""
    sampled_with: tuple[str, ...] = TOKEN_LEVEL
    """What it was sampled with, of `TOKEN_LEVEL`: what its sampler could do (a turn recorded without saying was
    sampled with all of them)."""
    use: str = SAMPLE
    """`sample`, or `score`: the logprobs the channel gave the prompt's tokens (`scores`), with nothing sampled."""
    scores: Scores | None = None
    """A scoring turn's scores."""
    trained: bool = True
    """Whether it may be trained on: false for a turn of a slot that is not trained (a judge, a fixed opponent), and
    for a turn a hosted API sampled."""
    spend: float | None = None
    """Dollars it cost, from its usage at its model's catalog prices, where its provider is metered and prices it."""

    @property
    def version(self) -> int:
        return self.depth

    @property
    def session_id(self) -> str:
        return str(SessionIdentity(self.run_id, self.slot))


@dataclass(frozen=True)
class Reply:
    """What a recorded turn answered, and who it answered."""

    effect_id: str
    slot: str
    checkpoint: str
    depth: int
    result: SampleResult
    replayed: bool = True
    """Whether it was recorded before (False: by the call that returned it)."""
    timings: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """The turn's timings (`TurnRecord.timings`), as recorded."""
    scores: Scores | None = None
    """A scoring turn's scores (`TurnRecord.scores`)."""


class TurnStore:
    """Turns in a ledger and a blob store. It keeps nothing that correctness depends on: the blobs it read are a
    cache."""

    def __init__(self, ledger: Ledger, blobs: Blobs, *, chunk_tokens: int = CHUNK_TOKENS, cached: int = 4096) -> None:
        self.ledger = ledger
        self.blobs = blobs
        self.chunk_tokens = chunk_tokens
        self._cached = cached
        self._read: OrderedDict[str, _Unpacked] = OrderedDict()
        """Turns' blobs read, by digest."""

    async def index(self, run: str, run_id: str) -> dict[str, JsonValue]:
        """The ledger's records of a program's run's turns, by effect id, in the order they were recorded."""
        return await self.ledger.read(turns_table(run, run_id))

    async def reply(
        self, run: str, run_id: str, effect_id: str, index: Mapping[str, JsonValue] | None = None
    ) -> Reply | None:
        """What the turn recorded under an effect id answered, if there is one (`index`: the run's records, if read)."""
        entry = (index if index is not None else await self.index(run, run_id)).get(effect_id)
        if entry is None:
            return None
        unpacked = await self._unpacked(entry)
        header = unpacked.header
        return Reply(
            effect_id,
            str(header["slot"]),
            str(header["checkpoint"]),
            int(header["depth"]),
            SampleResult.model_validate(header["result"]),
            timings=header.get("timings", {}),
            scores=unpacked.scores,
        )

    async def record(self, turn: TurnRecord, fence: Fence, index: Mapping[str, JsonValue] | None = None) -> Reply:
        """Keep a turn, unless one was recorded under its effect id first; returns what the turn recorded answers
        (`replayed` if it was not this one). `index` is the run's records, if they were read. Raises `Fenced` if
        `fence` was taken again."""
        index = index if index is not None else await self.index(turn.run, turn.run_id)
        mine = [(effect, entry) for effect, entry in index.items() if _of_slot(entry, turn.slot)][-LOOKED_AT:]
        earlier = await asyncio.gather(*(self._unpacked(entry) for _, entry in mine))
        packed = await asyncio.to_thread(
            self._packed, turn, [(effect, each) for (effect, _), each in zip(mine, earlier, strict=True)]
        )
        reference = await self.blobs.put(packed, TURN)
        entry: dict[str, JsonValue] = {
            "blob": _BLOB.dump_python(reference, mode="json"),
            "slot": turn.slot,
            "episode": turn.episode,
            "attempt": turn.attempt,
            "checkpoint": turn.checkpoint,
            "depth": turn.depth,
            "prompt": len(turn.prompt),
            "sampled": sum(turn.mask),
            "at": round(time.time(), 3),
            **({} if turn.trained else {"trained": False}),
            **({"spend": round(turn.spend, 8)} if turn.spend is not None else {}),
        }
        if turn.use != SAMPLE:
            entry["use"] = turn.use
        if await self.ledger.append(turns_table(turn.run, turn.run_id), turn.effect_id, entry, fence):
            return Reply(
                turn.effect_id, turn.slot, turn.checkpoint, turn.depth, turn.result, replayed=False, scores=turn.scores
            )
        found = await self.reply(turn.run, turn.run_id, turn.effect_id)  # (ours, from an append retried, or another's)
        assert found is not None, "an append refused for its key leaves the key there"
        return found

    async def turns(self, run: str, run_id: str) -> list[TurnRecord]:
        """A program's run's turns, in the order they were recorded."""
        index = await self.index(run, run_id)
        unpacked = await asyncio.gather(*(self._unpacked(entry) for entry in index.values()))
        held: dict[str, array[int]] = {}  # each turn's tokens: its prompt and its completion
        turns: list[TurnRecord] = []
        for effect, each in zip(index, unpacked, strict=True):
            header = each.header
            prompt = array("i")
            if (parent := header.get("parent")) is not None:
                if parent not in held:
                    raise ValueError(f"turn {effect} continues {parent}, which its run did not record before it")
                prompt.extend(held[parent][: int(header["shared"])])
            prompt.extend(each.suffix)
            tokens = array("i", prompt)
            tokens.extend(each.completion)
            if _digest(_bytes(tokens)) != each.marks[-MARK:]:
                raise ValueError(f"turn {effect} does not read back as it was recorded")
            held[effect] = tokens
            forced = [False] * len(each.completion)
            for start, end in header["forced"]:
                forced[start:end] = [True] * (end - start)
            values = iter(each.sampled)
            turns.append(
                TurnRecord(
                    effect_id=effect,
                    run=str(header["run"]),
                    run_id=str(header["run_id"]),
                    slot=str(header["slot"]),
                    channel=str(header["channel"]),
                    checkpoint=str(header["checkpoint"]),
                    depth=int(header["depth"]),
                    prompt=prompt,
                    completion=list(each.completion),
                    mask=[not flag for flag in forced],
                    logprobs=[math.nan if flag else next(values) for flag in forced],
                    result=SampleResult.model_validate(header["result"]),
                    episode=str(header["episode"]),
                    attempt=int(header["attempt"]),
                    links=tuple(Link(str(link["type"]), str(link["source"])) for link in header["links"]),
                    timings=header["timings"],
                    sampled_with=tuple(header.get("sampled_with", TOKEN_LEVEL)),
                    use=str(header.get("use", SAMPLE)),
                    scores=each.scores,
                    trained=bool(header.get("trained", True)),
                    spend=header.get("spend"),
                )
            )
        return turns

    async def sessions(self, run: str, run_id: str, *, accepted_only: bool = False) -> dict[str, list[Segment]]:
        """What each model slot of a program's run exports, by slot (`segments_of` its samples: scoring turns are left
        out, and so are a hosted API's, which hold no tokens). The segments of a slot that is not trained are kept,
        marked so (`Segment.trained`). With `accepted_only`, what a compaction attempt sampled is trained on only if
        its harness went on from it."""
        by_slot: dict[str, list[TurnRecord]] = {}
        turns = [
            turn for turn in await self.turns(run, run_id) if turn.use == SAMPLE and (turn.prompt or turn.completion)
        ]
        for turn in turns:
            by_slot.setdefault(turn.slot, []).append(turn)
        untrained = unaccepted(turns) if accepted_only else set[str]()
        sessions: dict[str, list[Segment]] = {}
        for slot, each in by_slot.items():
            segments = segments_of(each, untrained=untrained)
            trained = all(turn.trained for turn in each)
            sessions[slot] = segments if trained else [replace(segment, trained=False) for segment in segments]
        return sessions

    def _packed(self, turn: TurnRecord, earlier: Sequence[tuple[str, "_Unpacked"]]) -> bytes:
        """A turn's blob, its prompt stored after what it shares with the earlier turn it shares the most with."""
        size = self.chunk_tokens
        tokens = array("i", turn.prompt)
        tokens.extend(turn.completion)
        data = _bytes(tokens)
        ends = {each.length for _, each in earlier if each.length <= len(turn.prompt)}
        wanted = sorted({*range(size, len(tokens) + 1, size), len(tokens), *ends})
        prefixes = _prefix_digests(data, wanted)
        marks = b"".join(prefixes[at] for at in [*range(size, len(tokens), size), len(tokens)])
        parent, shared = None, 0
        for effect, each in reversed(
            earlier
        ):  # (the latest first: of two that share as much, the latest is the parent)
            for at, mark in each.prefixes(size).items():
                if shared < at <= len(turn.prompt) and prefixes.get(at) == mark:
                    parent, shared = effect, at
        sampled = [value for value, kept in zip(turn.logprobs, turn.mask, strict=True) if kept]
        narrow = array("f", sampled)
        kind = "f" if array("d", narrow).tobytes() == array("d", sampled).tobytes() else "d"
        header: dict[str, Any] = {
            "format": FORMAT,
            "effect_id": turn.effect_id,
            "run": turn.run,
            "run_id": turn.run_id,
            "slot": turn.slot,
            "channel": turn.channel,
            "episode": turn.episode,
            "attempt": turn.attempt,
            "checkpoint": turn.checkpoint,
            "depth": turn.depth,
            "parent": parent,
            "shared": shared,
            "prompt": len(turn.prompt),
            "completion": len(turn.completion),
            "forced": _stretches(turn.mask, False),
            "logprobs": kind,
            "marks": len(marks) // MARK,
            "result": turn.result.model_dump(mode="json"),
            "links": [{"type": link.type, "source": link.source} for link in turn.links],
            "timings": dict(turn.timings),
            "sampled_with": list(turn.sampled_with),
            "trained": turn.trained,
        }
        if turn.spend is not None:
            header["spend"] = turn.spend
        if turn.use != SAMPLE:
            header["use"] = turn.use
        scored: list[bytes] = []
        if (scores := turn.scores) is not None:
            values = [*scores.logprobs, *(value for each in scores.top_logprobs for value in each)]
            narrow_scores = array("f", values)
            exact = array("d", narrow_scores).tobytes() == array("d", values).tobytes()
            top = len(scores.top_tokens[0]) if scores.top_tokens else 0
            header["scores"] = {"start": scores.start, "count": len(scores.logprobs), "top": top,
                                "logprobs": "f" if exact else "d"}  # fmt: skip
            top_tokens = array("i", (token for each in scores.top_tokens for token in each))
            scored = [_bytes(narrow_scores if exact else array("d", values)), _bytes(top_tokens)]
        head = json.dumps(header, separators=(",", ":")).encode()
        body = [
            len(head).to_bytes(4, "little"),
            head,
            data[shared * 4 : len(turn.prompt) * 4],  # the prompt after what it shares, then the completion
            data[len(turn.prompt) * 4 :],
            _bytes(narrow if kind == "f" else array("d", sampled)),
            marks,
            *scored,  # a scoring turn's logprobs, its most likely tokens' logprobs, and those tokens
        ]
        return lzma.compress(b"".join(body), format=lzma.FORMAT_RAW, filters=COMPRESSION)

    async def _unpacked(self, entry: JsonValue) -> "_Unpacked":
        assert isinstance(entry, dict)
        reference = _BLOB.validate_python(entry["blob"])
        known = self._read.get(reference.sha256)
        if known is not None:
            self._read.move_to_end(reference.sha256)
            return known
        packed = await self.blobs.read(reference)
        unpacked = await asyncio.to_thread(_Unpacked.of, packed)
        self._read[reference.sha256] = unpacked
        while len(self._read) > self._cached:
            self._read.popitem(last=False)
        return unpacked


def _of_slot(entry: JsonValue, slot: str) -> bool:
    return isinstance(entry, dict) and entry.get("slot") == slot


def unaccepted(turns: Sequence[TurnRecord]) -> set[str]:
    """The compaction attempts no turn went on from (by effect id): a request linked from an earlier one as its
    `compaction_attempt`, and to no later one as the source of a `compaction`."""
    attempts = {turn.effect_id for turn in turns if any(link.type == COMPACTION_ATTEMPT for link in turn.links)}
    accepted = {link.source for turn in turns for link in turn.links if link.type == COMPACTION}
    return attempts - accepted


@dataclass(frozen=True)
class _Unpacked:
    """A turn's blob, read: its header, the arrays it holds, and its marks."""

    header: Mapping[str, Any]
    suffix: "array[int]"
    """Its prompt after what it shares with its parent."""
    completion: "array[int]"
    sampled: "array[float]"
    """The logprobs of the sampled tokens."""
    marks: bytes
    scores: Scores | None = None
    """A scoring turn's scores."""

    @property
    def length(self) -> int:
        """How many tokens it holds: its prompt and its completion."""
        return int(self.header["prompt"]) + int(self.header["completion"])

    def prefixes(self, size: int) -> dict[int, bytes]:
        """Its marks, by the length of the prefix each is the digest of."""
        count = len(self.marks) // MARK
        lengths = [*[*range(size, self.length, size)][: count - 1], self.length]
        return {at: self.marks[index * MARK : (index + 1) * MARK] for index, at in enumerate(lengths)}

    @classmethod
    def of(cls, packed: bytes) -> "_Unpacked":
        body = lzma.decompress(packed, format=lzma.FORMAT_RAW, filters=COMPRESSION)
        size = int.from_bytes(body[:4], "little")
        header: dict[str, Any] = json.loads(body[4 : 4 + size])
        at = 4 + size
        suffix, at = _array("i", body, at, int(header["prompt"]) - int(header["shared"]))
        completion, at = _array("i", body, at, int(header["completion"]))
        sampled_count = int(header["completion"]) - sum(end - start for start, end in header["forced"])
        sampled, at = _array(str(header["logprobs"]), body, at, sampled_count)
        marks, at = body[at : at + int(header["marks"]) * MARK], at + int(header["marks"]) * MARK
        return cls(header, suffix, completion, sampled, marks, _scores(header.get("scores"), body, at))


def _scores(said: Any, data: bytes, at: int) -> Scores | None:
    """A scoring turn's scores, read from its blob at `at` as its header says (`said`: none for a sample)."""
    if said is None:
        return None
    count, top = int(said["count"]), int(said["top"])
    values, at = _array(str(said["logprobs"]), data, at, count * (1 + top))
    tokens, _ = _array("i", data, at, count * top)
    rows = range(count) if top else range(0)
    return Scores(
        int(said["start"]),
        list(values[:count]),
        top_tokens=[list(tokens[row * top : (row + 1) * top]) for row in rows],
        top_logprobs=[list(values[count + row * top : count + (row + 1) * top]) for row in rows],
    )


def _prefix_digests(data: bytes, lengths: Collection[int]) -> dict[int, bytes]:
    """The digests of the prefixes of `data` (tokens, 4 bytes each) that are `lengths` tokens long."""
    hashing = hashlib.blake2b(digest_size=MARK)
    found: dict[int, bytes] = {}
    done = 0
    for length in sorted(lengths):
        hashing.update(data[done * 4 : length * 4])
        done = length
        found[length] = hashing.copy().digest()
    return found


def _digest(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=MARK).digest()


def _stretches(mask: Sequence[bool], value: bool) -> list[list[int]]:
    """The `[start, end)` stretches of `mask` that are `value`."""
    found: list[list[int]] = []
    start: int | None = None
    for index, each in enumerate([*mask, not value]):
        if each == value and start is None:
            start = index
        elif each != value and start is not None:
            found.append([start, index])
            start = None
    return found


def _bytes(values: "array[Any]") -> bytes:
    """An array's bytes, little-endian."""
    if sys.byteorder == "big":
        values = array(values.typecode, values)
        values.byteswap()
    return values.tobytes()


def _array(kind: str, data: bytes, at: int, count: int) -> "tuple[array[Any], int]":
    """`count` values of type `kind` read from `data` at `at` (little-endian), and where they end."""
    values: array[Any] = array(kind)
    end = at + count * values.itemsize
    values.frombytes(data[at:end])
    if sys.byteorder == "big":
        values.byteswap()
    return values, end
