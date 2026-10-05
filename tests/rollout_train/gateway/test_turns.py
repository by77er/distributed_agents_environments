"""The turn store and segment assembly: turns read back are the turns recorded, and the segments assembled from them
are the segments of the turns as they were sampled, whatever the session did."""

import asyncio
import json
import lzma
import math
import random
from array import array
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import JsonValue

from rollout.contracts import FinishReason, Message, SampleResult, Usage
from rollout_train.gateway import Link, TurnRecord, TurnStore, turns_table, unaccepted
from rollout_train.gateway.turns import (
    COMPRESSION,
    SAMPLE,
    SCORE,
    TURN,
    _Unpacked,  # pyright: ignore[reportPrivateUsage]
)
from rollout_train.inference import Scores
from rollout_train.ledger import Fenced
from rollout_train.recorder.segments import TOKEN_LEVEL, segments_of
from tests.rollout_train.gateway.support import stores


def turn(
    effect_id: str,
    prompt: list[int],
    completion: list[int],
    mask: list[bool],
    *,
    depth: int = 0,
    links: tuple[Link, ...] = (),
    slot: str = "policy",
    sampled_with: tuple[str, ...] = TOKEN_LEVEL,
) -> TurnRecord:
    logprobs = [-(token % 5 + 1) / 4 if sampled else math.nan for token, sampled in zip(completion, mask, strict=True)]
    result = SampleResult(
        message=Message.assistant(f"reply {effect_id}"),
        finish_reason=FinishReason.STOP,
        usage=Usage(context_used=len(prompt) + len(completion), context_limit=4096, input_tokens=len(prompt)),
    )
    return TurnRecord(
        effect_id, "train", "r_1", slot, "policy", f"c{depth}", depth, array("i", prompt), completion, mask, logprobs,
        result, episode="1/1", attempt=1, links=links, sampled_with=sampled_with,
    )  # fmt: skip


def session(seed: int, count: int = 24) -> list[TurnRecord]:
    """Turns a harness might make: contexts that only grow, edited ones (a shortened observation, a compaction),
    exact retries, and thinking closed by force."""
    chance = random.Random(seed)
    system = [chance.randrange(1, 500) for _ in range(chance.randrange(5, 40))]
    turns: list[TurnRecord] = []
    for index in range(count):
        held = [*turns[-1].prompt, *turns[-1].completion] if turns else system
        action = chance.random()
        if not turns:
            prompt = list(system)
        elif action < 0.5:  # the context only grew
            prompt = [*held, *(chance.randrange(1, 500) for _ in range(chance.randrange(1, 30)))]
        elif action < 0.7:  # an observation shortened: the context shares a prefix only
            cut = chance.randrange(len(system), len(held))
            prompt = [*held[:cut], *(chance.randrange(1, 500) for _ in range(chance.randrange(1, 20)))]
        elif action < 0.8:  # a retry of an earlier prompt, exactly
            prompt = list(chance.choice(turns).prompt)
        elif action < 0.9:  # compacted: the system prompt and a summary
            prompt = [*system, *(chance.randrange(1, 500) for _ in range(chance.randrange(1, 10)))]
        else:  # continues an earlier turn, not the latest
            earlier = chance.choice(turns)
            prompt = [*earlier.prompt, *earlier.completion, chance.randrange(1, 500)]
        completion = [chance.randrange(1, 500) for _ in range(chance.randrange(1, 25))]
        mask = [True] * len(completion)
        if chance.random() < 0.3:  # a close forced in the middle
            at = chance.randrange(len(completion))
            mask[at : at + 3] = [False] * len(mask[at : at + 3])
        turns.append(turn(f"r_1:0:{index}", prompt, completion, mask, depth=index // 8))
    return turns


@pytest.mark.parametrize("seed", range(6))
async def test_turns_read_back_are_the_turns_recorded_and_assemble_to_the_same_segments(
    tmp_path: Path, seed: int
) -> None:
    ledger, blobs = stores(tmp_path)
    store = TurnStore(ledger, blobs, chunk_tokens=8)  # (small chunks: prefixes are found at many lengths)
    fence = await ledger.take("runs/train/episodes/1/1")
    turns = session(seed)
    for each in turns:
        await store.record(each, fence)
    back = await store.turns("train", "r_1")
    assert [(each.prompt, each.completion, each.mask, each.result) for each in back] == [
        (each.prompt, each.completion, each.mask, each.result) for each in turns
    ]
    assert [[value for value in each.logprobs if not math.isnan(value)] for each in back] == [
        [value for value in each.logprobs if not math.isnan(value)] for each in turns
    ]
    assert (await store.sessions("train", "r_1"))["policy"] == segments_of(turns)


async def test_a_turn_stores_only_what_it_adds_to_the_turn_it_shares_the_most_with(tmp_path: Path) -> None:
    ledger, blobs = stores(tmp_path)
    store = TurnStore(ledger, blobs, chunk_tokens=4)
    fence = await ledger.take("runs/train/episodes/1/1")
    first = turn("a", list(range(1, 41)), [90, 91], [True, True])
    grown = turn("b", [*range(1, 41), 90, 91, 7, 8], [92], [True])  # continues a
    edited = turn("c", [*range(1, 30), 5, 5, 5], [93], [True])  # shares 29 tokens of b: 28, at the marks
    for each in (first, grown, edited):
        await store.record(each, fence)
    index = await ledger.read(turns_table("train", "r_1"))
    headers = [(await store._unpacked(index[effect])).header for effect in "abc"]  # pyright: ignore[reportPrivateUsage]
    assert [(header["parent"], header["shared"]) for header in headers] == [(None, 0), ("a", 42), ("b", 28)]
    assert [each.prompt for each in await store.turns("train", "r_1")] == [first.prompt, grown.prompt, edited.prompt]


async def test_the_first_turn_recorded_under_an_effect_id_is_the_one_kept(tmp_path: Path) -> None:
    ledger, blobs = stores(tmp_path)
    fence = await ledger.take("runs/train/episodes/1/1")
    one, two = TurnStore(ledger, blobs), TurnStore(ledger, blobs)  # (two replicas)
    first = turn("r_1:0:0", [1, 2, 3], [4], [True])
    second = turn("r_1:0:0", [1, 2, 3], [5, 6], [True, True])  # the same request, sampled again elsewhere
    kept, other = await asyncio.gather(one.record(first, fence), two.record(second, fence))
    assert kept.result == other.result and {kept.replayed, other.replayed} == {False, True}
    (back,) = await one.turns("train", "r_1")
    assert back.completion in ([4], [5, 6]) and back.result == kept.result


async def test_a_turn_under_a_fence_taken_again_is_refused(tmp_path: Path) -> None:
    ledger, blobs = stores(tmp_path)
    store = TurnStore(ledger, blobs)
    stale = await ledger.take("runs/train/episodes/1/1")
    await ledger.take("runs/train/episodes/1/1")  # a newer attempt
    with pytest.raises(Fenced):
        await store.record(turn("r_1:0:0", [1, 2], [3], [True]), stale)
    assert await store.turns("train", "r_1") == []


async def test_a_turn_that_does_not_read_back_as_recorded_is_refused(tmp_path: Path) -> None:
    ledger, blobs = stores(tmp_path)
    store = TurnStore(ledger, blobs, chunk_tokens=4)
    fence = await ledger.take("runs/train/episodes/1/1")
    await store.record(turn("a", [1, 2, 3, 4, 5], [6], [True]), fence)
    other = _Unpacked.of(store._packed(turn("x", [9, 9, 9, 9, 9], [6], [True]), []))  # pyright: ignore[reportPrivateUsage]
    wrong = store._packed(turn("b", [9, 9, 9, 9, 9, 6, 7], [8], [True]), [("a", other)])  # pyright: ignore[reportPrivateUsage]
    reference = await blobs.put(wrong, TURN)  # (it says it shares six tokens with a, which a does not have)
    entry: dict[str, JsonValue] = {"blob": reference.model_dump(mode="json"), "slot": "policy"}
    await ledger.append(turns_table("train", "r_1"), "b", entry, fence)
    with pytest.raises(ValueError, match="does not read back"):
        await store.turns("train", "r_1")


def test_a_compaction_attempt_is_trained_on_only_if_its_harness_went_on_from_it() -> None:
    conversation = turn("t1", [1, 2, 3], [4], [True])
    attempt = turn("t2", [1, 2, 3, 4, 9], [5, 5], [True, True], links=(Link("compaction_attempt", "t1"),))
    abandoned = turn("t3", [1, 2, 3, 4, 9, 9], [6], [True], links=(Link("compaction_attempt", "t1"),))
    resumed = turn("t4", [1, 7, 5, 5], [8], [True], links=(Link("compaction", "t2"), Link("vendor:note", "t1")))
    turns = [conversation, attempt, abandoned, resumed]
    assert unaccepted(turns) == {"t3"}
    every = segments_of(turns)
    accepted = segments_of(turns, untrained=unaccepted(turns))
    assert [len(segment.spans) for segment in every] == [2, 2, 1]  # (t2 and t3 each continue t1)
    assert [[span.effect_id for span in segment.spans] for segment in accepted] == [["t1", "t2"], ["t1"], ["t4"]]
    assert accepted[1].tokens == every[1].tokens  # (what t3 sampled is kept as context)


async def test_a_turn_says_what_it_was_sampled_with_and_a_segment_what_all_its_turns_were(tmp_path: Path) -> None:
    ledger, blobs = stores(tmp_path)
    store = TurnStore(ledger, blobs)
    fence = await ledger.take("runs/train/episodes/1/1")
    exact = turn("a", [1, 2, 3], [4], [True])
    text = turn("b", [1, 2, 3, 4, 5], [6], [True], sampled_with=())  # (continues a, sampled by what returns text)
    alone = turn("c", [7, 8], [9], [True], sampled_with=("token_exact",))
    for each in (exact, text, alone):
        await store.record(each, fence)
    back = await store.turns("train", "r_1")
    assert [each.sampled_with for each in back] == [TOKEN_LEVEL, (), ("token_exact",)]
    segments = (await store.sessions("train", "r_1"))["policy"]
    assert [(each.sampled_with, each.lacks) for each in segments] == [
        ((), ("token_exact", "sampled_logprobs")),  # (what one of its turns lacked, the segment lacks)
        (("token_exact",), ("sampled_logprobs",)),
    ]
    assert segments_of([exact])[0].lacks == ()


async def test_a_turn_recorded_without_saying_what_it_was_sampled_with_was_sampled_token_by_token(
    tmp_path: Path,
) -> None:
    ledger, blobs = stores(tmp_path)
    store = TurnStore(ledger, blobs)
    fence = await ledger.take("runs/train/episodes/1/1")
    recorded = store._packed(turn("a", [1, 2], [3], [True]), [])  # pyright: ignore[reportPrivateUsage]
    body = lzma.decompress(recorded, format=lzma.FORMAT_RAW, filters=COMPRESSION)
    size = int.from_bytes(body[:4], "little")
    header = json.loads(body[4 : 4 + size])
    del header["sampled_with"]  # (as turns were recorded before they said)
    head = json.dumps(header).encode()
    unsaid = len(head).to_bytes(4, "little") + head + body[4 + size :]
    packed = lzma.compress(unsaid, format=lzma.FORMAT_RAW, filters=COMPRESSION)
    reference = await blobs.put(packed, TURN)
    await ledger.append(
        turns_table("train", "r_1"), "a", {"blob": reference.model_dump(mode="json"), "slot": "policy"}, fence
    )
    (back,) = await store.turns("train", "r_1")
    assert back.sampled_with == TOKEN_LEVEL


async def test_a_scoring_turn_reads_back_with_its_scores_and_is_left_out_of_the_segments(tmp_path: Path) -> None:
    ledger, blobs = stores(tmp_path)
    store = TurnStore(ledger, blobs, chunk_tokens=4)
    fence = await ledger.take("runs/train/episodes/1/1")
    sample = turn("a", list(range(1, 11)), [20, 21], [True, True])
    exact = Scores(3, [-0.5, -0.25], [[4, 9, 8], [5, 1, 2]], [[-0.5, -1.0, -2.0], [-0.25, -3.0, -4.0]])
    inexact = Scores(1, [-0.1, -0.2, -1 / 3])  # (not exact in 32 bits, and no most likely tokens)
    teacher = replace(turn("s", [*range(1, 11), 20, 21], [], []), slot="teacher", use=SCORE, scores=exact)
    short = replace(turn("t", [1, 2, 3, 4], [], []), slot="teacher", use=SCORE, scores=inexact)
    grown = turn("b", [*range(1, 11), 20, 21, 30], [31], [True])
    for each in (sample, teacher, short, grown):
        await store.record(each, fence)
    back = await store.turns("train", "r_1")
    assert [(each.use, each.scores) for each in back] == [(SAMPLE, None), (SCORE, exact), (SCORE, inexact),
                                                          (SAMPLE, None)]  # fmt: skip
    assert [list(each.prompt) for each in back] == [list(each.prompt) for each in (sample, teacher, short, grown)]
    assert (await store.sessions("train", "r_1")) == {"policy": segments_of([sample, grown])}
    recorded = await store.reply("train", "r_1", "s")
    assert recorded is not None and recorded.scores == exact
    index = await ledger.read(turns_table("train", "r_1"))
    assert [entry.get("use") for entry in index.values() if isinstance(entry, dict)] == [None, SCORE, SCORE, None]
