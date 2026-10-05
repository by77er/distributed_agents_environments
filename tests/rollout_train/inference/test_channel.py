"""A channel: which engine serves a session, pausing, publishing weights, scoring tokens, and what it counts."""

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from rollout_train.inference import Channel, Generation, Limits, NotLoaded
from rollout_train.inference.channel import most_likely
from rollout_train.testing import scripted_engine

OPTIONS: dict[str, Any] = {"max_tokens": 50, "temperature": 1.0, "top_p": 1.0, "stop_token_ids": [], "adapter": None}


class Slow:
    """An engine that takes a moment, and keeps what it was asked and told."""

    max_model_len = 8192
    processes: Sequence[int] = ()

    def __init__(self, seconds: float = 0.1, tokens: int = 1) -> None:
        self.seconds, self.tokens = seconds, tokens
        self.started: list[int] = []
        self.told: list[str] = []

    async def generate(self, prompt: Sequence[int], **options: Any) -> Generation:
        self.started.append(prompt[0])
        await asyncio.sleep(self.seconds)
        return Generation(tokens=[1] * self.tokens, logprobs=[0.0] * self.tokens, finish_reason="stop")

    async def load_adapter(self, name: str, path: str) -> None:
        self.told.append(f"load {name}")

    async def remove_adapter(self, name: str) -> None:
        self.told.append(f"remove {name}")

    async def sleep(self) -> None:
        self.told.append("sleep")

    async def wake(self) -> None:
        self.told.append("wake")

    def close(self) -> None:
        self.told.append("close")


def channel(*engines: Slow, limits: Limits | None = None) -> Channel:
    return Channel("policy", list(engines), renderer=None, limits=limits or Limits())  # type: ignore[arg-type]


async def test_a_channel_counts_tokens_and_throughput() -> None:
    served = channel(Slow(0.2, tokens=50))
    await asyncio.gather(*(served.generate([0] * 100, **OPTIONS) for _ in range(4)))  # four at once
    counts = served.take()
    assert counts["requests"] == 4 and counts["prompt_tokens"] == 400 and counts["generated_tokens"] == 200
    assert 0.15 < counts["busy_seconds"] < 0.5  # the four overlapped: busy once, not four times
    assert counts["tokens_per_second"] > 3 * counts["tokens_per_second_per_stream"]
    assert 3.0 < counts["mean_concurrency"] <= 4.0
    assert served.take()["requests"] == 0  # taking resets
    running = asyncio.create_task(served.generate([0] * 100, **OPTIONS))
    await asyncio.sleep(0.1)
    assert served.take()["busy_seconds"] >= 0.1  # a channel that never falls idle is busy all the same
    await running


async def test_a_paused_channel_finishes_what_is_in_flight_and_holds_the_rest_back() -> None:
    engine = Slow()
    served = channel(engine)
    first = asyncio.create_task(served.generate([1], **OPTIONS))
    await asyncio.sleep(0.02)
    await served.pause()  # returns once the request in flight is done
    assert first.done() and engine.started == [1]
    second = asyncio.create_task(served.generate([2], **OPTIONS))
    await asyncio.sleep(0.05)
    assert engine.started == [1] and not second.done()  # held back: the engines may sleep now
    await served.sleep()
    await served.wake()
    served.resume()
    await second
    assert engine.started == [1, 2] and engine.told == ["sleep", "wake"]


async def test_a_session_keeps_to_one_engine_and_sessions_spread_over_them() -> None:
    engines = [Slow(0.0), Slow(0.0), Slow(0.0)]
    served = channel(*engines)
    for _ in range(4):
        for session in range(30):
            await served.generate([session], **OPTIONS, session=f"r_{session}/policy")
    assert sum(len(engine.started) for engine in engines) == 120
    assert all(len(engine.started) >= 8 for engine in engines)  # every engine takes a share
    for engine in engines:  # and each session's four turns went to the same one
        assert all(engine.started.count(session) in (0, 4) for session in range(30))


async def test_publishing_loads_the_adapter_everywhere_and_keeps_the_one_before() -> None:
    engines = [Slow(), Slow()]
    served = channel(*engines)
    assert (served.adapter, served.version) == (None, 0)
    assert await served.publish("step-1", "/adapters/step-1") == 1
    assert await served.publish("step-2", "/adapters/step-2") == 2
    assert (served.adapter, served.version) == ("step-2", 2)
    assert engines[0].told == engines[1].told == ["load step-1", "load step-2"]  # a turn in flight ends under step-1
    await served.publish("step-3", "/adapters/step-3")
    assert engines[0].told[-2:] == ["load step-3", "remove step-1"]
    served.close()
    assert engines[1].told[-1] == "close"


def test_a_channel_takes_the_shortest_limit_among_its_engines_and_the_trainer() -> None:
    small, large = Slow(), Slow()
    small.max_model_len = 4096
    assert channel(small, large).context_limit == 4096
    assert channel(large, limits=Limits(sequence=8000)).context_limit == 8000


async def test_a_channel_scores_on_the_sessions_engine_under_what_it_serves_and_counts_tokens_in() -> None:
    engines = [scripted_engine("m"), scripted_engine("m")]
    served = Channel("policy", engines, renderer=None, model="m")  # type: ignore[arg-type]
    tokens = [65, 66, 67, 68, 69]
    scores = await served.score(tokens, start=2, end=4, top=2, adapter=None, session="s")
    assert (scores.start, scores.end, scores.logprobs) == (2, 4, [-0.25, -0.25])
    assert (scores.top_tokens, scores.top_logprobs) == ([[67, 68], [68, 69]], [[-0.25, -1.25], [-0.25, -1.25]])
    asked = [engine for engine in engines if engine.scored]
    assert len(asked) == 1 and asked[0].scored == [([65, 66, 67, 68], 2, 2)]  # (the tokens up to the end, no more)
    counted = served.take()
    assert (counted["requests"], counted["prompt_tokens"], counted["generated_tokens"]) == (1, 4, 0)
    named = await served.scored(tokens, start=1, name=None, session="s")
    assert named.model == "m" and named.top_tokens == [] and len(named.logprobs) == 4
    with pytest.raises(NotLoaded):
        await served.scored(tokens, start=1, name="not-loaded")
    with pytest.raises(ValueError, match="not a range"):
        await served.score(tokens, start=0, adapter=None)  # (the first token has nothing before it)


async def test_a_generation_carries_the_most_likely_tokens_when_asked() -> None:
    served = Channel("policy", [scripted_engine("m")], renderer=None)  # type: ignore[arg-type]
    plain = await served.generate([65], **OPTIONS)
    assert plain.tokens and plain.top_tokens == [] and plain.top_logprobs == []
    asked = await served.generate([65], top=3, **OPTIONS)
    assert asked.top_tokens == [[token, token + 1, token + 2] for token in asked.tokens]
    assert all(values == [-0.5, -1.5, -2.5] for values in asked.top_logprobs)


def test_the_most_likely_tokens_leave_out_the_positions_own_token_when_it_is_one_more() -> None:
    entry = {5: -0.1, 9: -0.7, 2: -3.0}  # (top 2, and the position's own token, 2, outside them)
    assert most_likely(entry, 2, 2) == ([5, 9], [-0.1, -0.7])
    assert most_likely({5: -0.1, 9: -0.7}, 9, 2) == ([5, 9], [-0.1, -0.7])  # (its own token among them)
    assert most_likely({4: -0.2}, 4, 0) == ([], [])
