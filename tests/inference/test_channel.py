"""A channel: which engine serves a session, pausing, publishing weights, and what it counts."""

import asyncio
from collections.abc import Sequence
from typing import Any

from rollout.inference import Channel, Generation, Limits

OPTIONS: dict[str, Any] = {"max_tokens": 50, "temperature": 1.0, "top_p": 1.0, "stop_token_ids": [], "adapter": None}


class Slow:
    """An engine that takes a moment, and keeps what it was asked and told."""

    max_model_len = 8192

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
