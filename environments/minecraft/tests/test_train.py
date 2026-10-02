"""The training driver's bookkeeping of overlapping groups (no engine, no server)."""

import asyncio

from minecraft_swarm.train import Flight


class Episode:
    """Stands in for a run handle: ends when told to."""

    def __init__(self) -> None:
        self.ended = asyncio.Event()

    async def result(self) -> None:
        await self.ended.wait()


async def test_the_next_group_starts_when_one_straggler_is_left() -> None:
    flight = Flight()
    episodes = [Episode() for _ in range(4)]
    followers: list[asyncio.Task[None]] = []
    for episode in episodes:
        flight.took_off()
        followers.append(asyncio.create_task(flight.follow(episode)))  # type: ignore[arg-type]
    room = asyncio.create_task(flight.at_most(1))
    for episode in episodes[:2]:
        episode.ended.set()
    await asyncio.sleep(0.01)
    assert flight.count == 2 and not room.done()  # two still running: the next group waits
    episodes[2].ended.set()
    await asyncio.wait_for(room, 1)  # one straggler left: go
    assert flight.count == 1
    episodes[3].ended.set()
    await asyncio.wait_for(flight.at_most(0), 1)
    await asyncio.gather(*followers)
