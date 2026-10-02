"""The training driver's bookkeeping: overlapping groups, the speed bonus, the turns of an update (no engine, no
server)."""

import asyncio
import random
from pathlib import Path
from typing import Any

from minecraft_swarm.train import Flight, last_iteration, speed_bonus, spread


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


def test_the_fastest_of_the_episodes_that_saturated_the_task_scores_a_point_more() -> None:
    def result(ended: str, minutes: float, turns: int) -> dict[str, Any]:
        return {"ended": ended, "game_minutes": minutes, "turns": turns}

    full = "nothing left to earn"
    assert speed_bonus([result(full, 0.4, 4), result(full, 0.1, 2), result(full, 0.1, 3)]) == [0.0, 1.0, 0.0]
    assert speed_bonus([result(full, 0.2, 3), result("turns", 0.1, 2), result(full, 0.2, 3)]) == [1.0, 0.0, 1.0]
    # One episode alone at the top already scores more than the rest; and an unfinished game is not a fast one.
    assert speed_bonus([result(full, 0.2, 3), result("turns", 3.0, 36)]) == [0.0, 0.0]
    assert speed_bonus([result("game time", 3.0, 30), result("turns", 3.0, 36)]) == [0.0, 0.0]
    assert speed_bonus([]) == []


def test_an_update_takes_an_even_share_of_every_episodes_turns() -> None:
    turns = [(episode, index) for episode, count in (("a", 300), ("b", 100), ("c", 40)) for index in range(count)]
    assert spread(turns, 500, random.Random(0)) == turns  # all of them, when they fit
    chosen = spread(turns, 110, random.Random(0))
    assert len(chosen) == len(set(chosen)) == 110
    share = {episode: sum(1 for name, _ in chosen if name == episode) for episode in "abc"}
    assert share == {"a": 75, "b": 25, "c": 10}
    first = [index for name, index in chosen if name == "a"]
    assert first == sorted(first) and first[0] < 4 and first[-1] > 295  # over the whole game, start to end


def test_a_run_goes_on_after_the_last_iteration_it_logged(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    assert last_iteration(metrics) == 0
    metrics.write_text('{"iteration": 1}\n{"iteration": 3}\n{"iteration": 2}\n')  # groups overlap: 3 ended before 2
    assert last_iteration(metrics) == 3
