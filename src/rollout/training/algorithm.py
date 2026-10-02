"""Group-relative policy optimisation over episodes: what to compare, how much each episode counts, what to train on.

- **Groups.** Episodes of one row from one start are compared with each other (`complete_groups` gathers them from
  a job's stream by a label).
- **Advantages** (Dr. GRPO): an episode's score minus its group's mean, without dividing by the group's spread
  (which favours groups that are nearly solved or nearly hopeless). Every token the policy sampled in the episode
  gets it: with several model slots, every slot's, so a team is rewarded together.
- **Dynamic sampling** (DAPO): a group whose scores are all equal has nothing to teach and is skipped.
- **The fastest of the saturated.** Episodes that reached everything their task has to give earned the same; the
  one that took the least (`info["duration"]`, in the task's own units) played better, and scores a point more.
  The task says what saturated means and how long it took; comparing across the group is done here.
- **What is trained on**: every sequence of the episodes whose advantage is not zero, up to what the trainer can
  afford in a step; beyond that, sequences are taken at even steps through the group, so that each episode and
  slot keeps its share, spread over its whole game.
"""

import random
import statistics
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from rollout.rollouts import Episode
from rollout.training.trainer import Budget, Weighted


async def complete_groups(
    episodes: AsyncIterator[Episode], *, by: str = "group", size: int
) -> AsyncIterator[list[Episode]]:
    """The episodes of a stream, gathered by the label `by`: each group is yielded when `size` of its episodes have
    ended (whatever their outcome)."""
    gathering: dict[str, list[Episode]] = {}
    async for episode in episodes:
        group = gathering.setdefault(episode.labels.get(by, ""), [])
        group.append(episode)
        if len(group) == size:
            yield gathering.pop(episode.labels.get(by, ""))


def group_advantages(scores: Sequence[float]) -> list[float] | None:
    """Each score minus the group's mean; None when all are equal (no signal)."""
    if len(scores) < 2 or max(scores) == min(scores):
        return None
    mean = statistics.fmean(scores)
    return [score - mean for score in scores]


def fastest_of_the_saturated(group: Sequence[Episode]) -> list[float]:
    """A point for the fastest of the episodes that saturated their task, when more than one did; episodes that tie
    for fastest all get it."""
    took = {
        index: float(episode.info.get("duration") or 0.0)  # type: ignore[arg-type]
        for index, episode in enumerate(group)
        if episode.info.get("saturated")
    }
    if len(took) < 2:
        return [0.0] * len(group)
    fastest = min(took.values())
    return [1.0 if took.get(index) == fastest else 0.0 for index in range(len(group))]


def spread[Item](items: Sequence[Item], limit: int | None, rng: random.Random) -> list[Item]:
    """At most `limit` of the items, taken at even steps through them from a random start."""
    if limit is None or len(items) <= limit:
        return list(items)
    step = len(items) / limit
    start = rng.random() * step
    return [items[int(start + index * step)] for index in range(limit)]


@dataclass(frozen=True)
class Grpo:
    group_size: int = 4
    tie_break: bool = True
    """Whether the fastest of a group's saturated episodes scores a point more."""

    def scores(self, group: Sequence[Episode]) -> list[float]:
        """What each episode of a group is compared by: its reward, and the tie-break."""
        bonus = fastest_of_the_saturated(group) if self.tie_break else [0.0] * len(group)
        return [episode.reward + extra for episode, extra in zip(group, bonus, strict=True)]

    def batch(self, group: Sequence[Episode], budget: Budget, rng: random.Random) -> list[Weighted] | None:
        """What to train on from a group of episodes (those fit to train on: completed, and not excluded), or None
        if the group has nothing to teach."""
        advantages = group_advantages(self.scores(group))
        if advantages is None:
            return None
        weighted = [
            Weighted(epoch, advantage)
            for episode, advantage in zip(group, advantages, strict=True)
            if advantage != 0.0
            for trace in episode.traces.values()
            for epoch in trace.epochs
        ]
        return spread(weighted, budget.sequences, rng)
