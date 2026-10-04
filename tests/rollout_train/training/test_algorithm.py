"""The group algorithm on made-up episodes: advantages, the tie-break and what a step trains on."""

import random
from typing import Any

from rollout_train import Budget, Grpo, group_advantages
from rollout_train.algorithm import fastest_of_the_saturated, spread
from rollout_train.recorder import TOKEN_LEVEL, Segment, Span
from rollout_train.rollouts import Episode, Outcome, Trajectory


def episode(
    reward: float, *, segments: int = 2, number: int = 1, sampled_with: tuple[str, ...] = TOKEN_LEVEL, **info: Any
) -> Episode:
    """An episode of two model slots that were rewarded together, each with `segments` segments."""
    made = [Segment([1, 2, 3], [Span(1, 3, 0)], [-0.5, -0.5], "policy", sampled_with) for _ in range(segments)]
    trajectories = {slot: Trajectory(list(made), {"default": reward}) for slot in ("ada", "ben")}
    return Episode("train", 1, number, f"r{number}", {}, Outcome.COMPLETED, info=info, trajectories=trajectories)


def test_advantages_are_centered_but_not_scaled() -> None:
    assert group_advantages([0, 0, 3, 1]) == [-1.0, -1.0, 2.0, 0.0]
    assert group_advantages([2, 2, 2, 2]) is None  # no signal: skipped
    assert group_advantages([5]) is None


def test_the_fastest_of_the_episodes_that_saturated_the_task_scores_a_point_more() -> None:
    def full(duration: float) -> Episode:
        return episode(5.0, saturated=True, duration=duration)

    assert fastest_of_the_saturated([full(0.4), full(0.1), full(0.3)]) == [0.0, 1.0, 0.0]
    assert fastest_of_the_saturated([full(0.2), episode(5.0, duration=0.1), full(0.2)]) == [1.0, 0.0, 1.0]
    # One episode alone at the top already scores more than the rest; and an unfinished game is not a fast one.
    assert fastest_of_the_saturated([full(0.2), episode(1.0, duration=3.0)]) == [0.0, 0.0]
    assert fastest_of_the_saturated([]) == []
    # An episode that does not say how long it took is not compared (it is not the fastest for saying nothing).
    assert fastest_of_the_saturated([full(0.2), episode(5.0, saturated=True), full(0.3)]) == [1.0, 0.0, 0.0]
    group = [full(0.4), full(0.1), full(0.3)]
    tied = Grpo().batch(group, Budget(), random.Random(0))
    assert tied.notes == {"speed_bonus": [0.0, 1.0, 0.0]} and len(tied.segments) == 12
    assert sorted({round(weighted.advantage, 3) for weighted in tied.segments}) == [-0.333, 0.667]
    untied = Grpo(tie_break=False).batch(group, Budget(), random.Random(0))  # nothing to compare them by
    assert not untied.segments and untied.skipped == "every episode scored the same"


def test_a_step_trains_on_every_slots_sequences_of_the_episodes_that_differ_from_the_mean() -> None:
    group = [episode(0.0), episode(0.0), episode(3.0), episode(1.0)]  # advantages -1, -1, 2, 0
    batch = Grpo().batch(group, Budget(), random.Random(0))
    assert batch.skipped is None and [weighted.advantage for weighted in batch.segments] == [-1.0] * 8 + [2.0] * 4
    limited = Grpo().batch(group, Budget(segments=6), random.Random(0))
    assert [w.advantage for w in limited.segments] == [-1.0] * 4 + [2.0] * 2  # each keeps its share
    failed = Episode("train", 1, 9, "r9", {}, Outcome.FAILED, detail="it raised")
    assert [w.advantage for w in Grpo().batch([*group, failed], Budget(), random.Random(0)).segments] == [
        weighted.advantage for weighted in batch.segments
    ]  # an episode that did not complete is no part of the comparison
    assert Grpo().batch([group[0], failed], Budget(), random.Random(0)).skipped == "1 of 2 episodes completed"


def test_an_update_takes_an_even_share_of_every_episodes_sequences() -> None:
    items = [(name, index) for name, count in (("a", 300), ("b", 100), ("c", 40)) for index in range(count)]
    assert spread(items, 500, random.Random(0)) == items and spread(items, None, random.Random(0)) == items
    chosen = spread(items, 110, random.Random(0))
    assert len(chosen) == len(set(chosen)) == 110
    assert {name: sum(1 for each, _ in chosen if each == name) for name in "abc"} == {"a": 75, "b": 25, "c": 10}
    first = [index for name, index in chosen if name == "a"]
    assert first == sorted(first) and first[0] < 4 and first[-1] > 295  # over the whole game, start to end


def test_a_group_with_a_segment_sampled_without_behaviour_logprobs_is_not_trained_on() -> None:
    refused = Grpo().batch([episode(0.0), episode(3.0, sampled_with=()), episode(0.0)], Budget(), random.Random(0))
    assert not refused.segments
    assert refused.skipped == "turns of channel `policy` were sampled without their exact tokens or behaviour logprobs"
    exact = [episode(0.0), episode(3.0, sampled_with=("token_exact",)), episode(0.0)]
    said = Grpo().batch(exact, Budget(), random.Random(0)).skipped
    assert said == "turns of channel `policy` were sampled without behaviour logprobs"
    # An episode whose advantage is zero is not trained on: what it was sampled with does not matter.
    middle = [episode(0.0), episode(1.0, sampled_with=()), episode(2.0)]
    assert len(Grpo().batch(middle, Budget(), random.Random(0)).segments) == 8
