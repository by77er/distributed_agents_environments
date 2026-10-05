"""The group algorithm on made-up episodes: advantages, the tiebreak and what a step trains on."""

import random
import statistics
from dataclasses import replace
from typing import Any

import pytest

from rollout_train import Budget, Grpo, group_advantages
from rollout_train.algorithm import (
    DEFAULT_ADVANTAGE,
    Distillations,
    Preferences,
    advantages_of,
    algorithm_for,
    equal_scores,
    group_mean,
    group_std,
    leave_one_out,
    spread,
    tiebreak,
    within,
)
from rollout_train.objectives import PRESETS, Advantage, resolved
from rollout_train.recorder import TOKEN_LEVEL, Segment, Span, TeacherScores
from rollout_train.rollouts import Episode, Outcome, Trajectory
from rollout_train.trainer import Distilled, Labelled, Pair


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


def test_a_tiebreak_favours_the_shortest_only_where_every_episode_saturated_its_task() -> None:
    def full(duration: float) -> Episode:
        return episode(1.0, saturated=True, duration=duration)

    assert tiebreak([full(40), full(12), full(30)], 0.05) == [0.0, 0.05, 0.0]
    assert tiebreak([full(12), full(20), full(12)], 0.05) == [0.05, 0.0, 0.05]  # a tie for shortest: both get it
    assert tiebreak([full(12), full(12)], 0.05) == [0.0, 0.0]  # nothing to break
    # One episode short of saturating: the rewards already differ, and the shortest is not compared with it.
    assert tiebreak([full(12), episode(0.5, duration=8), full(30)], 0.05) == [0.0, 0.0, 0.0]
    # An episode that does not say how long it took leaves nothing to compare; and none is given by default.
    assert tiebreak([full(12), episode(1.0, saturated=True), full(30)], 0.05) == [0.0, 0.0, 0.0]
    assert tiebreak([full(40), full(12)], 0.0) == [0.0, 0.0] and tiebreak([], 0.05) == []
    group = [full(40), full(12), full(30)]
    untied = Grpo().batch(group, Budget(), random.Random(0))  # the default: how long an episode took counts for nothing
    assert not untied.items and untied.skipped == "every episode scored the same" and untied.notes == {}
    assert DEFAULT_ADVANTAGE.tiebreak == 0.0 and PRESETS["default"].objective.advantage.tiebreak == 0.0
    tied = Grpo(advantage=Advantage(tiebreak=0.06)).batch(group, Budget(), random.Random(0))
    assert tied.notes == {"tiebreak": [0.0, 0.06, 0.0]} and len(tied.items) == 12
    assert sorted({round(weighted.advantage, 3) for weighted in tied.items}) == [-0.02, 0.04]
    made = algorithm_for(resolved("default", {"advantage.tiebreak": 0.06}))
    assert isinstance(made, Grpo) and made.advantage == Advantage(tiebreak=0.06)


def test_a_step_trains_on_every_slots_sequences_of_the_episodes_that_differ_from_the_mean() -> None:
    group = [episode(0.0), episode(0.0), episode(3.0), episode(1.0)]  # advantages -1, -1, 2, 0
    batch = Grpo().batch(group, Budget(), random.Random(0))
    assert batch.skipped is None and [weighted.advantage for weighted in batch.items] == [-1.0] * 8 + [2.0] * 4
    limited = Grpo().batch(group, Budget(segments=6), random.Random(0))
    assert [w.advantage for w in limited.items] == [-1.0] * 4 + [2.0] * 2  # each keeps its share
    failed = Episode("train", 1, 9, "r9", {}, Outcome.FAILED, detail="it raised")
    assert [w.advantage for w in Grpo().batch([*group, failed], Budget(), random.Random(0)).items] == [
        weighted.advantage for weighted in batch.items
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
    assert not refused.items
    assert refused.skipped == "turns of channel `policy` were sampled without their exact tokens or behaviour logprobs"
    exact = [episode(0.0), episode(3.0, sampled_with=("token_exact",)), episode(0.0)]
    said = Grpo().batch(exact, Budget(), random.Random(0)).skipped
    assert said == "turns of channel `policy` were sampled without behaviour logprobs"
    # An episode whose advantage is zero is not trained on: what it was sampled with does not matter.
    middle = [episode(0.0), episode(1.0, sampled_with=()), episode(2.0)]
    assert len(Grpo().batch(middle, Budget(), random.Random(0)).items) == 8


def test_the_advantage_components() -> None:
    scores = [1.0, 0.0, 0.0, 1.0, 0.5]
    assert group_mean(scores) == pytest.approx([0.5, -0.5, -0.5, 0.5, 0.0])
    assert leave_one_out(scores) == pytest.approx([1 - 1.5 / 4, -2.5 / 4, -2.5 / 4, 1 - 1.5 / 4, 0.5 - 2 / 4])
    deviation = statistics.stdev(scores)  # (the sample's)
    assert group_std(group_mean(scores), scores) == pytest.approx([each / deviation for each in group_mean(scores)])
    assert group_std([0.0, 0.0], [1.0, 1.0]) == [0.0, 0.0]  # (scores that do not vary: nothing to compare)
    assert equal_scores([2.0, 2.0]) and equal_scores([3.0]) and not equal_scores([1.0, 2.0])
    assert advantages_of([2.0, 2.0], Advantage(filter="none")) == [0.0, 0.0]
    assert advantages_of([2.0, 2.0], Advantage(baseline="none", filter="none")) == [2.0, 2.0]  # (REINFORCE)
    assert advantages_of([2.0, 2.0]) is None  # (DAPO's dynamic sampling: the default)
    assert advantages_of([1.0, 0.0], PRESETS["rloo"].objective.advantage) == [1.0, -1.0]


def test_the_algorithm_follows_the_objectives_family_and_what_it_reads() -> None:
    assert isinstance(algorithm_for(PRESETS["default"].objective), Grpo)
    assert isinstance(algorithm_for(PRESETS["dpo"].objective), Preferences)
    assert algorithm_for(PRESETS["kto"].objective) == Preferences(labelled=True)
    corrected = algorithm_for(PRESETS["reinforce"].objective)
    assert isinstance(corrected, Grpo) and corrected.needs == ("token_exact", "sampled_logprobs")
    reinforce = algorithm_for(resolved("reinforce", {"importance.paper_exact": True}))
    assert isinstance(reinforce, Grpo) and reinforce.needs == ("token_exact",)  # (no correction: no logprobs needed)
    # A policy gradient with no importance correction trains on turns without behaviour logprobs, not on inexact ones.
    without = [episode(0.0, sampled_with=("token_exact",)), episode(3.0, sampled_with=("token_exact",))]
    assert reinforce.batch(without, Budget(), random.Random(0)).items
    inexact = [episode(0.0, sampled_with=()), episode(3.0, sampled_with=())]
    assert reinforce.batch(inexact, Budget(), random.Random(0)).skipped is not None
    assert algorithm_for(PRESETS["sft"].objective).batch(inexact, Budget(), random.Random(0)).items


def test_a_groups_best_episode_is_preferred_to_its_worst() -> None:
    group = [episode(0.5, number=1), episode(1.0, number=2), episode(0.0, number=3, sampled_with=()), episode(1.0)]
    batch = Preferences().batch(group, Budget(), random.Random(0))
    (pair,) = batch.items
    assert isinstance(pair, Pair) and pair.source == "train/1/2>3"  # (the first of the best; the worst)
    assert len(pair.chosen) == len(pair.rejected) == 4  # (every turn of each, both slots': the start is shared)
    assert pair.rejected[0].lacks  # (turns without behaviour logprobs: a preference loss does not read them)
    assert Preferences().batch([episode(1.0), episode(1.0)], Budget(), random.Random(0)).skipped == (
        "every episode scored the same"
    )
    labelled = Preferences(labelled=True).batch(group, Budget(), random.Random(0)).items
    assert [(each.source, each.desirable) for each in labelled if isinstance(each, Labelled)] == [
        ("train/1/1", False), ("train/1/2", True), ("train/1/3", False), ("train/1/1", True)
    ]  # fmt: skip
    # Within a budget of segments: a pair holds both sides'.
    pairs = [Pair((segment,) * 3, (segment,) * 3) for segment in group[0].trajectories["ada"].segments * 5]
    assert len(within(pairs, 30, random.Random(0))) == 5 and len(within(pairs, None, random.Random(0))) == 10


def scored(found: Episode, top: int = 0) -> Episode:
    """The episode with every segment scored by a teacher (its top `top` tokens at each sampled one)."""
    scores = TeacherScores("teacher", [-1.0, -2.0], [[5] * top, [6] * top] if top else [],
                           [[-0.1] * top, [-0.2] * top] if top else [])  # fmt: skip
    trajectories = {
        slot: replace(trajectory, segments=[replace(each, teacher=scores) for each in trajectory.segments])
        for slot, trajectory in found.trajectories.items()
    }
    return replace(found, trajectories=trajectories)


def test_a_distillation_trains_on_every_segment_of_one_episode_with_its_teachers_scores() -> None:
    algorithm = algorithm_for(PRESETS["mopd"].objective)
    assert isinstance(algorithm, Distillations) and algorithm.group_size == 1
    made = algorithm.batch([scored(episode(0.0))], Budget(), random.Random(0))
    assert len(made.items) == 4 and all(isinstance(each, Distilled) and each.advantage == 0.0 for each in made.items)
    assert made.items[0].source == "train/1/1/ada/0" and made.items[0].scores.teacher == "teacher"
    assert len(algorithm.batch([scored(episode(0.0))], Budget(segments=2), random.Random(0)).items) == 2
    unscored = algorithm.batch([episode(1.0)], Budget(), random.Random(0))
    assert not unscored.items and unscored.skipped == "turns of channel `policy` were not scored by a teacher"
    top = algorithm_for(PRESETS["mopd_top_k"].objective)
    short = top.batch([scored(episode(0.0))], Budget(), random.Random(0))
    assert short.skipped == "teacher `teacher` gave no top tokens, and the objective reads 64"
    assert len(top.batch([scored(episode(0.0), top=3)], Budget(), random.Random(0)).items) == 4
    failed = replace(episode(0.0), outcome=Outcome.FAILED)
    assert algorithm.batch([failed], Budget(), random.Random(0)).skipped == "0 of 1 episodes completed"
    inexact = scored(episode(0.0, sampled_with=("sampled_logprobs",)))
    assert "without their exact tokens" in str(algorithm.batch([inexact], Budget(), random.Random(0)).skipped)


def test_a_policy_gradient_with_a_distillation_term_keeps_every_segment_with_its_advantage() -> None:
    objective = resolved("default", {"distillation.coefficient": 0.5})
    algorithm = algorithm_for(objective)
    assert isinstance(algorithm, Grpo) and algorithm.distills and algorithm.group_size == 4
    group = [scored(episode(1.0, number=1)), scored(episode(0.0, number=2)), scored(episode(0.0, number=3))]
    made = algorithm.batch(group, Budget(), random.Random(0))
    assert len(made.items) == 12 and all(isinstance(each, Distilled) for each in made.items)
    assert sorted({round(each.advantage, 3) for each in made.items}) == [-0.333, 0.667]
    same = algorithm.batch([scored(episode(1.0, number=n)) for n in (1, 2)], Budget(), random.Random(0))
    assert len(same.items) == 8 and {each.advantage for each in same.items} == {0.0}  # (the teacher still trains)
    assert algorithm.batch([episode(1.0), episode(0.0)], Budget(), random.Random(0)).skipped is not None
