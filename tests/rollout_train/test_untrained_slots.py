"""Slots that are not trained (a judge, a fixed opponent): the binding says so for each slot, the key carries it, every
turn records it, and segments, episodes and the algorithm keep such turns without training on them."""

import random
from collections.abc import Mapping
from pathlib import Path

import pytest

from rollout.contracts import Message
from rollout.harness import ModelSlot, Program, ProgramReference, RunContext, bind, register
from rollout.harness.blobs import FileBlobStore
from rollout_train import Budget, Grpo
from rollout_train.gateway import Grant
from rollout_train.gateway.turns import turns_table
from rollout_train.ledger import Fence
from rollout_train.recorder import TOKEN_LEVEL, Segment, Span
from rollout_train.rollouts import Episode, Outcome, Trajectory
from rollout_train.rollouts.episodes import Record, loaded, stored
from rollout_train.testing import keyring, sample_request
from tests.rollout_train.gateway.support import echo_channel, gateway_over, grant, stores


class Judged(Program):
    """A policy that answers and a judge that scores it."""

    def model_slots(self) -> Mapping[str, ModelSlot]:
        return {"policy": ModelSlot(), "judge": ModelSlot(trained=False, judge=True)}

    async def main(self, run: RunContext) -> None:
        raise NotImplementedError


def test_a_judge_is_declared_untrained_and_bound_to_the_channel_the_run_names() -> None:
    with pytest.raises(ValueError, match="never trained"):
        ModelSlot(judge=True)
    binding = bind(ProgramReference(program=register(Judged)), "policy", slots={"judge": "judge"})
    policy, judge = (binding.models[slot].recorded for slot in ("policy", "judge"))
    assert policy is not None and (policy.channel, policy.trained) == ("policy", True)
    assert judge is not None and (judge.channel, judge.trained) == ("judge", False)
    unbound = bind(ProgramReference(program=register(Judged)), "policy").models["judge"].recorded
    assert unbound is not None and unbound.channel == "policy"  # (whether that may be is validation's to say)


def test_a_key_carries_whether_its_slot_is_trained() -> None:
    granted = Grant("train", "r_1", "judge", "judge", Fence("f", 1), 9e9, trained=False)
    assert keyring().verify(keyring().mint(granted)).trained is False
    trained = Grant("train", "r_1", "policy", "policy", Fence("f", 1), 9e9)
    assert "trained" not in trained.to_json() and keyring().verify(keyring().mint(trained)).trained


async def test_a_judges_turns_are_recorded_and_its_segments_kept_untrained(tmp_path: Path) -> None:
    ledger, blobs = stores(tmp_path)
    gateway = gateway_over(echo_channel(), ledger, blobs)
    fence = await ledger.take("runs/train/episodes/1/1")
    for slot, trained in (("policy", True), ("judge", False)):
        said = await grant(ledger, slot=slot, trained=trained, fence=fence)
        request = sample_request([Message.user("Score it.")], f"r_1:{slot}:0", session_id=f"r_1/{slot}")
        await gateway.sample(said, request)
    recorded = await ledger.read(turns_table("train", "r_1"))
    assert [entry.get("trained", True) for entry in recorded.values()] == [True, False]  # type: ignore[union-attr]
    assert [turn.trained for turn in await gateway.store.turns("train", "r_1")] == [True, False]
    sessions = await gateway.store.sessions("train", "r_1")
    assert [segment.trained for segment in sessions["policy"]] == [True]
    assert [segment.trained for segment in sessions["judge"]] == [False]
    assert sessions["judge"][0].sampled > 0  # (what it sampled still counts)


def judged(reward: float, number: int, *, judge_with: tuple[str, ...] = TOKEN_LEVEL) -> Episode:
    """An episode whose policy was rewarded and whose judge, sampled with `judge_with`, was not."""
    answer = Segment([1, 2, 3], [Span(1, 3, 0)], [-0.5, -0.5], "policy")
    verdict = Segment([4, 5, 6], [Span(1, 3, 0)], [-0.5, -0.5], "judge", judge_with, trained=False)
    trajectories = {
        "policy": Trajectory([answer], {"default": reward}),
        "judge": Trajectory([verdict], {}, trained=False),
    }
    return Episode("train", 1, number, f"r{number}", {}, Outcome.COMPLETED, trajectories=trajectories)


def test_the_algorithm_never_trains_on_a_judges_turns_and_the_reward_is_the_trained_slots() -> None:
    group = [judged(0.0, 1), judged(1.0, 2, judge_with=())]  # (a judge that returns text cannot be weighed)
    assert [episode.reward for episode in group] == [0.0, 1.0]  # (not halved by the judge's empty reward)
    batch = Grpo().batch(group, Budget(), random.Random(0))
    assert batch.skipped is None
    assert [weighted.segment.channel for weighted in batch.items] == ["policy", "policy"]


async def test_an_episode_keeps_which_of_its_slots_are_trained_once_stored(tmp_path: Path) -> None:
    blobs = FileBlobStore(tmp_path / "blobs")
    episode = judged(1.0, 1)
    record = stored(episode, [], blobs)
    kept = (await record).to_json()
    back = Record.from_json(kept)
    assert back.episode.reward == 1.0 and not back.episode.trajectories["judge"].trained
    assert back.sampled == {"policy": 2, "judge": 2}  # (spend counts the judge's tokens)
    again = await loaded(back, blobs)
    assert [segment.trained for segment in again.trajectories["judge"].segments] == [False]
