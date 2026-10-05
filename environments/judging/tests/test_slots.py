"""The environment declares its judge so that a run cannot bind it to the trained channel unless it says so."""

from judging.environment import environment
from rollout.environment import first_program
from rollout.harness import instantiate
from rollout_train.run_settings import RunSettings
from rollout_train.slots import Declared, problems

declared = Declared.of(instantiate(first_program(environment)).model_slots())
JUDGE = {
    "channels.policy.model": "Qwen/Qwen3-0.6B",
    "channels.judge.model": "Qwen/Qwen3-8B",
    "channels.judge.mode": "fixed",
    "slots.judge": "judge",
}


def test_the_judge_is_declared_untrained_and_judging() -> None:
    assert declared == Declared(frozenset({"policy", "judge"}), frozenset({"judge"}), frozenset({"judge"}))


def test_a_run_binds_the_judge_to_a_channel_of_its_own_or_says_it_judges_itself() -> None:
    assert problems(RunSettings(JUDGE), declared) == []
    assert problems(RunSettings({"channels.policy.model": "Qwen/Qwen3-0.6B"}), declared) == [
        ("slots.judge", "slot judge is not trained, so it samples no channel by default: bind it to one (slots.judge)")
    ]
    itself = {"channels.policy.model": "Qwen/Qwen3-0.6B", "slots.judge": "policy"}
    ((key, reason),) = problems(RunSettings(itself), declared)
    assert key == "slots.judge" and "the policy would judge itself" in reason
    assert problems(RunSettings({**itself, "self_judging": True}), declared) == []
