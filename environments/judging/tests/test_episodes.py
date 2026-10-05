"""Whole episodes through the real program on the local runner, the policy and the judge each answered by a script: a
well-formed verdict is the reward, a malformed one is asked again once, two malformed ones end the episode unrewarded
and say why, and a judge asked twice is averaged. The environment's rows and eval data hold, and `rollout env check`
passes."""

import json
import random
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import JsonValue

from judging.environment import EVAL, ROWS, TRAIN, Judging, environment, twice
from judging.episode import SYSTEM
from judging.rubric import EXPLAIN
from rollout.contracts import RunEventType
from rollout.environment import held_out, start_key, train_start
from rollout.harness import DirectModel, ModelBinding, RunBinding, RunSpecification, RunStatus, with_row
from rollout.local import LocalRunner
from rollout.testing import ScriptedModelEndpoint, payload
from rollout_train.check import checked, scripted

ANSWER = "Plants catch sunlight with their green leaves and use it to turn air and water into sugar, their food."


def verdict(score: int) -> str:
    return json.dumps({"criteria": {each.name: score for each in EXPLAIN.criteria}, "score": score, "reason": "Fair."})


async def play(
    judge: Sequence[str], judged: Judging = environment
) -> tuple[dict[str, JsonValue], dict[str, float], ScriptedModelEndpoint, ScriptedModelEndpoint]:
    """One episode of a training start of `explain`: the policy says `ANSWER`, the judge says `judge`, in turn. Its
    result, its rewards by slot, and the two endpoints."""
    start = train_start(judged, ROWS[0], random.Random(0), held_out(judged))
    policy, judging = ScriptedModelEndpoint([ANSWER]), ScriptedModelEndpoint(list(judge))
    binding = RunBinding(
        models={
            "policy": ModelBinding(direct=DirectModel(provider="policy", model="policy")),
            "judge": ModelBinding(direct=DirectModel(provider="judge", model="judge")),
        }
    )
    runner = LocalRunner(providers={"policy": lambda model: policy, "judge": lambda model: judging})
    handle = await runner.start(RunSpecification(program=with_row(judged.program, start), binding=binding))
    assert (await handle.result()).status is RunStatus.COMPLETED
    events = handle.recorded_events()
    rewards = {
        str(payload(event)["slot"]): float(str(payload(event)["value"]))
        for event in events
        if event.type is RunEventType.REWARD_ASSIGNED
    }
    (result,) = [payload(event)["payload"] for event in events if event.type is RunEventType.OUTPUT_EMITTED]
    assert isinstance(result, dict)
    return result, rewards, policy, judging


async def test_a_well_formed_verdict_is_the_reward_and_the_result_records_it() -> None:
    result, rewards, policy, judge = await play([verdict(10)])
    assert rewards == {"policy": 1.0}  # (the judge, not trained, is not rewarded)
    assert result["rubric"] == "explain-1" and result["verdicts"] == [verdict(10)] and result["score"] == 10
    assert result["judged"] is True and result["words"] == len(ANSWER.split()) and "why" not in result
    (asked,) = policy.requests
    assert asked.context.append[0].text == SYSTEM
    (shown,) = judge.requests
    system, user = shown.context.append
    assert system.text == EXPLAIN.instructions() and ANSWER in user.text and "ten-year-old" in user.text


async def test_a_malformed_verdict_is_asked_again_once() -> None:
    result, rewards, _, judge = await play(["I would give it an 8.", verdict(4)])
    assert rewards == {"policy": pytest.approx(3 / 9)} and result["score"] == 4 and result["judged"] is True
    assert result["verdicts"] == ["I would give it an 8.", verdict(4)]
    retried = judge.requests[1].context.append
    assert retried[-1].text.startswith("Your verdict could not be read: the reply is not one JSON object")


async def test_two_malformed_verdicts_end_the_episode_unrewarded_and_say_why() -> None:
    result, rewards, _, judge = await play(["8/10", '{"score": 8}'])
    assert rewards == {"policy": 0.0} and result["judged"] is False and result["score"] is None
    assert result["why"] == (
        "the judge's verdict could not be read after 2 replies: the object has keys ['score'], not criteria, score "
        "and reason"
    )
    assert result["verdicts"] == ["8/10", '{"score": 8}'] and len(judge.requests) == 2


async def test_a_judge_asked_twice_is_averaged() -> None:
    result, rewards, _, judge = await play([verdict(6), "?", verdict(9)], twice)
    assert result["scores"] == [6, 9] and result["score"] == 7.5 and rewards == {"policy": pytest.approx(6.5 / 9)}
    assert [len(each.context.append) for each in judge.requests] == [2, 2, 4]  # (the second afresh, then asked again)


def test_training_never_draws_what_the_eval_asks() -> None:
    eval_starts = environment.evals()["judging-eval"]
    assert len(eval_starts) == sum(len(each) for each in EVAL.values())
    abouts = {str(start.parameters["about"]) for start in eval_starts}  # type: ignore[index]
    trained = {each if isinstance(each, str) else each[0] for items in TRAIN.values() for each in items}
    assert not abouts & trained
    held = held_out(environment)
    for row in ROWS:
        for seed in range(30):
            assert start_key(train_start(environment, row, random.Random(seed), held)) not in held


async def test_rollout_env_check_passes() -> None:
    found = [*checked(environment), await scripted(environment)]
    assert all(each.passed for each in found), [str(each) for each in found]


def test_the_command_checks_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from rollout_train.cli import main

    monkeypatch.setenv("HOME", str(tmp_path))  # (its scratch files go under ~/.cache/rollout/checks)
    monkeypatch.setattr("sys.argv", ["rollout", "env", "check", "judging.environment:twice"])
    with pytest.raises(SystemExit) as ended:
        main()
    out = capsys.readouterr().out
    assert int(ended.value.code or 0) == 0 and [line.split()[0] for line in out.splitlines()] == ["ok"] * 5, out
