"""Whole episodes through the real program on the local runner: the scripted team solves a start of every row and
every agent is rewarded, a team that never acts runs out of turns, and `rollout env check` passes."""

import random

import pytest
from pydantic import JsonValue

from gridworld.environment import ROWS, environment
from gridworld.episode import TEAM, GridEpisode
from gridworld.prompts import observe, parse
from gridworld.scripted import ScriptedTeam, reply
from rollout.contracts import ModelEndpoint, RunEvent, RunEventType
from rollout.environment import Row, held_out, train_start
from rollout.harness import DirectModel, ModelBinding, RunBinding, RunSpecification, RunStatus, with_row
from rollout.local import LocalRunner
from rollout.testing import ScriptedModelEndpoint, payload
from rollout_train.check import checked, scripted


async def play(
    row: Row, endpoint: ModelEndpoint, seed: int = 0
) -> tuple[dict[str, JsonValue], dict[str, float], list[RunEvent]]:
    """One episode of a training start of `row`, every slot answered by `endpoint`: its result, its rewards by slot,
    and its events."""
    start = train_start(environment, row, random.Random(seed), held_out(environment))
    model = ModelBinding(direct=DirectModel(provider="scripted", model="team"))
    binding = RunBinding(models=dict.fromkeys(TEAM, model))
    runner = LocalRunner(providers={"scripted": lambda model: endpoint})
    handle = await runner.start(RunSpecification(program=with_row(environment.program, start), binding=binding))
    outcome = await handle.result()
    assert outcome.status is RunStatus.COMPLETED, outcome
    events = handle.recorded_events()
    rewards = {
        str(payload(event)["slot"]): float(str(payload(event)["value"]))
        for event in events
        if event.type is RunEventType.REWARD_ASSIGNED
    }
    (result,) = [payload(event)["payload"] for event in events if event.type is RunEventType.OUTPUT_EMITTED]
    assert isinstance(result, dict)
    return result, rewards, events


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row.key)
async def test_the_scripted_team_solves_every_row_and_every_agent_is_rewarded(row: Row) -> None:
    result, rewards, events = await play(row, ScriptedTeam())
    agents = int(str(row.parameters["agents"]))
    assert result["solved"] is True and result["ended"] == "every final plate pressed"
    assert rewards == dict.fromkeys(TEAM[:agents], 1.0)
    duration = result["duration"]
    assert isinstance(duration, int) and 0 < duration <= int(str(row.parameters["turns"]))
    doors = {"open": 0, "door": 1, "gate": 1, "vault": 2}[str(row.parameters["layout"])]  # (opened for good: not gates)
    assert len(result["opened"] or {}) == doors  # pyright: ignore[reportArgumentType]
    assert result["chat"], "the team says who goes where"
    samples = [event for event in events if event.type is RunEventType.EFFECT_REQUESTED]
    assert samples, "every agent's turn is a recorded sample"


async def test_a_team_that_never_acts_runs_out_of_turns_unrewarded() -> None:
    row = ROWS[0]
    turns = int(str(row.parameters["turns"]))
    endpoint = ScriptedModelEndpoint(["I am thinking."] * 2 * turns)
    result, rewards, _ = await play(row, endpoint)
    assert result["solved"] is False and result["ended"] == "turns" and result["duration"] == turns
    assert rewards == {"agent-1": 0.0, "agent-2": 0.0}
    assert result["actions"] == {"none": 2 * turns, "blocked": 0}
    assert len(endpoint.requests) == 2 * turns
    first = endpoint.requests[0]
    assert [tool.name for tool in first.tools] == ["move", "wait", "say"]
    assert [message.role.value for message in first.context.append] == ["system", "user"]


def test_the_scripted_team_solves_many_starts_of_every_row_within_its_budget() -> None:
    """The game alone, without the runner: twenty starts of each row."""
    for row in ROWS:
        for seed in range(20):
            episode = GridEpisode({**row.parameters, "seed": seed})
            game = episode.game()
            while not game.over:
                game.step([parse(reply(observe(game, agent))) for agent in range(episode.agents)])
            assert game.solved, (row.key, seed)


def test_too_few_or_too_many_agents_are_refused() -> None:
    with pytest.raises(ValueError, match="from two to 4 agents"):
        GridEpisode({"layout": "open", "agents": 1})
    with pytest.raises(ValueError, match="from two to 4 agents"):
        GridEpisode({"layout": "open", "agents": 5})


async def test_the_environment_passes_its_check() -> None:
    found = [*checked(environment), await scripted(environment)]
    assert all(finding.passed for finding in found), [str(finding) for finding in found]
    assert len(environment.evals()["gridworld-eval"]) == 3 * len(ROWS)
