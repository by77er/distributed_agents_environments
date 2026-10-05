"""A run over a profile with a judge: its settings bind the judge's slot to the profile's judge channel, its start
records them, its runner's key for the slot routes there, and the judge's turns are recorded and never trained on. A
launch passes the same settings on to the run it starts."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from rollout_train.cli import _train  # pyright: ignore[reportPrivateUsage]
from rollout_train.launcher import Launcher
from rollout_train.launches import Asked, launches_of
from rollout_train.ledger import FileLedger
from rollout_train.presence import presence_of
from rollout_train.record import RESULTS, STARTS, newest_record, table
from rollout_train.rollouts import Record
from rollout_train.rollouts.scheduler import EPISODES, PLANS
from tests.rollout_train.support import Process, profiles, write

JUDGED = "tests.rollout_train.rollouts.games:judged"


async def test_a_run_over_a_profile_binds_its_judge_to_the_channel_its_settings_name(tmp_path: Path) -> None:
    path = write(tmp_path)
    sets = ["slots.judge=judge", "channels.judge.mode=fixed"]
    await _train(path, None, JUDGED, groups=2, groups_per_step=1, seed=1, sets=sets)
    ledger = FileLedger(tmp_path / "run" / "ledger")
    (run,) = {name.split("/")[1] for name in await ledger.tables() if name.endswith(f"/{RESULTS}")}
    start = newest_record(await ledger.read(table(run, STARTS)))
    fixed: dict[str, Any] = start["run_settings"]["fixed"]
    assert (fixed["slots.judge"], fixed["channels.judge.mode"], fixed["self_judging"]) == ("judge", "fixed", False)
    binding = newest_record(await ledger.read(table(run, PLANS)))["binding"]["models"]
    assert binding["judge"]["recorded"]["channel"] == "judge" and binding["judge"]["recorded"]["trained"] is False
    assert binding["policy"]["recorded"]["channel"] == "policy" and binding["policy"]["recorded"]["trained"] is True
    records = [Record.from_json(each) for each in (await ledger.read(table(run, EPISODES))).values()]  # type: ignore[arg-type]
    assert records and all(not each.episode.trajectories["judge"].trained for each in records)
    assert all(each.sampled["judge"] > 0 for each in records)  # (the judge's tokens count)


@pytest.mark.parametrize(
    ("sets", "refused"),
    [
        ([], "slots.judge: slot judge is not trained, so it samples no channel by default"),
        (["slots.judge=policy"], "the policy would judge itself (self_judging allows it)"),
        (["slots.judge=elsewhere"], "slots.judge: channel elsewhere is not one of the profile's (policy, judge)"),
        (["slots.judge=judge", "channels.policy.mode=fixed"], "policy is the trained channel"),
    ],
)
async def test_a_run_over_a_profile_refuses_what_its_slots_cannot_be(
    tmp_path: Path, sets: list[str], refused: str
) -> None:
    with pytest.raises(SystemExit, match=refused.replace("(", r"\(").replace(")", r"\)")):
        await _train(write(tmp_path), None, JUDGED, groups=1, groups_per_step=1, seed=1, sets=sets)


async def test_a_launch_passes_the_slots_and_channels_it_sets_to_its_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    launches, heartbeats = launches_of(ledger), presence_of(ledger)
    assert launches is not None and heartbeats is not None
    commands: list[list[str]] = []

    async def spawn(*command: str, **_: Any) -> Process:
        commands.append(list(command))
        return Process(0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    found = Launcher("launcher/here", launches, heartbeats, profiles(tmp_path), [JUDGED], tmp_path / "runs", every=0.01)
    settings = {"slots.judge": "judge", "channels.judge.mode": "follows", "channels.judge.follows": "policy"}
    await launches.ask(Asked("small", JUDGED, "Judged", settings={**settings, "channels.judge.lag": 2}))
    serving = asyncio.create_task(found.serve())
    try:
        async with asyncio.timeout(5):
            while not commands:  # noqa: ASYNC110 (the launcher starts it)
                await asyncio.sleep(0.01)
    finally:
        serving.cancel()
    (command,) = commands
    said = {command[index + 1] for index, each in enumerate(command) if each == "--set"}
    assert {'slots.judge="judge"', 'channels.judge.mode="follows"', "channels.judge.lag=2"} <= said
