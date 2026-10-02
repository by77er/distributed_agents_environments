"""The training loop under two wirings: everything in this process, and a durable runner with the rollout jobs
behind HTTP. The loop's code is the same; so is what it does."""

import json
import random
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import JsonValue

from rollout.core.harness import ProgramReference, Runner, agent_program
from rollout.core.local import LocalRunner
from rollout.recorder import Recorder
from rollout.rollouts import JobHooks, Jobs, RolloutJobs, Row
from rollout.training import Budget, Colocated, Directory, Step, Weighted, iterations, train
from tests.rollouts.games import Guess
from tests.support import ScriptedEngine, plain_channel


class Words:
    """A catalog of the guessing game: three rows, each a word to say."""

    program: ProgramReference = agent_program(Guess)

    def rows(self) -> Sequence[Row]:
        return [Row(f"say-{word}", f"say {word}", {"word": word}) for word in ("yes", "no", "maybe")]

    def start(self, row: Row, rng: random.Random) -> JsonValue:
        return {**row.parameters, "seed": rng.randrange(1000)}


class Counting:
    """A trainer that trains nothing and keeps what it was given."""

    budget = Budget(sequences=3)

    def __init__(self) -> None:
        self.batches: list[list[Weighted]] = []

    async def step(self, batch: Sequence[Weighted], *, seed: int) -> Step:
        self.batches.append(list(batch))
        name = f"step-{len(self.batches)}"
        return Step(name, f"/adapters/{name}", {"sequences": float(len(batch))})


class Notes(JobHooks):
    def __init__(self) -> None:
        self.kinds: list[str] = []

    def on_job(self, event: Mapping[str, JsonValue]) -> None:
        self.kinds.append(str(event["kind"]))


@pytest.fixture(params=["in process", "durable runner, jobs over HTTP"])
async def wiring(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[tuple[Jobs, Recorder, Notes]]:
    recorder = Recorder({"policy": plain_channel(always=[("yes\n", "stop"), ("no\n", "stop")])})
    notes = Notes()
    if request.param == "in process":
        local = RolloutJobs(LocalRunner(recorder=recorder), recorder, hooks=[notes])
        yield local, recorder, notes
        await local.close()
        return
    pytest.importorskip("dbos")
    pytest.importorskip("starlette")
    from rollout.durable import DurableRunner
    from rollout.rollouts.service import RolloutClient, create_app

    runner = DurableRunner(tmp_path / "runs", recorder=recorder)
    await runner.launch()
    served = RolloutJobs(cast_runner(runner), recorder, log=tmp_path / "log", hooks=[notes])
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(served)), base_url="http://rollouts")
    try:
        yield RolloutClient("http://rollouts", client=client), recorder, notes
    finally:
        await served.close()
        await runner.close()


def cast_runner(runner: Any) -> Runner:
    return runner


async def test_the_loop_trains_groups_as_they_finish_and_goes_on_where_it_stopped(
    wiring: tuple[Jobs, Recorder, Notes], tmp_path: Path
) -> None:
    jobs, recorder, notes = wiring
    store, trainer = Directory(tmp_path / "run"), Counting()
    await train(jobs, Words(), trainer, store, groups=3, seed=1)

    lines = iterations(store)
    assert [line["iteration"] for line in lines] == [1, 2, 3]
    assert all(len(line["rewards"]) == 4 and line["failed"] == 0 for line in lines)
    # Half of each group said the word (the policy says yes and no in turn): rewards differ, so every group trains.
    assert all(sorted(line["rewards"]) == [0.0, 0.0, 1.0, 1.0] for line in lines if line["task"] != "say-maybe")
    trained = [line for line in lines if isinstance(line["update"], dict)]
    assert len(trained) == len(trainer.batches) >= 2
    assert [line["version"] for line in trained] == list(range(1, len(trained) + 1))
    assert recorder.channels["policy"].version == len(trained)
    for line, batch in zip(trained, trainer.batches, strict=True):
        assert line["sequences_recorded"] == 4 and line["sequences_trained"] == len(batch) == 3  # the trainer's budget
        assert line["speed_bonus"] == [1.0 if reward else 0.0 for reward in line["rewards"]]  # both were as fast
        said = {"".join(chr(t) for t in w.epoch.tokens[w.epoch.spans[0].start :]).strip(): w.advantage for w in batch}
        word = line["task"].removeprefix("say-")
        assert said[word] == 1.0 and all(value == -1.0 for other, value in said.items() if other != word)
    saved = json.loads(store.read("curriculum.json") or "{}")
    assert {entry["title"] for entry in saved.values()} <= {"say yes", "say no", "say maybe"}
    assert notes.kinds.count("iteration") == 3 and notes.kinds.count("published") == len(trained)
    assert notes.kinds.count("episode") == 12

    await train(jobs, Words(), trainer, store, groups=2, seed=1)  # started again: it goes on after the last group
    assert [line["iteration"] for line in iterations(store)] == [1, 2, 3, 4, 5]


async def test_a_trainer_that_shares_the_engines_gpu_puts_them_to_sleep_around_each_step(tmp_path: Path) -> None:
    recorder = Recorder({"policy": plain_channel(always=[("yes\n", "stop"), ("no\n", "stop")])})
    rollouts = RolloutJobs(LocalRunner(recorder=recorder), recorder)
    channel = recorder.channels["policy"]
    engine: Any = channel.engines[0]
    assert isinstance(engine, ScriptedEngine)
    guarded: list[str] = []
    trainer = Colocated(Counting(), [channel], guard=lambda: guarded.append(engine.told[-1]))
    await train(rollouts, Words(), trainer, Directory(tmp_path / "run"), groups=2)
    await rollouts.close()
    steps = [line for line in iterations(Directory(tmp_path / "run")) if isinstance(line["update"], dict)]
    assert steps and engine.told[:4] == ["sleep", "wake", "load step-1", "sleep"][: len(engine.told[:4])]
    assert guarded and set(guarded) == {"sleep"}  # the guard runs once the engines are asleep
    assert all("update_seconds" in line["update"] for line in steps)
