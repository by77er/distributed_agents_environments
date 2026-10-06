# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (Ray is partly untyped.)
"""A run built from its settings and the cluster config, on the session's Ray, with scripted engines and a trainer
that trains nothing: its engine hosts and its trainer are actors it asks Ray for, the trainer colocated with the
trained channel's hosts; its gateway samples every channel its settings name; its start records them. A judging run
launched by its settings binds its judge's slot to a channel of its own; settings the cluster refuses end the run with
the reasons; a run that cannot have its GPUs yet waits, and says what for. A trainer in an actor says how far its step
has got to the driver, which says it in a line and in its runner's beats."""

import asyncio
import dataclasses
import inspect
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import JsonValue

from rollout_train import jobs
from rollout_train.cluster import Cluster
from rollout_train.jobs import Run, Stepping, TrainerActor, TrainerClient, driven, ran
from rollout_train.launches import ENDED, launch_of, launches_of
from rollout_train.launching import Refused
from rollout_train.record import ENDS, RESULTS, STARTS, newest_record, table
from rollout_train.rollouts import Record
from rollout_train.rollouts.scheduler import EPISODES, PLANS
from rollout_train.run_settings import RunSettings
from rollout_train.serving import SERVING
from rollout_train.stores import Stores
from rollout_train.submitting import ask
from rollout_train.trainer import MINIBATCH, Budget, Files, Item, Progress, Progressing, Step
from tests.local_ray import LocalRay
from tests.rollout_train.clusters import JUDGED, POLICY, WORDS, a_cluster


async def a_run(cluster: Cluster, settings: dict[str, JsonValue], name: str = "built") -> Run:
    stores = Stores.open(cluster)
    entry = await stores.registry.create(name)
    return Run(cluster, stores, RunSettings({**settings, "name": name}), entry)


def starts(run: Run) -> Any:
    return run.ledger.read(table(run.run.id, STARTS))


async def test_a_run_built_from_its_settings_trains_with_actors_it_asks_ray_for(
    tmp_path: Path, local_ray: LocalRay
) -> None:
    cluster = a_cluster(tmp_path)
    run = await a_run(cluster, {**POLICY, "kind": "train", "environment": WORDS})
    await ran(run)
    ledger = run.ledger
    assert len(await ledger.read(table(run.run.id, RESULTS))) == 2
    made = [each for each in await run.checkpoints.all() if each.run == run.run.id]
    assert made and all(each.weights is not None for each in made)  # (a group all of one reward trains nothing)
    start = newest_record(await starts(run))
    fixed: dict[str, Any] = start["run_settings"]["fixed"]
    assert (fixed["trainer.provider"], fixed["channels.policy.provider"]) == ("steps", "local")
    assert start["environment"] == WORDS and start["directory"].endswith(f"scratch/runs/{run.run.id}")
    served = await ledger.read(table(run.run.id, SERVING))
    assert {each["layout"] for each in served.values() if each.get("checkpoint")} == {"verbatim"}  # (bridged)
    assert newest_record(await ledger.read(table(run.run.id, ENDS)))["how"] == "finished"
    assert run.hosts["policy"] and run.trainer_handle is not None  # (actors of its own, ended with it)


async def test_an_eval_built_from_its_settings_plays_a_suite_with_a_checkpoint_on_engine_hosts_of_its_own(
    tmp_path: Path, local_ray: LocalRay
) -> None:
    from rollout.names import named
    from rollout_train.evals import make_suite, suite_entry

    cluster = a_cluster(tmp_path)
    trained = await a_run(cluster, {**POLICY, "kind": "train", "environment": WORDS})
    await ran(trained)
    (checkpoint, *_) = [each for each in await trained.checkpoints.all() if each.run == trained.run.id]
    entry = suite_entry(WORDS, named(WORDS), eval_data="words-held-out", episodes=1)
    suite = await make_suite(trained.ledger, "held-out", [entry])
    channel = {key: value for key, value in POLICY.items() if key.startswith("channels.")}
    evaluation = await a_run(cluster, {
        **channel, "kind": "eval", "environment": WORDS, "eval.suite": "held-out", "start": checkpoint.id,
    }, "held out on the checkpoint")  # fmt: skip
    await ran(evaluation)
    ledger, run = evaluation.ledger, evaluation.run.id
    assert len(await ledger.read(table(run, RESULTS))) == len(suite.starts)
    (served,) = (await ledger.read(table(run, SERVING))).values()
    assert isinstance(served, dict) and served["checkpoint"] == checkpoint.id and served["files"]  # (bridged)
    start = newest_record(await ledger.read(table(run, STARTS)))
    assert start["kind"] == "eval" and start["checkpoint"] == checkpoint.id
    assert start["run_settings"]["fixed"]["eval.suite"] == "held-out"
    assert evaluation.hosts["policy"] and evaluation.trainer_handle is None  # (nothing trained)
    assert newest_record(await ledger.read(table(run, ENDS)))["how"] == "finished"


async def test_a_judging_run_launched_by_its_settings_samples_its_judge_on_the_channel_they_bind_it_to(
    tmp_path: Path, local_ray: LocalRay
) -> None:
    cluster = a_cluster(tmp_path)
    stores = Stores.open(cluster)
    judge: dict[str, JsonValue] = {
        "channels.judge.provider": "local", "channels.judge.model": "tiny", "channels.judge.mode": "fixed",
        "channels.judge.renderer": "rollout_train.testing:plain_renderer", "slots.judge": "judge",
    }  # fmt: skip
    settings = RunSettings({**POLICY, **judge, "kind": "train", "name": "judged", "environment": JUDGED})
    launch = await ask(settings, stores.ledger)
    await driven(launch.id, cluster, stores)
    launches = launches_of(stores.ledger)
    assert launches is not None and launch.run is not None
    assert (await launch_of(launches, launch.id)).state == ENDED
    ledger, run = stores.ledger, launch.run
    binding = newest_record(await ledger.read(table(run, PLANS)))["binding"]["models"]
    assert binding["judge"]["recorded"]["channel"] == "judge" and binding["judge"]["recorded"]["trained"] is False
    records = [Record.from_json(each) for each in (await ledger.read(table(run, EPISODES))).values()]  # type: ignore[arg-type]
    assert records and all(not each.episode.trajectories["judge"].trained for each in records)
    assert all(each.sampled["judge"] > 0 for each in records)
    start = newest_record(await ledger.read(table(run, STARTS)))
    assert start["launch"] == launch.id and start["run_settings"]["fixed"]["slots.judge"] == "judge"


@pytest.mark.parametrize(
    ("changed", "key", "reason"),
    [
        ({"channels.policy.provider": "elsewhere"}, "channels.policy.provider", "offers no inference provider"),
        ({"environment": "tests.rollout_train.rollouts.games:guess"}, "environment", "does not offer"),
        ({"slots.judge": "policy", "environment": JUDGED}, "slots.judge", "the policy would judge itself"),
    ],
)
async def test_settings_the_cluster_refuses_end_the_run_with_the_reasons(
    tmp_path: Path, local_ray: LocalRay, changed: dict[str, JsonValue], key: str, reason: str
) -> None:
    cluster = a_cluster(tmp_path)
    run = await a_run(cluster, {**POLICY, "kind": "train", "environment": WORDS, **changed})
    with pytest.raises(Refused) as refused:
        await ran(run)
    assert any(each.key == key and reason in each.reason for each in refused.value.refusals)
    ended = newest_record(await run.ledger.read(table(run.run.id, ENDS)))
    assert ended["how"] == "failed" and key in ended["detail"]
    assert not run.hosts and run.trainer_handle is None  # (it asked Ray for nothing)


async def test_a_run_that_cannot_have_its_gpus_yet_waits_and_says_what_for(tmp_path: Path, local_ray: LocalRay) -> None:
    cluster = a_cluster(tmp_path, gpus=2)
    run = await a_run(cluster, {**POLICY, "kind": "train", "environment": WORDS}, "waiting")
    said: list[str] = []

    async def noted(detail: str) -> None:
        said.append(detail)

    run.noted = noted
    going = asyncio.create_task(ran(run))
    try:
        async with asyncio.timeout(60):
            while not said:  # noqa: ASYNC110 (it says so once it has looked)
                await asyncio.sleep(0.1)
        assert said[0].startswith("waits for ") and f"run/{run.run.id}/engine/policy/0 (1 GPU" in said[0]
        presence = run.stores.ledger.presence  # pyright: ignore[reportAttributeAccessIssue]
        (beat,) = [each for each in await presence.beats() if each.runner == f"run/{run.run.id}"]
        assert beat.about["kind"] == "run" and beat.about["waiting"]
        demanded = cast(dict[str, Any], beat.about["demand"])  # (what the monitor's queue shows it waits with)
        assert demanded["gpus"] > 1 and isinstance(beat.about["asked"], float) and beat.about["reserved"] is None
    finally:
        going.cancel()
        await asyncio.gather(going, return_exceptions=True)


HALFWAY = Progress(MINIBATCH, 3, 6, packs=10, packs_total=20, fraction=0.5, tokens_per_second=5900.0, kl=0.012,
                   max_kl=0.05, eta_seconds=2040.0)  # fmt: skip


class Saying:
    """A trainer that says how far its step has got, then waits until it is let go on."""

    budget = Budget()
    weights = "lora"

    def __init__(self, model: str) -> None:
        self.told: Callable[[Progress], None] | None = None
        self.go = asyncio.Event()

    def watch(self, told: Callable[[Progress], None] | None) -> None:
        self.told = told

    async def step(self, batch: Sequence[Item], *, seed: int, parent: Files | None, into: Path) -> Step:
        assert self.told is not None
        self.told(HALFWAY)
        await self.go.wait()
        return Step({"loss": 0.5})


class Handle:
    """An actor's handle as Ray gives it, over an object in this process: each method's `remote` gives what awaits its
    answer."""

    def __init__(self, actor: object) -> None:
        self.actor = actor

    def __getattr__(self, name: str) -> Any:
        method = getattr(self.actor, name)

        async def answer(*given: Any) -> Any:
            found = method(*given)
            return await found if inspect.isawaitable(found) else found

        return SimpleNamespace(remote=answer)


async def test_a_trainer_in_an_actor_says_how_far_its_step_has_got_to_the_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jobs, "PROGRESS_LOOK", 0.01)
    actor = TrainerActor("tests.rollout_train.test_jobs:Saying", "tiny", {})
    client = TrainerClient(Handle(actor), actor.described())
    told: list[Progress] = []
    heard = asyncio.Event()

    def hear(progress: Progress) -> None:
        told.append(progress)
        heard.set()

    assert isinstance(client, Progressing)
    client.watch(hear)
    stepping = asyncio.create_task(client.step([], seed=0, parent=None, into=tmp_path))
    await asyncio.wait_for(heard.wait(), 10)
    await asyncio.sleep(0.05)
    assert told == [HALFWAY]  # (asked again and again, told once: it did not change)
    cast(Saying, actor.trainer).go.set()
    assert (await stepping).metrics == {"loss": 0.5}
    assert actor.progress() is None  # (none between steps)


async def test_a_driver_says_each_progress_noted_in_a_line_and_in_its_runners_beats(
    capsys: pytest.CaptureFixture[str],
) -> None:
    beaten: list[float] = []

    async def beat() -> None:
        beaten.append(time.monotonic())

    stepping = Stepping()
    stepping.beat = beat
    later = dataclasses.replace(HALFWAY, minibatch=4, fraction=0.51)
    stepping.on_note({"kind": "result", "group": 1})
    stepping.on_note({"kind": "progress", "step": 3, "progress": HALFWAY.to_json()})
    stepping.on_note({"kind": "progress", "step": 3, "progress": later.to_json()})
    await asyncio.sleep(0)
    assert capsys.readouterr().out.splitlines() == [
        "step 3: minibatch 3/6 · 50% · 5.9k tok/s · KL 0.012/0.05 · ETA 34 min",
        "step 3: minibatch 4/6 · 51% · 5.9k tok/s · KL 0.012/0.05 · ETA 34 min",
    ]
    assert stepping.latest == {"step": 3, **later.to_json()}
    assert len(beaten) == 1  # (beaten at once, then not again for PROGRESS_BEAT seconds)
    stepping.on_note({"kind": "progress", "step": 3, "progress": None})
    assert stepping.latest is None and capsys.readouterr().out == ""
