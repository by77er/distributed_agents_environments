"""Evaluations: a suite (an eval configuration, kept in versions), played by a checkpoint (or a base model) with
nothing trained; asked for from the page by its settings and submitted as a job; and made by a training run of its own
checkpoints, between its steps."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from rollout.curriculum import Curriculum
from rollout.harness.blobs import FileBlobStore
from rollout.local import LocalRunner
from rollout_train import train
from rollout_train.checkpoints import Checkpoints, new_id
from rollout_train.cluster import Cluster
from rollout_train.evals import (
    DRAWN,
    EVAL,
    EVAL_DATA,
    NOTHING_TRAINED,
    Fetched,
    edit_suite,
    evaluate,
    make_suite,
    subject_table,
    suite_entry,
    suite_for,
    suite_of,
    suites_in,
)
from rollout_train.jobs import Run, ran
from rollout_train.launches import SUBMITTED
from rollout_train.ledger import FileLedger
from rollout_train.monitor.system import System
from rollout_train.record import EVALS, GROUPS, RESULTS, STARTS, STEPS, results, scope, table
from rollout_train.resuming import _evaluated  # pyright: ignore[reportPrivateUsage]
from rollout_train.rollouts import EpisodeRunner, Record, loaded, playing
from rollout_train.rollouts.scheduler import EPISODES
from rollout_train.run_settings import RunSettings, key_of
from rollout_train.stores import Stores
from rollout_train.submitting import RayJobs
from tests.local_ray import LocalRay
from tests.rollout_train.clusters import POLICY, WORDS, a_cluster
from tests.rollout_train.rollouts.games import words
from tests.rollout_train.support import Counting, answering, here, made_by
from tests.rollout_train.test_full_weights import seeded
from tests.rollout_train.test_submitting import Jobs

pytest.importorskip("starlette")
from tests.rollout_train.support import ENVIRONMENT, a_schedule, monitor_client


async def test_a_suite_is_a_list_of_starts_of_an_environments_rows(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    made = await make_suite(
        ledger, "words-v1", [suite_entry(ENVIRONMENT, words, rows=["say-yes", "say-no"], seeds=[1, 2])]
    )
    assert [(start.task, start.seed) for start in made.starts] == [
        ("say-yes", 1), ("say-yes", 2), ("say-no", 1), ("say-no", 2),
    ]  # fmt: skip
    again = await suite_of(ledger, "words-v1")
    assert again is not None and again.starts == made.starts and again.environments == [ENVIRONMENT]
    (entry,) = again.entries
    assert entry.rows == ["say-yes", "say-no"] and entry.seeds == [1, 2]
    assert await suites_in(ledger) == ["words-v1"]
    assert (again.id, again.number, entry.chosen, entry.episodes) == ("words-v1@1", 1, DRAWN, 1)
    assert again.held_out  # (seeds 1 and 2 of these rows are its eval data's starts: training never draws them)
    with pytest.raises(ValueError, match="already: edit it"):
        await make_suite(ledger, "words-v1", [suite_entry(ENVIRONMENT, words, rows=None, seeds=[3])])
    with pytest.raises(ValueError, match="no row say-perhaps"):
        await make_suite(ledger, "other", [suite_entry(ENVIRONMENT, words, rows=["say-perhaps"], seeds=[1])])
    with pytest.raises(ValueError, match="cannot be a name"):
        await make_suite(ledger, "a/b", [suite_entry(ENVIRONMENT, words, rows=None, seeds=[1])])
    every = await make_suite(ledger, "every-row", [suite_entry(ENVIRONMENT, words, rows=None, seeds=[7])])
    assert [start.task for start in every.starts] == ["say-yes", "say-no", "say-maybe"]


async def test_an_environments_eval_data_is_frozen_as_a_suite_the_first_time_it_is_played(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    assert await suite_of(ledger, "words-held-out") is None
    made = await suite_for(ledger, "words-held-out", ENVIRONMENT, words)
    (entry,) = made.entries
    assert made.starts == list(words.evals()["words-held-out"]) and made.held_out and entry.environment_version == "1"
    assert (made.id, entry.chosen, entry.eval_data) == ("words-held-out@1", EVAL_DATA, "words-held-out")
    again = await suite_of(ledger, "words-held-out")
    assert again is not None and (again.starts, again.held_out, again.entries) == (made.starts, True, made.entries)
    assert (await suite_for(ledger, "words-held-out", ENVIRONMENT, words)).made == made.made  # (frozen: not again)
    assert (await suite_for(ledger, "words-held-out", "other:environment", words)).made == made.made  # (the ledger's)
    with pytest.raises(KeyError, match="no eval data of that name"):
        await suite_for(ledger, "words-v9", ENVIRONMENT, words)
    by_hand = await make_suite(ledger, "words-v1", [suite_entry(ENVIRONMENT, words, rows=["say-yes"], seeds=[5])])
    assert not by_hand.held_out and by_hand.entries[0].environment_version == "1"


async def test_a_full_checkpoint_is_evaluated_in_place_of_the_engines_weights(tmp_path: Path) -> None:
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    checkpoints = Checkpoints(ledger, blobs)
    suite = await make_suite(ledger, "words-v1", [suite_entry(ENVIRONMENT, words, rows=["say-yes"], seeds=[1])])
    weights = tmp_path / "w"
    weights.mkdir()
    (weights / "model.safetensors").write_text("every weight")
    fence = await ledger.take(scope("merges"))
    subject = await checkpoints.add(fence, new_id(), weights=weights, run=None, base="tiny", kind="full")
    published: list[tuple[str, bool]] = []

    async def publish(
        channel: str, adapter: str, files: Fetched, version: int | None = None, *, full: bool = False
    ) -> int:
        published.append((adapter, full))
        return version or 0

    async with here(ledger, answering(), blobs):
        await evaluate(
            checkpoints, run="eval-1", suite=suite, subject=subject.id, base="tiny", channel="policy",
            directory=tmp_path / "files", publish=publish,
        )  # fmt: skip
    assert published == [(subject.id, True)]


async def test_an_eval_plays_a_suite_with_a_checkpoint_and_records_how_it_went(tmp_path: Path) -> None:
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    checkpoints = Checkpoints(ledger, blobs)
    suite = await make_suite(
        ledger, "words-v1", [suite_entry(ENVIRONMENT, words, rows=["say-yes", "say-no"], seeds=[1])]
    )
    fence = await ledger.take(scope("trained"))
    weights = tmp_path / "w"
    weights.mkdir()
    (weights / "adapter.bin").write_text("weights")
    subject = await checkpoints.add(fence, new_id(), weights=weights, run="trained", step=3, base="tiny")
    published: list[tuple[str, str, int | None]] = []

    async def publish(
        channel: str, adapter: str, files: Fetched, version: int | None = None, *, full: bool = False
    ) -> int:
        published.append((channel, adapter, version))
        assert (Path(await files()) / "adapter.bin").read_text() == "weights" and not full
        return version or 0

    recorder = answering()
    async with here(ledger, recorder, blobs, places=1):  # (one episode at a time, in order: yes, no, yes, no)
        said = await evaluate(
            checkpoints, run="eval-1", suite=suite, subject=subject.id, base="tiny", channel="policy",
            directory=tmp_path / "files", publish=publish, episodes=2,
        )  # fmt: skip
    assert published == [("policy", subject.id, 1)]
    assert said["played"] == 4 and 0 < said["solved"] < 4  # (it says yes, then no: the yes starts are half solved)
    results: Any = await ledger.read(subject_table("words-v1", "eval-1", "results"))
    assert sorted(results) == ["1-1", "1-2", "2-1", "2-2"] and all("run_id" in each for each in results.values())
    who: Any = (await ledger.read(subject_table("words-v1", "eval-1", "subject")))["subject"]
    assert (who["kind"], who["checkpoint"], who["episodes"], who["model"]) == ("checkpoint", subject.id, 2, "tiny")
    start: Any = next(iter((await ledger.read(table("eval-1", STARTS))).values()))
    assert (start["kind"], start["suite"], start["checkpoint"]) == (EVAL, "words-v1", subject.id)
    assert (start["suite_version"], start["version"]) == ("words-v1@1", "1")  # (the suite's, and its environment's)
    lines: Any = await ledger.read(table("eval-1", RESULTS))
    assert sorted(lines) == ["1", "2"] and all(line["skipped"] == NOTHING_TRAINED for line in lines.values())
    assert await checkpoints.all() == [subject]  # nothing trained, nothing made

    async with here(ledger, recorder, blobs):  # started again: what it decided and recorded is not done twice
        again = await evaluate(
            checkpoints, run="eval-1", suite=suite, subject=subject.id, base="tiny", channel="policy",
            directory=tmp_path / "files", publish=publish, episodes=2,
        )  # fmt: skip
    assert again == said and len(await ledger.read(table("eval-1", GROUPS))) == 2

    system = await System(ledger=ledger).evals()
    (listed,) = system["suites"]
    assert listed["suite"] == "words-v1" and listed["environments"] == [ENVIRONMENT] and len(listed["starts"]) == 2
    (played,) = listed["subjects"]
    assert played["checkpoint"] == subject.id and played["played"] == 4 and played["solved"] == said["solved"]
    (run,) = system["evals"]
    assert (run["run"], run["suite"], run["checkpoint"], run["played"], run["expected"], run["done"]) == (
        "eval-1", "words-v1", subject.id, 4, 4, True,
    )  # fmt: skip
    snapshot = await System(ledger=ledger).snapshot()
    assert {each["run"]: each["kind"] for each in snapshot["runs"]} == {"eval-1": EVAL}
    (shown,) = snapshot["runs"]  # (what its environment's results say, as its start records it)
    assert shown["version"] == "1" and shown["description"] == words.description.to_json()


async def test_an_eval_started_again_asks_for_the_version_it_played_whatever_its_start_says(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    first = await make_suite(ledger, "words-v1", [suite_entry(ENVIRONMENT, words, rows=["say-yes"], seeds=[1])])
    await edit_suite(ledger, "words-v1", [suite_entry(ENVIRONMENT, words, rows=["say-no"], seeds=[1])])
    fence = await ledger.take(scope("eval-1"))
    who: JsonValue = {"kind": "model", "episodes": 2, "run": "eval-1", "version": first.id}
    await ledger.append(subject_table("words-v1", "eval-1", "subject"), "subject", who, fence)
    said: dict[str, Any] = {"kind": EVAL, "suite": "words-v1", "checkpoint": None, "environment": ENVIRONMENT}
    # (a start written before starts said their suite's version: its environment's version is under `version`)
    for start in ({**said, "version": "1"}, {**said, "version": "1", "suite_version": first.id}):
        assert await _evaluated(ledger, "eval-1", start, {}) == {"eval.suite": "words-v1@1", "eval.episodes": 2}


async def test_an_eval_of_the_base_model_serves_nothing(tmp_path: Path) -> None:
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    suite = await make_suite(ledger, "words-v1", [suite_entry(ENVIRONMENT, words, rows=["say-no"], seeds=[1])])

    async def publish(
        channel: str, adapter: str, files: Fetched, version: int | None = None, *, full: bool = False
    ) -> int:
        raise AssertionError("the base model is served as it is")

    async with here(ledger, answering(), blobs):
        said = await evaluate(
            Checkpoints(ledger, blobs), run="eval-base", suite=suite, subject=None, base="tiny",
            channel="policy", directory=tmp_path / "files", publish=publish,
        )  # fmt: skip
    assert said["played"] == 1
    who: Any = (await ledger.read(subject_table("words-v1", "eval-base", "subject")))["subject"]
    assert (who["kind"], who["checkpoint"], who["model"]) == ("model", None, "tiny")


OTHER = """
[inference.other]
kind = "vllm"
engine = "rollout_train.testing:scripted_engine"
gpus = 0.25
[inference.other.models."org/another-base"]
context = 4096
"""
"""A provider of another base model, beside the test cluster's."""
EVALUATED: dict[str, JsonValue] = {
    key: value for key, value in POLICY.items() if key.startswith("channels.") or key == "episodes_at_once"
}
"""The channel an eval on the test cluster plays on."""


async def an_eval(cluster: Cluster, name: str, settings: dict[str, JsonValue]) -> Run:
    stores = Stores.open(cluster)
    given = RunSettings({**EVALUATED, "kind": EVAL, "name": name, "environment": WORDS, **settings})
    return Run(cluster, stores, given, await stores.registry.create(name))


async def test_an_eval_is_asked_for_from_the_page_by_its_settings(tmp_path: Path) -> None:
    cluster, jobs = a_cluster(tmp_path, more=OTHER), Jobs()
    stores = Stores.open(cluster)
    await make_suite(stores.ledger, "words-v1", [suite_entry(WORDS, words, rows=["say-yes"], seeds=[1])])
    given: dict[str, JsonValue] = {**EVALUATED, "eval.suite": "words-v1", "eval.episodes": 2}

    def asked(name: str, **settings: JsonValue) -> dict[str, Any]:
        return {"kind": EVAL, "name": name, "settings": {**given, **settings}}

    another: dict[str, JsonValue] = {"channels.policy.provider": "other", "channels.policy.model": "org/another-base"}
    backends = {"ray": RayJobs("x", jobs)}
    where = f"sqlite:///{tmp_path}/ledger.db"
    async with monitor_client(where, beat=0.0, cluster=cluster, backends=backends) as client:
        answer = await client.post("/api/launches", json=asked("on words"))
        assert answer.status_code == 200, answer.text
        launch = answer.json()["launch"]
        assert launch["state"] == SUBMITTED and launch["asked"]["kind"] == EVAL
        assert launch["asked"]["settings"]["environment"] == WORDS  # (the suite's environment)
        missing = await client.post("/api/launches", json=asked("x", **{"eval.suite": "none-such"}))
        assert missing.status_code == 404 and "no suite" in missing.json()["error"]
        none = await client.post("/api/launches", json=asked("y", **{"eval.episodes": 0}))
        assert none.status_code == 422 and [each["key"] for each in none.json()["refusals"]] == ["eval.episodes"]
        based = await client.post("/api/launches", json=asked("on another base", **another))
        assert based.status_code == 200, based.text
        elsewhere = {**another, "channels.policy.model": "org/elsewhere"}
        refused = await client.post("/api/launches", json=asked("elsewhere", **elsewhere))
        assert refused.status_code == 422
        assert [each["key"] for each in refused.json()["refusals"]] == ["channels.policy.model"]
        trained = await client.post("/api/launches", json=asked("z", **{"trainer.rank": 8}))
        assert trained.status_code == 422 and "trains nothing" in trained.json()["refusals"][0]["reason"]
        listed = (await client.get("/api/evals")).json()
        assert [each["suite"] for each in listed["suites"]] == ["words-v1"] and listed["evals"] == []
    assert [each["entrypoint"].split()[-1] for each in jobs.submitted] == [launch["id"], based.json()["launch"]["id"]]


def test_the_command_makes_and_lists_suites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rollout_train.cli import main

    def run(*arguments: str) -> str:
        monkeypatch.setattr("sys.argv", ["rollout", "suite", *arguments, "--ledger", str(tmp_path / "ledger")])
        main()
        return capsys.readouterr().out

    made = run("make", "words-v1", "--environment", ENVIRONMENT, "--rows", "say-yes, say-no", "--seeds", "1,2,3")
    assert made == f"the suite words-v1@1: 6 starts of {ENVIRONMENT}\n"
    assert run("list").split() == ["words-v1@1", "6", "starts", ENVIRONMENT]
    with pytest.raises(SystemExit, match="already: edit it"):
        run("make", "words-v1", "--environment", ENVIRONMENT, "--seeds", "4")
    with pytest.raises(SystemExit, match="invalid literal"):
        run("make", "other", "--environment", ENVIRONMENT, "--seeds", "one")
    unplayed = run("list", "--environment", ENVIRONMENT).splitlines()
    assert unplayed[-1].split() == [
        "words-held-out",
        "6",
        "starts",
        ENVIRONMENT,
        "(eval",
        "data,",
        "not",
        "a",
        "suite)",
    ]
    held = run("make", "words-held-out", "--environment", ENVIRONMENT)
    assert held == f"the suite words-held-out@1: 6 starts of {ENVIRONMENT}, held out of training\n"
    assert [line.split()[0] for line in run("list", "--environment", ENVIRONMENT).splitlines()] == [
        "words-held-out@1", "words-v1@1",
    ]  # fmt: skip
    with pytest.raises(SystemExit, match="no eval data 'words-v9'"):
        run("make", "words-v9", "--environment", ENVIRONMENT)
    edited = run("edit", "words-v1", "--rows", "say-yes", "--seeds", "7", "--episodes", "3")
    assert edited == f"the suite words-v1@2: 1 starts of {ENVIRONMENT}\n"
    assert run("edit", "words-v1", "--thinking-tokens", "64") == f"the suite words-v1@3: 1 starts of {ENVIRONMENT}\n"
    assert run("list").splitlines()[-1].split() == ["words-v1@3", "1", "starts", ENVIRONMENT, "(3", "versions)"]
    with pytest.raises(SystemExit, match="nothing changed"):
        run("edit", "words-v1", "--thinking-tokens", "64", "--episodes", "3")
    guessing = "tests.rollout_train.rollouts.games:guessing"  # (an entry added, then the first dropped)
    added = run("edit", "words-v1", "--environment", guessing, "--seeds", "1", "--answer-tokens", "9")
    assert added == f"the suite words-v1@4: 4 starts of {ENVIRONMENT}, {guessing}\n"
    with pytest.raises(SystemExit, match="say the entry's environment"):
        run("edit", "words-v1", "--episodes", "2")
    dropped = run("edit", "words-v1", "--drop", ENVIRONMENT)
    assert dropped == f"the suite words-v1@5: 3 starts of {guessing}, held out of training\n"


async def test_an_eval_by_its_settings_plays_a_suite_with_an_adapter_over_full_weights_and_makes_no_trainer(
    tmp_path: Path, local_ray: LocalRay
) -> None:
    cluster, made = await seeded(tmp_path)
    stores = Stores.open(cluster)
    await make_suite(stores.ledger, "words-v1", [suite_entry(WORDS, words, rows=None, seeds=[1])])
    run = await an_eval(cluster, "stacked-on-words", {"eval.suite": "words-v1", "start": "stacked", "eval.episodes": 2})
    await ran(run)
    assert run.origin == made["stacked"] and run.trainer is None and run.trainer_handle is None
    listed = (await System(ledger=stores.ledger).evals())["evals"]
    assert [(each["name"], each["checkpoint"], each["played"], each["done"]) for each in listed] == [
        ("stacked-on-words", made["stacked"], 6, True)
    ]
    (started,) = (await stores.ledger.read(table(run.run.id, STARTS))).values()
    recorded: Any = started["run_settings"]  # type: ignore[index]
    assert recorded["fixed"]["start"] == "stacked" and recorded["fixed"]["eval.suite"] == "words-v1"
    assert recorded["fixed"]["eval.episodes"] == 2 and recorded["fixed"]["kind"] == "eval"


async def test_a_run_evaluates_the_checkpoints_its_schedule_names_between_their_step_and_the_next(
    tmp_path: Path,
) -> None:
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    checkpoints = Checkpoints(ledger, blobs)
    suite = await make_suite(
        ledger, "words-v1", [suite_entry(ENVIRONMENT, words, rows=["say-yes", "say-no"], seeds=[1])]
    )
    recorder, curriculum = answering(), Curriculum(words.rows())
    async with here(ledger, recorder, blobs):
        await train(
            words, Counting(), checkpoints, base="tiny", channel="policy", directory=tmp_path / "files",
            publish=recorder.publish, groups=8, groups_per_step=1, seed=1, curriculum=curriculum,
            evals=a_schedule(suite, every=2),
        )  # fmt: skip
    made = {checkpoint.step: checkpoint for checkpoint in await made_by(checkpoints)}
    evaluated: Any = await ledger.read(table("train", EVALS))
    assert len(made) >= 2 and sorted(int(key) for key in evaluated) == [step for step in made if step and step % 2 == 0]
    steps: Any = await ledger.read(table("train", STEPS))
    for key, said in evaluated.items():
        checkpoint = made[int(key)]
        assert (said["suite"], said["checkpoint"], said["run"], said["played"]) == (
            "words-v1", checkpoint.id, f"eval-{key}", 4,
        )  # fmt: skip
        if str(int(key) + 1) in steps:  # the next step waited for the eval
            assert steps[str(int(key) + 1)]["decided"] >= said["at"]
        for record in (await ledger.read(table(said["run"], EPISODES))).values():  # each played under that checkpoint
            episode = await loaded(Record.from_json(record), blobs)  # type: ignore[arg-type]
            spans = [span for each in episode.trajectories.values() for part in each.segments for span in part.spans]
            assert spans and {span.version for span in spans} == {checkpoint.depth}
        who: Any = (await ledger.read(subject_table("words-v1", said["run"], "subject")))["subject"]
        assert (who["checkpoint"], who["episodes"], who["asked_by"]) == (checkpoint.id, 2, "by its run's schedule")
        start: Any = next(iter((await ledger.read(table(said["run"], STARTS))).values()))
        assert (start["kind"], start["by"], start["from"]) == (EVAL, "train", None)
    newest = max(int(key) for key in evaluated)
    last, lines = curriculum.evaluations[("words-v1", ENVIRONMENT)]
    assert last == made[newest].id and [line.task for line in lines] == ["say-yes", "say-no"]
    assert lines == await results(ledger, f"eval-{newest}")
    listed = (await System(ledger=ledger).evals())["suites"][0]["subjects"]
    assert {each["checkpoint"] for each in listed} == {made[int(key)].id for key in evaluated}


async def test_a_run_started_again_finishes_the_eval_it_left_before_it_steps_again(tmp_path: Path) -> None:
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    checkpoints = Checkpoints(ledger, blobs)
    suite = await make_suite(
        ledger, "words-v1", [suite_entry(ENVIRONMENT, words, rows=["say-yes", "say-no"], seeds=[1])]
    )
    recorder, trainer = answering(), Counting()

    async def training(groups: int, curriculum: Curriculum | None = None) -> None:
        await train(
            words, trainer, checkpoints, base="tiny", channel="policy", directory=tmp_path / "files",
            publish=recorder.publish, groups=groups, groups_per_step=1, seed=1, curriculum=curriculum,
            evals=a_schedule(suite, every=1),
        )  # fmt: skip

    # A runner that plays the training run's episodes and not the eval's: the loop waits in the first eval.
    recorded = recorder.gateway(ledger, blobs)
    only = EpisodeRunner("training", ledger, LocalRunner(gateway=recorded), recorded, blobs, 6, runs={"train"})
    async with playing(only):
        going = asyncio.create_task(training(8))
        async with asyncio.timeout(10):
            while not await ledger.read(table("eval-1", GROUPS)):  # noqa: ASYNC110 (the first step's eval is asked for)
                await asyncio.sleep(0.01)
        await asyncio.sleep(0.3)  # (training groups go on being played meanwhile)
        going.cancel()
        await asyncio.gather(going, return_exceptions=True)
    (first,) = await made_by(checkpoints)
    assert list(await ledger.read(table("train", STEPS))) == ["1"] and not await ledger.read(table("train", EVALS))

    curriculum = Curriculum(words.rows())
    async with here(ledger, recorder, blobs):
        await training(0, curriculum)  # (no more groups: it trains on those played and evaluates what that makes)
    evaluated: Any = await ledger.read(table("train", EVALS))
    assert evaluated["1"]["checkpoint"] == first.id and evaluated["1"]["played"] == 4
    steps: Any = await ledger.read(table("train", STEPS))
    assert all(step["decided"] >= evaluated["1"]["at"] for key, step in steps.items() if key != "1")
    newest = max(evaluated, key=int)
    assert curriculum.evaluations[("words-v1", ENVIRONMENT)][0] == evaluated[newest]["checkpoint"]

    again = Curriculum(words.rows())  # a run started again folds its evals into its curriculum
    async with here(ledger, recorder, blobs):
        await training(0, again)
    assert again.evaluations[("words-v1", ENVIRONMENT)] == curriculum.evaluations[("words-v1", ENVIRONMENT)]


async def test_each_eval_a_runs_schedule_asks_for_is_a_run_of_its_own_that_its_runner_plays(tmp_path: Path) -> None:
    cluster = a_cluster(tmp_path)
    stores = Stores.open(cluster)
    settings = RunSettings({**POLICY, "kind": "train", "name": "scheduled", "environment": WORDS,
                            "evals.suite": "words-v1", "evals.every": 2})  # fmt: skip
    run = Run(cluster, stores, settings, await stores.registry.create("scheduled"))
    made = await run.eval_run(2)
    assert await run.eval_run(2) == made and made in run.runs and made == f"{run.run.id}-eval-2"
    names = {entry.id: entry.name for entry in await stores.registry.runs()}
    assert names[made] == "scheduled-eval-2" and await run.eval_run(2, 1) == f"{made}-1"
    every = key_of("evals.every")
    assert every is not None and "at least 1" in str(every.problem(0))


async def test_a_training_run_by_its_settings_evaluates_its_checkpoints_on_the_suite_they_name(
    tmp_path: Path, local_ray: LocalRay
) -> None:
    cluster = a_cluster(tmp_path)
    stores = Stores.open(cluster)
    await make_suite(stores.ledger, "words-v1", [suite_entry(WORDS, words, rows=["say-yes"], seeds=[1, 2])])
    settings: dict[str, JsonValue] = {**POLICY, "kind": "train", "name": "scheduled", "environment": WORDS,
                                      "groups": 4, "evals.suite": "words-v1", "evals.episodes": 3}  # fmt: skip
    run = Run(cluster, stores, RunSettings(settings), await stores.registry.create("scheduled"))
    await ran(run)
    evaluated: Any = await stores.ledger.read(table(run.run.id, EVALS))
    assert evaluated and all(each["suite"] == "words-v1" and each["played"] == 6 for each in evaluated.values())
    unmade = Run(cluster, stores, RunSettings({**settings, "name": "unmade", "evals.suite": "words-held-out"}),
                 await stores.registry.create("unmade"))  # fmt: skip
    from rollout_train.launching import Refused

    with pytest.raises(Refused, match="there is no suite words-held-out"):  # (a name never becomes a suite by itself)
        await ran(unmade)


async def test_an_eval_by_its_settings_plays_a_base_model_they_name_and_records_it_as_the_subject(
    tmp_path: Path, local_ray: LocalRay
) -> None:
    cluster = a_cluster(tmp_path, more=OTHER)
    stores = Stores.open(cluster)
    await make_suite(stores.ledger, "words-v1", [suite_entry(WORDS, words, rows=None, seeds=[1])])
    another = {"channels.policy.provider": "other", "channels.policy.model": "org/another-base"}
    run = await an_eval(cluster, "words on another base", {"eval.suite": "words-v1", **another})
    await ran(run)
    assert set(run.hosts) == {"policy"}
    system = System(ledger=stores.ledger)
    history = await system.history("model", "org/another-base")
    assert history is not None and [each["name"] for each in history["evals"]] == ["words on another base"]
    assert (await system.lineage())["bases"] == ["org/another-base"]  # (a root of the graph)
