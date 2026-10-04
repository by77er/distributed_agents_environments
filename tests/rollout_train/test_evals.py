"""Evaluations: a frozen suite of starts, played by a checkpoint (or the base model) with nothing trained; asked for
from the page and started by a launcher; and made by a training run of its own checkpoints, between its steps."""

import asyncio
import functools
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import JsonValue

from rollout.harness.blobs import FileBlobStore
from rollout.local import LocalRunner
from rollout_train import evals as evals_module
from rollout_train import loop as loop_module
from rollout_train import testing as support
from rollout_train import train
from rollout_train.checkpoints import Checkpoints, new_id
from rollout_train.curriculum import Curriculum
from rollout_train.evals import (
    EVAL,
    NOTHING_TRAINED,
    Schedule,
    Suite,
    evaluate,
    make_suite,
    subject_table,
    suite_of,
    suites_in,
)
from rollout_train.launcher import LAUNCHER, Launcher
from rollout_train.launches import Asked, launches_of
from rollout_train.ledger import FileLedger
from rollout_train.monitor.system import System
from rollout_train.presence import presence_of
from rollout_train.profile import Profile
from rollout_train.record import EVALS, GROUPS, RESULTS, STARTS, STEPS, results, scope, table
from rollout_train.registry import registry_of
from rollout_train.rollouts import EpisodeRunner, Record, loaded, playing
from rollout_train.rollouts.scheduler import EPISODES, episodes_of
from tests.rollout_train.monitor.test_launching import OFFERED
from tests.rollout_train.rollouts.games import words
from tests.rollout_train.test_full_weights import a_ledger, a_profile
from tests.rollout_train.test_launches import Process, profiles
from tests.rollout_train.test_profile import write
from tests.rollout_train.training.test_loop import Counting, answering, here, made_by

pytest.importorskip("starlette")
from rollout_train.monitor.app import create_app

CATALOG = "tests.rollout_train.rollouts.games:words"


@pytest.fixture(autouse=True)
def quickly(monkeypatch: pytest.MonkeyPatch) -> None:
    """An eval, and a run, look for their groups' episodes in the ledger often (a run looks twice a second)."""
    monkeypatch.setattr(evals_module, "episodes_of", functools.partial(episodes_of, every=0.01))
    monkeypatch.setattr(loop_module, "episodes_of", functools.partial(episodes_of, every=0.01))


async def test_a_suite_is_a_frozen_list_of_starts_of_a_catalogs_rows(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    made = await make_suite(ledger, "words-v1", CATALOG, words, rows=["say-yes", "say-no"], seeds=[1, 2])
    assert [(start.task, start.seed) for start in made.starts] == [
        ("say-yes", 1), ("say-yes", 2), ("say-no", 1), ("say-no", 2),
    ]  # fmt: skip
    again = await suite_of(ledger, "words-v1")
    assert again is not None and again.starts == made.starts and again.catalog == CATALOG
    assert again.rows == ["say-yes", "say-no"] and again.seeds == [1, 2]
    assert await suites_in(ledger) == ["words-v1"]
    with pytest.raises(ValueError, match="never changed"):
        await make_suite(ledger, "words-v1", CATALOG, words, rows=None, seeds=[3])
    with pytest.raises(ValueError, match="no row say-perhaps"):
        await make_suite(ledger, "other", CATALOG, words, rows=["say-perhaps"], seeds=[1])
    with pytest.raises(ValueError, match="cannot be a name"):
        await make_suite(ledger, "a/b", CATALOG, words, rows=None, seeds=[1])
    every = await make_suite(ledger, "every-row", CATALOG, words, rows=None, seeds=[7])
    assert [start.task for start in every.starts] == ["say-yes", "say-no", "say-maybe"]


async def test_a_full_checkpoint_is_evaluated_in_place_of_the_engines_weights(tmp_path: Path) -> None:
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    checkpoints = Checkpoints(ledger, blobs)
    suite = await make_suite(ledger, "words-v1", CATALOG, words, rows=["say-yes"], seeds=[1])
    weights = tmp_path / "w"
    weights.mkdir()
    (weights / "model.safetensors").write_text("every weight")
    fence = await ledger.take(scope("merges"))
    subject = await checkpoints.add(fence, new_id(), weights=weights, run=None, base="tiny", kind="full")
    published: list[tuple[str, bool]] = []

    async def publish(channel: str, adapter: str, path: str, version: int | None = None, *, full: bool = False) -> int:
        published.append((adapter, full))
        return version or 0

    async with here(ledger, answering(), blobs):
        await evaluate(
            words, checkpoints, run="eval-1", suite=suite, subject=subject.id, base="tiny", channel="policy",
            directory=tmp_path / "files", publish=publish,
        )  # fmt: skip
    assert published == [(subject.id, True)]


async def test_an_eval_plays_a_suite_with_a_checkpoint_and_records_how_it_went(tmp_path: Path) -> None:
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    checkpoints = Checkpoints(ledger, blobs)
    suite = await make_suite(ledger, "words-v1", CATALOG, words, rows=["say-yes", "say-no"], seeds=[1])
    fence = await ledger.take(scope("trained"))
    weights = tmp_path / "w"
    weights.mkdir()
    (weights / "adapter.bin").write_text("weights")
    subject = await checkpoints.add(fence, new_id(), weights=weights, run="trained", step=3, base="tiny")
    published: list[tuple[str, str, int | None]] = []

    async def publish(channel: str, adapter: str, path: str, version: int | None = None, *, full: bool = False) -> int:
        published.append((channel, adapter, version))
        assert (Path(path) / "adapter.bin").read_text() == "weights" and not full
        return version or 0

    recorder = answering()
    async with here(ledger, recorder, blobs):
        said = await evaluate(
            words, checkpoints, run="eval-1", suite=suite, subject=subject.id, base="tiny", channel="policy",
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
    lines: Any = await ledger.read(table("eval-1", RESULTS))
    assert sorted(lines) == ["1", "2"] and all(line["skipped"] == NOTHING_TRAINED for line in lines.values())
    assert await checkpoints.all() == [subject]  # nothing trained, nothing made

    async with here(ledger, recorder, blobs):  # started again: what it decided and recorded is not done twice
        again = await evaluate(
            words, checkpoints, run="eval-1", suite=suite, subject=subject.id, base="tiny", channel="policy",
            directory=tmp_path / "files", publish=publish, episodes=2,
        )  # fmt: skip
    assert again == said and len(await ledger.read(table("eval-1", GROUPS))) == 2

    system = await System(ledger=ledger).evals()
    (listed,) = system["suites"]
    assert listed["suite"] == "words-v1" and listed["catalog"] == CATALOG and len(listed["starts"]) == 2
    (played,) = listed["subjects"]
    assert played["checkpoint"] == subject.id and played["played"] == 4 and played["solved"] == said["solved"]
    (run,) = system["evals"]
    assert (run["run"], run["suite"], run["checkpoint"], run["played"], run["expected"], run["done"]) == (
        "eval-1", "words-v1", subject.id, 4, 4, True,
    )  # fmt: skip
    snapshot = await System(ledger=ledger).snapshot()
    assert {each["run"]: each["kind"] for each in snapshot["runs"]} == {"eval-1": EVAL}


async def test_an_eval_of_the_base_model_serves_nothing(tmp_path: Path) -> None:
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    suite = await make_suite(ledger, "words-v1", CATALOG, words, rows=["say-no"], seeds=[1])

    async def publish(channel: str, adapter: str, path: str, version: int | None = None, *, full: bool = False) -> int:
        raise AssertionError("the base model is served as it is")

    async with here(ledger, answering(), blobs):
        said = await evaluate(
            words, Checkpoints(ledger, blobs), run="eval-base", suite=suite, subject=None, base="tiny",
            channel="policy", directory=tmp_path / "files", publish=publish,
        )  # fmt: skip
    assert said["played"] == 1
    who: Any = (await ledger.read(subject_table("words-v1", "eval-base", "subject")))["subject"]
    assert (who["kind"], who["checkpoint"], who["model"]) == ("model", None, "tiny")


async def test_a_launcher_starts_an_eval_launch_as_rollout_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    launches, heartbeats = launches_of(ledger), presence_of(ledger)
    assert launches is not None and heartbeats is not None
    started: list[list[str]] = []

    async def spawn(*command: str, stdout: Any, **_: Any) -> Process:
        started.append(list(command))
        process = Process(0)
        process.done.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    runs = tmp_path / "runs"
    found = Launcher("launcher/here", launches, heartbeats, profiles(tmp_path), [CATALOG], runs, every=0.01)
    asked = Asked("small", CATALOG, "words, best", start="best", kind=EVAL, suite="words-v1", episodes=3)
    await launches.ask(asked)
    serving = asyncio.create_task(found.serve())
    try:
        async with asyncio.timeout(5):
            while not started:  # noqa: ASYNC110 (the launcher starts it)
                await asyncio.sleep(0.01)
    finally:
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)
    (command,) = started
    assert command[2:4] == ["rollout_train.cli", "eval"] and command[5] == "words-v1"
    assert command[command.index("--episodes") + 1] == "3" and command[command.index("--checkpoint") + 1] == "best"
    assert "--groups" not in command and not any(each.startswith("trainer.start") for each in command)


async def test_an_eval_is_asked_for_from_the_page(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    await make_suite(ledger, "words-v1", CATALOG, words, rows=["say-yes"], seeds=[1])
    heartbeats, registry = presence_of(ledger), registry_of(ledger)
    assert heartbeats is not None and registry is not None
    about: JsonValue = {"kind": LAUNCHER, "profiles": [OFFERED], "catalogs": [CATALOG], "at_once": 1, "playing": 0}
    await heartbeats.beat("launcher/far", about)
    transport = httpx.ASGITransport(app=create_app(str(tmp_path / "ledger"), beat=0.0))
    async with httpx.AsyncClient(transport=transport, base_url="http://monitor") as client:
        body = {"kind": EVAL, "suite": "words-v1", "profile": OFFERED["profile"], "name": "on words", "episodes": 2}
        answer = await client.post("/api/launches", json=body)
        assert answer.status_code == 200, answer.text
        launch = answer.json()["launch"]
        assert launch["asked"]["catalog"] == CATALOG and launch["asked"]["kind"] == EVAL  # (the suite's catalog)
        missing = await client.post("/api/launches", json=body | {"suite": "no-such-suite", "name": "x"})
        assert missing.status_code == 404 and "no suite" in missing.json()["error"]
        none = await client.post("/api/launches", json=body | {"episodes": 0, "name": "y"})
        assert none.status_code == 409
        listed = (await client.get("/api/evals")).json()
        assert [each["suite"] for each in listed["suites"]] == ["words-v1"] and listed["evals"] == []


def test_the_command_makes_and_lists_suites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rollout_train.cli import main

    def run(*arguments: str) -> str:
        monkeypatch.setattr("sys.argv", ["rollout", "suite", *arguments, "--ledger", str(tmp_path / "ledger")])
        main()
        return capsys.readouterr().out

    made = run("make", "words-v1", "--catalog", CATALOG, "--rows", "say-yes, say-no", "--seeds", "1,2,3")
    assert made == f"the suite words-v1: 6 starts of {CATALOG}\n"
    assert run("list").split() == ["words-v1", "6", "starts", CATALOG]
    with pytest.raises(SystemExit, match="never changed"):
        run("make", "words-v1", "--catalog", CATALOG, "--seeds", "4")
    with pytest.raises(SystemExit, match="invalid literal"):
        run("make", "other", "--catalog", CATALOG, "--seeds", "one")


class Unmade:
    """A trainer an eval must not make."""

    weights = "lora"

    def __init__(self, model: str, **settings: Any) -> None:
        raise AssertionError("an eval makes no trainer")


def test_the_command_plays_a_suite_with_an_adapter_over_full_weights_and_makes_no_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rollout_train.cli import main

    shared, made = asyncio.run(a_ledger(tmp_path))
    asyncio.run(make_suite(FileLedger(tmp_path / "ledger"), "words-v1", CATALOG, words, rows=None, seeds=[1]))
    profile = a_profile(tmp_path, shared, "Unmade", "plain")
    profile.write_text(profile.read_text().replace("test_full_weights:Unmade", "test_evals:Unmade"))
    support.STARTED.clear()
    directory = tmp_path / "eval"
    arguments = [str(profile), "words-v1", "--checkpoint", "stacked", "--episodes", "2", "--directory", str(directory)]
    monkeypatch.setattr("sys.argv", ["rollout", "eval", *arguments, "--name", "stacked-on-words"])
    with pytest.raises(SystemExit) as exited:
        main()
    assert exited.value.code == 0
    assert capsys.readouterr().out.startswith("words-v1: solved ")
    policy = support.STARTED[0]
    assert policy.told[0].startswith(f"started {directory / 'bases' / made['merged']}")  # (what the adapter is over)
    assert f"load {made['stacked']}" in policy.told
    ledger = FileLedger(tmp_path / "ledger")
    listed = asyncio.run(System(ledger=ledger).evals())["evals"]
    assert [(each["name"], each["checkpoint"], each["played"], each["done"]) for each in listed] == [
        ("stacked-on-words", made["stacked"], 6, True)
    ]


def a_schedule(suite: Suite, every: int) -> Schedule:
    """Evals of `suite` every `every` steps, two episodes of each start, each eval the run `eval-STEP`."""

    async def run(step: int) -> str:
        return f"eval-{step}"

    return Schedule(suite, words, run, every=every, episodes=2)


async def test_a_run_evaluates_the_checkpoints_its_schedule_names_between_their_step_and_the_next(
    tmp_path: Path,
) -> None:
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    checkpoints = Checkpoints(ledger, blobs)
    suite = await make_suite(ledger, "words-v1", CATALOG, words, rows=["say-yes", "say-no"], seeds=[1])
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
        assert who["checkpoint"] == checkpoint.id and who["episodes"] == 2
        start: Any = next(iter((await ledger.read(table(said["run"], STARTS))).values()))
        assert (start["kind"], start["by"], start["from"]) == (EVAL, "train", None)
    newest = max(int(key) for key in evaluated)
    last, lines = curriculum.evaluations["words-v1"]
    assert last == made[newest].id and [line.task for line in lines] == ["say-yes", "say-no"]
    assert lines == await results(ledger, f"eval-{newest}")
    listed = (await System(ledger=ledger).evals())["suites"][0]["subjects"]
    assert {each["checkpoint"] for each in listed} == {made[int(key)].id for key in evaluated}


async def test_a_run_started_again_finishes_the_eval_it_left_before_it_steps_again(tmp_path: Path) -> None:
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    checkpoints = Checkpoints(ledger, blobs)
    suite = await make_suite(ledger, "words-v1", CATALOG, words, rows=["say-yes", "say-no"], seeds=[1])
    recorder, trainer = answering(), Counting()

    async def training(groups: int, curriculum: Curriculum | None = None) -> None:
        await train(
            words, trainer, checkpoints, base="tiny", channel="policy", directory=tmp_path / "files",
            publish=recorder.publish, groups=groups, groups_per_step=1, seed=1, curriculum=curriculum,
            evals=a_schedule(suite, every=1),
        )  # fmt: skip

    # A runner that plays the training run's episodes and not the eval's: the loop waits in the first eval.
    only = EpisodeRunner("training", ledger, LocalRunner(recorder=recorder), recorder, blobs, 6, runs={"train"})
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
    assert curriculum.evaluations["words-v1"][0] == evaluated[newest]["checkpoint"]

    again = Curriculum(words.rows())  # a run started again folds its evals into its curriculum
    async with here(ledger, recorder, blobs):
        await training(0, again)
    assert again.evaluations["words-v1"] == curriculum.evaluations["words-v1"]


async def test_a_profile_says_what_its_run_evaluates_and_each_eval_is_a_run_its_runner_plays(tmp_path: Path) -> None:
    path = write(tmp_path)
    path.write_text(path.read_text() + '\n[evals]\nsuite = "words-v1"\nevery = 2\nepisodes = 3\n')
    profile = Profile.load(path)
    assert profile.evals is not None and (profile.evals.suite, profile.evals.every, profile.evals.episodes) == (
        "words-v1", 2, 3,
    )  # fmt: skip
    async with profile.open() as platform:
        assert platform.registry is not None
        made = await platform.eval_run(2)
        assert await platform.eval_run(2) == made and made in (platform.runner.runs or ())
        names = {entry.id: entry.name for entry in await platform.registry.runs()}
        assert names[made] == f"{platform.run.name}-eval-2"
    with pytest.raises(ValueError, match="1 at least"):
        Profile.load(path, settings={"evals.every": 0})
