"""Evaluations: a suite (an eval configuration, kept in versions), played by a checkpoint (or the base model) with
nothing trained; asked for from the page and started by a launcher; and made by a training run of its own checkpoints,
between its steps."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from rollout.curriculum import Curriculum
from rollout.harness.blobs import FileBlobStore
from rollout.local import LocalRunner
from rollout_train import testing as support
from rollout_train import train
from rollout_train.checkpoints import Checkpoints, new_id
from rollout_train.evals import (
    DRAWN,
    EVAL,
    EVAL_DATA,
    NOTHING_TRAINED,
    edit_suite,
    evaluate,
    make_suite,
    subject_table,
    suite_entry,
    suite_for,
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
from rollout_train.resuming import _asked  # pyright: ignore[reportPrivateUsage]
from rollout_train.rollouts import EpisodeRunner, Record, loaded, playing
from rollout_train.rollouts.scheduler import EPISODES
from tests.rollout_train.rollouts.games import words
from tests.rollout_train.support import (
    OFFERED,
    Counting,
    Process,
    a_ledger,
    a_profile,
    answering,
    here,
    made_by,
    profiles,
    write,
)

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

    async def publish(channel: str, adapter: str, path: str, version: int | None = None, *, full: bool = False) -> int:
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

    async def publish(channel: str, adapter: str, path: str, version: int | None = None, *, full: bool = False) -> int:
        published.append((channel, adapter, version))
        assert (Path(path) / "adapter.bin").read_text() == "weights" and not full
        return version or 0

    recorder = answering()
    async with here(ledger, recorder, blobs):
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
        asked = await _asked(ledger, "eval-1", start, None)
        assert (asked.kind, asked.suite, asked.episodes, asked.environment) == (EVAL, "words-v1@1", 2, ENVIRONMENT)


async def test_an_eval_of_the_base_model_serves_nothing(tmp_path: Path) -> None:
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    suite = await make_suite(ledger, "words-v1", [suite_entry(ENVIRONMENT, words, rows=["say-no"], seeds=[1])])

    async def publish(channel: str, adapter: str, path: str, version: int | None = None, *, full: bool = False) -> int:
        raise AssertionError("the base model is served as it is")

    async with here(ledger, answering(), blobs):
        said = await evaluate(
            Checkpoints(ledger, blobs), run="eval-base", suite=suite, subject=None, base="tiny",
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
    found = Launcher("launcher/here", launches, heartbeats, profiles(tmp_path), [ENVIRONMENT], runs, every=0.01)
    asked = Asked("small", ENVIRONMENT, "words, best", start="best", kind=EVAL, suite="words-v1", episodes=3)
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
    assert command[command.index("--environment") + 1] == ENVIRONMENT
    assert "--groups" not in command and not any(each.startswith("trainer.start") for each in command)


async def test_an_eval_is_asked_for_from_the_page(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    await make_suite(ledger, "words-v1", [suite_entry(ENVIRONMENT, words, rows=["say-yes"], seeds=[1])])
    heartbeats, registry = presence_of(ledger), registry_of(ledger)
    assert heartbeats is not None and registry is not None
    about: JsonValue = {
        "kind": LAUNCHER,
        "profiles": [OFFERED],
        "environments": [ENVIRONMENT],
        "at_once": 1,
        "playing": 0,
    }
    await heartbeats.beat("launcher/far", about)
    async with monitor_client(str(tmp_path / "ledger"), beat=0.0) as client:
        body = {"kind": EVAL, "suite": "words-v1", "profile": OFFERED["profile"], "name": "on words", "episodes": 2}
        answer = await client.post("/api/launches", json=body)
        assert answer.status_code == 200, answer.text
        launch = answer.json()["launch"]
        assert (
            launch["asked"]["environment"] == ENVIRONMENT and launch["asked"]["kind"] == EVAL
        )  # (the suite's environment)
        missing = await client.post("/api/launches", json=body | {"suite": "no-such-suite", "name": "x"})
        assert missing.status_code == 404 and "no suite" in missing.json()["error"]
        none = await client.post("/api/launches", json=body | {"episodes": 0, "name": "y"})
        assert none.status_code == 409
        unplayed = body | {"suite": "words-held-out", "environment": ENVIRONMENT, "name": "held out"}
        answer = await client.post("/api/launches", json=unplayed)  # (frozen when it is first played)
        assert answer.status_code == 200 and answer.json()["launch"]["asked"]["suite"] == "words-held-out"
        listed = (await client.get("/api/evals")).json()
        assert [each["suite"] for each in listed["suites"]] == ["words-v1"] and listed["evals"] == []


async def test_a_profile_without_a_trainer_is_asked_for_evals_only_and_a_published_environment_with_its_profiles(
    tmp_path: Path,
) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    await make_suite(ledger, "words-v1", [suite_entry(ENVIRONMENT, words, rows=["say-yes"], seeds=[1])])
    heartbeats = presence_of(ledger)
    assert heartbeats is not None
    published = f"words@{'0' * 64}"
    serving: dict[str, Any] = {
        **OFFERED,
        "profile": "serving",
        "kinds": [EVAL],
        "settings": {"episodes_at_once": 6},
        "published": [],
    }
    boxed: dict[str, Any] = {**OFFERED, "profile": "boxed", "pools": ["box"], "published": [published]}
    about: dict[str, Any] = {
        "kind": LAUNCHER, "profiles": [serving, boxed], "environments": [ENVIRONMENT, published], "at_once": 1,
        "playing": 0,
    }  # fmt: skip
    await heartbeats.beat("launcher/far", about)
    async with monitor_client(str(tmp_path / "ledger"), beat=0.0) as client:
        evaluating = {"kind": EVAL, "suite": "words-v1", "profile": "serving", "name": "on words"}
        answer = await client.post("/api/launches", json=evaluating)
        assert answer.status_code == 200, answer.text
        training: dict[str, Any] = {"profile": "serving", "environment": ENVIRONMENT, "name": "trained"}
        training["settings"] = {"evals.suite": None}
        refused = await client.post("/api/launches", json=training)
        assert refused.status_code == 409 and "launches evals only" in refused.json()["error"]
        unplayed = await client.post("/api/launches", json=training | {"environment": published})
        assert unplayed.status_code == 409  # (evals only, whatever it plays)
        elsewhere = training | {"profile": "boxed", "environment": published, "name": "boxed"}
        assert (await client.post("/api/launches", json=elsewhere)).status_code == 200
        on_serving = evaluating | {"suite": "words-held-out", "environment": published, "name": "unplayed"}
        missing = await client.post("/api/launches", json=on_serving)
        assert missing.status_code == 404 and published in missing.json()["error"]  # (not with that profile)


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
    assert unplayed[-1].split() == ["words-held-out", "6", "starts", ENVIRONMENT, "(not", "played", "yet)"]
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
    asyncio.run(
        make_suite(FileLedger(tmp_path / "ledger"), "words-v1", [suite_entry(ENVIRONMENT, words, rows=None, seeds=[1])])
    )
    profile = a_profile(tmp_path, shared, "Unmade", "plain")
    profile.write_text(profile.read_text().replace("test_full_weights:Unmade", "test_evals:Unmade"))
    support.STARTED.clear()
    directory = tmp_path / "eval"
    arguments = [str(profile), "words-v1", "--checkpoint", "stacked", "--episodes", "2", "--directory", str(directory)]
    monkeypatch.setattr("sys.argv", ["rollout", "eval", *arguments, "--name", "stacked-on-words"])
    with pytest.raises(SystemExit) as exited:
        main()
    assert exited.value.code == 0
    assert capsys.readouterr().out.startswith(f"words-v1@1 {ENVIRONMENT}: solved ")
    policy = support.STARTED[0]
    assert policy.told[0].startswith(f"started {directory / 'bases' / made['merged']}")  # (what the adapter is over)
    assert f"load {made['stacked']}" in policy.told
    assert not (directory / "bases").exists() and not (directory / "checkpoints").exists()  # (deleted once it ended)
    ledger = FileLedger(tmp_path / "ledger")
    listed = asyncio.run(System(ledger=ledger).evals())["evals"]
    assert [(each["name"], each["checkpoint"], each["played"], each["done"]) for each in listed] == [
        ("stacked-on-words", made["stacked"], 6, True)
    ]
    registry = registry_of(ledger)
    assert registry is not None
    (run,) = [each.id for each in asyncio.run(registry.runs()) if each.name == "stacked-on-words"]
    (started,) = asyncio.run(ledger.read(table(run, STARTS))).values()
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
    only = EpisodeRunner("training", ledger, LocalRunner(recorder=recorded), recorded, blobs, 6, runs={"train"})
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


SCHEDULED = """
directory = "{directory}"

[channels.policy]
model = "a-checkpoint"
renderer = "rollout_train.testing:plain_renderer"
engine = "rollout_train.testing:scripted_engine"

[trainer]
kind = "tests.rollout_train.support:Steps"
channel = "policy"
segment_tokens = 900
segments_per_step = 3

[evals]
suite = "words-held-out"
"""


async def test_a_scheduled_eval_names_its_environments_eval_data_frozen_on_first_use(tmp_path: Path) -> None:
    from rollout_train.cli import _train  # pyright: ignore[reportPrivateUsage]

    path = write(tmp_path, SCHEDULED)
    await _train(path, None, ENVIRONMENT, groups=2, groups_per_step=1, seed=1)
    ledger = FileLedger(tmp_path / "run" / "ledger")
    suite = await suite_of(ledger, "words-held-out")
    assert suite is not None and suite.held_out and suite.starts == list(words.evals()["words-held-out"])
    (run,) = [name.split("/")[1] for name in await ledger.tables() if name.endswith(f"/{EVALS}")]
    evaluated: Any = await ledger.read(table(run, EVALS))
    assert evaluated and all(each["suite"] == "words-held-out" and each["played"] == 6 for each in evaluated.values())


def test_the_command_plays_a_suite_with_a_base_model_it_names_and_records_it_as_the_subject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rollout_train.cli import main

    shared, _ = asyncio.run(a_ledger(tmp_path))
    asyncio.run(
        make_suite(FileLedger(tmp_path / "ledger"), "words-v1", [suite_entry(ENVIRONMENT, words, rows=None, seeds=[1])])
    )
    profile = a_profile(tmp_path, shared, "Unmade", "plain")
    profile.write_text(profile.read_text().replace("test_full_weights:Unmade", "test_evals:Unmade"))
    support.STARTED.clear()
    arguments = [str(profile), "words-v1", "--model", "org/another-base", "--directory", str(tmp_path / "eval")]
    monkeypatch.setattr("sys.argv", ["rollout", "eval", *arguments, "--name", "words on another base"])
    with pytest.raises(SystemExit) as exited:
        main()
    assert exited.value.code == 0 and capsys.readouterr().out.startswith(f"words-v1@1 {ENVIRONMENT}: solved ")
    assert support.STARTED[0].told[0].startswith("started org/another-base")  # (not the profile's model)
    system = System(ledger=FileLedger(tmp_path / "ledger"))
    history = asyncio.run(system.history("model", "org/another-base"))
    assert history is not None and [each["name"] for each in history["evals"]] == ["words on another base"]
    assert asyncio.run(system.lineage())["bases"] == ["a-checkpoint", "org/another-base"]  # (a root of the graph)


CLUSTER = """
name = "here"
[ledger]
url = "sqlite:///~/ledger.db"
[inference.local]
kind = "vllm"
[inference.local.models."org/base-a"]
context = 4096
[inference.local.models."org/base-b"]
context = 4096
[inference.tinker]
kind = "tinker"
[inference.tinker.models."org/elsewhere"]
context = 4096
"""


async def test_an_eval_on_a_base_model_goes_from_the_page_to_the_launchers_ray_job(tmp_path: Path) -> None:
    import shlex
    import tomllib

    from rollout_train.cluster import parsed
    from rollout_train.launches import ASKED, ENDED
    from tests.rollout_train.support import PROFILE
    from tests.rollout_train.test_ray_launcher import Jobs, until_state

    ledger = FileLedger(tmp_path / "ledger")
    await make_suite(ledger, "words-v1", [suite_entry(ENVIRONMENT, words, rows=["say-yes"], seeds=[1])])
    launches, heartbeats = launches_of(ledger), presence_of(ledger)
    assert launches is not None and heartbeats is not None
    offering = tmp_path / "profiles"
    offering.mkdir()
    served = PROFILE.replace("rollout_train.testing:scripted_engine", "rollout_vllm:VllmEngine", 1)  # (the policy's)
    (offering / "vllm.toml").write_text(served.format(directory=tmp_path / "run"))
    jobs = Jobs(steps=0)
    found = Launcher(
        "launcher/here", launches, heartbeats, offering, [ENVIRONMENT], tmp_path / "runs",
        ray="http://127.0.0.1:8265", every=0.01, cluster=parsed(tomllib.loads(CLUSTER)),
    )  # fmt: skip
    found._client = lambda: jobs  # type: ignore[method-assign]  # (no Ray cluster: the stand-in)
    await found._beat()  # pyright: ignore[reportPrivateUsage]  (it says what it offers)
    async with monitor_client(str(tmp_path / "ledger"), beat=0.0) as client:
        (launcher,) = (await client.get("/api/launches")).json()["launchers"]
        assert launcher["profiles"][0]["models"] == ["a-checkpoint", "org/base-a", "org/base-b"]  # (not Tinker's)
        body = {"kind": EVAL, "suite": "words-v1", "profile": "vllm", "name": "words on b", "model": "org/base-b"}
        answer = await client.post("/api/launches", json=body)
        assert answer.status_code == 200, answer.text
        made = answer.json()["launch"]
        asked = made["asked"]
        assert (asked["model"], asked["start"], asked["environment"]) == ("org/base-b", None, ENVIRONMENT)
        unoffered = await client.post("/api/launches", json=body | {"model": "org/elsewhere", "name": "elsewhere"})
        assert unoffered.status_code == 404
        said = "no launcher alive offers the base model 'org/elsewhere' with the profile 'vllm'"
        assert unoffered.json()["error"] == said
        both = await client.post("/api/launches", json=body | {"start": "best", "name": "both"})
        assert both.status_code == 409 and "not both" in both.json()["error"]
        training = {"profile": "vllm", "environment": ENVIRONMENT, "name": "trained", "model": "org/base-a"}
        trained = await client.post("/api/launches", json=training | {"settings": {"evals.suite": None}})
        assert trained.status_code == 409 and "only an eval" in trained.json()["error"]
    other = Asked("vllm", ENVIRONMENT, "unoffered", kind=EVAL, suite="words-v1", model="org/elsewhere")
    elsewhere = await launches.ask(other)
    await found._step()  # pyright: ignore[reportPrivateUsage]
    (submitted,) = jobs.submitted  # (the launch whose base model it offers, and not the other)
    assert submitted["submission_id"] == f"run-{made['id']}"
    command = shlex.split(submitted["entrypoint"].split(" && exec ", 1)[1])
    assert command[2:4] == ["rollout_train.cli", "eval"] and command[command.index("--model") + 1] == "org/base-b"
    assert "--checkpoint" not in command and command[command.index("--name") + 1] == "words on b"
    assert (await until_state(launches, made["id"], ENDED)).state == ENDED
    assert next(each for each in await launches.all() if each.id == elsewhere.id).state == ASKED
