"""A suite of several environments: a version is a list of entries, each an environment once with its own starts,
episodes and limits; an eval plays each entry in a run of its own (its parts) and scores each apart; a training run's
schedule plays each on its channel and folds each into its curriculum, the entry named; launchers claim only what they
offer every environment of; and the page makes, edits and lists such suites, and lists the environments it knows."""

import asyncio
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import JsonValue

from rollout.curriculum import Curriculum
from rollout.harness.blobs import FileBlobStore
from rollout_train import train
from rollout_train.checkpoints import Checkpoints
from rollout_train.evals import (
    DRAWN,
    EVAL_DATA,
    GIVEN,
    SCORES,
    Schedule,
    edit_suite,
    evaluate,
    make_suite,
    subject_table,
    suite_entry,
    suite_of,
    suite_table,
)
from rollout_train.launcher import LAUNCHER, Launcher
from rollout_train.launches import EVAL, Asked, launches_of
from rollout_train.ledger import FileLedger
from rollout_train.monitor.scores import evals_of, path_of
from rollout_train.monitor.system import System
from rollout_train.presence import presence_of
from rollout_train.record import EVALS, GROUPS, RESULTS, STARTS, table
from rollout_train.rollouts.scheduler import PLANS, Plan
from tests.rollout_train.rollouts.games import guessing, words
from tests.rollout_train.support import Counting, Process, answering, here, made_by, profiles

pytest.importorskip("starlette")
from rollout_train.monitor.app import create_app

WORDS = "tests.rollout_train.rollouts.games:words"
GUESSING = "tests.rollout_train.rollouts.games:guessing"


def two_entries() -> list[Any]:
    """Two seeds of saying yes, played twice each; and one guess of each word, played once, with limits of its own."""
    return [
        suite_entry(WORDS, words, rows=["say-yes"], seeds=[1, 2], episodes=2),
        suite_entry(GUESSING, guessing, rows=None, seeds=[7], thinking_tokens=10, answer_tokens=7),
    ]


async def test_a_version_is_a_list_of_entries_each_environment_once(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    made = await make_suite(ledger, "mixed", two_entries())
    assert made.environments == [WORDS, GUESSING] and len(made.starts) == 2 + 3  # (numbered across, in order)
    record: Any = (await ledger.read(suite_table("mixed", "suite")))["suite"]
    assert [each["environment"] for each in record["entries"]] == [WORDS, GUESSING]
    assert record["entries"][1]["thinking_tokens"] == 10 and record["entries"][0]["episodes"] == 2
    again = await suite_of(ledger, "mixed")
    assert again is not None and again.entries == made.entries and again.configured() == made.configured()
    assert [entry.limits for entry in again.entries] == [{}, {"thinking_tokens": 10, "answer_tokens": 7}]
    assert not again.held_out  # (seed 7 of guessing is none of its eval starts)
    with pytest.raises(ValueError, match="in a version once"):
        await make_suite(ledger, "twice", [two_entries()[0], two_entries()[0]])

    first = made.entries[0]  # an edit that adds an entry keeps the other as it was, and one that drops it
    held = suite_entry(GUESSING, guessing, eval_data="guessing-held-out")
    added = await edit_suite(ledger, "mixed", [suite_entry(WORDS, words, starts=first.starts, episodes=2), held])
    assert (added.id, added.entries[0].chosen, added.entries[1].chosen) == ("mixed@2", DRAWN, EVAL_DATA)
    dropped = await edit_suite(ledger, "mixed", [added.entries[1]], base=2)
    assert dropped.environments == [GUESSING] and dropped.held_out
    with pytest.raises(ValueError, match="nothing changed"):
        await edit_suite(ledger, "mixed", list(dropped.entries))


async def test_an_eval_of_two_environments_plays_each_in_a_part_and_scores_each_apart(tmp_path: Path) -> None:
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    suite = await make_suite(ledger, "mixed", two_entries())

    async def publish(channel: str, adapter: str, path: str, version: int | None = None, *, full: bool = False) -> int:
        raise AssertionError("the base model is served as it is")

    recorder = answering()
    async with here(ledger, recorder, blobs):
        said = await evaluate(
            Checkpoints(ledger, blobs), run="eval-1", suite=suite, subject=None, base="tiny", channel="policy",
            directory=tmp_path / "files", publish=publish,
        )  # fmt: skip
    entries = {each["environment"]: each for each in said["entries"]}
    assert [each["run"] for each in said["entries"]] == ["eval-1-1", "eval-1-2"]  # (a run of its own each)
    assert entries[WORDS]["played"] == 4 and entries[GUESSING]["played"] == 3 and said["played"] == 7
    assert entries[GUESSING]["solved"] == 0 and entries[GUESSING]["reward"] == 0.0  # (it says yes and no: no word)
    assert "score" not in said and "score" not in entries[WORDS]  # (environments' rewards do not compare)
    assert [line.task for line in entries[GUESSING]["results"]] == ["guess-apple", "guess-river", "guess-candle"]

    for number, program in ((1, words.program), (2, guessing.program)):  # each part: its own plan, starts, groups
        part = f"eval-1-{number}"
        plans: Any = await ledger.read(table(part, PLANS))
        assert Plan.from_json(plans[max(plans, key=int)]).program == program
        start: Any = next(iter((await ledger.read(table(part, STARTS))).values()))
        assert (start["kind"], start["part_of"], start["entry"], start["suite"]) == (EVAL, "eval-1", number, "mixed")
        assert len(await ledger.read(table(part, GROUPS))) == len(suite.entries[number - 1].starts)
        assert len(await ledger.read(table(part, RESULTS))) == len(suite.entries[number - 1].starts)
    guessed: Any = await ledger.read(table("eval-1-2", PLANS))
    (sampling,) = [each.recorded for each in Plan.from_json(guessed[max(guessed, key=int)]).binding.models.values()]
    assert sampling is not None and (sampling.sampling.thinking_tokens, sampling.sampling.answer_tokens) == (10, 7)
    budgets = cast(list[int], recorder.channels["policy"].engines[0].budgets)  # type: ignore[attr-defined]
    assert 17 in budgets  # the guesses' limits
    assert max(budgets) > 30_000  # and the channel's own for the words: none, all the context leaves

    results: Any = await ledger.read(subject_table("mixed", "eval-1", "results"))
    assert sorted(results) == [
        "1-1",
        "1-2",
        "2-1",
        "2-2",
        "3-1",
        "4-1",
        "5-1",
    ]  # (each start by its number in the version)
    who: Any = await ledger.read(subject_table("mixed", "eval-1", "subject"))
    assert [(each["environment"], each["run"], each["episodes"]) for each in who["subject"]["parts"]] == [
        (WORDS, "eval-1-1", 2), (GUESSING, "eval-1-2", 1),
    ]  # fmt: skip
    assert who["subject"]["episodes"] is None  # (not the same for every entry)
    assert [each["environment"] for each in who[SCORES]["entries"]] == [WORDS, GUESSING] and "score" not in who[SCORES]
    assert not await ledger.read(table("eval-1", GROUPS))  # (the eval's own run plays nothing)

    system = await System(ledger=ledger).evals()
    (listed,) = system["evals"]  # (one eval: its parts are not listed apart)
    assert (listed["run"], listed["played"], listed["expected"], listed["done"]) == ("eval-1", 7, 7, True)
    assert [(each["environment"], each["played"], each["solved"]) for each in listed["entries"]] == [
        (WORDS, 4, entries[WORDS]["solved"]), (GUESSING, 3, 0),
    ]  # fmt: skip
    (shown,) = system["suites"]
    assert shown["environments"] == [WORDS, GUESSING]
    (version,) = shown["versions"]
    assert [(each["environment"], each["offset"], each["starts"]) for each in version["entries"]] == [
        (WORDS, 0, 2), (GUESSING, 2, 3),
    ]  # fmt: skip
    assert [each["environment"] for each in version["starts"]] == [WORDS] * 2 + [GUESSING] * 3
    (subject,) = shown["subjects"]
    assert [(each["environment"], each["episodes"], each["played"], each["solved"]) for each in subject["entries"]] == [
        (WORDS, 2, 4, entries[WORDS]["solved"]), (GUESSING, 1, 3, 0),
    ]  # fmt: skip
    snapshot = await System(ledger=ledger).snapshot()
    parts = {each["run"]: each["part_of"] for each in snapshot["runs"]}
    assert parts == {"eval-1": None, "eval-1-1": "eval-1", "eval-1-2": "eval-1"}


async def test_a_scheduled_eval_of_two_environments_plays_each_on_the_trained_channel(tmp_path: Path) -> None:
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    checkpoints = Checkpoints(ledger, blobs)
    suite = await make_suite(ledger, "mixed", two_entries())

    async def run_of(step: int, part: int | None = None) -> str:
        return f"eval-{step}" if part is None else f"eval-{step}-{part}"

    recorder, curriculum = answering(), Curriculum(words.rows())
    async with here(ledger, recorder, blobs):
        await train(
            words, Counting(), checkpoints, base="tiny", channel="policy", directory=tmp_path / "files",
            publish=recorder.publish, groups=4, groups_per_step=1, seed=1, curriculum=curriculum,
            evals=Schedule(suite, run_of, every=2, episodes=1),
        )  # fmt: skip
    made = {checkpoint.step: checkpoint for checkpoint in await made_by(checkpoints)}
    evaluated: Any = await ledger.read(table("train", EVALS))
    assert evaluated and sorted(int(key) for key in evaluated) == [step for step in made if step and step % 2 == 0]
    for key, said in evaluated.items():
        assert said["played"] == 2 + 3  # (one episode of each start, as the schedule says)
        assert [(each["environment"], each["run"], each["played"]) for each in said["entries"]] == [
            (WORDS, f"eval-{key}-1", 2), (GUESSING, f"eval-{key}-2", 3),
        ]  # fmt: skip
        for part in (f"eval-{key}-1", f"eval-{key}-2"):
            start: Any = next(iter((await ledger.read(table(part, STARTS))).values()))
            assert (start["by"], start["step"], start["part_of"]) == ("train", int(key), f"eval-{key}")
    newest = max(int(key) for key in evaluated)
    assert set(curriculum.evaluations) == {("mixed", WORDS), ("mixed", GUESSING)}  # (each entry, named)
    checkpoint, lines = curriculum.evaluations[("mixed", GUESSING)]
    assert checkpoint == made[newest].id and [line.task for line in lines] == [
        "guess-apple",
        "guess-river",
        "guess-candle",
    ]

    again = Curriculum(words.rows())  # a run started again folds each entry of each eval in again
    async with here(ledger, recorder, blobs):
        await train(
            words, Counting(), checkpoints, base="tiny", channel="policy", directory=tmp_path / "files",
            publish=recorder.publish, groups=0, groups_per_step=1, seed=1, curriculum=again,
            evals=Schedule(suite, run_of, every=2, episodes=1),
        )  # fmt: skip
    assert again.evaluations == curriculum.evaluations

    tables = {name: await ledger.read(name) for name in await ledger.tables()}
    (shown, *_) = evals_of(tables, made[newest].id)
    assert [(each["environment"], each["played"]) for each in shown["entries"]] == [(WORDS, 2), (GUESSING, 3)]
    line = path_of(tables, {each.id: each for each in await checkpoints.all()}, made[newest].id)
    scored = line["points"][-1]["scores"]["mixed@1"]
    assert set(scored["entries"]) == {WORDS, GUESSING} and scored["entries"][GUESSING]["solved"] == 0.0
    assert line["suites"][0]["environments"] == [WORDS, GUESSING]


async def test_a_launcher_claims_only_what_it_offers_every_environment_of(
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
    found = Launcher("launcher/here", launches, heartbeats, profiles(tmp_path), [WORDS], tmp_path / "runs", every=0.01)
    both = await launches.ask(Asked("small", WORDS, "both", kind=EVAL, suite="mixed@1", environments=[GUESSING]))
    await launches.ask(Asked("small", WORDS, "words only", kind=EVAL, suite="words@1"))
    serving = asyncio.create_task(found.serve())
    try:
        async with asyncio.timeout(5):
            while not started:  # noqa: ASYNC110 (the launcher starts the one it can)
                await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
    finally:
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)
    assert [command[command.index("--name") + 1] for command in started] == ["words only"]
    assert next(each for each in await launches.all() if each.id == both.id).state == "asked"


async def test_the_page_makes_and_edits_a_suite_of_entries_and_lists_the_environments_it_knows(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    await ledger.take("suites/none")
    heartbeats = presence_of(ledger)
    assert heartbeats is not None
    offered: JsonValue = {
        "kind": LAUNCHER,
        "profiles": [{"profile": "small", "path": "/p/small.toml", "kinds": ["eval"], "model": "m", "settings": {}}],
        "environments": [WORDS],
        "at_once": 1,
        "playing": 0,
    }
    await heartbeats.beat("launcher/far", offered)
    transport = httpx.ASGITransport(app=create_app(str(tmp_path / "ledger"), beat=0.0))
    words_entry = {"environment": WORDS, "chosen": DRAWN, "rows": ["say-yes"], "seeds": [5], "episodes": 2}
    guesses = {"environment": GUESSING, "chosen": GIVEN, "starts": [{"task": "guess-river", "seed": 3}]}
    async with httpx.AsyncClient(transport=transport, base_url="http://monitor") as client:
        answer = await client.post("/api/suites/mixed", json={"entries": [words_entry, guesses | {"answer_tokens": 9}]})
        assert answer.status_code == 200, answer.text
        kept = {"environment": WORDS, "chosen": "same", "episodes": 3}  # (an edit keeps an entry's starts)
        edited = await client.post("/api/suites/mixed", json={"entries": [kept], "base": 1})
        assert edited.status_code == 200 and edited.json()["version"] == "mixed@2"
        for why, body in {
            "an entry of an environment it keeps no starts of": {"entries": [{**kept, "environment": GUESSING}]},
            "an entry whose environment does not load": {"entries": [{**words_entry, "environment": "no.such:x"}]},
        }.items():
            answer = await client.post("/api/suites/mixed", json=body)
            assert answer.status_code == 409 and answer.json()["error"], why
        listed = (await client.get("/api/evals")).json()
        known = (await client.get("/api/environments")).json()["environments"]
        launched = await client.post(
            "/api/launches", json={"kind": EVAL, "suite": "mixed@1", "profile": "small", "name": "on both"}
        )
    (suite,) = listed["suites"]
    assert [each["environments"] for each in suite["versions"]] == [[WORDS, GUESSING], [WORDS]]
    assert suite["versions"][0]["entries"][1]["answer_tokens"] == 9
    found = await suite_of(ledger, "mixed@2")
    assert found is not None and found.entries[0].episodes == 3 and found.entries[0].chosen == DRAWN
    assert {each["environment"]: (each["name"], each["offered"]) for each in known} == {
        WORDS: ("words", True), GUESSING: ("guessing", False),
    }  # fmt: skip
    assert next(each for each in known if each["environment"] == WORDS)["versions"] == ["1"]
    assert launched.status_code == 404 and GUESSING in launched.json()["error"]  # (no launcher offers it)
