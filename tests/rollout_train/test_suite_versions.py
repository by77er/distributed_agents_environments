"""A suite's versions: an edit makes a new version and moves the suite's name to it, every eval keeps the version it
played, a suite made before versions reads as its version 1, and a training run plays the version its suite's name
points to when each step is decided. The page makes and edits suites, and checks the evals a training run asked for
says: a suite the ledger has, of environments the cluster offers."""

import random
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from rollout.harness.blobs import FileBlobStore
from rollout_train import train
from rollout_train.checkpoints import Checkpoints
from rollout_train.evals import (
    DRAWN,
    EVAL_DATA,
    GIVEN,
    Schedule,
    edit_suite,
    evaluate,
    make_suite,
    subject_table,
    suite_entry,
    suite_for,
    suite_of,
    suite_table,
    versions_of,
)
from rollout_train.ledger import FileLedger
from rollout_train.monitor.system import System
from rollout_train.record import EVALS, STEPS, table
from rollout_train.registry import Taken, registry_of
from rollout_train.stores import Stores
from tests.rollout_train.clusters import POLICY, a_cluster
from tests.rollout_train.rollouts.games import guessing, words
from tests.rollout_train.support import Counting, answering, here, signed_in

pytest.importorskip("starlette")
from rollout_train.monitor.app import create_app
from tests.rollout_train.support import monitor_client

ENVIRONMENT = "tests.rollout_train.rollouts.games:words"
GUESSING = "tests.rollout_train.rollouts.games:guessing"


async def test_an_edit_makes_a_new_version_and_moves_the_suites_name_to_it(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    first = await make_suite(
        ledger, "words-v1", [suite_entry(ENVIRONMENT, words, rows=["say-yes", "say-no"], seeds=[5])]
    )
    same = first.starts
    second = await edit_suite(ledger, "words-v1", [suite_entry(ENVIRONMENT, words, starts=same, episodes=3)], base=1)
    assert (second.id, second.edited_from, second.starts, second.entries[0].episodes) == ("words-v1@2", 1, same, 3)
    assert second.entries[0].chosen == DRAWN  # (its starts as they were: chosen as they were)
    redrawn = suite_entry(ENVIRONMENT, words, rows=["say-maybe"], seeds=[5, 6], thinking_tokens=64)
    third = await edit_suite(ledger, "words-v1", [redrawn], base=2)
    assert [(start.task, start.seed) for start in third.starts] == [("say-maybe", 5), ("say-maybe", 6)]
    (entry,) = third.entries
    assert entry.limits == {"thinking_tokens": 64} and entry.episodes == 1 and entry.chosen == DRAWN
    found = await suite_of(ledger, "words-v1")  # (the name: its newest version)
    assert found is not None and found.id == "words-v1@3"
    older = await suite_of(ledger, "words-v1@1")  # (a version, by id: never changed, never deleted)
    assert older is not None and older.starts == first.starts and older.entries[0].episodes == 1
    assert [each.id for each in await versions_of(ledger, "words-v1")] == ["words-v1@1", "words-v1@2", "words-v1@3"]
    assert await suite_of(ledger, "words-v1@9") is None
    records: Any = await ledger.read(suite_table("words-v1", "suite"))
    assert list(records) == ["suite", "2", "3"]  # (version 1 where suites always were; each later one by number)
    registry = registry_of(ledger)
    assert registry is not None
    assert [(each.name, each.version) for each in await registry.suites()] == [("words-v1", "words-v1@3")]

    kept = third.starts
    with pytest.raises(ValueError, match="edited meanwhile"):  # (made from version 2: the newest is 3)
        await edit_suite(ledger, "words-v1", [suite_entry(ENVIRONMENT, words, starts=kept, episodes=4)], base=2)
    with pytest.raises(ValueError, match="nothing changed"):
        await edit_suite(ledger, "words-v1", [suite_entry(ENVIRONMENT, words, starts=kept, thinking_tokens=64)])
    with pytest.raises(ValueError, match="no row say-perhaps"):
        suite_entry(ENVIRONMENT, words, rows=["say-perhaps"], seeds=[1])
    with pytest.raises(ValueError, match="1 at least"):
        suite_entry(ENVIRONMENT, words, starts=kept, episodes=0)
    with pytest.raises(KeyError, match="no eval data 'nothing'"):
        suite_entry(ENVIRONMENT, words, eval_data="nothing")
    with pytest.raises(KeyError, match="no suite"):
        await edit_suite(ledger, "never-made", [suite_entry(ENVIRONMENT, words, starts=kept)])
    with pytest.raises(ValueError, match="an entry at least"):
        await edit_suite(ledger, "words-v1", [])
    held = await edit_suite(ledger, "words-v1", [suite_entry(ENVIRONMENT, words, eval_data="words-held-out")])
    (entry,) = held.entries
    assert (held.id, entry.chosen, entry.eval_data, held.held_out) == ("words-v1@4", EVAL_DATA, "words-held-out", True)

    await registry.point_suite("words-v1", "words-v1@2")  # (a name moved back by hand is read where it points)
    assert (found := await suite_of(ledger, "words-v1")) is not None and found.id == "words-v1@2"
    assert (await registry.point_suite("words-v1", "words-v1@1", forward=True)).version == "words-v1@2"


@pytest.mark.parametrize("kind", ["files", "database"])
async def test_a_suites_name_points_to_a_version_beside_the_ledger(tmp_path: Path, kind: str) -> None:
    from rollout_train.database import DatabaseLedger

    ledger = FileLedger(tmp_path / "files") if kind == "files" else DatabaseLedger(f"sqlite:///{tmp_path / 'db'}")
    registry = registry_of(ledger)
    assert registry is not None and await registry.suites() == []
    await registry.create("a-run")
    await registry.name_dataset("guesses", "kkkkmmmm")
    assert (await registry.point_suite("words", "words@2")).version == "words@2"
    assert (await registry.point_suite("words", "words@3", forward=True)).version == "words@3"
    assert (await registry.point_suite("words", "words@2", forward=True)).version == "words@3"  # (not back)
    assert (await registry.point_suite("words", "words@1")).version == "words@1"  # (moved by hand, anywhere)
    await registry.point_suite("other", "other@1")
    pointed = [(each.name, each.version) for each in await registry.suites()]
    assert pointed == [("other", "other@1"), ("words", "words@1")]
    assert [each.name for each in await registry.runs()] == ["a-run"]  # (the rest is as it was)
    assert [each.name for each in await registry.datasets()] == ["guesses"]
    with pytest.raises(Taken):
        await registry.point_suite("a/b", "a/b@1")


async def test_every_eval_keeps_the_version_it_played(tmp_path: Path) -> None:
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    checkpoints = Checkpoints(ledger, blobs)
    first = await make_suite(
        ledger, "words-v1", [suite_entry(ENVIRONMENT, words, rows=["say-yes", "say-no"], seeds=[5])]
    )
    second = await edit_suite(
        ledger, "words-v1", [suite_entry(ENVIRONMENT, words, rows=["say-no"], seeds=[5], episodes=2)]
    )

    async def publish(channel: str, adapter: str, path: str, version: int | None = None, *, full: bool = False) -> int:
        raise AssertionError("the base model is served as it is")

    async with here(ledger, answering(), blobs):
        for run, suite in (("eval-first", first), ("eval-second", second)):
            await evaluate(
            checkpoints, run=run, suite=suite, subject=None, base="tiny", channel="policy",
                directory=tmp_path / "files", publish=publish,
            )  # fmt: skip
    who: Any = (await ledger.read(subject_table("words-v1", "eval-second", "subject")))["subject"]
    assert (who["version"], who["episodes"]) == ("words-v1@2", 2)  # (the version's episodes of each start)
    listed = await System(ledger=ledger).evals()
    (suite,) = listed["suites"]
    assert suite["version"] == "words-v1@2" and [len(each["starts"]) for each in suite["versions"]] == [2, 1]
    assert {each["subject"]: (each["version"], each["played"]) for each in suite["subjects"]} == {
        "eval-first": ("words-v1@1", 2), "eval-second": ("words-v1@2", 2),
    }  # fmt: skip
    assert {each["run"]: (each["version"], each["expected"]) for each in listed["evals"]} == {
        "eval-first": ("words-v1@1", 2), "eval-second": ("words-v1@2", 2),
    }  # fmt: skip
    identities = [[start["identity"] for start in each["starts"]] for each in suite["versions"]]
    assert identities[1] == identities[0][1:]  # (say-no with seed 5 is the same start in both)


@pytest.mark.parametrize("named", ["words-v1", "words-v1@1"])
async def test_a_scheduled_eval_plays_the_version_its_suites_name_points_to_when_its_step_is_decided(
    tmp_path: Path, named: str
) -> None:
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    checkpoints = Checkpoints(ledger, blobs)
    suite = await make_suite(
        ledger, "words-v1", [suite_entry(ENVIRONMENT, words, rows=["say-yes", "say-no"], seeds=[5])]
    )
    edited: list[str] = []

    async def wanted() -> dict[str, JsonValue]:
        if not edited and await ledger.read(table("train", STEPS)):  # (once the first step is decided, it is edited)
            again = [suite_entry(ENVIRONMENT, words, rows=["say-no"], seeds=[5])]
            edited.append((await edit_suite(ledger, "words-v1", again)).id)
        return {}

    async def run_of(step: int, part: int | None = None) -> str:
        return f"eval-{step}"

    async def scheduled(name: str, every: int, episodes: int | None) -> Schedule | None:
        return Schedule(await suite_for(ledger, name, ENVIRONMENT, words), run_of, every, episodes)

    recorder = answering()
    async with here(ledger, recorder, blobs):
        await train(
            words, Counting(), checkpoints, base="tiny", channel="policy", directory=tmp_path / "files",
            publish=recorder.publish, groups=6, groups_per_step=1, seed=1, desired=wanted, scheduled=scheduled,
            evals=Schedule(suite, run_of, named=named),
        )  # fmt: skip
    assert edited == ["words-v1@2"]
    steps: Any = await ledger.read(table("train", STEPS))
    versions = [steps[key]["suite_version"] for key in sorted(steps, key=int)]
    assert all(steps[key]["settings"]["evals.suite"] == named for key in steps)  # (as it was set)
    if named == "words-v1@1":  # (one version, for good: the edit changes nothing it plays)
        assert len(versions) >= 2 and set(versions) == {"words-v1@1"}
        return
    assert len(versions) >= 2 and versions[0] == "words-v1@1" and set(versions[1:]) == {"words-v1@2"}
    evaluated: Any = await ledger.read(table("train", EVALS))
    assert evaluated["1"]["version"] == "words-v1@1" and evaluated["1"]["played"] == 2
    later = [each for key, each in evaluated.items() if key != "1"]
    assert later and all(each["version"] == "words-v1@2" and each["played"] == 1 for each in later)
    for each in evaluated.values():
        who: Any = (await ledger.read(subject_table("words-v1", each["run"], "subject")))["subject"]
        assert who["version"] == each["version"]


async def test_the_page_makes_and_edits_suites_and_refuses_what_cannot_be(tmp_path: Path) -> None:
    ledger = FileLedger(tmp_path / "ledger")
    await ledger.take("suites/none")  # (a ledger of files, as the monitor finds it)
    async with monitor_client(str(tmp_path / "ledger"), beat=0.0) as client:
        said = (await client.get(f"/api/environments/{ENVIRONMENT}")).json()
        assert [row["key"] for row in said["rows"]] == ["say-yes", "say-no", "say-maybe"]
        assert said["evals"] == {"words-held-out": 6} and said["version"] == "1"
        assert (await client.get("/api/environments/no.such:thing")).status_code == 404

        drawn = {"environment": ENVIRONMENT, "chosen": DRAWN, "rows": ["say-yes"], "seeds": [5], "episodes": 2}
        answer = await client.post("/api/suites/words-new", json=drawn)
        assert answer.status_code == 200, answer.text
        assert answer.json() == {"suite": "words-new", "version": "words-new@1", "number": 1}
        given = {"environment": ENVIRONMENT, "chosen": GIVEN, "starts": [{"task": "say-maybe", "seed": 3}]}
        assert (await client.post("/api/suites/words-given", json=given)).status_code == 200
        held = {"environment": ENVIRONMENT, "chosen": EVAL_DATA, "eval_data": "words-held-out"}
        assert (await client.post("/api/suites/words-held", json=held)).status_code == 200
        refused: dict[str, tuple[dict[str, Any], str]] = {
            "an environment that does not load": ({**drawn, "environment": "no.such:thing"}, "words-x"),
            "no environment": ({key: value for key, value in drawn.items() if key != "environment"}, "words-x"),
            "a row it lacks": ({**drawn, "rows": ["say-perhaps"]}, "words-x"),
            "seeds that are no numbers": ({**drawn, "seeds": ["one"]}, "words-x"),
            "no seeds": ({**drawn, "seeds": []}, "words-x"),
            "eval data it lacks": ({**held, "eval_data": "nothing"}, "words-x"),
            "no episodes": ({**drawn, "episodes": 0}, "words-x"),
            "a limit below 1": ({**drawn, "thinking_tokens": 0}, "words-x"),
            "a start of a row it lacks": ({**given, "starts": [{"task": "say-perhaps", "seed": 1}]}, "words-x"),
            "no way of choosing": ({**drawn, "chosen": "somehow"}, "words-x"),
            "a name that is no name": (drawn, "a@b"),
            "the same starts of no suite": ({**drawn, "chosen": "same"}, "words-x"),
            "an environment in two entries": ({"entries": [drawn, drawn]}, "words-x"),
            "no entries": ({"entries": []}, "words-x"),
            "an edit made from another version": ({"chosen": "same", "episodes": 3, "base": 2}, "words-new"),
            "an edit that changes nothing": ({"chosen": "same", "episodes": 2, "base": 1}, "words-new"),
        }
        for why, (body, name) in refused.items():
            answer = await client.post(f"/api/suites/{name}", json=body)
            assert answer.status_code == 409 and answer.json()["error"], why
        assert (await client.post("/api/suites/words-x", content=b"not json")).status_code == 400

        edited = await client.post("/api/suites/words-new", json={"chosen": "same", "episodes": 3, "base": 1})
        assert edited.status_code == 200 and edited.json()["version"] == "words-new@2"
        redrawn = {"chosen": DRAWN, "rows": None, "seeds": [5], "answer_tokens": 32, "base": 2}
        assert (await client.post("/api/suites/words-new", json=redrawn)).json()["version"] == "words-new@3"
        listed = (await client.get("/api/evals")).json()
    suites = {each["suite"]: each for each in listed["suites"]}
    assert set(suites) == {"words-new", "words-given", "words-held"}
    shown = suites["words-new"]
    assert shown["version"] == "words-new@3" and len(shown["starts"]) == 3 and shown["environments"] == [ENVIRONMENT]
    entries = [(each["id"], each["entries"][0]) for each in shown["versions"]]
    assert [(version, entry["episodes"], entry["answer_tokens"]) for version, entry in entries] == [
        ("words-new@1", 2, None), ("words-new@2", 3, None), ("words-new@3", 1, 32),
    ]  # fmt: skip
    second = await suite_of(ledger, "words-new@2")
    assert second is not None and second.edited_from == 1
    given_suite = await suite_of(ledger, "words-given")
    assert given_suite is not None and given_suite.entries[0].chosen == GIVEN
    assert given_suite.starts[0].parameters == words.start(words.rows()[2], random.Random(3))  # (drawn as `drawn` does)
    assert suites["words-held"]["versions"][0]["held_out"] is True


async def test_the_evals_a_training_run_asked_for_says_are_checked(tmp_path: Path) -> None:
    cluster = a_cluster(tmp_path)  # (it offers the words, and not the guessing game)
    ledger = Stores.open(cluster).ledger
    await make_suite(ledger, "words-v1", [suite_entry(ENVIRONMENT, words, rows=["say-yes"], seeds=[5])])
    await make_suite(ledger, "guesses", [suite_entry(GUESSING, guessing, rows=["guess-apple"], seeds=[5])])
    app = create_app(f"sqlite:///{tmp_path}/ledger.db", beat=0.0, cluster=cluster)
    asked: dict[str, Any] = {"kind": "train", "name": "a run", "environment": ENVIRONMENT}
    cases: dict[str, tuple[dict[str, Any], str | None]] = {
        "none": ({}, None),
        "none, said so": ({"evals.suite": None}, None),
        "a suite of its environment": ({"evals.suite": "words-v1", "evals.every": 2}, None),
        "one version of it": ({"evals.suite": "words-v1@1"}, None),
        "a version it does not have": ({"evals.suite": "words-v1@4"}, "has versions 1 to 1"),
        "its environment's eval data, which is no suite": ({"evals.suite": "words-held-out"}, "there is no suite"),
        "a suite the ledger does not have": ({"evals.suite": "words-v9"}, "there is no suite"),
        "a suite of an environment the cluster does not offer": ({"evals.suite": "guesses"}, "does not offer"),
        "a suite named by no name": ({"evals.suite": 3}, "is text or null"),
    }
    async with signed_in(app) as client:
        for why, (settings, refused) in cases.items():
            body: dict[str, Any] = {**asked, "settings": {**POLICY, **settings}}
            said = (await client.post("/api/launches/check", json=body)).json()
            reasons = [each["reason"] for each in said["refusals"] if each["key"] == "evals.suite"]
            assert any(refused in each for each in reasons) if refused else reasons == [], (why, said)
