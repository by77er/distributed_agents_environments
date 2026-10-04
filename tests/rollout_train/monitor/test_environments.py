"""The environments' list and an environment's page: every environment the system knows of, with its runs and suites
counted; and one's rows with what the training runs played of each, its eval data, curriculum, runs, suites, the evals
of its suites at its entry, and its newest check."""

import functools
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from rollout.environment import binding_for
from rollout.harness.blobs import FileBlobStore
from rollout_train import evals as evals_module
from rollout_train import loop as loop_module
from rollout_train import train
from rollout_train.check import NOTHING_TAUGHT, played
from rollout_train.checkpoints import Checkpoints
from rollout_train.evals import evaluate, make_suite, suite_entry
from rollout_train.launcher import LAUNCHER
from rollout_train.ledger import FileLedger
from rollout_train.monitor.environments import Read, Sighting, listed
from rollout_train.presence import presence_of
from rollout_train.record import GROUPS, RESULTS, STARTS, scope, table
from rollout_train.registry import registry_of
from rollout_train.rollouts.scheduler import episodes_of
from tests.rollout_train.rollouts.games import words
from tests.rollout_train.training.test_loop import Counting, answering, here, made_by

pytest.importorskip("starlette")
from rollout_train.monitor.app import create_app

WORDS = "tests.rollout_train.rollouts.games:words"
GUESSING = "tests.rollout_train.rollouts.games:guessing"


@pytest.fixture(autouse=True)
def quickly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evals_module, "episodes_of", functools.partial(episodes_of, every=0.01))
    monkeypatch.setattr(loop_module, "episodes_of", functools.partial(episodes_of, every=0.01))


async def scratch(directory: Path, groups: int = 4) -> FileLedger:
    """A ledger with two training runs on the words (`first`, then `second` from its newest checkpoint), a suite of
    two of its rows, an eval of `first`'s newest checkpoint on it, a check of the words, and a launcher alive that
    offers the words."""
    ledger, blobs = FileLedger(directory / "ledger"), FileBlobStore(directory / "blobs")
    checkpoints, recorder = Checkpoints(ledger, blobs), answering()
    started: Any = {"environment": WORDS}
    async with here(ledger, recorder, blobs):
        await train(
            words, Counting(), checkpoints, base="tiny", channel="policy", directory=directory / "first",
            publish=recorder.publish, run="first", groups=groups, groups_per_step=1, seed=1, started=started,
        )  # fmt: skip
        newest = (await made_by(checkpoints, "first"))[-1]
        await train(
            words, Counting(), checkpoints, start=newest.id, channel="policy", directory=directory / "second",
            publish=recorder.publish, run="second", groups=groups, groups_per_step=2, seed=2, started=started,
        )  # fmt: skip
        suite = await make_suite(
            ledger, "words-yes", [suite_entry(WORDS, words, rows=["say-yes", "say-no"], seeds=[1])]
        )
        await evaluate(
            checkpoints, run="eval-1", suite=suite, subject=newest.id, base="tiny", channel="policy",
            directory=directory / "eval", publish=recorder.publish, started=started,
        )  # fmt: skip
        binding = binding_for(words, "policy")
        await played(words, ledger, blobs, run="check-1", binding=binding, groups=3, episodes=4, started=started)
    registry, heartbeats = registry_of(ledger), presence_of(ledger)
    assert registry is not None and heartbeats is not None
    await registry.create("first words", "first")
    await heartbeats.beat("launcher/far", {"kind": LAUNCHER, "profiles": [], "environments": [WORDS, GUESSING]})
    return ledger


async def test_the_list_and_an_environments_page_say_what_was_done_with_it(tmp_path: Path) -> None:
    await scratch(tmp_path)
    transport = httpx.ASGITransport(app=create_app(str(tmp_path / "ledger"), beat=0.0))
    async with httpx.AsyncClient(transport=transport, base_url="http://monitor") as client:
        known = (await client.get("/api/environments")).json()["environments"]
        answer = await client.get(f"/api/environments/{WORDS}")
        assert answer.status_code == 200, answer.text
        page = answer.json()
        again = await client.get(f"/api/environments/{WORDS}", headers={"If-None-Match": answer.headers["ETag"]})
        loose = (await client.get(f"/api/environments/{GUESSING}")).json()
        missing = await client.get("/api/environments/no.such:thing")
    assert again.status_code == 304  # (served by the hub, as every page's topic is)
    assert missing.status_code == 404

    by_name = {each["environment"]: each for each in known}
    assert set(by_name) == {WORDS, GUESSING}
    words_line = by_name[WORDS]
    assert (words_line["runs"], words_line["suites"], words_line["offered"]) == (
        ["first", "second"],
        ["words-yes"],
        True,
    )
    assert by_name[WORDS]["versions"] == ["1"] and by_name[WORDS]["used"] is not None
    assert (by_name[GUESSING]["runs"], by_name[GUESSING]["suites"], by_name[GUESSING]["used"]) == ([], [], None)

    assert (page["name"], page["loads"], page["offered"], page["version"]) == ("words", True, True, "1")
    assert page["description"]["duration"] == "turns" and page["description"]["solved"] is True
    assert page["curriculum"] == {"name": "Curriculum", "own": False, "start": 3, "reach": 4}
    assert page["evals"] == {"words-held-out": 6}
    rows = page["rows"]
    assert [row["key"] for row in rows] == ["say-yes", "say-no", "say-maybe"]  # (as the environment orders them)
    assert all(row["held"] == 2 and row["trains"] for row in rows)  # (two seeds of each row are eval data)
    assert sum(row["groups"] for row in rows) == 8  # (four groups of each run)
    assert sum(row["played"] for row in rows) == sum(run["played"] for run in page["runs"])
    yes = next(row for row in rows if row["key"] == "say-yes")
    maybe = next(row for row in rows if row["key"] == "say-maybe")
    assert yes["solved"] and yes["said"] == yes["played"]  # (it says yes, then no: some solve saying yes)
    assert maybe["groups"] == 0 or maybe["solved"] == 0  # (no one says maybe)

    assert [(run["run"], run["name"]) for run in page["runs"]] == [("second", "second"), ("first", "first words")]
    assert all(run["groups"] == 4 and run["version"] == "1" for run in page["runs"])
    ((suite),) = page["suites"]
    assert (suite["suite"], suite["version"], suite["current"]) == ("words-yes", "words-yes@1", True)
    assert [(each["id"], each["starts"]) for each in suite["versions"]] == [("words-yes@1", 2)]
    ((scored),) = page["scores"]
    assert (scored["run"], scored["suite"], scored["version"], scored["kind"]) == (
        "eval-1", "words-yes", "words-yes@1", "checkpoint",
    )  # fmt: skip
    assert scored["played"] == 2 and scored["done"] and scored["checkpoint"]

    check = page["check"]
    assert check["run"] == "check-1" and check["version"] == "1" and check["ended"] is None
    assert [group["group"] for group in check["groups"]] == [1, 2, 3]
    assert all(len(group["rewards"]) == 4 for group in check["groups"])
    for group in check["groups"]:  # (a group is flagged where every episode scored the same)
        assert group["flagged"] == (len(set(group["rewards"])) == 1)
        assert group["skipped"] == (NOTHING_TAUGHT if group["flagged"] else None)

    assert (loose["loads"], loose["offered"], loose["runs"], loose["check"]) == (True, True, [], None)
    assert [row["groups"] for row in loose["rows"]] == [0, 0, 0]


async def test_an_environment_that_does_not_load_here_is_shown_from_its_runs(tmp_path: Path) -> None:
    ledger, gone = FileLedger(tmp_path / "ledger"), "far.away:maze"
    fence = await ledger.take(scope("walk"))
    said: Any = {"environment": gone, "version": "7", "description": {"solved": False, "duration": "steps"}}
    await ledger.append(table("walk", STARTS), str(fence.number), {**said, "started": time.time()}, fence)
    for number, (task, rewards) in enumerate([("hall", [1.0, 0.0]), ("cellar", [0.5]), ("hall", [1.0])], start=1):
        await ledger.append(table("walk", GROUPS), str(number), {"task": task, "title": f"the {task}"}, fence)
        line: Any = {"time": time.time(), "rewards": rewards, "solved": [False] * len(rewards), "failed": 1}
        await ledger.append(table("walk", RESULTS), str(number), line, fence)
    transport = httpx.ASGITransport(app=create_app(str(tmp_path / "ledger"), beat=0.0))
    async with httpx.AsyncClient(transport=transport, base_url="http://monitor") as client:
        page = (await client.get(f"/api/environments/{gone}")).json()
    assert (page["loads"], page["version"], page["versions"], page["curriculum"]) == (False, None, ["7"], None)
    assert "does not load here" in page["error"]
    assert page["description"]["duration"] == "steps"  # (as its newest run's start says)
    assert [(row["key"], row["title"], row["groups"], row["played"]) for row in page["rows"]] == [
        ("hall", "the hall", 2, 5),
        ("cellar", "the cellar", 1, 2),
    ]  # fmt: skip (in the order first played; failed episodes were played too)
    assert all(row["solved"] is None for row in page["rows"])  # (its results do not say)
    assert page["rows"][0]["reward"] == round(2 / 3, 4) and page["evals"] == {}
    ((run),) = page["runs"]
    assert (run["run"], run["groups"], run["played"], run["solved"]) == ("walk", 3, 7, None)


def test_the_list_folds_every_sources_sightings() -> None:
    def packages(read: Read) -> list[Sighting]:  # (another source: a table of published environments, say)
        return [Sighting("pkg:maze", version="3"), Sighting(WORDS, version="2")]

    def runs(read: Read) -> list[Sighting]:
        return [
            Sighting(WORDS, version="1", run="a", at=5.0),
            Sighting(WORDS, run="b", at=9.0),
            Sighting(WORDS, at=7.0),
        ]

    found = {each["environment"]: each for each in listed(Read({}), [packages, runs])}
    assert found[WORDS] | {"name": None} == {
        "environment": WORDS, "name": None, "versions": ["1", "2"], "offered": False, "runs": ["a", "b"], "suites": [],
        "used": 9.0,
    }  # fmt: skip
    assert (found["pkg:maze"]["name"], found["pkg:maze"]["versions"], found["pkg:maze"]["used"]) == (
        "maze",
        ["3"],
        None,
    )
