"""Rewriting a database ledger for what moved: the locations of the stores of files that moved name the new store, the
blob references into them carry its URIs, paths under a moved directory are under its new one, what did not move is
left as it was, and running it again changes nothing more."""

import json
from pathlib import Path

import pytest
from pydantic import JsonValue

from rollout_train.database import DatabaseLedger
from rollout_train.launches import Asked
from rollout_train.ledger import Fence
from rollout_train.relocating import Relocation, main, relocated
from rollout_train.stores import FILES

HOME = Path("/home/someone")
CACHE = HOME / ".cache" / "rollout"
INTO = {"kind": "rollout_s3:S3BlobStore", "bucket": "blobs", "prefix": "blobs/"}
DIGEST = "ab" + "c" * 62


def reference(directory: Path) -> dict[str, JsonValue]:
    return {"uri": (directory / DIGEST[:2] / DIGEST).as_uri(), "sha256": DIGEST, "size": 3, "media_type": "text/plain"}


def relocation() -> Relocation:
    return Relocation(
        INTO, "s3://blobs/blobs/", frozenset({CACHE / "gateway" / "blobs", CACHE / "datasets" / "blobs"}),
        {str(CACHE): "/root/.cache/rollout"}, HOME,
    )  # fmt: skip


def test_locations_references_and_paths_move_and_the_rest_stays() -> None:
    start = {
        "directory": str(CACHE / "runs" / "eval"),
        "profile": "/home/someone/Code/profile.toml",
        "blobs": {"kind": FILES, "directory": "~/.cache/rollout/gateway/blobs"},
    }
    assert relocation().moved(start) == {
        "directory": "/root/.cache/rollout/runs/eval",
        "profile": "/home/someone/Code/profile.toml",
        "blobs": INTO,
    }
    turn = {"blob": reference(CACHE / "gateway" / "blobs"), "text": "a rollout"}
    assert relocation().moved(turn) == {"blob": reference(CACHE / "gateway" / "blobs") | {
        "uri": f"s3://blobs/blobs/ab/{DIGEST}"
    }, "text": "a rollout"}  # fmt: skip
    in_a_run = reference(CACHE / "runs" / "old" / "blobs")  # (a run whose start names no store: its directory's)
    assert relocation().moved({"events": in_a_run}) == {
        "events": in_a_run | {"uri": Path(f"/root/.cache/rollout/runs/old/blobs/ab/{DIGEST}").as_uri()}
    }
    elsewhere = {"kind": FILES, "directory": "/elsewhere/blobs"}
    assert relocation().moved({"blobs": elsewhere, "n": 3, "list": [str(CACHE)]}) == {
        "blobs": elsewhere, "n": 3, "list": ["/root/.cache/rollout"],
    }  # fmt: skip
    assert relocation().moved(str(CACHE) + "-other") == str(CACHE) + "-other"  # (only what is under the directory)


async def test_a_ledgers_records_and_launches_are_rewritten_once(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'ledger.db'}"
    ledger = DatabaseLedger(url)
    fence: Fence = await ledger.take("runs/eval")
    start: dict[str, JsonValue] = {
        "directory": str(CACHE / "runs" / "eval"),
        "blobs": {"kind": FILES, "directory": str(CACHE / "gateway" / "blobs")},
    }
    await ledger.append("runs/eval/starts", "1", start, fence)
    await ledger.append("runs/eval/episodes", "1/1", {"events": reference(CACHE / "gateway" / "blobs")}, fence)
    await ledger.append("runs/eval/groups", "1", {"episodes": 1}, fence)
    launch = await ledger.launches.ask(Asked("one", "e:e", "eval", directory=str(CACHE / "runs" / "eval")))
    ledger.close()
    assert relocated(url, relocation()) == {"ledger_records": 2, "launches": 1, "run_settings": 0}
    assert relocated(url, relocation()) == {"ledger_records": 0, "launches": 0, "run_settings": 0}
    ledger = DatabaseLedger(url)
    assert (await ledger.read("runs/eval/starts"))["1"] == {
        "directory": "/root/.cache/rollout/runs/eval",
        "blobs": INTO,
    }
    assert (await ledger.read("runs/eval/episodes"))["1/1"] == {
        "events": reference(CACHE / "gateway" / "blobs") | {"uri": f"s3://blobs/blobs/ab/{DIGEST}"}
    }
    (asked,) = await ledger.launches.all()
    assert asked.id == launch.id and asked.asked.directory == "/root/.cache/rollout/runs/eval"
    ledger.close()


def test_the_command_says_what_it_rewrote(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    url = f"sqlite:///{tmp_path / 'ledger.db'}"
    DatabaseLedger(url).close()
    main([url, "--into", json.dumps(INTO), "--uris", "s3://blobs/blobs/", "--store", str(CACHE / "gateway" / "blobs")])
    assert "0 rows of ledger_records rewritten" in capsys.readouterr().out
