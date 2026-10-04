"""Heartbeats beside the ledger, where a run's blobs are, and how a machine is doing."""

import subprocess
import time
from pathlib import Path

import pytest

from rollout.harness.blobs import FileBlobStore
from rollout_train import machine, presence
from rollout_train.database import DatabaseLedger
from rollout_train.ledger import FileLedger, Ledger
from rollout_train.presence import STALE, Beat, alive, presence_of
from rollout_train.stores import FILES, location, opened


def ledgers(tmp_path: Path) -> list[Ledger]:
    return [FileLedger(tmp_path / "files"), DatabaseLedger(f"sqlite:///{tmp_path / 'ledger.db'}")]


@pytest.mark.parametrize("kind", [0, 1], ids=["files", "database"])
async def test_a_runner_beats_and_its_newest_beat_is_kept_with_its_recent_measurements(
    tmp_path: Path, kind: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(presence, "KEPT", 3)
    found = presence_of(ledgers(tmp_path)[kind])
    assert found is not None and await found.beats() == []
    for number in range(5):
        await found.beat("here", {"run": "train", "playing": number, "machine": {"at": number}, "host": "h"})
    await found.beat("there", {"run": "other"})
    here, there = await found.beats()
    assert (here.runner, there.runner) == ("here", "there")
    assert here.about == {"run": "train", "playing": 4, "machine": {"at": 4}, "host": "h"}
    assert [point["playing"] for point in here.history] == [2, 3, 4]  # (the newest KEPT, oldest first)
    assert set(here.history[-1]) == {"at", "machine", "playing"}  # (what is measured, not who it is)
    assert alive(here) and there.history == [{"at": there.at}]


def test_a_runner_is_alive_while_it_beat_within_the_last_stale_seconds() -> None:
    beat = Beat("here", 1000.0, {})
    assert alive(beat, now=1000.0 + STALE) and not alive(beat, now=1000.0 + STALE + 1) and not alive(None)


async def test_a_store_is_noted_without_its_credentials_and_opened_again_from_the_note(tmp_path: Path) -> None:
    assert location({}, tmp_path / "blobs") == {"kind": FILES, "directory": str(tmp_path / "blobs")}
    given = {"kind": "rollout_s3:S3BlobStore", "bucket": "b", "prefix": "runs", "secret_access_key": "x", "Token": "y"}
    assert location(given, tmp_path) == {"kind": "rollout_s3:S3BlobStore", "bucket": "b", "prefix": "runs"}
    store = opened(location({}, tmp_path / "blobs"))
    assert isinstance(store, FileBlobStore)
    written = await FileBlobStore(tmp_path / "blobs").put(b"events", "application/octet-stream")
    assert await store.read(written) == b"events"
    named = opened({"kind": "rollout.harness.blobs:FileBlobStore", "directory": str(tmp_path / "blobs")})
    assert isinstance(named, FileBlobStore)


def test_a_machine_is_measured_and_one_without_accelerators_has_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(*_: object, **__: object) -> None:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", missing)
    measured = machine.measured(tmp_path)
    assert measured["accelerators"] == [] and measured["disk"]["total"] > 0
    assert measured["memory"]["total"] is None or measured["memory"]["total"] > 0
    assert abs(measured["at"] - time.time()) < 5 and machine.measured()["disk"] is None
