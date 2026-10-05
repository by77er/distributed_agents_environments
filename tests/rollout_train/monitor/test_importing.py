"""Importing an environment from git through the monitor: `POST /api/environments/import` imports it (the version, or
why it was refused), the published versions are listed and read one by one, the import's stages are said as they go,
and a published environment is in the environments' list and has a page, from what its check recorded."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from rollout.harness.blobs import FileBlobStore
from rollout_train import publishing
from rollout_train.launcher import LAUNCHER
from rollout_train.ledger import FileLedger
from rollout_train.presence import presence_of
from rollout_train.publishing import Importer
from rollout_train.record import scope
from tests.rollout_train.sources import TINY, git, repository

pytest.importorskip("starlette")
from tests.rollout_train.support import OFFERED, monitor_client


@pytest.fixture
def checked_here(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Checks run in this process, on the source as the test made it, rather than in a Ray job."""
    asked: list[dict[str, Any]] = []

    async def here(jobs: str, entry_point: str, runtime_env: Any, **_: Any) -> dict[str, Any]:
        asked.append({"jobs": jobs, "entry_point": entry_point, "runtime_env": runtime_env})
        return await asyncio.to_thread(publishing.report, entry_point)

    monkeypatch.setattr(publishing, "checked_on_ray", here)
    return asked


async def test_an_import_makes_a_version_listed_read_and_shown_as_an_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checked_here: list[dict[str, Any]]
) -> None:
    source = repository(tmp_path / "source", TINY)
    monkeypatch.syspath_prepend(str(source))  # pyright: ignore[reportUnknownMemberType]
    ledger = FileLedger(tmp_path / "ledger")
    await ledger.take(scope("elsewhere"))  # (a ledger of files, with a fence)
    importer = Importer(FileBlobStore(tmp_path / "blobs"), "http://ray:8265", tmp_path / "scratch")
    async with monitor_client(str(tmp_path / "ledger"), beat=0.0, importer=importer) as client:
        answer = await client.post("/api/environments/import", json={"url": str(source), "ref": "main"})
        assert answer.status_code == 200, answer.text
        made = answer.json()
        version = made["version"]
        assert not made["existing"] and version["name"] == "words" and version["ref"] == "main"
        assert version["commit"] == git(source, "rev-parse", "HEAD").strip()
        assert [each["check"] for each in version["check"]][-1] == "episode"
        again = (await client.post("/api/environments/import", json={"url": str(source)})).json()
        assert again["existing"] and again["version"]["version"] == version["version"] and len(checked_here) == 1
        listed = (await client.get("/api/environments/versions")).json()
        assert [each["reference"] for each in listed["versions"]] == [version["reference"]] and listed["importing"]
        one = await client.get(f"/api/environments/versions/{version['reference']}")
        assert one.json()["entry_point"] == "words:environment"
        assert (await client.get(f"/api/environments/versions/{'0' * 64}")).status_code == 404
        imports = (await client.get("/api/environments/imports")).json()["imports"]
        assert [(each["stage"], each["version"]) for each in imports] == [("done", version["reference"])] * 2

        heartbeats = presence_of(ledger)
        assert heartbeats is not None
        offered = {**OFFERED, "published": [version["reference"]]}  # (the profile plays it)
        offer: dict[str, Any] = {"kind": LAUNCHER, "profiles": [offered], "environments": [version["reference"]]}
        await heartbeats.beat("launcher/far", offer)
        asked = {"profile": "one-gpu", "environment": version["reference"], "name": "on words"}
        refused = await client.post("/api/launches", json={**asked, "settings": {"evals.suite": "other"}})
        assert refused.status_code == 409 and "no eval data of that name" in refused.json()["error"]
        launched = await client.post("/api/launches", json={**asked, "settings": {"evals.suite": "words-eval"}})
        assert launched.status_code == 200 and launched.json()["launch"]["asked"]["environment"] == version["reference"]
        (line,) = (await client.get("/api/environments")).json()["environments"]
        assert (line["environment"], line["name"], line["offered"]) == (
            version["reference"], f"words@{version['version'][:12]}", True,
        )  # fmt: skip
        assert line["published"]["source"] == str(source) and line["published"]["passed"]
        page = (await client.get(f"/api/environments/{version['reference']}")).json()
        assert page["loads"] and [row["key"] for row in page["rows"]] == ["say-yes", "say-no"]
        assert page["evals"] == {"words-eval": 4} and page["published"]["commit"] == version["commit"]


async def test_an_import_refused_says_why_and_a_monitor_without_an_importer_imports_nothing(
    tmp_path: Path, checked_here: list[dict[str, Any]]
) -> None:
    await FileLedger(tmp_path / "ledger").take(scope("elsewhere"))
    async with monitor_client(str(tmp_path / "ledger"), beat=0.0) as client:
        answer = await client.post("/api/environments/import", json={"url": str(tmp_path / "nowhere")})
        assert answer.status_code == 409 and "imports no environment" in answer.json()["error"]
        assert not (await client.get("/api/environments/versions")).json()["importing"]
    importer = Importer(FileBlobStore(tmp_path / "blobs"), "http://ray:8265", tmp_path / "scratch")
    bare = repository(tmp_path / "bare", {"README.md": "no project here"})
    async with monitor_client(str(tmp_path / "ledger"), beat=0.0, importer=importer) as client:
        answer = await client.post("/api/environments/import", json={"url": str(tmp_path / "nowhere")})
        assert answer.status_code == 422 and "does not clone" in answer.json()["error"]
        answer = await client.post("/api/environments/import", json={"url": str(bare)})
        assert answer.status_code == 422 and "no pyproject.toml" in answer.json()["error"]
        assert (await client.post("/api/environments/import", json={"ref": "main"})).status_code == 409
        assert (await client.post("/api/environments/import", content=b"[")).status_code == 400
        imports = (await client.get("/api/environments/imports")).json()["imports"]
        assert [each["stage"] for each in imports] == ["refused", "refused"]
        assert "no pyproject.toml" in imports[0]["error"]
    assert checked_here == []
