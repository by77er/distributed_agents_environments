"""Reporting a run: the summary in words, the chart, and the post to a webhook (on made-up groups)."""

from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import JsonValue

from rollout.catalog import Row
from rollout.harness.blobs import FileBlobStore
from rollout_train import Curriculum, FileLedger, Result, Trained, Version, Versions, results, trained
from rollout_train.record import GROUPS, RESULTS, STEPS, table
from rollout_train.report import chart, hours, post, report, summary

MADE = "kmnopqrstuvwxyzk"
"""The version the run's step made."""
ROWS = [Row(f"r{number}", f"row {number}") for number in range(1, 8)]
METRICS = {
    "loss": -0.002,
    "clip_fraction": 0.004,
    "mean_ratio": 1.0002,
    "kl_moved": 0.0123,
    "kl_floor": 0.0009,
    "optimizer_steps": 9.0,
    "tokens": 6800.0,
    "segments": 384.0,
    "gradient_norm": 0.42,
}


def groups() -> list[dict[str, Any]]:
    return [
        {"group": 1, "time": 1800.0, "task": "r1", "title": "row 1", "rewards": [9.0, 9.0], "solved": [True, True],
         "failed": 0, "rollout_seconds": 1800, "unlocked": 3, "skipped": "every episode scored the same"},
        {"group": 2, "time": 3600.0, "task": "r3", "title": "row 3", "rewards": [0.0, 2.0, 10.0, 11.0],
         "solved": [False, True, True, True], "failed": 0, "rollout_seconds": 1800, "unlocked": 7,
         "segments_recorded": 680, "segments": 384},
    ]  # fmt: skip


async def run(tmp_path: Path) -> tuple[Path, list[Result], Curriculum, dict[int, Trained], dict[str, Version]]:
    """A run's directory with two groups in its ledger, the second trained on in a step that made a version; its
    results and what was done with them; and the curriculum they fold to."""
    directory = tmp_path / "run-1"
    ledger = FileLedger(directory / "ledger")
    fence = await ledger.take("runs/train")
    for line in reversed(groups()):  # (groups do not always end in the order they were started in)
        key, decided = str(line["group"]), {"task": line["task"], "title": line["title"]}
        await ledger.append(
            table("train", GROUPS), key, {**decided, "decided": line["time"] - line["rollout_seconds"]}, fence
        )
        kept = {
            name: value for name, value in line.items() if name not in ("group", "task", "title", "rollout_seconds")
        }
        await ledger.append(table("train", RESULTS), key, kept, fence)
    step: JsonValue = {"groups": [2], "parent": None, "makes": MADE, "segments": 384, "seed": 1}
    await ledger.append(table("train", STEPS), "1", step, fence)
    versions = Versions(ledger, FileBlobStore(directory / "blobs"))
    weights = tmp_path / "adapter.bin"
    weights.write_text("weights")
    version = await versions.add(fence, MADE, weights=weights, run="train", step=1, metrics=METRICS)
    lines = await results(ledger)
    curriculum = Curriculum(ROWS)
    for line in lines:
        curriculum.recorded(line)
    return directory, lines, curriculum, await trained(ledger), {version.id: version}


async def test_the_summary_gives_the_latest_group_what_was_done_with_it_and_each_rows_record(tmp_path: Path) -> None:
    _, lines, curriculum, steps, versions = await run(tmp_path)
    assert [line.group for line in lines] == [1, 2] and hours(lines) == [0.5, 1.0]
    text = summary("run-1", lines, curriculum, steps, versions)
    assert f"**run-1** — group 2, 1.0 h in, 1 updates (serving {MADE})" in text
    assert "rewards 0 / 2 / 10 / 11 (mean 5.75, sd 4.82); solved 3/4" in text
    assert "kmnopqrs (depth 1), 384 of 680 segments, 6800 sampled tokens" in text
    assert "moved the policy by KL ≈ 0.0123 (floor 0.0009), 9 steps" in text and "clipped 0.4%" in text
    assert "**Rows:** 7 of 7 unlocked" in text
    assert "`r1` row 1 — 1 groups, solved 100%, mean reward 9.0" in text
    assert "`r3` row 3 — 1 groups, solved 75%, mean reward 5.8" in text and "`r2` row 2 — not tried yet" in text
    assert len(text) <= 2000
    assert "segments waiting for a step" in summary("run-1", lines, curriculum)  # (before its step)
    assert "no group has finished yet" in summary("empty", [], Curriculum(ROWS))


async def test_a_report_writes_the_summary_and_the_chart_into_the_runs_directory(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    directory, lines, _, _, versions = await run(tmp_path)
    image = chart(lines, ROWS, list(versions.values()))
    assert image.startswith(b"\x89PNG") and len(image) > 10_000
    await report(directory, ROWS, None, run="train")
    assert (directory / "progress.png").read_bytes().startswith(b"\x89PNG")
    written = (directory / "progress.md").read_text()
    assert "`r3` row 3 — 1 groups, solved 75%" in written and f"(serving {MADE})" in written  # (from the ledger)


async def test_a_report_is_posted_with_the_chart_attached() -> None:
    received: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(204)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await post("https://discord.example/api/webhooks/1/secret", "progress", b"\x89PNG fake", client=client)
    (request,) = received
    body = request.content
    assert request.headers["content-type"].startswith("multipart/form-data")
    assert b'"content": "progress"' in body and b"progress.png" in body and b"\x89PNG fake" in body
