"""Reporting a run: the summary in words, the chart, and the post to a webhook (on made-up groups)."""

from pathlib import Path
from typing import Any

import httpx
import pytest

from rollout.catalog import Row
from rollout_train import Curriculum, FileLedger, Iteration, iterations
from rollout_train.record import ITERATIONS, table
from rollout_train.report import chart, hours, post, report, summary

ROWS = [Row(f"r{number}", f"row {number}") for number in range(1, 8)]


def groups() -> list[dict[str, Any]]:
    update = {
        "loss": -0.002,
        "clip_fraction": 0.004,
        "mean_ratio": 1.0002,
        "kl_moved": 0.0123,
        "kl_floor": 0.0009,
        "optimizer_steps": 9.0,
        "tokens": 6800.0,
    }
    return [
        {"iteration": 1, "time": 1800.0, "task": "r1", "title": "row 1", "rewards": [9.0, 9.0],
         "solved": [True, True], "failed": 0, "seconds": 1800, "unlocked": 3,
         "skipped": "every episode scored the same"},
        {"iteration": 2, "time": 3600.0, "task": "r3", "title": "row 3", "rewards": [0.0, 2.0, 10.0, 11.0],
         "solved": [False, True, True, True], "failed": 0, "seconds": 1800, "unlocked": 7, "adapter": "step-1",
         "version": 1, "segments_recorded": 680, "segments_trained": 384,
         "update": {**update, "gradient_norm": 0.42}},
    ]  # fmt: skip


async def run(tmp_path: Path) -> tuple[Path, list[Iteration], Curriculum]:
    """A run's directory with two groups in its ledger; its iterations; and the curriculum they fold to."""
    directory = tmp_path / "run-1"
    ledger = FileLedger(directory / "ledger")
    fence = await ledger.take("runs/train")
    for line in reversed(groups()):  # (groups do not always end in the order they were started in)
        await ledger.append(table("train", ITERATIONS), str(line["iteration"]), line, fence)
    lines = await iterations(ledger)
    curriculum = Curriculum(ROWS)
    for line in lines:
        curriculum.recorded(line)
    return directory, lines, curriculum


async def test_the_summary_gives_the_latest_group_what_its_update_did_and_each_rows_record(tmp_path: Path) -> None:
    _, lines, curriculum = await run(tmp_path)
    assert [line.iteration for line in lines] == [1, 2] and hours(lines) == [0.5, 1.0]
    text = summary("run-1", lines, curriculum)
    assert "**run-1** — group 2, 1.0 h in, 1 updates (serving step-1)" in text
    assert "rewards 0 / 2 / 10 / 11 (mean 5.75, sd 4.82); solved 3/4" in text
    assert "384 of 680 segments, 6800 sampled tokens" in text
    assert "moved the policy by KL ≈ 0.0123 (floor 0.0009), 9 steps" in text and "clipped 0.4%" in text
    assert "**Rows:** 7 of 7 unlocked" in text
    assert "`r1` row 1 — 1 groups, solved 100%, mean reward 9.0" in text
    assert "`r3` row 3 — 1 groups, solved 75%, mean reward 5.8" in text and "`r2` row 2 — not tried yet" in text
    assert len(text) <= 2000
    assert "no group has finished yet" in summary("empty", [], Curriculum(ROWS))


async def test_a_report_writes_the_summary_and_the_chart_into_the_runs_directory(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    directory, lines, _ = await run(tmp_path)
    image = chart(lines, ROWS)
    assert image.startswith(b"\x89PNG") and len(image) > 10_000
    await report(directory, ROWS, None)
    assert (directory / "progress.png").read_bytes().startswith(b"\x89PNG")
    assert "`r3` row 3 — 1 groups, solved 75%" in (directory / "progress.md").read_text()  # (folded from the ledger)


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
