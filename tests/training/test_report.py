"""Reporting a run: the summary in words, the chart, and the post to a webhook (on made-up groups)."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from rollout.rollouts import Row
from rollout.training import Curriculum, Directory, iterations
from rollout.training.report import chart, hours, post, report, summary

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
        {"iteration": 1, "time": 1800.0, "task": "r1", "rewards": [9.0, 9.0], "solved": [True, True], "failed": 0,
         "seconds": 1800, "unlocked": 3, "update": "skipped: every episode scored the same"},
        {"iteration": 2, "time": 3600.0, "task": "r3", "rewards": [0.0, 2.0, 10.0, 11.0],
         "solved": [False, True, True, True], "failed": 0, "seconds": 1800, "unlocked": 7, "adapter": "step-1",
         "version": 1, "sequences_recorded": 680, "sequences_trained": 384,
         "update": {**update, "gradient_norm": 0.42}},
    ]  # fmt: skip


def run(tmp_path: Path) -> tuple[Path, Curriculum]:
    directory = tmp_path / "run-1"
    store = Directory(directory)
    for line in groups():
        store.append("metrics.jsonl", json.dumps(line))
    curriculum = Curriculum(ROWS)
    curriculum.update(ROWS[0], [9.0, 9.0], [True, True])
    curriculum.update(ROWS[2], [0.0, 2.0, 10.0, 11.0], [False, True, True, True])
    store.write("curriculum.json", json.dumps(curriculum.saved()))
    return directory, curriculum


def test_the_summary_gives_the_latest_group_what_its_update_did_and_each_rows_record(tmp_path: Path) -> None:
    directory, curriculum = run(tmp_path)
    lines = iterations(Directory(directory))
    assert hours(lines) == [0.5, 1.0]
    text = summary("run-1", lines, curriculum)
    assert "**run-1** — group 2, 1.0 h in, 1 updates (serving step-1)" in text
    assert "rewards 0 / 2 / 10 / 11 (mean 5.75, sd 4.82); solved 3/4" in text
    assert "384 of 680 sequences, 6800 sampled tokens" in text
    assert "moved the policy by KL ≈ 0.0123 (floor 0.0009), 9 steps" in text and "clipped 0.4%" in text
    assert "**Rows:** 7 of 7 unlocked" in text
    assert "`r1` row 1 — 1 groups, solved 100%, mean reward 9.0" in text
    assert "`r3` row 3 — 1 groups, solved 75%, mean reward 5.8" in text and "`r2` row 2 — not tried yet" in text
    assert len(text) <= 2000
    assert "no group has finished yet" in summary("empty", [], Curriculum(ROWS))


async def test_a_report_writes_the_summary_and_the_chart_into_the_runs_directory(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    directory, _ = run(tmp_path)
    image = chart(iterations(Directory(directory)), ROWS)
    assert image.startswith(b"\x89PNG") and len(image) > 10_000
    await report(directory, ROWS, None)
    assert (directory / "progress.png").read_bytes().startswith(b"\x89PNG")
    assert "`r3` row 3 — 1 groups, solved 75%" in (directory / "progress.md").read_text()  # (the saved curriculum)


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
