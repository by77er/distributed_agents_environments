"""Reporting a run: the summary in words, the chart, and the post to a webhook (on made-up iterations)."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from minecraft_swarm.curriculum import Curriculum
from minecraft_swarm.report import chart, hours, load, post, summary
from minecraft_swarm.tasks import catalog


def iterations() -> list[dict[str, Any]]:
    update = {"loss": -0.002, "clip_fraction": 0.004, "mean_ratio": 1.0002, "approx_kl": 0.0123, "tokens": 6800.0}
    return [
        {"iteration": 1, "task": "t001", "rewards": [9.0, 9.0], "solved": [True, True], "failed": 0, "seconds": 1800,
         "unlocked": 7, "adapter_step": 0, "update": "skipped: every episode scored the same"},
        {"iteration": 2, "task": "t003", "rewards": [0.0, 2.0, 10.0, 11.0], "solved": [False, True, True, True],
         "failed": 0, "seconds": 1800, "unlocked": 7, "adapter_step": 1,
         "update": {**update, "sequences": 96.0, "gradient_norm": 0.42},
         "inference": {"tokens_per_second": 512.3, "mean_concurrency": 9.1, "generated_tokens": 120000}},
    ]  # fmt: skip


def run(tmp_path: Path) -> Path:
    directory = tmp_path / "run-1"
    directory.mkdir()
    (directory / "metrics.jsonl").write_text("".join(json.dumps(line) + "\n" for line in iterations()))
    tasks = catalog()
    curriculum = Curriculum(tasks)
    curriculum.update(tasks[0], [9.0, 9.0], [True, True])
    curriculum.update(tasks[2], [0.0, 2.0, 10.0, 11.0], [False, True, True, True])
    curriculum.save(directory / "curriculum.json")
    return directory


def test_the_summary_gives_the_latest_group_the_task_set_and_the_trainers_statistics(tmp_path: Path) -> None:
    directory = run(tmp_path)
    lines = load(directory)
    assert hours(lines) == [0.5, 1.0]
    text = summary(directory, lines)
    assert "iteration 2, 1.0 h in, 1 updates (adapter step 1)" in text
    assert "rewards 0 / 2 / 10 / 11 (mean 5.75, sd 4.82); solved 3/4" in text
    assert "KL to the sampling policy ≈ 0.0123" in text and "clipped 0.4%" in text and "96 turns" in text
    assert "**Task set:** 7 of 57 unlocked" in text
    assert "`t001` skills, diamonds, items, kit none — 1 groups, solved 100%, mean reward 9.0" in text
    assert "`t003` skills, diamonds, chests, kit none — 1 groups, solved 75%, mean reward 5.8" in text
    assert "`t002` skills, diamonds, items, kit none, easy — not tried yet" in text
    assert "512.3 tokens/s" in text
    assert len(text) <= 2000
    assert "no iteration has finished yet" in summary(tmp_path / "empty", [])


def test_the_chart_is_a_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    image = chart(load(run(tmp_path)))
    assert image.startswith(b"\x89PNG") and len(image) > 10_000


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
