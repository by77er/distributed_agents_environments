"""Posting a run's report to a webhook: a post the webhook refuses, or that cannot reach it, is said by its status
and never by the webhook's address (a secret); a report that watches says so and posts again after the next group."""

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import JsonValue

from rollout.testing import until
from rollout_train import report as report_module
from rollout_train.record import RESULTS, table
from rollout_train.report import NotPosted, post, report
from tests.rollout_train.training.test_report import ROWS, run

WEBHOOK = "https://discord.example/api/webhooks/1234/the-secret-token"


async def test_a_post_the_webhook_refuses_says_its_status_and_not_its_address() -> None:
    def busy(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"retry_after": 1.0})

    async with httpx.AsyncClient(transport=httpx.MockTransport(busy)) as client:
        with pytest.raises(NotPosted, match="429") as refused:
            await post(WEBHOOK, "progress", None, client=client)
    assert "the-secret-token" not in str(refused.value) and refused.value.__cause__ is None

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {request.url}", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(unreachable)) as client:
        with pytest.raises(NotPosted, match="could not be reached") as lost:
            await post(WEBHOOK, "progress", None, client=client)
    assert "the-secret-token" not in str(lost.value) and lost.value.__cause__ is None


async def test_a_report_that_watches_goes_on_past_a_post_the_webhook_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("matplotlib")
    directory, *_ = await run(tmp_path)
    posted: list[str] = []

    async def refused(webhook: str, text: str, image: bytes | None, **_: Any) -> None:
        posted.append(text)
        raise NotPosted("the webhook refused the post (429 Too Many Requests)")

    monkeypatch.setattr(report_module, "post", refused)
    with pytest.raises(NotPosted):  # (once, not watching: it fails)
        await report(directory, ROWS, WEBHOOK, run="train")
    watching = asyncio.create_task(report(directory, ROWS, WEBHOOK, run="train", watch=True, interval=0.01))
    try:
        await until(lambda: len(posted) == 2)
        ledger = report_module.of_run(directory)
        fence = await ledger.take("runs/train")
        line: JsonValue = {"time": 5400.0, "rewards": [1.0], "solved": [True], "failed": 0, "unlocked": 7}
        await ledger.append(table("train", RESULTS), "3", line, fence)
        await until(lambda: len(posted) == 3)  # (the next group posted all the same)
        assert not watching.done()
    finally:
        watching.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watching
    assert "not posted: the webhook refused the post (429 Too Many Requests)" in capsys.readouterr().err
