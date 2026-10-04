"""Two replicas of the gateway, each its own process started by `rollout gateway PROFILE`, sharing a ledger and a
blob store, behind a pass-through proxy that sends requests to each in turn: one is killed in the middle of a turn,
and the session still records exactly the segments a single gateway records when nothing fails."""

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
import time
from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import httpx
import pytest

from rollout.harness.blobs import FileBlobStore
from rollout_train.gateway import TurnStore
from rollout_train.ledger import FileLedger
from tests.rollout_train.gateway.support import SECRETS, converse, grant, keyring, recorded_undisturbed

pytest.importorskip("uvicorn")
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

ROOT = Path(__file__).resolve().parents[3]
REPLICAS = [8831, 8832]
PROXY = 8833
PROFILE = """
directory = "{directory}/run"
ledger = "{directory}/ledger"

[channels.policy]
model = "echo"
renderer = "rollout_train.testing:plain_renderer"
engine = "tests.rollout_train.gateway.support:echo_engine"
engines = [{{ delay = 0.4 }}]

[gateway]
keys = "{directory}/keys"
"""


def proxy(upstreams: list[str], failed: list[str]) -> Starlette:
    """Passes `/gw/...` on to the upstreams in turn, as `/...`; an upstream that refuses the connection is skipped
    (nothing was sent), and one that fails after it was sent the request is a 502 (noted in `failed`)."""
    turn = iter(range(1_000_000))
    http = httpx.AsyncClient(timeout=30.0)

    async def passed(request: Request) -> Response:
        start = next(turn)
        body = await request.body()
        headers = {name: value for name, value in request.headers.items() if name not in ("host", "content-length")}
        for offset in range(len(upstreams)):
            upstream = upstreams[(start + offset) % len(upstreams)]
            try:
                answer = await http.request(
                    request.method, f"{upstream}/{request.path_params['rest']}", content=body, headers=headers
                )
            except httpx.ConnectError:
                continue
            except httpx.TransportError as error:
                failed.append(upstream)
                return Response(f"upstream failed: {error!r}", status_code=502)
            kept = {name: value for name, value in answer.headers.items() if name.lower() != "content-length"}
            return Response(answer.content, status_code=answer.status_code, headers=kept)
        return Response("no upstream", status_code=502)

    return Starlette(routes=[Route("/gw/{rest:path}", passed, methods=["GET", "POST"])])


@contextlib.contextmanager
def replicas(profile: Path) -> Generator[list[subprocess.Popen[bytes]]]:
    started = [
        subprocess.Popen(
            [sys.executable, "-m", "rollout_train.cli", "gateway", str(profile), "--listen", f"127.0.0.1:{port}"],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        for port in REPLICAS
    ]
    try:
        deadline = time.monotonic() + 60
        for port in REPLICAS:
            while True:
                with contextlib.suppress(httpx.TransportError):
                    if httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=1).status_code == 200:
                        break
                if time.monotonic() > deadline or any(each.poll() is not None for each in started):
                    raise RuntimeError("a replica did not start")
                time.sleep(0.2)
        yield started
    finally:
        for each in started:
            each.kill()
            each.wait()


@contextlib.asynccontextmanager
async def serving(app: Starlette, port: int) -> AsyncGenerator[None]:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(server.serve())
    while not server.started:  # noqa: ASYNC110 (uvicorn says it has started by a flag)
        await asyncio.sleep(0.05)
    try:
        yield
    finally:
        server.should_exit = True
        await task


async def test_replicas_behind_a_proxy_one_killed_mid_turn_record_what_one_undisturbed_records(tmp_path: Path) -> None:
    profile = tmp_path / "profile.toml"
    profile.write_text(PROFILE.format(directory=tmp_path))
    (tmp_path / "keys").write_text("".join(f"{name} {secret}\n" for name, secret in SECRETS))
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "run" / "blobs")
    key = keyring().mint(await grant(ledger))
    with replicas(profile) as started:
        first = started[0]

        async def kill_the_first(number: int) -> None:
            if number == 2:  # the third request goes to the first replica: it dies while sampling it
                await asyncio.sleep(0.2)
                first.send_signal(signal.SIGKILL)

        upstreams = [f"http://127.0.0.1:{port}" for port in REPLICAS]
        failed: list[str] = []
        async with serving(proxy(upstreams, failed), PROXY), httpx.AsyncClient(timeout=30.0) as http:
            await converse(http, key, prefix=f"http://127.0.0.1:{PROXY}/gw", during=kill_the_first)
        assert first.poll() is not None and started[1].poll() is None
        assert failed == [upstreams[0]]  # (the turn the first was sampling when it died was asked again of the other)
    segments = (await TurnStore(ledger, blobs).sessions("train", "r_1"))["policy"]
    assert segments == await recorded_undisturbed(tmp_path / "undisturbed")
    turns = await TurnStore(ledger, blobs).turns("train", "r_1")
    assert [turn.effect_id for turn in turns] == ["e0", "e1", "e2", "e3", "e4", "e-again", "e-again-later"]
