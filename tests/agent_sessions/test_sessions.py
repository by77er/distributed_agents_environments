"""End to end: a session creates another, which works in its own environment and reports back."""

import asyncio
import os
import random
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest
from session_scripts import script

from agent_sessions.faults import Server, Servers
from agent_sessions.service import SessionInfo, SessionsService, Settings
from rollout.core.contracts import Text, ToolResultBlock
from rollout.core.harness import DirectModel
from rollout.core.testing import ScriptedModelEndpoint

IMAGE_CACHE = Path.home() / ".cache" / "rollout" / "images"


def namespaces_available() -> bool:
    if shutil.which("unshare") is None:
        return False
    probe = ["unshare", "--user", "--map-root-user", "--mount", "--pid", "--fork", "true"]
    return subprocess.run(probe, check=False).returncode == 0


NAMESPACES = pytest.param(
    "namespaces",
    marks=pytest.mark.skipif(not namespaces_available(), reason="unprivileged namespaces are not available"),
)


@pytest.mark.parametrize("environment", [NAMESPACES, "local"])
@pytest.mark.parametrize("kind", ["in-memory", "durable", "durable-postgres"])
async def test_a_session_spawns_a_worker_that_reports_back(
    tmp_path: Path, kind: str, environment: str, request: pytest.FixtureRequest
) -> None:
    durable = kind.startswith("durable")
    database = request.getfixturevalue("postgres") if kind == "durable-postgres" else None
    endpoint = ScriptedModelEndpoint([script] * 20)

    def factory(model: DirectModel) -> ScriptedModelEndpoint:
        return endpoint

    settings = Settings(
        state=tmp_path,
        durable=durable,
        image_cache=IMAGE_CACHE,
        environment="local" if environment == "local" else "namespaces",
        database=database,
    )
    service = SessionsService(settings, {"scripted": factory})
    await service.start()
    try:
        await service.create("lead", "Find out what 6*7 is, using a worker session.")
        replies: list[object] = []
        for _ in range(300):
            replies = [e["text"] for e in service.activity("lead") if e["kind"] == "reply"]
            if "The worker reports 42." in replies:
                break
            await asyncio.sleep(0.1)
        assert replies == ["I started a worker.", "The worker reports 42."]

        def worker() -> SessionInfo:
            return {s.name: s for s in service.sessions()}["worker"]

        for _ in range(100):  # the worker may still be finishing its last turn ("Reported to lead.")
            if worker().status == "waiting":
                break
            await asyncio.sleep(0.1)
        assert (worker().parent, worker().status) == ("lead", "waiting")
        worker_activity = service.activity("worker")
        assert worker_activity[0] == {**worker_activity[0], "kind": "message", "from": "lead"}
        shell_outputs = [
            part.text
            for request in endpoint.requests
            for message in request.context.append
            for block in message.content
            if isinstance(block, ToolResultBlock)
            for part in block.result.content
            if isinstance(part, Text)
        ]
        # 42, then the Alpine release inside namespaces, or the workspace on the host
        after = "3." if environment == "namespaces" else f"{tmp_path / 'environments'}/e_"
        assert any(output.startswith(f"exit 0\n42\n{after}") for output in shell_outputs)
        system_prompt = endpoint.requests[0].context.append[0].text
        assert ("Alpine" in system_prompt) == (environment == "namespaces")
        environments = list((tmp_path / "environments").iterdir())
        assert len([e for e in environments if e.name.startswith("e_")]) == 2  # one computer per session

        assert await service.stop("worker")
        assert service.status("worker") == "stopped"
        assert len([e for e in (tmp_path / "environments").iterdir() if e.name.startswith("e_")]) == 1
    finally:
        await service.close()


async def test_a_state_directory_keeps_its_environment_backend(tmp_path: Path) -> None:
    SessionsService(Settings(state=tmp_path, durable=False, environment="local"), {"scripted": lambda model: None})  # type: ignore[arg-type, return-value]
    with pytest.raises(ValueError, match="use 'local' environments"):
        SessionsService(Settings(state=tmp_path, durable=False), {"scripted": lambda model: None})  # type: ignore[arg-type, return-value]


def server_logs(directory: Path) -> str:
    return "".join(path.read_text() for path in directory.glob("server-*.log"))


async def test_sessions_carry_on_across_servers_when_one_dies(tmp_path: Path, postgres: str) -> None:
    """Three servers share a database and a state directory. The lead is created through one, which is killed at
    once and stays down: another server takes the lead over, the relay moves to a live server, and the fan-out
    completes. Every request goes to a random live server."""
    here = str(Path(__file__).parent)
    variables = {"PYTHONPATH": os.pathsep.join([here, os.environ.get("PYTHONPATH", "")])}
    servers = [
        Server(
            tmp_path,
            "local",
            database=postgres,
            runner_id=f"server-{index}",
            takeover_after=3,
            providers="session_scripts:PROVIDERS",
            variables=variables,
        )
        for index in range(3)
    ]
    await asyncio.gather(*(server.start() for server in servers))
    group = Servers(servers, random.Random(7))
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            created = await client.post(
                f"{servers[0].url}/sessions", json={"name": "lead", "instructions": "Find out 6*7 with a worker."}
            )
            assert created.status_code == 201
            await servers[0].kill()  # the lead's run was started on this server

            async def lead_replies() -> list[str]:
                activity = (await group.request(client, "GET", "/sessions/lead/activity")).json()["activity"]
                return [entry["text"] for entry in activity if entry["kind"] == "reply"]

            for _ in range(300):
                if "The worker reports 42." in await lead_replies():
                    break
                await asyncio.sleep(0.3)
            assert await lead_replies() == ["I started a worker.", "The worker reports 42."]
            assert "recovered the runs of runner server-0" in server_logs(tmp_path)  # a peer took the lead over
            sessions = (await group.request(client, "GET", "/sessions")).json()["sessions"]
            assert sorted(session["name"] for session in sessions) == ["lead", "worker"]
            environments = [
                path for path in (tmp_path / "state" / "environments").iterdir() if path.name.startswith("e_")
            ]
            assert len(environments) == 2  # one per session, though the lead's run moved between servers

            await servers[0].start()  # back, with its runner id
            stopped = await group.request(client, "POST", "/sessions/worker/stop")
            assert stopped.json() == {"stopped": True}
            remaining = [path for path in (tmp_path / "state" / "environments").iterdir() if path.name.startswith("e_")]
            assert len(remaining) == 1
    finally:
        for server in servers:
            await server.stop()
