"""End to end: a session creates another, which works in its own environment and reports back."""

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from agent_sessions.service import SessionsService, Settings
from rollout.core.contracts import Message, Role, SampleRequest, Text, ToolCall, ToolResultBlock
from rollout.core.harness import DirectModel
from rollout.core.testing import ScriptedModelEndpoint, tool_call_reply

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


def call(tool: str, /, **arguments: object) -> Message:
    return tool_call_reply(ToolCall(call_id=f"c-{tool}", name=tool, arguments=arguments))  # type: ignore[arg-type]


def script(request: SampleRequest) -> Message:
    """Each session's model: decided by who is asking and what it last saw."""
    messages = request.context.append
    me = messages[0].text.split('"')[1]
    last = messages[-1]
    results = [b.result for b in last.content if isinstance(b, ToolResultBlock)]
    last_text = "".join(p.text for r in results for p in r.content if isinstance(p, Text)) or last.text
    if me == "lead":
        if last.role is Role.USER and last.text.startswith("[message from worker]"):
            return Message.assistant("The worker reports 42.")
        if last.role is Role.USER:
            return call("create_session", name="worker", instructions="Compute 6*7 in your shell and send it to lead.")
        return Message.assistant("I started a worker.")
    if last.role is Role.USER:
        return call("shell", command="echo $((6*7)) && (cat /etc/alpine-release 2>/dev/null || echo $WORKSPACE)")
    if last_text.startswith("exit 0"):
        return call("send_message", to="lead", text=last_text.splitlines()[1])
    return Message.assistant("Reported to lead.")


@pytest.mark.parametrize("environment", [NAMESPACES, "local"])
@pytest.mark.parametrize("durable", [False, True], ids=["in-memory", "durable"])
async def test_a_session_spawns_a_worker_that_reports_back(tmp_path: Path, durable: bool, environment: str) -> None:
    endpoint = ScriptedModelEndpoint([script] * 20)

    def factory(model: DirectModel) -> ScriptedModelEndpoint:
        return endpoint

    settings = Settings(state=tmp_path, durable=durable, image_cache=IMAGE_CACHE, environment=environment)  # type: ignore[arg-type]
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

        worker = {s.name: s for s in service.sessions()}["worker"]
        assert (worker.parent, worker.status) == ("lead", "waiting")
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
