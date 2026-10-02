import asyncio
import subprocess
from pathlib import Path

import httpx
import pytest

from project_assistant.assistant import ProjectAgent, ProjectAssistant
from project_assistant.http import create_app
from project_assistant.notes import NotesStore
from project_assistant.repository import RepositoryTools
from project_assistant.service import AssistantService, Settings
from rollout.contracts import Conflict, RunEventType, Text, ToolCall
from rollout.harness import DeliveryMode, DirectModel, Envelope, rollout
from rollout.testing import ScriptedModelEndpoint, ScriptedReply, local_run, tool_call_reply


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "demo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "billing.py").write_text("def invoice_total(items):\n    return sum(items)\n")
    (root / "README.md").write_text("# Demo\nBilling lives in src/billing.py.\n")
    for command in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "Add billing"],
    ):
        subprocess.run(["git", "-C", str(root), *command], check=True)
    return root


async def call(tools: RepositoryTools, name: str, **arguments: object) -> str:
    result = await tools.call(name, arguments, effect_id="r:0:0", arguments_digest="d")  # type: ignore[arg-type]
    return "".join(block.text for block in result.content if isinstance(block, Text))


async def test_repository_tools(repository: Path) -> None:
    tools = RepositoryTools(repository)
    assert (await call(tools, "list_files")).splitlines() == ["README.md", "src/billing.py"]
    assert await call(tools, "search", pattern="invoice_") == "src/billing.py:1: def invoice_total(items):"
    assert "    2      return sum(items)" in await call(tools, "read_file", path="src/billing.py")
    assert "Add billing" in await call(tools, "git_log")
    assert "outside the repository" in await call(tools, "read_file", path="../../etc/passwd")


async def test_notes_deduplicate_by_effect_and_reject_conflicts(tmp_path: Path) -> None:
    notes = NotesStore(tmp_path / "notes.sqlite")
    arguments = {"title": "Decision", "body": "Ship on Fridays"}
    first = await notes.call("save_note", arguments, effect_id="r:0:3", arguments_digest="digest-a")
    again = await notes.call("save_note", arguments, effect_id="r:0:3", arguments_digest="digest-a")
    assert first == again
    assert len(notes.notes()) == 1
    with pytest.raises(Conflict):
        await notes.call("save_note", {"title": "Other", "body": "x"}, effect_id="r:0:3", arguments_digest="digest-b")
    found = await notes.call("search_notes", {"query": "fridays"}, effect_id="r:0:4", arguments_digest="d")
    assert "Ship on Fridays" in found.content[0].text  # type: ignore[union-attr]


async def test_a_follow_up_wakes_the_assistant_when_no_message_arrives() -> None:
    task = ProjectAssistant()
    schedule = ToolCall(
        call_id="c1", name="schedule_follow_up", arguments={"minutes_from_now": 0.001, "about": "check the build"}
    )
    run, endpoint = local_run(task, [tool_call_reply(schedule), "I'll check back.", "The build is green."])

    episode = asyncio.create_task(rollout(task, ProjectAgent({"repository_name": "demo"}), run))
    await asyncio.sleep(0)
    run.deliver(Envelope(content=[Text(text="Remind me to check the build.")]), DeliveryMode.QUEUE)
    await asyncio.sleep(0.5)
    episode.cancel()
    replies = [e.payload["payload"] for e in run.events if e.type is RunEventType.OUTPUT_EMITTED]  # type: ignore[index, call-overload]
    assert replies == ["I'll check back.", "The build is green."]
    assert "Follow-up due" in endpoint.requests[2].context.append[-1].text
    assert task.follow_ups == []


def service_with(repository: Path, replies: list[ScriptedReply], tmp_path: Path) -> AssistantService:
    endpoint = ScriptedModelEndpoint(replies)

    def factory(model: DirectModel) -> ScriptedModelEndpoint:
        return endpoint

    return AssistantService(
        Settings(repository=repository, notes_path=tmp_path / "notes.sqlite"), {"scripted": factory}
    )


async def test_the_http_api_answers_and_keeps_a_transcript(repository: Path, tmp_path: Path) -> None:
    search = ToolCall(call_id="c1", name="search", arguments={"pattern": "invoice"})
    service = service_with(repository, [tool_call_reply(search), "It is in src/billing.py:1."], tmp_path)
    transport = httpx.ASGITransport(app=create_app(service))
    async with httpx.AsyncClient(transport=transport, base_url="http://assistant") as client:
        response = await client.post("/conversations/c1/messages", json={"text": "Where is invoice_total?"})
        assert response.status_code == 200
        assert response.json()["reply"] == "It is in src/billing.py:1."
        transcript = (await client.get("/conversations/c1/transcript")).json()["messages"]
        assert [(entry["role"], entry["text"]) for entry in transcript] == [
            ("user", "Where is invoice_total?"),
            ("assistant", "It is in src/billing.py:1."),
        ]
        assert (await client.post("/conversations/c1/cancel")).json() == {"cancelled": True}
        bad = await client.post("/conversations/c1/messages", json={"text": ""})
        assert bad.status_code == 400
