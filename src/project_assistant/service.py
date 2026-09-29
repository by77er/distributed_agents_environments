"""Assembles the project assistant: a runner, its tool sets and the deployment, plus conversation helpers."""

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from project_assistant.assistant import ProjectAgent, ProjectAssistant
from project_assistant.notes import NotesStore
from project_assistant.repository import RepositoryTools
from rollout.core.contracts import RunEvent, RunEventType, Text
from rollout.core.harness import (
    Address,
    Deployment,
    DirectModel,
    Envelope,
    ModelBinding,
    Priority,
    RunBinding,
    RunSpecification,
    SamplingParameters,
    ToolBinding,
    agent_program,
)
from rollout.core.local import EndpointFactory, LocalRunHandle, LocalRunner

NAMESPACE = "assistant"


@dataclass(frozen=True)
class Settings:
    repository: Path
    model: str = "gpt-6-astra"
    reasoning_effort: str | None = "low"
    notes_path: Path | None = None
    """Default: `.rollout/notes.sqlite` inside the repository."""


@dataclass(frozen=True)
class TranscriptEntry:
    role: str  # "user" | "assistant"
    text: str
    at: str
    run_id: str
    seq: int


class AssistantService:
    def __init__(self, settings: Settings, providers: Mapping[str, EndpointFactory]) -> None:
        self.settings = settings
        repository = settings.repository.resolve()
        self.repository_tools = RepositoryTools(repository)
        self.notes = NotesStore(settings.notes_path or repository / ".rollout" / "notes.sqlite")
        self.runner = LocalRunner(
            providers=providers, tool_sets={"repository": self.repository_tools, "notes": self.notes}
        )
        self.deployment_name = f"{NAMESPACE}/{repository.name}"
        provider = next(iter(providers))
        binding = RunBinding(
            models={
                "policy": ModelBinding(
                    direct=DirectModel(
                        provider=provider,
                        model=settings.model,
                        sampling=SamplingParameters(reasoning_effort=settings.reasoning_effort),
                    )
                )
            },
            imports={"repository": ToolBinding(local="repository"), "notes": ToolBinding(local="notes")},
        )
        program = agent_program(
            ProjectAssistant, ProjectAgent, agent_configuration={"repository_name": repository.name}
        )
        self.runner.deploy(
            Deployment(name=self.deployment_name, specification=RunSpecification(program=program, binding=binding))
        )

    def address(self, conversation: str) -> Address:
        return Address(kind="conversation", value=f"{self.deployment_name}/{conversation}")

    def runs(self, conversation: str) -> list[LocalRunHandle]:
        return self.runner.conversation_runs(self.deployment_name, conversation)

    async def send(
        self, conversation: str, text: str, *, priority: Priority = Priority.NORMAL, idempotency_key: str | None = None
    ) -> tuple[str, LocalRunHandle]:
        """Deliver a message; returns its message_id and the run that received it."""
        envelope = Envelope(content=[Text(text=text)], reply_to=Address(kind="external", value="http"))
        message_id = await self.runner.send(
            self.address(conversation), envelope, priority=priority, idempotency_key=idempotency_key, sender="http"
        )
        return message_id, self.runs(conversation)[-1]

    async def reply_to(self, run: LocalRunHandle, message_id: str) -> str | None:
        """The first reply the run emits after it received `message_id`, or None if the run ends first."""
        received = False
        async for event in run.events():
            data = event.payload if isinstance(event.payload, dict) else {}
            if event.type is RunEventType.MESSAGE_RECEIVED:
                envelope = data.get("envelope")
                if isinstance(envelope, dict) and envelope.get("message_id") == message_id:
                    received = True
            elif event.type is RunEventType.OUTPUT_EMITTED and received and data.get("kind") == "reply":
                return str(data.get("payload"))
        return None

    def transcript(self, conversation: str) -> list[TranscriptEntry]:
        entries: list[TranscriptEntry] = []
        for run in self.runs(conversation):
            for event in run.context.events:
                entry = _transcript_entry(event)
                if entry is not None:
                    entries.append(entry)
        return entries

    async def events(self, conversation: str, *, from_seq: int = 0) -> AsyncIterator[RunEvent]:
        """The live run's events: replayed from `from_seq`, then followed."""
        runs = self.runs(conversation)
        if runs:
            async for event in runs[-1].events(from_seq=from_seq):
                yield event

    async def cancel(self, conversation: str) -> bool:
        runs = self.runs(conversation)
        if not runs or runs[-1].done:
            return False
        await self.runner.cancel(runs[-1].run_id, reason="cancelled through the API")
        return True


def _transcript_entry(event: RunEvent) -> TranscriptEntry | None:
    data: Mapping[str, JsonValue] = event.payload if isinstance(event.payload, dict) else {}
    at = event.recorded_at.isoformat(timespec="seconds")
    if event.type is RunEventType.MESSAGE_RECEIVED:
        envelope = data.get("envelope")
        content = envelope.get("content") if isinstance(envelope, dict) else None
        text = "".join(
            str(block.get("text", ""))
            for block in (content if isinstance(content, list) else [])
            if isinstance(block, dict)
        )
        return TranscriptEntry("user", text, at, event.run_id, event.seq)
    if event.type is RunEventType.OUTPUT_EMITTED and data.get("kind") == "reply":
        return TranscriptEntry("assistant", str(data.get("payload")), at, event.run_id, event.seq)
    return None
