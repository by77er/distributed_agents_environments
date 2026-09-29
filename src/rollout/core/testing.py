"""Test doubles for code built on the core: a model endpoint that replies from a script."""

import inspect
import json
import time
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path

from pydantic import JsonValue

from rollout.core.contracts import (
    CapabilityContract,
    FinishReason,
    Message,
    ModelEndpoint,
    Role,
    RunEvent,
    RunEventType,
    SampleRequest,
    SampleResult,
    Text,
    ToolCall,
    Usage,
    new_run_id,
)
from rollout.core.harness.environments import EnvironmentService, EnvironmentSpecification, ExecutionResult
from rollout.core.harness.task import Task
from rollout.core.local.context import LocalRunContext

__all__ = [
    "LedgerEndpoint",
    "LedgerEnvironments",
    "ScriptedModelEndpoint",
    "ScriptedReply",
    "events_of",
    "local_run",
    "payload",
    "read_ledger",
    "tool_call_reply",
]

type ScriptedReply = Message | str | Callable[[SampleRequest], Message | Awaitable[Message]]
"""A reply, its text, or a function of the request (which may await, e.g. to hold a sample open)."""


class ScriptedModelEndpoint:
    """Replies with the scripted entries in order and records every request and cancellation."""

    def __init__(self, replies: Iterable[ScriptedReply], *, contract: CapabilityContract | None = None) -> None:
        self._replies = list(replies)
        self._contract = contract or CapabilityContract(context_limit=32_768, max_output_tokens=4_096)
        self.requests: list[SampleRequest] = []
        self.cancelled: list[str] = []

    def describe(self, session_id: str) -> CapabilityContract:
        return self._contract

    async def sample(self, request: SampleRequest) -> SampleResult:
        self.requests.append(request)
        if not self._replies:
            raise AssertionError(f"the script has no reply left for request {request.effect_id}")
        entry = self._replies.pop(0)
        if isinstance(entry, str):
            reply = Message.assistant(entry)
        elif isinstance(entry, Message):
            reply = entry
        else:
            produced = entry(request)
            reply = await produced if inspect.isawaitable(produced) else produced
        finish_reason = FinishReason.TOOL_USE if reply.tool_calls else FinishReason.STOP
        context_used = sum(len(message.text) for message in request.context.append)
        return SampleResult(
            message=reply,
            finish_reason=finish_reason,
            usage=Usage(context_used=context_used, context_limit=self._contract.context_limit),
        )

    async def cancel(self, effect_id: str) -> None:
        self.cancelled.append(effect_id)


def tool_call_reply(*calls: ToolCall, text: str = "") -> Message:
    """An assistant reply that makes tool calls."""
    return Message(role=Role.ASSISTANT, content=[*([Text(text=text)] if text else []), *calls])


def local_run(task: Task, replies: Iterable[ScriptedReply] = ()) -> tuple[LocalRunContext, ScriptedModelEndpoint]:
    """A local run context for `task` whose model slots all reply from one script."""
    endpoint = ScriptedModelEndpoint(replies)
    run = LocalRunContext(new_run_id(), dict.fromkeys(task.models, endpoint), context_hints=task.context_hints)
    return run, endpoint


def events_of(run: LocalRunContext, event_type: RunEventType) -> list[RunEvent]:
    """The run's events of one type, in order."""
    return [event for event in run.events if event.type is event_type]


def payload(event: RunEvent) -> dict[str, JsonValue]:
    """An event's payload as a JSON object."""
    if not isinstance(event.payload, dict):
        raise TypeError(f"{event.type} payload is not an object")
    return event.payload


class LedgerEndpoint:
    """Wraps a model endpoint and appends every sample's `effect_id` to a file: to count calls across processes."""

    def __init__(self, inner: ModelEndpoint, ledger: Path) -> None:
        self._inner = inner
        self._ledger = ledger

    def describe(self, session_id: str) -> CapabilityContract:
        return self._inner.describe(session_id)

    async def sample(self, request: SampleRequest) -> SampleResult:
        _append(self._ledger, {"effect_id": request.effect_id, "session_id": request.session_id})
        return await self._inner.sample(request)

    async def cancel(self, effect_id: str) -> None:
        await self._inner.cancel(effect_id)


class LedgerEnvironments:
    """Wraps an environment service and appends every command it starts to a file, with its `effect_id`."""

    def __init__(self, inner: EnvironmentService, ledger: Path) -> None:
        self._inner = inner
        self._ledger = ledger

    async def create(self, environment_id: str, specification: EnvironmentSpecification) -> None:
        await self._inner.create(environment_id, specification)

    async def execute(
        self, environment_id: str, command: str, *, timeout_seconds: float, cwd: str | None, effect_id: str = ""
    ) -> ExecutionResult:
        _append(self._ledger, {"effect_id": effect_id, "environment_id": environment_id, "command": command})
        return await self._inner.execute(
            environment_id, command, timeout_seconds=timeout_seconds, cwd=cwd, effect_id=effect_id
        )

    async def put(self, environment_id: str, path: str, data: bytes) -> None:
        await self._inner.put(environment_id, path, data)

    async def get(self, environment_id: str, path: str) -> bytes:
        return await self._inner.get(environment_id, path)

    async def destroy(self, environment_id: str) -> None:
        await self._inner.destroy(environment_id)


def read_ledger(ledger: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []


def _append(ledger: Path, entry: dict[str, str]) -> None:
    with ledger.open("a") as file:
        file.write(json.dumps({**entry, "at": time.time()}) + "\n")
