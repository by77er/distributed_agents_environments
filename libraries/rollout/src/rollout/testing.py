"""Test doubles for code built on the core: a model endpoint that replies from a script, and sandboxes that are only
records."""

import inspect
import json
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import JsonValue

from rollout.contracts import (
    CapabilityContract,
    FinishReason,
    Message,
    ModelAddress,
    ModelEndpoint,
    RetryClass,
    Role,
    RunEvent,
    RunEventType,
    SampleRequest,
    SampleResult,
    Text,
    ToolCall,
    ToolResult,
    ToolSpecification,
    Usage,
    address_of,
    new_run_id,
)
from rollout.harness.environments import EnvironmentService, EnvironmentSpecification, ExecutionResult
from rollout.harness.sandboxes import Process, Reach, SandboxSpec
from rollout.harness.task import Task
from rollout.local.context import LocalRunContext

__all__ = [
    "FakeSandbox",
    "FakeSandboxes",
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

    def address(self, session_id: str) -> ModelAddress:
        return address_of(self._inner, session_id)

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


@dataclass
class FakeSandbox:
    """One of `FakeSandboxes`: what it was made from and given, the process it runs, and every operation asked of
    it."""

    handle: str
    spec: SandboxSpec
    environment: Mapping[str, str]
    process: Process | None = None
    """What its spec says to run, with the lease's environment added to the process's own."""
    written: int = 0
    """Bytes written to its scratch directory."""
    calls: list[tuple[str, Mapping[str, JsonValue]]] = field(default_factory=list[tuple[str, Mapping[str, JsonValue]]])


type Operation = Callable[[FakeSandbox, Mapping[str, JsonValue]], JsonValue | Awaitable[JsonValue]]
"""What an operation of `FakeSandboxes` answers, given the sandbox and the call's arguments. One that raises
`PermissionError` answers with an error: the sandbox's spec does not allow it."""


def _describe(sandbox: FakeSandbox, arguments: Mapping[str, JsonValue]) -> JsonValue:
    process = sandbox.process
    return {
        "handle": sandbox.handle,
        "parameters": dict(sandbox.spec.parameters),
        "environment": dict(sandbox.environment),
        "process": {"command": list(process.command), "environment": dict(process.environment)} if process else None,
    }


def _write(sandbox: FakeSandbox, arguments: Mapping[str, JsonValue]) -> JsonValue:
    path, size = str(arguments["path"]), int(str(arguments["bytes"]))
    spec = sandbox.spec
    if any(_under(path, mount.target) for mount in spec.mounts):
        raise PermissionError(f"{path} is on a read-only mount")
    if spec.scratch is None or not _under(path, spec.scratch.path):
        raise PermissionError(f"{path} is not in the sandbox's scratch directory")
    if sandbox.written + size > spec.scratch.mib * 2**20:
        raise PermissionError(f"the scratch directory holds {spec.scratch.mib} MiB")
    sandbox.written += size
    return {"written": sandbox.written}


def _fetch(sandbox: FakeSandbox, arguments: Mapping[str, JsonValue]) -> JsonValue:
    host = str(arguments["host"])
    if host not in sandbox.spec.network.allow:
        raise PermissionError(f"the sandbox may not reach {host}")
    return {"reached": host}


def _under(path: str, directory: str) -> bool:
    return path == directory or path.startswith(directory.rstrip("/") + "/")


class FakeSandboxes:
    """A sandbox `Provider` whose sandboxes are only records, honouring what their specs allow: at most `size` of
    them, of `kind`. A spec's process is "launched" with the lease's environment added to its own, and the runner
    reaches it at the address `process`. Its operations are `describe` (the sandbox's handle, parameters, environment
    and process), `write` (`path`, `bytes`: only within its scratch directory and its size, never on a mount), `fetch`
    (`host`: only a host its network allows), and any in `operations`; it keeps every sandbox it made and deleted.
    Lease them out with `SandboxPool(FakeSandboxes())`."""

    def __init__(self, kind: str = "fake", size: int = 4, operations: Mapping[str, Operation] | None = None) -> None:
        self.kind = kind
        self.size = size
        self.performs: dict[str, Operation] = {
            "describe": _describe,
            "write": _write,
            "fetch": _fetch,
            **(operations or {}),
        }
        self.sandboxes: dict[str, FakeSandbox] = {}
        """The sandboxes it has, by handle."""
        self.made: list[str] = []
        self.deleted: list[str] = []

    def operations(self) -> Sequence[ToolSpecification]:
        return [ToolSpecification(name=name, retry_class=RetryClass.PURE) for name in self.performs]

    async def create(self, handle: str, spec: SandboxSpec, environment: Mapping[str, str]) -> Reach:
        if handle not in self.sandboxes:
            process = spec.process
            if process is not None:
                process = process.model_copy(update={"environment": {**process.environment, **environment}})
            self.sandboxes[handle] = FakeSandbox(handle, spec, dict(environment), process)
            self.made.append(handle)
        addresses = {"fake": f"fake://{handle}"}
        if spec.process is not None:
            addresses["process"] = f"fake://{handle}/process"
        return Reach(addresses=addresses)

    async def delete(self, handle: str) -> None:
        if self.sandboxes.pop(handle, None) is not None:
            self.deleted.append(handle)

    async def held(self) -> Sequence[str]:
        return list(self.sandboxes)

    async def call(
        self, handle: str, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
        sandbox = self.sandboxes.get(handle)
        if sandbox is None:
            raise KeyError(f"there is no sandbox {handle}")
        perform = self.performs.get(name)
        if perform is None:
            return ToolResult(content=[Text(text=f"unknown operation {name}")], is_error=True)
        sandbox.calls.append((name, dict(arguments)))
        try:
            value = perform(sandbox, arguments)
            answered: JsonValue = await value if inspect.isawaitable(value) else value  # pyright: ignore[reportUnknownVariableType]
        except PermissionError as error:
            return ToolResult(content=[Text(text=str(error))], is_error=True)
        return ToolResult(content=[Text(text=json.dumps(answered))], structured=answered)


def read_ledger(ledger: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []


def _append(ledger: Path, entry: dict[str, str]) -> None:
    with ledger.open("a") as file:
        file.write(json.dumps({**entry, "at": time.time()}) + "\n")
