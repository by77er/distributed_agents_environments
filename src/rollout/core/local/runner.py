"""`LocalRunner`: runs programs on the current asyncio loop. Nothing persists; a process crash loses in-flight runs."""

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field

from pydantic import JsonValue

from rollout.core.contracts import (
    TERMINAL_EVENT_TYPES,
    ModelEndpoint,
    RunEvent,
    RunEventType,
    RunFailureClass,
    new_run_id,
    new_ulid,
)
from rollout.core.harness.conversations import Address, ConversationKey, Envelope, Priority
from rollout.core.harness.environments import EnvironmentService
from rollout.core.harness.imports import ToolSet
from rollout.core.harness.observation import InvalidObservation
from rollout.core.harness.program import Program
from rollout.core.harness.runner import (
    Deployment,
    DirectModel,
    RunBinding,
    RunOutcome,
    RunSpecification,
    RunStatus,
    instantiate,
)
from rollout.core.local.context import LocalRunContext

type EndpointFactory = Callable[[DirectModel], ModelEndpoint]
"""Creates the endpoint for a direct model binding; registered with the runner by provider name."""


class RunNotLive(Exception):
    """A message was addressed to a run that has ended."""


class LocalRunHandle:
    """A run started by a `LocalRunner`. Its context is available for inspection in tests and tools."""

    def __init__(self, run_id: str, specification: RunSpecification, conversation: ConversationKey | None) -> None:
        self._run_id = run_id
        self.specification = specification
        self.conversation = conversation
        self.context: LocalRunContext
        self._task: asyncio.Task[None] | None = None
        self._outcome: RunOutcome | None = None
        self._signal = asyncio.Event()

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def done(self) -> bool:
        return self._outcome is not None

    @property
    def outcome(self) -> RunOutcome | None:
        return self._outcome

    async def result(self) -> RunOutcome:
        if self._task is not None:
            await asyncio.shield(self._task)
        assert self._outcome is not None
        return self._outcome

    async def events(self, *, from_seq: int = 0) -> AsyncIterator[RunEvent]:
        index = from_seq
        while True:
            signal = self._signal
            events = self.context.events
            while index < len(events):
                event = events[index]
                index += 1
                yield event
                if event.type in TERMINAL_EVENT_TYPES:
                    return
            if self.done:
                return
            await signal.wait()

    def recorded_events(self) -> list[RunEvent]:
        """Every event recorded so far."""
        return list(self.context.events)

    # For the runner

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    def attach(self, task: asyncio.Task[None]) -> None:
        self._task = task

    def finish(self, outcome: RunOutcome) -> None:
        self._outcome = outcome
        self.notify()

    def notify(self, event: RunEvent | None = None) -> None:
        """Wake event streams: a new event was recorded, or the run ended."""
        signal, self._signal = self._signal, asyncio.Event()
        signal.set()


@dataclass
class _Conversation:
    key: ConversationKey
    live: LocalRunHandle | None = None
    runs: list[str] = field(default_factory=list[str])
    seen: set[str] = field(default_factory=set[str])


class LocalRunner:
    """Implements `Runner` in process.

    Direct model bindings are served by endpoint factories registered by provider name. Conversations addressed to a
    deployment start a run of its specification when none is live; one run consumes a conversation's messages at a
    time, and messages a run never consumed start the conversation's next run.
    """

    def __init__(
        self,
        *,
        providers: Mapping[str, EndpointFactory] | None = None,
        tool_sets: Mapping[str, ToolSet] | None = None,
        environments: EnvironmentService | None = None,
    ) -> None:
        self._providers = dict(providers or {})
        self._tool_sets = dict(tool_sets or {})
        self._environment_service = environments
        self._runs: dict[str, LocalRunHandle] = {}
        self._deployments: dict[str, Deployment] = {}
        self._conversations: dict[str, _Conversation] = {}
        self._background: set[asyncio.Task[None]] = set()

    # Deployments and inspection

    def deploy(self, deployment: Deployment) -> None:
        """Register or replace a deployment; a conversation's next run uses the current version."""
        self._deployments[deployment.name] = deployment

    def run(self, run_id: str) -> LocalRunHandle:
        return self._runs[run_id]

    def conversation_runs(self, deployment: str, key: str) -> list[LocalRunHandle]:
        """The conversation's runs, oldest first."""
        conversation = self._conversations.get(f"{deployment}/{key}")
        return [self._runs[run_id] for run_id in conversation.runs] if conversation else []

    # Runner

    async def start(
        self,
        specification: RunSpecification,
        *,
        run_id: str | None = None,
        conversation: ConversationKey | None = None,
        labels: Mapping[str, str] | None = None,
    ) -> LocalRunHandle:
        run_id = run_id or new_run_id()
        if run_id in self._runs:
            raise ValueError(f"run {run_id} already exists")
        program = instantiate(specification.program)
        endpoints = resolve_endpoints(program, specification.binding, self._providers)
        tool_sets = resolve_tool_sets(program, specification.binding, self._tool_sets)
        handle = LocalRunHandle(run_id, specification, conversation)
        handle.context = LocalRunContext(
            run_id,
            endpoints,
            context_hints=program.context_hints(),
            tool_sets=tool_sets,
            environment_service=self._environment_service,
            conversation=conversation,
            on_event=handle.notify,
        )
        handle.context.record_event(
            RunEventType.RUN_CREATED,
            {
                "specification": specification.model_dump(mode="json", exclude_none=True),
                "conversation": conversation.model_dump(mode="json", exclude_none=True) if conversation else None,
                "labels": dict(labels or {}),
            },
        )
        specifications = program.tool_specifications() + handle.context.tools.specifications()
        if specifications:
            resolved: list[JsonValue] = [
                specification.model_dump(mode="json", exclude_none=True) for specification in specifications
            ]
            handle.context.record_event(RunEventType.TOOLS_RESOLVED, {"specifications": resolved})
        self._runs[run_id] = handle
        handle.attach(asyncio.create_task(self._execute(handle, program), name=f"run {run_id}"))
        return handle

    async def send(
        self,
        to: Address,
        envelope: Envelope,
        *,
        priority: Priority = Priority.NORMAL,
        idempotency_key: str | None = None,
        sender: str | None = None,
    ) -> str:
        message_id = idempotency_key or f"m_{new_ulid()}"
        envelope = envelope.model_copy(update={"message_id": message_id, "sender": sender})
        if to.kind == "run":
            handle = self._runs.get(to.value)
            if handle is None or handle.done:
                raise RunNotLive(to.value)
            handle.context.deliver(envelope, handle.specification.binding.delivery.mode(priority, sender))
            return message_id
        if to.kind != "conversation":
            raise ValueError(f"the local runner cannot deliver to {to.kind} addresses")
        conversation = self._conversation(to.value, envelope.reply_to)
        if message_id in conversation.seen:
            return message_id  # deduplicated by message_id
        conversation.seen.add(message_id)
        handle = conversation.live
        if handle is None or handle.done:
            handle = await self._start_conversation_run(conversation)
        handle.context.deliver(envelope, handle.specification.binding.delivery.mode(priority, sender))
        return message_id

    async def cancel(self, run_id: str, *, reason: str) -> None:
        handle = self._runs[run_id]
        task = handle.task
        if handle.done or task is None:
            return
        handle.context.record_event(RunEventType.RUN_CANCEL_REQUESTED, {"reason": reason, "by": "runner"})
        task.cancel()
        await asyncio.wait([task])

    # Internals

    def _conversation(self, address: str, reply_to: Address | None) -> _Conversation:
        existing = self._conversations.get(address)
        if existing is not None:
            return existing
        namespace, name, key = [*address.split("/", 2), "", ""][:3]
        deployment = f"{namespace}/{name}"
        if not key or deployment not in self._deployments:
            raise ValueError(f"{address!r} does not name a conversation of a deployed agent")
        conversation = _Conversation(ConversationKey(deployment=deployment, key=key, origin=reply_to))
        self._conversations[address] = conversation
        return conversation

    async def _start_conversation_run(self, conversation: _Conversation) -> LocalRunHandle:
        deployment = self._deployments[conversation.key.deployment]
        handle = await self.start(deployment.specification, conversation=conversation.key)
        conversation.live = handle
        conversation.runs.append(handle.run_id)
        return handle

    async def _execute(self, handle: LocalRunHandle, program: Program) -> None:
        context = handle.context
        try:
            await program.main(context)
            outcome = RunOutcome(status=RunStatus.COMPLETED)
            context.record_event(RunEventType.RUN_COMPLETED, {"outcome": "success"})
        except asyncio.CancelledError:
            outcome = RunOutcome(status=RunStatus.CANCELLED)
            context.record_event(RunEventType.RUN_CANCELLED, None)
        except InvalidObservation as error:
            detail = str(error)
            outcome = RunOutcome(
                status=RunStatus.FAILED, failure_class=RunFailureClass.INVALID_OBSERVATION, detail=detail
            )
            context.record_event(RunEventType.RUN_FAILED, {"class": "invalid_observation", "detail": detail})
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            outcome = RunOutcome(status=RunStatus.FAILED, failure_class=RunFailureClass.TASK_ERROR, detail=detail)
            context.record_event(RunEventType.RUN_FAILED, {"class": "task_error", "detail": detail})
        if context.environments is not None:
            await context.environments.release_all()  # environments the run still owns (P12)
        handle.finish(outcome)
        self._hand_over(handle)

    def _hand_over(self, handle: LocalRunHandle) -> None:
        """Messages the finished run never consumed start the conversation's next run."""
        if handle.conversation is None:
            return
        address = f"{handle.conversation.deployment}/{handle.conversation.key}"
        conversation = self._conversations.get(address)
        if conversation is None or conversation.live is not handle:
            return
        conversation.live = None
        undelivered = handle.context.take_undelivered()
        if not undelivered:
            return

        async def continue_conversation() -> None:
            successor = await self._start_conversation_run(conversation)
            for envelope in undelivered:
                successor.context.deliver(envelope, successor.specification.binding.delivery.mode(Priority.LOW))

        task = asyncio.create_task(continue_conversation())
        self._background.add(task)
        task.add_done_callback(self._background.discard)


def resolve_endpoints(
    program: Program, binding: RunBinding, providers: Mapping[str, EndpointFactory]
) -> dict[str, ModelEndpoint]:
    """An endpoint for each model slot of the program, from the binding and the registered providers."""
    endpoints: dict[str, ModelEndpoint] = {}
    for slot in program.model_slots():
        model = binding.models.get(slot)
        if model is None:
            raise ValueError(f"the binding has no model for slot {slot!r}")
        if model.direct is None:
            raise NotImplementedError("recorded model bindings need the recorder (milestone M1)")
        factory = providers.get(model.direct.provider)
        if factory is None:
            raise ValueError(f"no endpoint factory registered for provider {model.direct.provider!r}")
        endpoints[slot] = factory(model.direct)
    return endpoints


def resolve_tool_sets(program: Program, binding: RunBinding, tool_sets: Mapping[str, ToolSet]) -> dict[str, ToolSet]:
    """The tool set serving each import of the program, from the binding and the registered tool sets."""
    resolved: dict[str, ToolSet] = {}
    for name in program.imports():
        tool_binding = binding.imports.get(name)
        if tool_binding is None or tool_binding.local is None:
            raise ValueError(f"the binding does not say how to serve the import {name!r}")
        tool_set = tool_sets.get(tool_binding.local)
        if tool_set is None:
            raise ValueError(f"no tool set registered as {tool_binding.local!r}")
        resolved[name] = tool_set
    return resolved
