"""`LocalRunner`: runs programs on the current asyncio loop. Nothing persists; a process crash loses in-flight runs."""

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass, field

from pydantic import JsonValue

from rollout.contracts import (
    TERMINAL_EVENT_TYPES,
    ModelEndpoint,
    RunEvent,
    RunEventType,
    RunFailureClass,
    new_run_id,
)
from rollout.harness.blobs import Blobs
from rollout.harness.conversations import Address, ConversationKey, DeliveryMode, DeliveryPolicy, Envelope
from rollout.harness.environments import EnvironmentService
from rollout.harness.hooks import RunHooks, observed, publish
from rollout.harness.imports import ToolSet
from rollout.harness.observation import InvalidObservation
from rollout.harness.program import Program
from rollout.harness.remote import remote_tool_set
from rollout.harness.runner import (
    Deployment,
    DirectModel,
    MessageRouter,
    RecordedEndpoints,
    RunBinding,
    RunOutcome,
    RunSpecification,
    RunStatus,
    instantiate,
)
from rollout.local.context import LocalRunContext

type EndpointFactory = Callable[[DirectModel], ModelEndpoint]
"""Creates the endpoint for a direct model binding; registered with the runner by provider name."""


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


class LocalRunner(MessageRouter):
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
        blobs: Blobs | None = None,
        recorder: RecordedEndpoints | None = None,
        hooks: Sequence[RunHooks] = (),
    ) -> None:
        """`recorder` serves recorded model bindings (trainable channels); direct bindings use `providers`. `hooks`
        watch every run: each event recorded and each model sample."""
        self._hooks = list(hooks)
        self._providers = dict(providers or {})
        self._tool_sets = dict(tool_sets or {})
        self._environment_service = environments
        self._blobs = blobs
        self._recorder = recorder
        self._runs: dict[str, LocalRunHandle] = {}
        self._deployments: dict[str, Deployment] = {}
        self._conversations: dict[str, _Conversation] = {}
        self._claimed: set[str] = set()
        self._background: set[asyncio.Task[None]] = set()

    def _recorded(self, handle: LocalRunHandle, event: RunEvent) -> None:
        handle.notify(event)
        publish(self._hooks, event)

    async def launch(self) -> None:
        """Nothing to start: runs execute on the caller's event loop."""

    async def close(self) -> None:
        """Nothing to release: nothing outlives the process."""

    # Deployments and inspection

    def deploy(self, deployment: Deployment) -> None:
        """Register or replace a deployment; a conversation's next run uses the current version."""
        self._deployments[deployment.name] = deployment

    def run(self, run_id: str) -> LocalRunHandle:
        return self._runs[run_id]

    def conversation_of(self, run_id: str) -> ConversationKey | None:
        """The conversation a run serves, if any."""
        handle = self._runs.get(run_id)
        return handle.conversation if handle is not None else None

    def conversation_runs(self, deployment: str, key: str) -> list[LocalRunHandle]:
        """The conversation's runs, oldest first."""
        conversation = self._conversations.get(ConversationKey(deployment=deployment, key=key).address)
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
        endpoints = resolve_endpoints(program, specification.binding, self._providers, self._recorder)
        endpoints = observed(endpoints, self._hooks, run_id)
        tool_sets = resolve_tool_sets(program, specification.binding, self._tool_sets)
        handle = LocalRunHandle(run_id, specification, conversation)
        handle.context = LocalRunContext(
            run_id,
            endpoints,
            context_hints=program.context_hints(),
            tool_sets=tool_sets,
            environment_service=self._environment_service,
            blobs=self._blobs,
            conversation=conversation,
            on_event=lambda event: self._recorded(handle, event),
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

    async def cancel(self, run_id: str, *, reason: str) -> None:
        handle = self._runs[run_id]
        task = handle.task
        if handle.done or task is None:
            return
        handle.context.record_event(RunEventType.RUN_CANCEL_REQUESTED, {"reason": reason, "by": "runner"})
        task.cancel()
        await asyncio.wait([task])

    # The transport of `MessageRouter`. Runs share one event loop and delivering never suspends, so nothing needs
    # holding while a message is delivered.

    def _exclusive(self, to: Address) -> AbstractAsyncContextManager[None]:
        return nullcontext()

    def _is_claimed(self, message_id: str) -> bool:
        return message_id in self._claimed

    def _claim(self, message_id: str, to: Address) -> None:
        self._claimed.add(message_id)

    async def _conversation_run(self, address: str, reply_to: Address | None) -> str:
        conversation = self._conversations.get(address)
        if conversation is None:
            key = ConversationKey.parse(address, origin=reply_to)
            if key.deployment not in self._deployments:
                raise ValueError(f"{address!r} does not name a conversation of a deployed agent")
            conversation = self._conversations[address] = _Conversation(key)
        if conversation.live is None or conversation.live.done:
            deployment = self._deployments[conversation.key.deployment]
            conversation.live = await self.start(deployment.specification, conversation=conversation.key)
            conversation.runs.append(conversation.live.run_id)
        return conversation.live.run_id

    def _delivery_policy(self, run_id: str) -> DeliveryPolicy | None:
        handle = self._runs.get(run_id)
        return handle.specification.binding.delivery if handle is not None and not handle.done else None

    async def _deliver(self, run_id: str, envelope: Envelope, mode: DeliveryMode) -> None:
        self._runs[run_id].context.deliver(envelope, mode)

    # Internals

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
            await context.environments.release_all()  # environments the run still owns
        handle.finish(outcome)
        self._leave_conversation(handle)

    def _leave_conversation(self, handle: LocalRunHandle) -> None:
        """Messages the finished run never consumed start the conversation's next run."""
        if handle.conversation is None:
            return
        address = handle.conversation.address
        conversation = self._conversations.get(address)
        if conversation is None or conversation.live is not handle:
            return
        conversation.live = None
        undelivered = handle.context.take_undelivered()
        if undelivered:
            task = asyncio.create_task(self._hand_over(address, undelivered))
            self._background.add(task)
            task.add_done_callback(self._background.discard)


def resolve_endpoints(
    program: Program,
    binding: RunBinding,
    providers: Mapping[str, EndpointFactory],
    recorder: RecordedEndpoints | None = None,
) -> dict[str, ModelEndpoint]:
    """An endpoint for each model slot of the program, from the binding: the recorder serves recorded models, the
    registered providers direct ones."""
    endpoints: dict[str, ModelEndpoint] = {}
    for slot in program.model_slots():
        model = binding.models.get(slot)
        if model is None:
            raise ValueError(f"the binding has no model for slot {slot!r}")
        if model.recorded is not None:
            if recorder is None:
                raise ValueError(f"slot {slot!r} is bound to a recorded channel, but the runner has no recorder")
            endpoints[slot] = recorder.endpoint(model.recorded)
            continue
        assert model.direct is not None  # a binding is one or the other
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
        if tool_binding is not None and tool_binding.url is not None:
            resolved[name] = remote_tool_set(tool_binding.url)
            continue
        if tool_binding is None or tool_binding.local is None:
            raise ValueError(f"the binding does not say how to serve the import {name!r}")
        tool_set = tool_sets.get(tool_binding.local)
        if tool_set is None:
            raise ValueError(f"no tool set registered as {tool_binding.local!r}")
        resolved[name] = tool_set
    return resolved
