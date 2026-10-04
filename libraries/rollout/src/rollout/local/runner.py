"""`LocalRunner`: runs programs on the current asyncio loop. Nothing persists; a process crash loses in-flight runs."""

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping, Sequence

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
from rollout.harness.hooks import RunHooks, observed, publish
from rollout.harness.imports import ToolSet
from rollout.harness.observation import InvalidObservation
from rollout.harness.program import Program
from rollout.harness.remote import remote_pool, remote_tool_set
from rollout.harness.runner import (
    DirectModel,
    RecordedEndpoints,
    RunBinding,
    RunOutcome,
    RunSpecification,
    RunStatus,
    instantiate,
)
from rollout.harness.sandboxes import Pool
from rollout.local.context import LocalRunContext

type EndpointFactory = Callable[[DirectModel], ModelEndpoint]
"""Creates the endpoint for a direct model binding; registered with the runner by provider name."""


class LocalRunHandle:
    """A run started by a `LocalRunner`. Its context is available for inspection in tests and tools."""

    def __init__(self, run_id: str, specification: RunSpecification, lease: str | None = None) -> None:
        self._run_id = run_id
        self.specification = specification
        self.lease = lease or run_id
        """What the run's sandboxes are acquired under."""
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


class LocalRunner:
    """Implements `Runner` in process. Direct model bindings are served by endpoint factories registered by provider
    name."""

    def __init__(
        self,
        *,
        providers: Mapping[str, EndpointFactory] | None = None,
        tool_sets: Mapping[str, ToolSet] | None = None,
        blobs: Blobs | None = None,
        recorder: RecordedEndpoints | None = None,
        hooks: Sequence[RunHooks] = (),
        pools: Mapping[str, Pool] | None = None,
    ) -> None:
        """`recorder` serves recorded model bindings (trainable channels); direct bindings use `providers`. `pools`
        are the sandbox pools a binding names as `local`. `hooks` watch every run: each event recorded and each model
        sample."""
        self._hooks = list(hooks)
        self._providers = dict(providers or {})
        self._tool_sets = dict(tool_sets or {})
        self._pools = dict(pools or {})
        self._blobs = blobs
        self._recorder = recorder
        self._runs: dict[str, LocalRunHandle] = {}

    def _recorded(self, handle: LocalRunHandle, event: RunEvent) -> None:
        handle.notify(event)
        publish(self._hooks, event)

    async def launch(self) -> None:
        """Nothing to start: runs execute on the caller's event loop."""

    async def close(self) -> None:
        """Nothing to release: nothing outlives the process."""

    # Runner

    def run(self, run_id: str) -> LocalRunHandle:
        return self._runs[run_id]

    async def start(
        self,
        specification: RunSpecification,
        *,
        run_id: str | None = None,
        labels: Mapping[str, str] | None = None,
        lease: str | None = None,
    ) -> LocalRunHandle:
        run_id = run_id or new_run_id()
        if run_id in self._runs:
            raise ValueError(f"run {run_id} already exists")
        program = instantiate(specification.program)
        endpoints = resolve_endpoints(program, specification.binding, self._providers, self._recorder)
        endpoints = observed(endpoints, self._hooks, run_id)
        tool_sets = resolve_tool_sets(program, specification.binding, self._tool_sets)
        pools = resolve_pools(program, specification.binding, self._pools)
        handle = LocalRunHandle(run_id, specification, lease)
        handle.context = LocalRunContext(
            run_id,
            endpoints,
            context_hints=program.context_hints(),
            tool_sets=tool_sets,
            blobs=self._blobs,
            on_event=lambda event: self._recorded(handle, event),
        )
        handle.context.record_event(
            RunEventType.RUN_CREATED,
            {
                "specification": specification.model_dump(mode="json", exclude_none=True),
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
        handle.attach(asyncio.create_task(self._execute(handle, program, pools), name=f"run {run_id}"))
        return handle

    async def cancel(self, run_id: str, *, reason: str) -> None:
        handle = self._runs[run_id]
        task = handle.task
        if handle.done or task is None:
            return
        handle.context.record_event(RunEventType.RUN_CANCEL_REQUESTED, {"reason": reason, "by": "runner"})
        task.cancel()
        await asyncio.wait([task])

    # Internals

    async def _execute(self, handle: LocalRunHandle, program: Program, pools: Mapping[str, Pool]) -> None:
        context = handle.context
        try:
            await context.acquire_sandboxes(program.sandboxes(), pools, handle.lease)
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
        await context.release_sandboxes()
        handle.finish(outcome)


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


def resolve_pools(program: Program, binding: RunBinding, pools: Mapping[str, Pool]) -> dict[str, Pool]:
    """The pool serving each kind of sandbox the program declares, from the binding and the registered pools."""
    resolved: dict[str, Pool] = {}
    for spec in program.sandboxes().values():
        pool_binding = binding.pools.get(spec.kind)
        if pool_binding is not None and pool_binding.url is not None:
            resolved[spec.kind] = remote_pool(pool_binding.url)
            continue
        if pool_binding is None or pool_binding.local is None:
            raise ValueError(f"the binding does not say which pool serves {spec.kind} sandboxes")
        pool = pools.get(pool_binding.local)
        if pool is None:
            raise ValueError(f"no pool registered as {pool_binding.local!r}")
        resolved[spec.kind] = pool
    return resolved
