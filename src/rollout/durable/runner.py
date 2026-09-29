"""`DurableRunner`: runs survive crashes and restarts (docs/durability/README.md).

Each run is a DBOS workflow whose id is the `run_id`. This first version runs the program inside the workflow, in the
runner's process ("trusted mode": platform and in-house code, trust tiers T0 and T1). The sandboxed task host that
speaks `HarnessHost` comes next; task and agent code will not change.

State lives in one directory:

- `dbos.sqlite`: the DBOS system database: workflow inputs, recorded steps, messages.
- `runs.sqlite`: run events, runs, conversations and delivered messages (see `RunStore`).

Only one `DurableRunner` can be active in a process, because DBOS is a process-wide singleton.
"""

import asyncio
import concurrent.futures
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from dbos import DBOS, SetWorkflowID, WorkflowHandleAsync
from pydantic import JsonValue

from rollout.core.contracts import (
    TERMINAL_EVENT_TYPES,
    RunEvent,
    RunEventType,
    RunFailureClass,
    new_run_id,
    new_ulid,
)
from rollout.core.harness.blobs import Blobs
from rollout.core.harness.conversations import Address, ConversationKey, DeliveryMode, Envelope, Priority
from rollout.core.harness.environments import EnvironmentService
from rollout.core.harness.imports import ToolSet
from rollout.core.harness.loop import UNLOAD
from rollout.core.harness.observation import InvalidObservation
from rollout.core.harness.runner import Deployment, RunOutcome, RunSpecification, RunStatus, instantiate
from rollout.core.local.runner import EndpointFactory, RunNotLive, resolve_endpoints, resolve_tool_sets
from rollout.durable.context import INBOX, INTERRUPT, DurableRunContext, RunCancelled
from rollout.durable.store import RunStore

_active: "DurableRunner | None" = None


def _runner() -> "DurableRunner":
    if _active is None:
        raise RuntimeError("no DurableRunner is active in this process; call launch() first")
    return _active


@DBOS.workflow(name="rollout.run")
async def run_workflow(
    run_id: str,
    specification: dict[str, Any],
    conversation: dict[str, Any] | None,
    labels: dict[str, str],
    started_at: str,
) -> dict[str, Any]:
    """One run. Re-executed from the start on recovery; recorded steps and receives return their recorded results."""
    runner = _runner()
    task = asyncio.current_task()
    if task is not None:
        runner.workflow_tasks[run_id] = task  # so eviction can unload it
    try:
        return await runner.execute(run_id, specification, conversation, labels, started_at)
    finally:
        if runner.workflow_tasks.get(run_id) is task:
            del runner.workflow_tasks[run_id]


class DurableRunHandle:
    """A durable run, read from the store: valid across processes and restarts."""

    def __init__(self, runner: "DurableRunner", run_id: str) -> None:
        self._runner = runner
        self._run_id = run_id

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def outcome(self) -> RunOutcome | None:
        record = self._runner.store.run(self._run_id)
        if record is None or record.outcome is None:
            return None
        return RunOutcome.model_validate_json(record.outcome)

    @property
    def done(self) -> bool:
        return self.outcome is not None

    async def result(self) -> RunOutcome:
        while (outcome := self.outcome) is None:
            await self._runner.store.changed(self._run_id, wait_seconds=0.5)
        return outcome

    async def events(self, *, from_seq: int = 0) -> AsyncIterator[RunEvent]:
        index = from_seq
        while True:
            for event in self._runner.store.events(self._run_id, index):
                index = event.seq + 1
                yield event
                if event.type in TERMINAL_EVENT_TYPES:
                    return
            await self._runner.store.changed(self._run_id, wait_seconds=0.5)

    def recorded_events(self) -> list[RunEvent]:
        """Every event recorded so far."""
        return self._runner.store.events(self._run_id)


class DurableRunner:
    """Implements `Runner` on DBOS. Call `await launch()` before use and `await close()` after."""

    def __init__(
        self,
        directory: Path,
        *,
        providers: Mapping[str, EndpointFactory] | None = None,
        tool_sets: Mapping[str, ToolSet] | None = None,
        environments: EnvironmentService | None = None,
        blobs: Blobs | None = None,
        application: str = "rollout",
        evict_after: timedelta | None = timedelta(minutes=5),
        eviction_interval: float = 5.0,
    ) -> None:
        """`evict_after`: unload runs that have waited this long for a message (None keeps every run resident);
        `eviction_interval`: how often, in seconds, to look for runs to evict or wake (docs/durability/eviction.md).
        """
        self._environment_service = environments
        self._blobs = blobs
        self._evict_after = evict_after
        self._eviction_interval = eviction_interval
        self._run_locks: dict[str, asyncio.Lock] = {}
        self._last_activity: dict[str, datetime] = {}
        """When each run last received a message or was woken; a run is evictable only if it suspended since."""
        self._evictor: asyncio.Task[None] | None = None
        self.workflow_tasks: dict[str, asyncio.Task[Any]] = {}
        """The asyncio task executing each resident run's workflow."""
        directory.mkdir(parents=True, exist_ok=True)
        self.store = RunStore(directory / "runs.sqlite")
        self._providers = dict(providers or {})
        self._tool_sets = dict(tool_sets or {})
        self._deployments: dict[str, Deployment] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._watchers: dict[str, asyncio.Task[None]] = {}
        DBOS(config={"name": application, "system_database_url": f"sqlite:///{directory / 'dbos.sqlite'}"})

    async def launch(self) -> None:
        """Start DBOS, which recovers the runs a crash left unfinished, and follow them."""
        global _active
        if _active is not None and _active is not self:
            raise RuntimeError("another DurableRunner is active in this process")
        _active = self
        DBOS.launch()  # recovers resident runs; evicted runs are CANCELLED in DBOS and stay unloaded
        for run_id in self.store.unfinished_runs():
            self._watch(run_id)
        if self._evict_after is not None:
            self._evictor = asyncio.create_task(self._evict_idle_runs(), name="evictor")

    async def close(self) -> None:
        global _active
        if self._evictor is not None:
            self._evictor.cancel()
        for watcher in list(self._watchers.values()):
            watcher.cancel()
        DBOS.destroy(destroy_registry=False)
        # DBOS installed its own thread pool as the loop's default executor and just shut it down; restore one so
        # the rest of the process (DNS lookups, to_thread) keeps working.
        asyncio.get_running_loop().set_default_executor(concurrent.futures.ThreadPoolExecutor())
        self.store.close()
        _active = None

    # Deployments and inspection

    def deploy(self, deployment: Deployment) -> None:
        self._deployments[deployment.name] = deployment

    def run(self, run_id: str) -> DurableRunHandle:
        if self.store.run(run_id) is None:
            raise KeyError(run_id)
        return DurableRunHandle(self, run_id)

    def conversation_of(self, run_id: str) -> ConversationKey | None:
        """The conversation a run serves, if any."""
        record = self.store.run(run_id)
        if record is None or record.conversation is None:
            return None
        stored = self.store.conversation_key(record.conversation)
        return ConversationKey.model_validate(stored) if isinstance(stored, dict) else None

    def conversation_runs(self, deployment: str, key: str) -> list[DurableRunHandle]:
        return [DurableRunHandle(self, run_id) for run_id in self.store.conversation_runs(f"{deployment}/{key}")]

    # Runner

    async def start(
        self,
        specification: RunSpecification,
        *,
        run_id: str | None = None,
        conversation: ConversationKey | None = None,
        labels: Mapping[str, str] | None = None,
    ) -> DurableRunHandle:
        run_id = run_id or new_run_id()
        specification_json = specification.model_dump(mode="json", exclude_none=True)
        conversation_json = conversation.model_dump(mode="json", exclude_none=True) if conversation else None
        address = f"{conversation.deployment}/{conversation.key}" if conversation else None
        self.store.create_run(run_id, specification_json, address, conversation_json)
        with SetWorkflowID(run_id):
            await DBOS.start_workflow_async(
                run_workflow, run_id, specification_json, conversation_json, dict(labels or {}), _now()
            )
        self._watch(run_id)
        return DurableRunHandle(self, run_id)

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
            record = self.store.run(to.value)
            if record is None or record.status != "running":
                raise RunNotLive(to.value)
            mode = RunSpecification.model_validate_json(record.specification).binding.delivery.mode(priority, sender)
            await self._deliver(to.value, envelope, mode)
            return message_id
        if to.kind != "conversation":
            raise ValueError(f"the durable runner cannot deliver to {to.kind} addresses")
        lock = self._locks.setdefault(to.value, asyncio.Lock())
        async with lock:
            if not self.store.claim_message(message_id, to.value):
                return message_id  # a retry of a message already delivered
            run_id = self._live_run(to.value)
            if run_id is None:
                run_id = (await self._start_conversation_run(to.value, envelope.reply_to)).run_id
            record = self.store.run(run_id)
            assert record is not None
            mode = RunSpecification.model_validate_json(record.specification).binding.delivery.mode(priority, sender)
            await self._deliver(run_id, envelope, mode)
        return message_id

    async def cancel(self, run_id: str, *, reason: str) -> None:
        """Ask the run to stop at its next effect, wait or turn boundary; `teardown` runs."""
        record = self.store.run(run_id)
        if record is None or record.status != "running":
            return
        item = {"type": "cancel", "reason": reason, "sent_at": _now()}
        async with self._run_lock(run_id):
            await DBOS.send_async(run_id, item, INBOX, idempotency_key=f"cancel:{run_id}")
            await DBOS.send_async(run_id, {"cancel": True}, INTERRUPT, idempotency_key=f"cancel-signal:{run_id}")
            await self._wake_if_evicted(run_id)  # an evicted run must run again to tear down
        await DurableRunHandle(self, run_id).result()

    # The workflow body

    async def execute(
        self,
        run_id: str,
        specification_json: dict[str, Any],
        conversation_json: dict[str, Any] | None,
        labels: dict[str, str],
        started_at: str,
    ) -> dict[str, Any]:
        specification = RunSpecification.model_validate(specification_json)
        conversation = ConversationKey.model_validate(conversation_json) if conversation_json else None
        program = instantiate(specification.program)
        context = DurableRunContext(
            run_id,
            resolve_endpoints(program, specification.binding, self._providers),
            started_at=datetime.fromisoformat(started_at),
            context_hints=program.context_hints(),
            tool_sets=resolve_tool_sets(program, specification.binding, self._tool_sets),
            environment_service=self._environment_service,
            blobs=self._blobs,
            conversation=conversation,
            on_event=self.store.append,
            mark_attempt=self.store.mark_attempt,
        )
        context.record_event(
            RunEventType.RUN_CREATED,
            {"specification": specification_json, "conversation": conversation_json, "labels": dict(labels)},
        )
        specifications = program.tool_specifications() + context.tools.specifications()
        if specifications:
            resolved: list[JsonValue] = [s.model_dump(mode="json", exclude_none=True) for s in specifications]
            context.record_event(RunEventType.TOOLS_RESOLVED, {"specifications": resolved})
        try:
            await program.main(context)
            outcome = RunOutcome(status=RunStatus.COMPLETED)
            context.record_event(RunEventType.RUN_COMPLETED, {"outcome": "success"})
        except RunCancelled:
            outcome = RunOutcome(status=RunStatus.CANCELLED)
            context.record_event(RunEventType.RUN_CANCELLED, None)
        except InvalidObservation as error:
            outcome = RunOutcome(
                status=RunStatus.FAILED, failure_class=RunFailureClass.INVALID_OBSERVATION, detail=str(error)
            )
            context.record_event(RunEventType.RUN_FAILED, {"class": "invalid_observation", "detail": str(error)})
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            outcome = RunOutcome(status=RunStatus.FAILED, failure_class=RunFailureClass.TASK_ERROR, detail=detail)
            context.record_event(RunEventType.RUN_FAILED, {"class": "task_error", "detail": detail})
        if context.environments is not None:
            await context.environments.release_all()  # environments the run still owns (P12)
        undelivered = await context.undelivered()
        self.store.finish_run(run_id, outcome.status.value, outcome.model_dump(mode="json", exclude_none=True))
        return {"undelivered": [envelope.model_dump(mode="json") for envelope in undelivered]}

    # Internals

    def _live_run(self, address: str) -> str | None:
        run_id = self.store.live_run(address)
        if run_id is None:
            return None
        record = self.store.run(run_id)
        return run_id if record is not None and record.status == "running" else None

    async def _start_conversation_run(self, address: str, reply_to: Address | None) -> DurableRunHandle:
        stored = self.store.conversation_key(address)
        if isinstance(stored, dict):
            conversation = ConversationKey.model_validate(stored)
        else:
            namespace, name, key = [*address.split("/", 2), "", ""][:3]
            if not key:
                raise ValueError(f"{address!r} does not name a conversation")
            conversation = ConversationKey(deployment=f"{namespace}/{name}", key=key, origin=reply_to)
        deployment = self._deployments.get(conversation.deployment)
        if deployment is None:
            raise ValueError(f"{address!r} does not name a conversation of a deployed agent")
        return await self.start(deployment.specification, conversation=conversation)

    async def _deliver(self, run_id: str, envelope: Envelope, mode: DeliveryMode) -> None:
        item = {"type": "message", "envelope": envelope.model_dump(mode="json"), "mode": mode.value, "sent_at": _now()}
        async with self._run_lock(run_id):
            await DBOS.send_async(run_id, item, INBOX, idempotency_key=f"{run_id}:{envelope.message_id}")
            if mode is DeliveryMode.INTERRUPT:
                signal = {"message_id": envelope.message_id}
                key = f"{run_id}:{envelope.message_id}:signal"
                await DBOS.send_async(run_id, signal, INTERRUPT, idempotency_key=key)
            self._last_activity[run_id] = datetime.now(UTC)
            await self._wake_if_evicted(run_id)

    # Eviction (docs/durability/eviction.md)

    def _run_lock(self, run_id: str) -> asyncio.Lock:
        return self._run_locks.setdefault(run_id, asyncio.Lock())

    async def _wake_if_evicted(self, run_id: str) -> None:
        """Resume an evicted run; its replay returns recorded steps and then takes the new message. Hold its lock."""
        if not self.store.is_evicted(run_id):
            return
        self._last_activity[run_id] = datetime.now(UTC)  # until it suspends again, it must not be re-evicted
        self.store.wake(run_id)
        await DBOS.resume_workflow_async(run_id)
        self._watch(run_id)

    async def _evict_idle_runs(self) -> None:
        assert self._evict_after is not None
        while True:
            await asyncio.sleep(self._eviction_interval)
            now = datetime.now(UTC)
            for run_id in self.store.due_for_waking(_timestamp(now)):
                async with self._run_lock(run_id):
                    await self._wake_if_evicted(run_id)
            for event in self.store.last_events():
                if event.type is not RunEventType.RUN_SUSPENDED or now - event.recorded_at < self._evict_after:
                    continue
                async with self._run_lock(event.run_id):
                    latest = self.store.events(event.run_id, event.seq)
                    active = self._last_activity.get(event.run_id)
                    if len(latest) != 1 or (active is not None and active >= event.recorded_at):
                        continue  # it moved on, or it was messaged or woken since it suspended
                    self.store.evict(event.run_id, _wake_at(event))
                    await DBOS.cancel_workflow_async(event.run_id)  # DBOS stops tracking it; wake resumes it
                    resident = self.workflow_tasks.pop(event.run_id, None)
                    if resident is not None:
                        resident.cancel(UNLOAD)  # unload now: the coroutine would otherwise linger in its receive
                    watcher = self._watchers.pop(event.run_id, None)
                    if watcher is not None:
                        watcher.cancel()  # a new watcher follows the run when it wakes

    def _watch(self, run_id: str) -> None:
        previous = self._watchers.get(run_id)
        if previous is not None and not previous.done():
            return
        task = asyncio.create_task(self._follow(run_id), name=f"follow {run_id}")
        self._watchers[run_id] = task
        task.add_done_callback(
            lambda done: self._watchers.pop(run_id, None) if self._watchers.get(run_id) is done else None
        )

    async def _follow(self, run_id: str) -> None:
        """When a conversation's run ends, messages it never consumed start the conversation's next run."""
        handle: WorkflowHandleAsync[dict[str, Any]] = await DBOS.retrieve_workflow_async(run_id)
        try:
            result = await handle.get_result()
        except Exception:
            if self.store.is_evicted(run_id):
                return  # evicted, not ended: a new watcher follows it when it wakes
            raise
        undelivered = [Envelope.model_validate(item) for item in result.get("undelivered", [])]
        record = self.store.run(run_id)
        if not undelivered or record is None or record.conversation is None:
            return
        async with self._locks.setdefault(record.conversation, asyncio.Lock()):
            successor = self._live_run(record.conversation)
            if successor is None:
                successor = (await self._start_conversation_run(record.conversation, None)).run_id
            for envelope in undelivered:
                await self._deliver(successor, envelope, DeliveryMode.QUEUE)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _timestamp(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="microseconds")


def _wake_at(suspended: RunEvent) -> str | None:
    """When an evicted run's wait times out, from its `run.suspended` event."""
    data = suspended.payload if isinstance(suspended.payload, dict) else {}
    waiting = data.get("waiting_for")
    timeout = waiting.get("timeout") if isinstance(waiting, dict) else None
    if not isinstance(timeout, int | float):
        return None
    return _timestamp(suspended.recorded_at + timedelta(seconds=timeout))
