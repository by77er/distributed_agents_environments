"""The run context inside a DBOS workflow: effects are recorded steps, messages arrive through a durable inbox.

Replay works by re-running the program: DBOS returns the recorded result of every step and receive it already
performed, so the program takes the same path and regenerates the same effect identities and events.

Messages reach a run on two DBOS topics:

- `inbox` carries every message and control request, and is read only at defined points: when the run waits
  (`WaitFor`), and at turn boundaries (steering). Reads are recorded, so replay sees the same messages.
- `interrupt` carries a signal (never the message itself) for `INTERRUPT` deliveries and cancellations. The agent's
  reply races that signal through `DBOS.asyncio_wait`, which records which finished first. A signal lost to that
  race is harmless: its message is still in the inbox.
"""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime, timedelta

from dbos import DBOS
from pydantic import JsonValue

from rollout.core.contracts import EffectKind, ModelEndpoint, OutcomeUnknown, RunEvent, RunEventType
from rollout.core.harness.blobs import Blobs
from rollout.core.harness.context import Interrupted
from rollout.core.harness.conversations import ConversationKey, DeliveryMode, Envelope
from rollout.core.harness.environments import EnvironmentService
from rollout.core.harness.history import ContextHints
from rollout.core.harness.imports import ToolSet
from rollout.core.harness.observation import WaitFor
from rollout.core.local.context import LocalRunContext

INBOX = "inbox"
INTERRUPT = "interrupt"
LONG_WAIT_SECONDS = 24 * 3600.0
"""DBOS receives need a timeout; waits without one are repeated receives of this length."""


class RunCancelled(Exception):
    """A cancellation request reached the run; the program unwinds and `teardown` runs."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def utc_now() -> str:
    """The wall clock, as the runner stamps messages and the context records it."""
    return datetime.now(UTC).isoformat()


CLOCK_STEP = "_utc_now"
"""The name a clock reading is recorded under in a run's journal; a replay checks it against the journal."""


class DurableRunContext(LocalRunContext):
    def __init__(
        self,
        run_id: str,
        endpoints: Mapping[str, ModelEndpoint],
        *,
        started_at: datetime,
        context_hints: ContextHints | None = None,
        tool_sets: Mapping[str, ToolSet] | None = None,
        environment_service: EnvironmentService | None = None,
        blobs: Blobs | None = None,
        conversation: ConversationKey | None = None,
        on_event: Callable[[RunEvent], None] | None = None,
        mark_attempt: Callable[[str], bool] = lambda effect_id: True,
    ) -> None:
        self._clock = started_at
        self._mark_attempt = mark_attempt
        super().__init__(
            run_id,
            endpoints,
            context_hints=context_hints,
            tool_sets=tool_sets,
            environment_service=environment_service,
            blobs=blobs,
            conversation=conversation,
            on_event=on_event,
            retain_events=False,  # the runner's store has them; an idle run should hold as little as possible
        )
        self._cancel_reason: str | None = None

    # Time: the time of the latest recorded input, so a replay sees the same clock.

    def now(self) -> datetime:
        return self._clock

    async def _record_clock(self) -> None:
        self._clock = datetime.fromisoformat(await DBOS.run_step_async({"name": CLOCK_STEP}, utc_now))

    # Effects: each one is a DBOS step.

    async def _execute_effect[T](
        self,
        kind: EffectKind,
        identifier: str,
        arguments_hash: str,
        execute: Callable[[str, str], Awaitable[T]],
        guard: bool,
    ) -> T:
        self._check_cancelled()

        async def step() -> tuple[T, str]:
            # The step body runs only if DBOS has no recorded result for it. For a guarded effect, a marker left by
            # an earlier attempt means a crash interrupted it: it may have happened, so it must not run again.
            if guard and not self._mark_attempt(identifier):
                raise OutcomeUnknown(identifier)
            return await execute(identifier, arguments_hash), utc_now()

        result, completed_at = await DBOS.run_step_async(None, step)
        self._clock = datetime.fromisoformat(completed_at)
        return result

    # Messages

    async def _receive(self, wait_seconds: float) -> bool:
        """Receive one inbox item, waiting up to `wait_seconds`. False when none arrived."""
        item = await DBOS.recv_async(INBOX, timeout_seconds=max(wait_seconds, 0.0))
        if item is None:
            if wait_seconds > 0:
                await self._record_clock()
            return False
        self._accept(item)
        return True

    def _accept(self, item: dict[str, JsonValue]) -> None:
        self._clock = datetime.fromisoformat(str(item["sent_at"]))
        if item["type"] == "cancel":
            if self._cancel_reason is None:
                self._cancel_reason = str(item.get("reason") or "cancelled")
                self.record_event(RunEventType.RUN_CANCEL_REQUESTED, {"reason": self._cancel_reason, "by": "runner"})
            return
        envelope = Envelope.model_validate(item["envelope"])
        mode = DeliveryMode(str(item["mode"]))
        self.record_event(
            RunEventType.MESSAGE_RECEIVED, {"envelope": envelope.model_dump(mode="json"), "mode": mode.value}
        )
        self._held.append((envelope, mode))

    async def _drain(self) -> None:
        """Take every item already in the inbox, without waiting."""
        while await self._receive(0):
            pass

    def _check_cancelled(self) -> None:
        if self._cancel_reason is not None:
            raise RunCancelled(self._cancel_reason)

    async def wait_for_message(self, wait: WaitFor) -> Envelope | None:
        await self._drain()
        self._check_cancelled()
        timeout = wait.timeout.total_seconds() if wait.timeout is not None else None
        deadline = self._clock + timedelta(seconds=timeout) if timeout is not None else None
        suspended = False
        while True:
            for index, (envelope, _) in enumerate(self._held):
                if envelope.kind == wait.kind:
                    del self._held[index]
                    return envelope
            if not suspended:
                self.record_event(RunEventType.RUN_SUSPENDED, {"waiting_for": {"kind": wait.kind, "timeout": timeout}})
                suspended = True
            remaining = (deadline - self._clock).total_seconds() if deadline is not None else LONG_WAIT_SECONDS
            if deadline is not None and remaining <= 0:
                return None
            arrived = await self._receive(remaining)
            self._check_cancelled()
            if not arrived and deadline is not None:
                return None

    async def take_steering_messages(self) -> list[Envelope]:
        await self._drain()
        self._check_cancelled()
        return await super().take_steering_messages()

    async def interruptible[T](self, reply: Awaitable[T]) -> T:
        await self._drain()
        self._check_cancelled()
        acting = asyncio.ensure_future(reply)
        while True:
            signal = asyncio.ensure_future(DBOS.recv_async(INTERRUPT, timeout_seconds=LONG_WAIT_SECONDS))
            done, _ = await DBOS.asyncio_wait([acting, signal], return_when="FIRST_COMPLETED")
            if acting in done:
                signal.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await signal
                return acting.result()
            await self._drain()
            interruption = next(((e, m) for e, m in self._held if m is DeliveryMode.INTERRUPT), None)
            if self._cancel_reason is None and interruption is None:
                continue  # a stale signal: its message was already consumed
            reply_effect_id = self._samples_in_flight[-1] if self._samples_in_flight else None
            acting.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await acting
            self._check_cancelled()
            assert interruption is not None
            self._held.remove(interruption)
            self.record_event(RunEventType.TURN_INTERRUPTED, {"reply_effect_id": reply_effect_id})
            raise Interrupted(interruption[0], reply_effect_id)

    async def undelivered(self) -> list[Envelope]:
        """Messages the run never consumed, including any still in the inbox; handed to the next run."""
        await self._drain()
        return self.take_undelivered()
