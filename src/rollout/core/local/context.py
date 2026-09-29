"""The in-process run context used by the `LocalRunner`: nothing persists, waits are in memory."""

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import JsonValue
from pydantic_core import to_jsonable_python

from rollout.core.contracts import (
    EffectKind,
    EffectStatus,
    Message,
    ModelEndpoint,
    OutcomeUnknown,
    RunEvent,
    RunEventType,
    arguments_digest,
    digest,
    effect_id,
    session_id,
)
from rollout.core.harness.context import Interrupted, Model
from rollout.core.harness.conversations import Address, ConversationKey, DeliveryMode, Envelope
from rollout.core.harness.environments import Environments, EnvironmentService
from rollout.core.harness.history import ContextHints, History, Turn
from rollout.core.harness.imports import Tools, ToolSet
from rollout.core.harness.model import EFFECT_ID_META, EndpointModel
from rollout.core.harness.observation import Observation, WaitFor


@dataclass(frozen=True)
class RewardAssignment:
    slot: str
    value: float
    key: str


class LocalRunContext:
    """Implements `RunContext` and `Effects` in process."""

    def __init__(
        self,
        run_id: str,
        endpoints: Mapping[str, ModelEndpoint],
        *,
        context_hints: ContextHints | None = None,
        tool_sets: Mapping[str, ToolSet] | None = None,
        environment_service: EnvironmentService | None = None,
        conversation: ConversationKey | None = None,
        generation: int = 0,
        on_event: Callable[[RunEvent], None] | None = None,
        retain_events: bool = True,
    ) -> None:
        self._run_id = run_id
        self._conversation = conversation
        self._generation = generation
        self._context_hints = context_hints or ContextHints()
        self._random = random.Random(run_id)
        self._history = History()
        self._turn = 0
        self._next_ordinal = 0
        self._on_event = on_event
        self.events: list[RunEvent] = []
        """Every event, in memory; empty when `retain_events` is False (a durable runner keeps them in its store)."""
        self._retain_events = retain_events
        self._next_seq = 0
        self.rewards: list[RewardAssignment] = []
        self.excluded_from_training: str | None = None

        self._models: dict[str, Model] = {
            slot: EndpointModel(endpoint, session_id(run_id, slot), self) for slot, endpoint in endpoints.items()
        }

        self._tools = Tools(tool_sets or {}, self)
        self._environments = Environments(environment_service, self) if environment_service is not None else None

        # Mailbox. `_held` keeps undelivered messages in arrival order with their delivery mode.
        self._held: list[tuple[Envelope, DeliveryMode]] = []
        self._waiting: tuple[str, asyncio.Future[Envelope]] | None = None
        self._acting: asyncio.Future[Any] | None = None
        self._interruption: Envelope | None = None
        self._interrupted_sample: str | None = None
        self._samples_in_flight: list[str] = []

    # RunContext, for task and agent code

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def conversation(self) -> ConversationKey | None:
        return self._conversation

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def history(self) -> History:
        return self._history

    @property
    def models(self) -> Mapping[str, Model]:
        return self._models

    @property
    def model(self) -> Model:
        return self._models["policy"]

    @property
    def tools(self) -> Tools:
        return self._tools

    @property
    def environments(self) -> Environments | None:
        return self._environments

    @property
    def random(self) -> random.Random:
        return self._random

    @property
    def context_hints(self) -> ContextHints:
        return self._context_hints

    def now(self) -> datetime:
        return datetime.now(UTC)

    def reward(self, value: float, *, slot: str = "policy", key: str = "default") -> None:
        if slot not in self._models:
            raise ValueError(f"unknown model slot {slot!r}")
        self.rewards.append(RewardAssignment(slot, value, key))
        self.record_event(RunEventType.REWARD_ASSIGNED, {"slot": slot, "value": value, "key": key})

    def exclude_from_training(self, reason: str) -> None:
        self.excluded_from_training = reason
        self.record_event(RunEventType.TRAINING_EXCLUDED, {"reason": reason})

    async def gather[T](self, *awaitables: Awaitable[T]) -> list[T]:
        return list(await asyncio.gather(*awaitables))

    def patched(self, change_id: str) -> bool:
        return True

    async def emit(self, kind: str, payload: JsonValue, *, to: Address | None = None) -> None:
        """Durable output, e.g. a reply that a connector delivers. Recorded as an `output.emit` effect."""
        arguments: dict[str, JsonValue] = {"kind": kind, "payload": payload}
        if to is not None:
            arguments["to"] = to.model_dump(mode="json")

        async def execute(effect_id: str, arguments_digest: str) -> str:
            return effect_id

        # Recorded after the effect, not inside it: a durable runner replays effects without executing them.
        identifier = await self.perform(EffectKind.OUTPUT_EMIT, arguments, execute, completion=lambda _: None)
        self.record_event(RunEventType.OUTPUT_EMITTED, {**arguments, "effect_id": identifier})

    # RunContext, for the loop

    def record(self, observation: Observation | WaitFor, *, reply: Message | None = None) -> None:
        if reply is not None:
            self._turn += 1
        if isinstance(observation, WaitFor):
            if reply is not None:
                self._history.append(Turn(reply=reply, observation=None))
            return
        self._history.append(Turn(reply=reply, observation=observation))
        messages: list[JsonValue] = [
            message.model_dump(mode="json", exclude_none=True) for message in observation.messages
        ]
        end = observation.end.value if observation.end is not None else None
        recorded: dict[str, JsonValue] = {"messages": messages, "reward": observation.reward, "end": end}
        self.record_event(
            RunEventType.OBSERVATION_RECORDED,
            {
                **recorded,
                "reply_effect_id": reply.meta.get(EFFECT_ID_META) if reply is not None else None,
                "info": to_jsonable_python(observation.info, fallback=repr),
                "digest": digest(recorded),
            },
        )

    async def wait_for_message(self, wait: WaitFor) -> Envelope | None:
        for index, (envelope, _) in enumerate(self._held):
            if envelope.kind == wait.kind:
                del self._held[index]
                return envelope
        timeout = wait.timeout.total_seconds() if wait.timeout is not None else None
        self.record_event(RunEventType.RUN_SUSPENDED, {"waiting_for": {"kind": wait.kind, "timeout": timeout}})
        future: asyncio.Future[Envelope] = asyncio.get_running_loop().create_future()
        self._waiting = (wait.kind, future)
        try:
            async with asyncio.timeout(timeout):
                return await future
        except TimeoutError:
            return None
        finally:
            self._waiting = None

    async def take_steering_messages(self) -> list[Envelope]:
        steering = [envelope for envelope, mode in self._held if mode is not DeliveryMode.QUEUE]
        self._held = [(envelope, mode) for envelope, mode in self._held if mode is DeliveryMode.QUEUE]
        return steering

    async def interruptible[T](self, reply: Awaitable[T]) -> T:
        acting = asyncio.ensure_future(reply)
        self._acting, self._interruption = acting, None
        try:
            result = await acting
        except asyncio.CancelledError:
            current = asyncio.current_task()
            interruption = self._interruption
            if interruption is None or (current is not None and current.cancelling()):
                raise
            reply_effect_id = self._interrupted_sample
            self.record_event(RunEventType.TURN_INTERRUPTED, {"reply_effect_id": reply_effect_id})
            raise Interrupted(interruption, reply_effect_id) from None
        finally:
            self._acting = None
        if self._interruption is not None:  # the reply finished before the cancellation landed
            self._held.append((self._interruption, DeliveryMode.STEER))
        return result

    # Delivery, for the runner

    def take_undelivered(self) -> list[Envelope]:
        """Messages the run never consumed; the runner hands them to the conversation's next run."""
        undelivered = [envelope for envelope, _ in self._held]
        self._held = []
        return undelivered

    def deliver(self, envelope: Envelope, mode: DeliveryMode) -> None:
        """Deliver a message to this run (docs/core/harness/conversations.md#priority-and-delivery-mode)."""
        self.record_event(
            RunEventType.MESSAGE_RECEIVED, {"envelope": envelope.model_dump(mode="json"), "mode": mode.value}
        )
        if self._waiting is not None and self._waiting[0] == envelope.kind and not self._waiting[1].done():
            self._waiting[1].set_result(envelope)
        elif mode is DeliveryMode.INTERRUPT and self._acting is not None and self._interruption is None:
            self._interruption = envelope
            self._interrupted_sample = self._samples_in_flight[-1] if self._samples_in_flight else None
            self._acting.cancel()
        else:
            self._held.append((envelope, mode))

    # Effects

    async def perform[T](
        self,
        kind: EffectKind,
        arguments: JsonValue,
        execute: Callable[[str, str], Awaitable[T]],
        *,
        completion: Callable[[T], JsonValue],
        guard: bool = False,
    ) -> T:
        identifier = effect_id(self._run_id, self._generation, self._next_ordinal)
        self._next_ordinal += 1
        arguments_hash = arguments_digest(arguments)
        self.record_event(
            RunEventType.EFFECT_REQUESTED,
            {"effect_id": identifier, "kind": kind.value, "arguments_digest": arguments_hash, "payload": arguments},
        )
        if kind is EffectKind.MODEL_SAMPLE:
            self._samples_in_flight.append(identifier)
        try:
            result = await self._execute_effect(kind, identifier, arguments_hash, execute, guard)
        except OutcomeUnknown:
            self._effect_completed(identifier, EffectStatus.OUTCOME_UNKNOWN, None, error_class="outcome_unknown")
            raise
        except asyncio.CancelledError:
            self._effect_completed(identifier, EffectStatus.FAILED, None, error_class="cancelled")
            raise
        except Exception as error:
            self._effect_completed(identifier, EffectStatus.FAILED, str(error), error_class=type(error).__name__)
            raise
        finally:
            if identifier in self._samples_in_flight:
                self._samples_in_flight.remove(identifier)
        self._effect_completed(identifier, EffectStatus.OK, completion(result))
        return result

    async def _execute_effect[T](
        self,
        kind: EffectKind,
        identifier: str,
        arguments_hash: str,
        execute: Callable[[str, str], Awaitable[T]],
        guard: bool,
    ) -> T:
        """Perform the effect. The durable run context overrides this to make it a recorded, guarded step."""
        return await execute(identifier, arguments_hash)

    def _effect_completed(
        self, identifier: str, status: EffectStatus, payload: JsonValue, *, error_class: str | None = None
    ) -> None:
        completed: dict[str, JsonValue] = {"effect_id": identifier, "status": status.value, "payload": payload}
        if error_class is not None:
            completed["error_class"] = error_class
        self.record_event(RunEventType.EFFECT_COMPLETED, completed)

    # Events

    def record_event(self, event_type: RunEventType, payload: JsonValue) -> RunEvent:
        event = RunEvent(
            run_id=self._run_id, seq=self._next_seq, type=event_type, recorded_at=self.now(), payload=payload
        )
        self._next_seq += 1
        if self._retain_events:
            self.events.append(event)
        if self._on_event is not None:
            self._on_event(event)
        return event
