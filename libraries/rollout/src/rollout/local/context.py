"""The in-process run context used by the `LocalRunner`: nothing persists."""

import asyncio
import contextlib
import random
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import JsonValue
from pydantic_core import to_jsonable_python

from rollout.contracts import (
    EffectKind,
    EffectStatus,
    Message,
    ModelEndpoint,
    RunEvent,
    RunEventType,
    arguments_digest,
    digest,
    effect_id,
    session_id,
)
from rollout.harness.blobs import Blobs
from rollout.harness.context import Model
from rollout.harness.history import ContextHints, History, Turn
from rollout.harness.imports import Tools, ToolSet
from rollout.harness.model import EFFECT_ID_META, EndpointModel
from rollout.harness.observation import Observation
from rollout.harness.sandboxes import Pool, Sandbox, SandboxSpec, acquire, harness_environment

GENERATION = 0
"""The generation in every `effect_id` (`{run_id}:{generation}:{ordinal}`): a run has one."""


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
        blobs: Blobs | None = None,
        on_event: Callable[[RunEvent], None] | None = None,
    ) -> None:
        self._run_id = run_id
        self._context_hints = context_hints or ContextHints()
        self._random = random.Random(run_id)
        self._history = History()
        self._turn = 0
        self._next_ordinal = 0
        self._on_event = on_event
        self.events: list[RunEvent] = []
        """Every event, in memory."""
        self._next_seq = 0
        self.rewards: list[RewardAssignment] = []
        self.excluded_from_training: str | None = None

        self._models: dict[str, Model] = {
            slot: EndpointModel(endpoint, session_id(run_id, slot), self) for slot, endpoint in endpoints.items()
        }

        self._tools = Tools(tool_sets or {}, self)
        self._blobs = blobs
        self._sandboxes: dict[str, Sandbox] = {}
        self._leased: list[tuple[str, Pool]] = []
        """The key of each sandbox the run acquired, or began to, and its pool: released there."""

    # RunContext, for task and agent code

    @property
    def run_id(self) -> str:
        return self._run_id

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

    def sandbox(self, name: str) -> Sandbox:
        if name not in self._sandboxes:
            raise KeyError(f"the run holds no sandbox {name!r}: a program declares its sandboxes in `sandboxes()`")
        return self._sandboxes[name]

    @property
    def blobs(self) -> Blobs | None:
        return self._blobs

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

    async def emit(self, kind: str, payload: JsonValue) -> None:
        """Output of the run, such as its result. Recorded as an `output.emit` effect."""
        arguments: dict[str, JsonValue] = {"kind": kind, "payload": payload}

        async def execute(effect_id: str, arguments_digest: str) -> str:
            return effect_id

        identifier = await self.perform(EffectKind.OUTPUT_EMIT, arguments, execute, completion=lambda _: None)
        self.record_event(RunEventType.OUTPUT_EMITTED, {**arguments, "effect_id": identifier})

    # RunContext, for the loop

    def record(self, observation: Observation, *, reply: Message | None = None) -> None:
        if reply is not None:
            self._turn += 1
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

    # Sandboxes, for the runner

    async def acquire_sandboxes(self, specs: Mapping[str, SandboxSpec], pools: Mapping[str, Pool], lease: str) -> None:
        """Acquire each declared sandbox from the pool of its kind, under `lease` and its name, giving a harness
        inside it its slots' model addresses; then record them."""
        for name, spec in specs.items():
            environment = harness_environment(spec.slots, lambda slot: self._models[slot].address())
            pool, key = pools[spec.kind], f"{lease}/{name}"
            self._leased.append((key, pool))
            self._sandboxes[name] = Sandbox(name, await acquire(pool, spec, key, environment), pool, self)
        if self._sandboxes:
            leases: dict[str, JsonValue] = {
                name: sandbox.lease.model_dump(mode="json") for name, sandbox in self._sandboxes.items()
            }
            self.record_event(RunEventType.SANDBOXES_ACQUIRED, {"sandboxes": leases})

    async def release_sandboxes(self) -> None:
        """Release every sandbox the run acquired, or began to: when the program has ended."""
        leased, self._leased, self._sandboxes = self._leased, [], {}
        for key, pool in leased:
            with contextlib.suppress(Exception):  # (one that is not released ends with its lease)
                await pool.release(key)

    # Effects

    async def perform[T](
        self,
        kind: EffectKind,
        arguments: JsonValue,
        execute: Callable[[str, str], Awaitable[T]],
        *,
        completion: Callable[[T], JsonValue],
    ) -> T:
        identifier = effect_id(self._run_id, GENERATION, self._next_ordinal)
        self._next_ordinal += 1
        arguments_hash = arguments_digest(arguments)
        self.record_event(
            RunEventType.EFFECT_REQUESTED,
            {"effect_id": identifier, "kind": kind.value, "arguments_digest": arguments_hash, "payload": arguments},
        )
        try:
            result = await execute(identifier, arguments_hash)
        except asyncio.CancelledError:
            self._effect_completed(identifier, EffectStatus.FAILED, None, error_class="cancelled")
            raise
        except Exception as error:
            self._effect_completed(identifier, EffectStatus.FAILED, str(error), error_class=type(error).__name__)
            raise
        self._effect_completed(identifier, EffectStatus.OK, completion(result))
        return result

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
        self.events.append(event)
        if self._on_event is not None:
            self._on_event(event)
        return event
