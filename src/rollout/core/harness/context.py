"""The run context: everything task and agent code can reach (docs/core/harness/task.md#runcontext).

Runners implement `RunContext`. Its first group of members is for task and agent code; the second is used only
by the loop.
"""

import random
from collections.abc import Awaitable, Mapping, Sequence
from datetime import datetime
from typing import Protocol

from pydantic import JsonValue

from rollout.core.contracts import CapabilityContract, Message, ModelAddress, ToolChoice, ToolSpecification, Usage
from rollout.core.harness.blobs import Blobs
from rollout.core.harness.conversations import Address, ConversationKey, Envelope
from rollout.core.harness.environments import Environments
from rollout.core.harness.history import ContextHints, History
from rollout.core.harness.imports import Tools
from rollout.core.harness.observation import Observation, WaitFor


class Model(Protocol):
    """A model slot as code sees it. Nothing here identifies the policy, weights or engine."""

    @property
    def capabilities(self) -> CapabilityContract: ...

    @property
    def usage(self) -> Usage | None:
        """Usage reported by the latest sample, if any: drives compaction decisions."""
        ...

    async def sample(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpecification] = (),
        max_output_tokens: int | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Message: ...

    def address(self) -> ModelAddress:
        """For a harness that brings its own loop (a coding agent running inside the environment, say): where it
        reaches this slot's model. Hand it the base URL and key; what it samples there is this slot's, recorded like
        any other sample. Raises if the deployment does not serve models over HTTP."""
        ...


class Interrupted(Exception):
    """The reply in progress was cancelled by a message delivered with mode `INTERRUPT`."""

    def __init__(self, envelope: Envelope, reply_effect_id: str | None) -> None:
        super().__init__(f"interrupted by a {envelope.kind!r} message")
        self.envelope = envelope
        self.reply_effect_id = reply_effect_id
        """The policy sample that was cancelled, if one was in flight."""


class RunContext(Protocol):
    """Everything task and agent code can reach during a run. Passed to every hook as `run`."""

    # For task and agent code.
    @property
    def run_id(self) -> str: ...
    @property
    def conversation(self) -> ConversationKey | None: ...
    @property
    def turn(self) -> int:
        """Completed model turns so far."""
        ...

    @property
    def history(self) -> History: ...
    @property
    def models(self) -> Mapping[str, Model]: ...
    @property
    def model(self) -> Model:
        """`models["policy"]`."""
        ...

    @property
    def tools(self) -> Tools:
        """Imported tools; each call is a `tool.call` effect."""
        ...

    @property
    def environments(self) -> Environments | None:
        """Creates environments the run owns; None when the runner has no environment backend."""
        ...

    @property
    def blobs(self) -> Blobs | None:
        """Stores bytes such as images for `Media` blocks; None when the runner has no blob store."""
        ...

    @property
    def random(self) -> random.Random:
        """Seeded from `run_id`."""
        ...

    @property
    def context_hints(self) -> ContextHints: ...

    def now(self) -> datetime:
        """The current time. Use it instead of the wall clock, which durable runs cannot replay."""
        ...

    def reward(self, value: float, *, slot: str = "policy", key: str = "default") -> None:
        """Assign a reward to a model slot outside an observation (e.g. to an opponent, or several keyed rewards)."""
        ...

    def exclude_from_training(self, reason: str) -> None:
        """Mark the run as unsuitable for training, e.g. after an infrastructure fault that is not the policy's."""
        ...

    async def gather[T](self, *awaitables: Awaitable[T]) -> list[T]:
        """Await concurrently, in order. Equivalent to `asyncio.gather`."""
        ...

    def patched(self, change_id: str) -> bool:
        """`True` unless replaying history recorded before the change (see docs/core/harness/determinism.md)."""
        ...

    async def emit(self, kind: str, payload: JsonValue, *, to: Address | None = None) -> None:
        """Durable output, such as a reply to a person; a connector or client delivers it."""
        ...

    # For the loop.
    def record(self, observation: Observation | WaitFor, *, reply: Message | None = None) -> None:
        """Append a turn to the history. A `WaitFor` records only the reply it answers."""
        ...

    async def wait_for_message(self, wait: WaitFor) -> Envelope | None:
        """Suspend until a message of `wait.kind` arrives; `None` on timeout."""
        ...

    async def take_steering_messages(self) -> list[Envelope]:
        """Messages delivered with mode `STEER` since the last turn boundary."""
        ...

    async def interruptible[T](self, reply: Awaitable[T]) -> T:
        """Await an agent's reply; raises `Interrupted` if a message with mode `INTERRUPT` arrives meanwhile."""
        ...
