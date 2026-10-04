"""The run context: everything task and agent code can reach (docs/guide/tasks.md#the-run-context).

Runners implement `RunContext`. Its members are for task and agent code, except `record`, which only the loop
uses.
"""

import random
from collections.abc import Awaitable, Mapping, Sequence
from datetime import datetime
from typing import Protocol

from pydantic import JsonValue

from rollout.contracts import (
    CapabilityContract,
    Message,
    ModelAddress,
    SampleLink,
    ToolChoice,
    ToolSpecification,
    Usage,
)
from rollout.harness.blobs import Blobs
from rollout.harness.history import ContextHints, History
from rollout.harness.imports import Tools
from rollout.harness.observation import Observation
from rollout.harness.sandboxes import Sandbox


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
        links: Sequence[SampleLink] = (),
    ) -> Message:
        """A reply to `messages`; `links` say how the request follows from earlier ones of the slot (by the effect
        ids its replies carry: `rollout.harness.model.EFFECT_ID_META`)."""
        ...

    def address(self) -> ModelAddress:
        """For a harness that brings its own loop (a coding agent running inside the environment, say): where it
        reaches this slot's model. Hand it the base URL and key; what it samples there is this slot's, recorded like
        any other sample. Raises if the deployment does not serve models over HTTP."""
        ...


class RunContext(Protocol):
    """Everything task and agent code can reach during a run. Passed to every hook as `run`."""

    # For task and agent code.
    @property
    def run_id(self) -> str: ...
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

    def sandbox(self, name: str) -> Sandbox:
        """A sandbox the program declared (`Program.sandboxes()`), acquired for this run: its addresses, its
        environment, its operations. `KeyError` for a name the program did not declare."""
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
        """The current time, in UTC: the time the run's events carry."""
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

    async def emit(self, kind: str, payload: JsonValue) -> None:
        """Output of the run, such as its result: recorded as an `output.emit` effect and an `output.emitted` event."""
        ...

    # For the loop.
    def record(self, observation: Observation, *, reply: Message | None = None) -> None:
        """Append a turn to the history."""
        ...
