"""Hooks: watch runs as they happen.

A runner calls its hooks for every run event it records and for every model sample it performs. Events are the
run's journal: they name a sample's context by digest only. A `ModelSample` carries the content: the messages the model
was sent, the tools it was offered and what it replied. Between them a hook can log, measure, or feed a live view
(`rollout_train.monitor`) without the program taking part.

Hooks run on the runner's event loop, in the middle of the run: they must be quick and must not block (hand slow work
to a queue or a file). A hook that raises is logged and otherwise ignored; a hook never fails a run. A durable runner
calls them for what this process records and performs: after a crash, another runner's hooks see the rest.
"""

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rollout.contracts import (
    CapabilityContract,
    ModelAddress,
    ModelEndpoint,
    RunEvent,
    SampleRequest,
    SampleResult,
    address_of,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSample:
    """One model sample: what a slot's model was sent and what it replied."""

    run_id: str
    slot: str
    request: SampleRequest
    """`request.context.append` holds the messages sent; `request.tools` the tools offered."""
    result: SampleResult
    seconds: float
    """How long the endpoint took."""


class RunHooks:
    """Subclass and override what you need; pass instances to a runner (`LocalRunner(hooks=[...])`)."""

    def on_event(self, event: RunEvent) -> None:
        """A run event was recorded (of any run of the runner; `event.run_id` says which)."""

    def on_sample(self, sample: ModelSample) -> None:
        """A model replied."""


def publish(hooks: Sequence[RunHooks], event: RunEvent) -> None:
    """Call every hook's `on_event`."""
    for hook in hooks:
        try:
            hook.on_event(event)
        except Exception:
            logger.exception("a hook failed on a %s event of run %s", event.type.value, event.run_id)


def observed(
    endpoints: Mapping[str, ModelEndpoint], hooks: Sequence[RunHooks], run_id: str
) -> Mapping[str, ModelEndpoint]:
    """The endpoints of a run's slots, reporting each sample to the hooks."""
    if not hooks:
        return endpoints
    return {slot: _ObservedEndpoint(endpoint, hooks, run_id, slot) for slot, endpoint in endpoints.items()}


class _ObservedEndpoint:
    """Implements `AddressableEndpoint` over another endpoint, telling hooks of each reply."""

    def __init__(self, endpoint: ModelEndpoint, hooks: Sequence[RunHooks], run_id: str, slot: str) -> None:
        self._endpoint = endpoint
        self._hooks = hooks
        self._run_id = run_id
        self._slot = slot

    def describe(self, session_id: str) -> CapabilityContract:
        return self._endpoint.describe(session_id)

    def address(self, session_id: str, *, through: ModelEndpoint | None = None) -> ModelAddress:
        return address_of(self._endpoint, session_id, through=through or self)  # (its samples reach the hooks too)

    async def cancel(self, effect_id: str) -> None:
        await self._endpoint.cancel(effect_id)

    async def sample(self, request: SampleRequest) -> SampleResult:
        started = time.monotonic()
        result = await self._endpoint.sample(request)
        sample = ModelSample(self._run_id, self._slot, request, result, time.monotonic() - started)
        for hook in self._hooks:
            try:
                hook.on_sample(sample)
            except Exception:
                logger.exception("a hook failed on a sample of %s in run %s", self._slot, self._run_id)
        return result
