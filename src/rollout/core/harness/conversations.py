"""Addressing, envelopes, priorities and delivery modes (docs/core/harness/conversations.md)."""

from collections.abc import Mapping
from enum import StrEnum
from typing import Literal

from pydantic import Field, JsonValue

from rollout.core.contracts import Block, ContractModel, FrozenSequence


class Address(ContractModel):
    kind: Literal["conversation", "run", "external"]
    value: str
    """`{deployment}/{key}`, a `run_id`, or a connector target."""


class ConversationKey(ContractModel):
    deployment: str
    """e.g. `acme/support-bot`."""
    key: str
    """Caller-chosen, e.g. `slack:T1/C2/171.2` or `user:42`."""
    origin: Address | None = None
    """Where replies go by default (e.g. a connector target)."""


class Envelope(ContractModel):
    kind: str = "message"
    content: FrozenSequence[Block] = ()
    data: JsonValue = None
    """A structured payload."""
    reply_to: Address | None = None
    message_id: str = ""
    """Set by the runner: the sender's `effect_id` or the caller's idempotency key."""
    sender: str | None = None
    """Set by the runner; never trusted from the payload."""


class Priority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class DeliveryMode(StrEnum):
    QUEUE = "queue"
    """Deliver only when the run next waits (`WaitFor`)."""
    STEER = "steer"
    """Add to the next observation at the next turn boundary; cancel nothing."""
    INTERRUPT = "interrupt"
    """Cancel the in-flight reply now; the message becomes the next observation."""


def _default_modes() -> dict[Priority, DeliveryMode]:
    return {
        Priority.LOW: DeliveryMode.QUEUE,
        Priority.NORMAL: DeliveryMode.STEER,
        Priority.HIGH: DeliveryMode.INTERRUPT,
    }


_PRIORITY_ORDER = (Priority.LOW, Priority.NORMAL, Priority.HIGH)


class DeliveryPolicy(ContractModel):
    modes: Mapping[Priority, DeliveryMode] = Field(default_factory=_default_modes)
    max_priority_by_sender: Mapping[str, Priority] = Field(default_factory=dict[str, Priority])
    """Caps per sender class, so an external system cannot interrupt when it should only queue."""

    def mode(self, priority: Priority, sender: str | None = None) -> DeliveryMode:
        cap = self.max_priority_by_sender.get(sender) if sender is not None else None
        if cap is not None and _PRIORITY_ORDER.index(priority) > _PRIORITY_ORDER.index(cap):
            priority = cap
        return self.modes[priority]
