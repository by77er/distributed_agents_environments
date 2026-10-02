"""Addressing, envelopes, priorities and delivery modes (docs/core/harness/conversations.md)."""

from collections.abc import Mapping
from enum import StrEnum
from typing import Literal

from pydantic import Field, JsonValue

from rollout.contracts import Block, ContractModel, FrozenSequence


class Address(ContractModel):
    """Where a message or output goes."""

    kind: Literal["conversation", "run", "external"]
    value: str
    """`{deployment}/{key}`, a `run_id`, or a connector target."""


class ConversationKey(ContractModel):
    """Identifies a conversation: a deployment and a caller-chosen key. One live run per conversation."""

    deployment: str
    """e.g. `acme/support-bot`."""
    key: str
    """Caller-chosen, e.g. `slack:T1/C2/171.2` or `user:42`."""
    origin: Address | None = None
    """Where replies go by default (e.g. a connector target)."""

    @property
    def address(self) -> str:
        """`{deployment}/{key}`: the value of the conversation's `Address`."""
        return f"{self.deployment}/{self.key}"

    @classmethod
    def parse(cls, address: str, *, origin: Address | None = None) -> "ConversationKey":
        """The conversation an address value names. A deployment is `{namespace}/{name}`; the rest is the key, which
        may itself contain `/`."""
        namespace, name, key = [*address.split("/", 2), "", ""][:3]
        if not namespace or not name or not key:
            raise ValueError(f"{address!r} does not name a conversation")
        return cls(deployment=f"{namespace}/{name}", key=key, origin=origin)


class Envelope(ContractModel):
    """A message delivered to a run."""

    kind: str = "message"
    """`message`, or an application-defined kind that `WaitFor` can select."""
    content: FrozenSequence[Block] = ()
    data: JsonValue = None
    """A structured payload."""
    reply_to: Address | None = None
    message_id: str = ""
    """Set by the runner: the caller's idempotency key (a sending run's `effect_id`, say), or a new `m_{ulid}`."""
    sender: str | None = None
    """Set by the runner; never trusted from the payload."""


class Priority(StrEnum):
    """A sender's priority; the run's `DeliveryPolicy` maps it to a delivery mode."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class DeliveryMode(StrEnum):
    """How a message reaches a run that is busy."""

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
    """Maps priorities to delivery modes. Default: LOW → QUEUE, NORMAL → STEER, HIGH → INTERRUPT."""

    modes: Mapping[Priority, DeliveryMode] = Field(default_factory=_default_modes)
    max_priority_by_sender: Mapping[str, Priority] = Field(default_factory=dict[str, Priority])
    """Caps per sender class, so an external system cannot interrupt when it should only queue."""

    def mode(self, priority: Priority, sender: str | None = None) -> DeliveryMode:
        """The delivery mode for a message, after capping the priority by sender."""
        cap = self.max_priority_by_sender.get(sender) if sender is not None else None
        if cap is not None and _PRIORITY_ORDER.index(priority) > _PRIORITY_ORDER.index(cap):
            priority = cap
        return self.modes[priority]
