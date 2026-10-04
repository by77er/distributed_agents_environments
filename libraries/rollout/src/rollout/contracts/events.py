"""Run events: the typed record of what happened in a run (docs/libraries/rollout/contracts/run-events.md).

The catalog is closed: a new event type requires a change to the contract.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import JsonValue

from rollout.contracts.base import ContractModel

RUN_EVENT_SCHEMA_VERSION = 1


class RunEventType(StrEnum):
    """The closed catalog of run events."""

    # Lifecycle
    RUN_CREATED = "run.created"
    RUN_SUSPENDED = "run.suspended"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCEL_REQUESTED = "run.cancel_requested"
    RUN_CANCELLED = "run.cancelled"
    # Episode
    OBSERVATION_RECORDED = "observation.recorded"
    REWARD_ASSIGNED = "reward.assigned"
    TRAINING_EXCLUDED = "training.excluded"
    OUTPUT_EMITTED = "output.emitted"
    # Effects
    EFFECT_REQUESTED = "effect.requested"
    EFFECT_COMPLETED = "effect.completed"
    # Conversations and messages
    MESSAGE_RECEIVED = "message.received"
    TURN_INTERRUPTED = "turn.interrupted"
    # Tools and sandboxes
    TOOLS_RESOLVED = "tools.resolved"
    SANDBOXES_ACQUIRED = "sandboxes.acquired"


TERMINAL_EVENT_TYPES = frozenset({RunEventType.RUN_COMPLETED, RunEventType.RUN_FAILED, RunEventType.RUN_CANCELLED})


class RunFailureClass(StrEnum):
    """Why a run failed (the `class` of a `run.failed` event)."""

    TASK_ERROR = "task_error"
    INVALID_OBSERVATION = "invalid_observation"


class RunEvent(ContractModel):
    """One entry in a run's event stream."""

    run_id: str
    seq: int
    """Position in this run's event stream, gapless from 0; `run.created` is 0."""
    type: RunEventType
    schema_version: int = RUN_EVENT_SCHEMA_VERSION
    recorded_at: datetime
    """Exposed to code as `run.now()` for inputs."""
    payload: JsonValue = None
    """Type-specific; see docs/libraries/rollout/contracts/run-events.md."""
