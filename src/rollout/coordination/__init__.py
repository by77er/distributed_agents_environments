"""Coordination between runs: a directory of participants, messages, and a shared board (docs/decisions/0025).

Tool calls write to a `CoordinationStore`; a `Relay` delivers its outbox through a runner. Identity, status and
delivery are injected, so the same pieces serve agent sessions, swarms and multi-agent RL tasks.
"""

from rollout.coordination.relay import Deliver, Relay
from rollout.coordination.store import CoordinationStore, Delivery, Participant, Post
from rollout.coordination.tools import BoardTools, Identify, SessionTools, Status, post, register, subscribe

__all__ = [
    "BoardTools",
    "CoordinationStore",
    "Deliver",
    "Delivery",
    "Identify",
    "Participant",
    "Post",
    "Relay",
    "SessionTools",
    "Status",
    "post",
    "register",
    "subscribe",
]
