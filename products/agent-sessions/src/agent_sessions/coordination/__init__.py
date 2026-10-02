"""Coordination between runs: a directory of participants, messages, and a shared board
(docs/products/agent-sessions.md).

Tool calls write to a `CoordinationStore`; a `Relay` delivers its outbox through a runner. Identity, status and
delivery are injected, so the same pieces serve agent sessions, swarms and multi-agent RL tasks.
"""

from agent_sessions.coordination.relay import Deliver, Relay
from agent_sessions.coordination.store import CoordinationStore, Delivery, Participant, Post
from agent_sessions.coordination.tools import BoardTools, Identify, SessionTools, Status, post, register

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
]
