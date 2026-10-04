"""Identifiers with a normative structure (docs/libraries/rollout/contracts/identifiers.md).

All other identifiers are opaque strings. Only the structures built and parsed here may be relied on.
"""

import os
import time
from dataclasses import dataclass

_CROCKFORD_BASE32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_ulid() -> str:
    """A ULID: 48 bits of Unix time in milliseconds, then 80 random bits, in Crockford base 32 (26 characters).

    Minted by runners and services, never by task code (which has no ambient randomness under a durable runner).
    """
    value = (time.time_ns() // 1_000_000) << 80 | int.from_bytes(os.urandom(10))
    characters = [_CROCKFORD_BASE32[(value >> shift) & 0x1F] for shift in range(125, -1, -5)]
    return "".join(characters)


def new_run_id() -> str:
    """`r_{ulid}`."""
    return f"r_{new_ulid()}"


def new_message_id() -> str:
    """`m_{ulid}`: the id of a message sent without an idempotency key."""
    return f"m_{new_ulid()}"


@dataclass(frozen=True)
class EffectIdentity:
    """The parts of an `effect_id`: `{run_id}:{generation}:{ordinal}`."""

    run_id: str
    generation: int
    ordinal: int

    def __post_init__(self) -> None:
        if not self.run_id or ":" in self.run_id:
            raise ValueError(f"invalid run_id {self.run_id!r}")
        if self.generation < 0 or self.ordinal < 0:
            raise ValueError("generation and ordinal must be non-negative")

    def __str__(self) -> str:
        return f"{self.run_id}:{self.generation}:{self.ordinal}"

    @classmethod
    def parse(cls, effect_id: str) -> "EffectIdentity":
        run_id, generation, ordinal = effect_id.rsplit(":", 2)
        return cls(run_id, int(generation), int(ordinal))


def effect_id(run_id: str, generation: int, ordinal: int) -> str:
    """`{run_id}:{generation}:{ordinal}`: the same on every re-execution, so it is the universal idempotency key."""
    return str(EffectIdentity(run_id, generation, ordinal))


@dataclass(frozen=True)
class SessionIdentity:
    """The parts of a `session_id`: `{run_id}/{model_slot}`."""

    owner: str
    model_slot: str

    def __post_init__(self) -> None:
        if not self.owner or "/" in self.owner or not self.model_slot:
            raise ValueError(f"invalid session identity {self.owner!r}/{self.model_slot!r}")

    def __str__(self) -> str:
        return f"{self.owner}/{self.model_slot}"

    @classmethod
    def parse(cls, session_id: str) -> "SessionIdentity":
        owner, model_slot = session_id.split("/", 1)
        return cls(owner, model_slot)


def session_id(run_id: str, model_slot: str) -> str:
    """`{run_id}/{model_slot}`: one recorded session per model slot per run."""
    return str(SessionIdentity(run_id, model_slot))
