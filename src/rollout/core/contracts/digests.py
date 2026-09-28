"""Digests over RFC 8785 canonical JSON (contracts evolution rule 6).

Digests are lowercase hexadecimal SHA-256. A contract model is canonicalized from its JSON form with fields whose
value is `None` omitted, so adding an optional field (default `None`) does not change the digest of existing
values. `null` inside JSON values (e.g. tool arguments) is kept.
"""

import hashlib
from collections.abc import Sequence

import rfc8785
from pydantic import BaseModel, JsonValue

from rollout.core.contracts.content import Message, ToolSpecification

EMPTY_DIGEST = hashlib.sha256(b"").hexdigest()
"""`d₀` of every context digest chain."""


def canonical_json(value: JsonValue | BaseModel) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    return rfc8785.dumps(value)


def digest(value: JsonValue | BaseModel) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def arguments_digest(arguments: JsonValue | BaseModel) -> str:
    """Sent with every `effect_id`; receivers reject a known `effect_id` whose arguments digest differs."""
    return digest(arguments)


def spec_hash(specification: ToolSpecification) -> str:
    """Covers only the model-visible fields of a tool specification."""
    return digest(specification.model_visible())


def message_digest(message: Message) -> str:
    """Covers what the model can see: `meta` is excluded."""
    return digest(message.model_dump(mode="json", exclude_none=True, exclude={"meta"}))


def context_digests(messages: Sequence[Message]) -> list[str]:
    """The digest chain `d₀ … dₙ` of a context: `dᵢ = sha256(dᵢ₋₁ ‖ sha256(JCS(itemᵢ)))` over raw digest bytes.

    `dₖ` identifies the prefix of length `k`, so a retained prefix is recognizable by its own chain value.
    """
    chain = [EMPTY_DIGEST]
    for message in messages:
        link = bytes.fromhex(chain[-1]) + bytes.fromhex(message_digest(message))
        chain.append(hashlib.sha256(link).hexdigest())
    return chain
