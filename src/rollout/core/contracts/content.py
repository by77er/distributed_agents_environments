"""Canonical content: the model-agnostic form of what a model reads or writes (docs/contracts/canonical-content.md)."""

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, JsonValue, field_validator, model_validator

from rollout.core.contracts.base import ContractModel, FrozenSequence


class BlobReference(ContractModel):
    """Content kept in object storage (any block larger than 64 KiB)."""

    uri: str
    sha256: str
    size: int
    media_type: str


class Text(ContractModel):
    type: Literal["text"] = "text"
    text: str


class Media(ContractModel):
    """An image, audio clip or document."""

    type: Literal["media"] = "media"
    media_type: str
    source: BlobReference


class ToolCall(ContractModel):
    type: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    arguments: Mapping[str, JsonValue]


class ReasoningScope(StrEnum):
    PORTABLE = "portable"
    """Plain text that any renderer may render or drop."""
    POLICY = "policy"
    """Opaque, valid only for its producer (e.g. encrypted provider reasoning)."""


class Reasoning(ContractModel):
    type: Literal["reasoning"] = "reasoning"
    scope: ReasoningScope
    text: str | None = None
    """`PORTABLE` only."""
    producer: str | None = None
    """`POLICY` only: the renderer or provider that can consume `opaque`."""
    opaque: BlobReference | None = None
    """`POLICY` only."""

    @model_validator(mode="after")
    def _fields_match_scope(self) -> Self:
        if self.scope is ReasoningScope.PORTABLE and (self.text is None or self.producer or self.opaque):
            raise ValueError("portable reasoning has text only")
        if self.scope is ReasoningScope.POLICY and (self.producer is None or self.opaque is None or self.text):
            raise ValueError("policy-scoped reasoning has a producer and opaque content only")
        return self


class Provenance(ContractModel):
    untrusted: bool = False
    """True for anything originating in a guest or a third-party server."""
    binding_kind: str | None = None
    """`task`, `environment`, `mcp`, `http`, `agent` or `human`."""


type ResultBlock = Annotated[Text | Media, Field(discriminator="type")]


class ToolResult(ContractModel):
    """What a tool produces. Platform failures are `tool.failed` events, not results."""

    content: FrozenSequence[ResultBlock] = ()
    structured: JsonValue = None
    """Optional; must validate against the tool's `output_schema` when it has one."""
    is_error: bool = False
    """A tool-level error the model should see and reason about (non-zero exit, file not found)."""
    truncated: bool = False
    """Content was cut to `max_result_bytes`; the full output is in `overflow`."""
    overflow: BlobReference | None = None
    provenance: Provenance = Provenance()


class ToolResultBlock(ContractModel):
    """A tool's result as it appears in a TOOL message, answering the `ToolCall` with the same `call_id`."""

    type: Literal["tool_result"] = "tool_result"
    call_id: str
    result: ToolResult


type Block = Annotated[Text | Media | ToolCall | ToolResultBlock | Reasoning, Field(discriminator="type")]


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(ContractModel):
    role: Role
    content: FrozenSequence[Block] = ()
    name: str | None = None
    """Optional speaker name (multi-agent)."""
    meta: Mapping[str, str] = Field(default_factory=dict[str, str])
    """Not model-visible; never rendered and not covered by the context digest."""

    @model_validator(mode="after")
    def _blocks_match_role(self) -> Self:
        for block in self.content:
            if self.role is Role.TOOL and not isinstance(block, ToolResultBlock):
                raise ValueError("TOOL messages contain only tool_result blocks")
            if self.role is not Role.TOOL and isinstance(block, ToolResultBlock):
                raise ValueError("tool_result blocks appear only in TOOL messages")
            if self.role is not Role.ASSISTANT and isinstance(block, ToolCall):
                raise ValueError("tool_call blocks appear only in ASSISTANT messages")
        return self

    @property
    def text(self) -> str:
        """The concatenated text blocks."""
        return "".join(block.text for block in self.content if isinstance(block, Text))

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [block for block in self.content if isinstance(block, ToolCall)]

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls(role=Role.USER, content=[Text(text=text)])

    @classmethod
    def assistant(cls, text: str) -> "Message":
        return cls(role=Role.ASSISTANT, content=[Text(text=text)])

    @classmethod
    def system(cls, text: str) -> "Message":
        return cls(role=Role.SYSTEM, content=[Text(text=text)])


class RetryClass(StrEnum):
    """What a durable runner may do with a tool call after a crash (docs/architecture/delivery-semantics.md)."""

    PURE = "pure"
    IDEMPOTENT = "idempotent"
    SIDE_EFFECTING = "side_effecting"
    UNKNOWN = "unknown"


class ToolAnnotations(ContractModel):
    """MCP-compatible hints; not model-visible, and ignored from untrusted servers."""

    title: str | None = None
    read_only_hint: bool | None = None
    destructive_hint: bool | None = None
    idempotent_hint: bool | None = None
    open_world_hint: bool | None = None


_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class ToolSpecification(ContractModel):
    """What the model sees about a tool, plus extensions that are never model-visible."""

    # Model-visible: covered by the spec hash.
    name: str
    description: str = ""
    input_schema: Mapping[str, JsonValue] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    """JSON Schema 2020-12 with `type: object`."""
    output_schema: Mapping[str, JsonValue] | None = None

    # Advisory, not model-visible.
    annotations: ToolAnnotations | None = None

    # Platform extensions, not model-visible.
    retry_class: RetryClass = RetryClass.UNKNOWN
    timeout_ms: int | None = None
    max_result_bytes: int | None = None

    @field_validator("name")
    @classmethod
    def _valid_name(cls, name: str) -> str:
        if not _TOOL_NAME.match(name):
            raise ValueError(f"tool name {name!r} does not match {_TOOL_NAME.pattern}")
        return name

    @field_validator("input_schema")
    @classmethod
    def _object_schema(cls, schema: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        if schema.get("type") != "object":
            raise ValueError("input_schema must have type: object")
        return schema

    def model_visible(self) -> dict[str, JsonValue]:
        """The fields a model sees and the spec hash covers; absent fields are omitted."""
        return self.model_dump(
            mode="json", exclude_none=True, include={"name", "description", "input_schema", "output_schema"}
        )
