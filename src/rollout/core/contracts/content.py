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
    """Plain text."""

    type: Literal["text"] = "text"
    text: str


class Media(ContractModel):
    """An image, audio clip or document."""

    type: Literal["media"] = "media"
    media_type: str
    source: BlobReference


class ToolCall(ContractModel):
    """A request by the model to call a tool. Appears only in ASSISTANT messages."""

    type: Literal["tool_call"] = "tool_call"
    call_id: str
    """Unique within the context; the `ToolResultBlock` that answers the call repeats it."""
    name: str
    arguments: Mapping[str, JsonValue]
    """A JSON object."""


class ReasoningScope(StrEnum):
    """Who can consume a reasoning block."""

    PORTABLE = "portable"
    """Plain text that any renderer may render or drop."""


class Reasoning(ContractModel):
    """The model's reasoning."""

    type: Literal["reasoning"] = "reasoning"
    scope: ReasoningScope
    text: str


type ResultBlock = Annotated[Text | Media, Field(discriminator="type")]


class ToolResult(ContractModel):
    """What a tool produces. Platform failures are exceptions, not results."""

    content: FrozenSequence[ResultBlock] = ()
    structured: JsonValue = None
    """Optional: the result as JSON, for code that reads it."""
    is_error: bool = False
    """A tool-level error the model should see and reason about (non-zero exit, file not found)."""
    truncated: bool = False
    """`content` is only part of what the tool produced."""


class ToolResultBlock(ContractModel):
    """A tool's result as it appears in a TOOL message, answering the `ToolCall` with the same `call_id`."""

    type: Literal["tool_result"] = "tool_result"
    call_id: str
    result: ToolResult


type Block = Annotated[Text | Media | ToolCall | ToolResultBlock | Reasoning, Field(discriminator="type")]


class Role(StrEnum):
    """Who a message is from. Observations contain only USER and TOOL messages."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(ContractModel):
    """A message in canonical form: a role and a sequence of content blocks.

    TOOL messages contain only `ToolResultBlock`s, and `ToolCall`s appear only in ASSISTANT messages.
    """

    role: Role
    content: FrozenSequence[Block] = ()
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
        """The tool calls, in order."""
        return [block for block in self.content if isinstance(block, ToolCall)]

    @classmethod
    def user(cls, text: str) -> "Message":
        """A USER message with one text block."""
        return cls(role=Role.USER, content=[Text(text=text)])

    @classmethod
    def assistant(cls, text: str) -> "Message":
        """An ASSISTANT message with one text block."""
        return cls(role=Role.ASSISTANT, content=[Text(text=text)])

    @classmethod
    def system(cls, text: str) -> "Message":
        """A SYSTEM message with one text block."""
        return cls(role=Role.SYSTEM, content=[Text(text=text)])


class RetryClass(StrEnum):
    """What a durable runner may do with a tool call after a crash (docs/architecture/overview.md)."""

    PURE = "pure"
    IDEMPOTENT = "idempotent"
    SIDE_EFFECTING = "side_effecting"
    UNKNOWN = "unknown"


_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class ToolSpecification(ContractModel):
    """What the model sees about a tool, plus an extension that is never model-visible."""

    # Model-visible: covered by the spec hash.
    name: str
    description: str = ""
    input_schema: Mapping[str, JsonValue] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    """JSON Schema 2020-12 with `type: object`."""

    # A platform extension, not model-visible.
    retry_class: RetryClass = RetryClass.UNKNOWN

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
        """The fields a model sees and the spec hash covers."""
        return self.model_dump(mode="json", exclude_none=True, include={"name", "description", "input_schema"})
