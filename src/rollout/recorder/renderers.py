"""Renderers: how a model family's tokens encode canonical content, pluggable per family.

A `Renderer` turns canonical messages and tool specifications into prompt tokens, says how thinking is delimited,
and parses sampled tokens back into a canonical message (reasoning, text, tool calls). The recorder and trainers
depend only on this protocol; a model family is supported by a function that makes its renderer from a checkpoint's
name (`qwen35`, `qwen3`), usually a `ChatTemplateRenderer` (the tokenizer's chat template) with that family's
`ToolCallFormat` and `ThinkingFormat`. A deployment's profile names the function.
"""

import json
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from pydantic import JsonValue

from rollout.core.contracts import (
    Message,
    Reasoning,
    ReasoningScope,
    Role,
    Text,
    ToolCall,
    ToolResultBlock,
    ToolSpecification,
)


@dataclass(frozen=True)
class ThinkingFormat:
    """How a family delimits thinking. Its generation prompt may already open the block."""

    open: str
    close: str
    prompt_opens: bool
    """Whether the generation prompt ends inside an opened thinking block (the model only closes it)."""
    forced_close: str
    """Text appended to end thinking that ran out of budget (masked from training)."""


class ToolCallFormat(Protocol):
    """How a family writes tool calls in its output."""

    def parse(self, text: str, tools: Sequence[ToolSpecification]) -> tuple[str, list[ToolCall]]:
        """Split text into what precedes the calls and the calls; arguments are converted to their schema types."""
        ...


class Tokenizer(Protocol):
    """The part of a Hugging Face tokenizer renderers use."""

    def apply_chat_template(self, conversation: Any, **options: Any) -> Any: ...
    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...
    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = ...) -> str: ...
    def convert_tokens_to_ids(self, tokens: str) -> Any: ...


class Renderer(Protocol):
    name: str
    thinking: ThinkingFormat | None

    def render(self, messages: Sequence[Message], tools: Sequence[ToolSpecification]) -> list[int]:
        """The prompt: every message, then the generation prompt for the assistant's next turn."""
        ...

    def encode(self, text: str) -> list[int]: ...

    def stop_token_ids(self) -> list[int]:
        """Tokens that end an assistant turn."""
        ...

    def thinking_end_token_ids(self) -> list[int]:
        """Tokens that end thinking (to stop a thinking phase on), or none if it is not a single token."""
        ...

    def parse(self, completion: Sequence[int], tools: Sequence[ToolSpecification]) -> Message:
        """A sampled turn as a canonical assistant message."""
        ...


# Tool call formats


class XmlFunctionCalls:
    """`<tool_call><function=name><parameter=key>value</parameter></function></tool_call>` (Qwen3.5, Qwen3-Coder)."""

    CALL = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
    PARAMETER = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL)

    def parse(self, text: str, tools: Sequence[ToolSpecification]) -> tuple[str, list[ToolCall]]:
        calls: list[ToolCall] = []
        for match in self.CALL.finditer(text):
            name, body = match.group(1), match.group(2)
            schema = _properties(tools, name)
            arguments = {key: _typed(value, schema.get(key)) for key, value in self.PARAMETER.findall(body)}
            calls.append(ToolCall(call_id=_call_id(), name=name, arguments=arguments))
        start = text.find("<tool_call>")
        return (text if start < 0 else text[:start]).strip(), calls


class JsonToolCalls:
    """`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` (Qwen3, Hermes)."""

    CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

    def parse(self, text: str, tools: Sequence[ToolSpecification]) -> tuple[str, list[ToolCall]]:
        calls: list[ToolCall] = []
        for match in self.CALL.finditer(text):
            try:
                payload: Any = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            call = cast(dict[str, Any], payload)
            name, arguments = call.get("name"), call.get("arguments")
            if isinstance(name, str):
                typed_arguments = cast(dict[str, JsonValue], arguments) if isinstance(arguments, dict) else {}
                calls.append(ToolCall(call_id=_call_id(), name=name, arguments=typed_arguments))
        start = text.find("<tool_call>")
        return (text if start < 0 else text[:start]).strip(), calls


# The chat-template renderer


class ChatTemplateRenderer:
    """Renders with the tokenizer's chat template; parses with a family's tool-call and thinking formats."""

    def __init__(
        self, name: str, tokenizer: Tokenizer, tool_calls: ToolCallFormat, thinking: ThinkingFormat | None, end: str
    ) -> None:
        self.name = name
        self.tokenizer = tokenizer
        self.tool_calls = tool_calls
        self.thinking = thinking
        self._end = end

    def render(self, messages: Sequence[Message], tools: Sequence[ToolSpecification]) -> list[int]:
        conversation = [_template_message(message) for message in messages]
        rendered = self.tokenizer.apply_chat_template(
            [entry for entries in conversation for entry in entries],
            tools=[_template_tool(tool) for tool in tools] or None,
            add_generation_prompt=True,
            tokenize=False,
        )
        return self.encode(str(rendered))

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def stop_token_ids(self) -> list[int]:
        return [int(self.tokenizer.convert_tokens_to_ids(self._end))]

    def thinking_end_token_ids(self) -> list[int]:
        if self.thinking is None:
            return []
        ids = self.encode(self.thinking.close)
        return ids if len(ids) == 1 else []

    def parse(self, completion: Sequence[int], tools: Sequence[ToolSpecification]) -> Message:
        text = self.tokenizer.decode(list(completion), skip_special_tokens=False)
        text = text.split(self._end)[0]
        blocks: list[Reasoning | Text | ToolCall] = []
        if self.thinking is not None and self.thinking.close in text:
            # The last close ends the thinking: a model whose thinking was closed for it may go on thinking, and
            # close it again itself.
            reasoning, text = text.rsplit(self.thinking.close, 1)
            reasoning = reasoning.replace(self.thinking.open, "").replace(self.thinking.close, "").strip()
            if reasoning:
                blocks.append(Reasoning(scope=ReasoningScope.PORTABLE, text=reasoning))
        elif self.thinking is not None and self.thinking.prompt_opens:  # thinking never closed: all of it is thought
            reasoning = text.replace(self.thinking.open, "").strip()
            return Message(
                role=Role.ASSISTANT,
                content=[Reasoning(scope=ReasoningScope.PORTABLE, text=reasoning)] if reasoning else [],
            )
        before, calls = self.tool_calls.parse(text, tools)
        if before:
            blocks.append(Text(text=before))
        blocks.extend(calls)
        return Message(role=Role.ASSISTANT, content=blocks)


# Model families


def tokenizer_of(model: str) -> Tokenizer:
    """The tokenizer of a checkpoint, by its name or path."""
    from transformers import AutoTokenizer

    return cast(Tokenizer, AutoTokenizer.from_pretrained(model))  # pyright: ignore[reportUnknownMemberType]


def qwen35(model: str | Tokenizer) -> Renderer:
    """Qwen3.5: XML function calls, and thinking the prompt opens. `model` is a checkpoint's name, or its tokenizer."""
    return ChatTemplateRenderer(
        "qwen3.5",
        tokenizer_of(model) if isinstance(model, str) else model,
        XmlFunctionCalls(),
        ThinkingFormat(open="<think>", close="</think>", prompt_opens=True, forced_close="\n</think>\n\n"),
        end="<|im_end|>",
    )


def qwen3(model: str | Tokenizer) -> Renderer:
    """Qwen3: JSON tool calls, and thinking the model opens. `model` is a checkpoint's name, or its tokenizer."""
    return ChatTemplateRenderer(
        "qwen3",
        tokenizer_of(model) if isinstance(model, str) else model,
        JsonToolCalls(),
        ThinkingFormat(open="<think>", close="</think>", prompt_opens=False, forced_close="\n</think>\n\n"),
        end="<|im_end|>",
    )


# Canonical content → chat template dictionaries


def _template_message(message: Message) -> list[dict[str, Any]]:
    if message.role is Role.TOOL:
        return [
            {"role": "tool", "content": "".join(part.text for part in block.result.content if isinstance(part, Text))}
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
    entry: dict[str, Any] = {"role": message.role.value, "content": message.text}
    if message.role is Role.ASSISTANT:
        reasoning = "\n".join(block.text for block in message.content if isinstance(block, Reasoning) and block.text)
        if reasoning:
            entry["reasoning_content"] = reasoning
        if message.tool_calls:
            entry["tool_calls"] = [
                {"type": "function", "function": {"name": call.name, "arguments": dict(call.arguments)}}
                for call in message.tool_calls
            ]
    return [entry]


def _template_tool(tool: ToolSpecification) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description, "parameters": dict(tool.input_schema)},
    }


def _properties(tools: Sequence[ToolSpecification], name: str) -> Mapping[str, Any]:
    for tool in tools:
        if tool.name == name:
            properties = tool.input_schema.get("properties")
            return properties if isinstance(properties, dict) else {}
    return {}


def _typed(value: str, schema: Any) -> JsonValue:
    """A parameter's text as its schema's type (XML parameters are text)."""
    kind = cast(dict[str, Any], schema).get("type") if isinstance(schema, dict) else None
    text = value.strip()
    try:
        if kind == "integer":
            return int(float(text))
        if kind == "number":
            return float(text)
        if kind == "boolean":
            return text.lower() in ("true", "1", "yes")
        if kind in ("object", "array"):
            parsed: JsonValue = json.loads(text)
            return parsed
    except ValueError:
        return text
    return text


def _call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"
