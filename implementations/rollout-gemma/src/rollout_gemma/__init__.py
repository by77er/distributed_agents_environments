"""Renderers for the Gemma model families: implementations of `rollout_train.recorder.Renderer`.

A profile names one for a channel (`renderer = "rollout_gemma:gemma4"`); it is called with the channel's model.

Gemma 4's turns are `<|turn>role ... <turn|>`. Its thinking is a channel, `<|channel>thought ... <channel|>`, which
the chat template turns on with `enable_thinking`. It calls tools as `<|tool_call>call:NAME{ARGUMENTS}<tool_call|>`,
and a tool's response comes back inside the model's own turn, after `<|tool_response>`: a sampled turn ends at
`<turn|>`, or at `<|tool_response>` (the model waits there for its tools' results), or at `<eos>`.
"""

import uuid
from collections.abc import Sequence
from typing import Any, cast

from pydantic import JsonValue

from rollout.contracts import ToolCall, ToolSpecification
from rollout_train.recorder.renderers import ChatTemplateRenderer, Renderer, ThinkingFormat, Tokenizer

__all__ = ["GemmaFunctionCalls", "arguments", "gemma4", "tokenizer_of"]

CALL_START, CALL_END = "<|tool_call>", "<tool_call|>"
QUOTE = '<|"|>'
"""How Gemma quotes a string in a call's arguments (and in a tool's declaration)."""


def tokenizer_of(model: str) -> Tokenizer:
    """The tokenizer of a checkpoint, by its name or path."""
    from transformers import AutoTokenizer

    return cast(Tokenizer, AutoTokenizer.from_pretrained(model))  # pyright: ignore[reportUnknownMemberType]


def gemma4(model: str | Tokenizer) -> Renderer:
    """Gemma 4, thinking: the template is asked for thinking, and the generation prompt opens the thought channel
    (as Gemma's template itself does after a tool's response), so that a thinking budget can close it. `model` is a
    checkpoint's name, or its tokenizer."""
    return ChatTemplateRenderer(
        "gemma4",
        tokenizer_of(model) if isinstance(model, str) else model,
        GemmaFunctionCalls(),
        ThinkingFormat(open="<|channel>thought\n", close="<channel|>", prompt_opens=True, forced_close="\n<channel|>"),
        end="<turn|>",
        stops=("<|tool_response>", "<eos>"),
        options={"enable_thinking": True},
        opens="<|channel>thought\n",
    )


class GemmaFunctionCalls:
    """`<|tool_call>call:name{key:<|"|>text<|"|>,count:3,flag:true,nested:{...},items:[...]}<tool_call|>`."""

    def parse(self, text: str, tools: Sequence[ToolSpecification]) -> tuple[str, list[ToolCall]]:
        calls: list[ToolCall] = []
        at = text.find(CALL_START)
        before = text if at < 0 else text[:at]
        while at >= 0:
            end = text.find(CALL_END, at)
            if end < 0:
                break  # (a call cut off: not a call)
            body = text[at + len(CALL_START) : end].strip()
            if body.startswith("call:") and "{" in body:
                name, _, rest = body[len("call:") :].partition("{")
                parsed = arguments(rest[:-1] if rest.endswith("}") else rest)
                calls.append(
                    ToolCall(
                        call_id=_call_id(), name=name.strip(), arguments=_schema_typed(parsed, tools, name.strip())
                    )
                )
            at = text.find(CALL_START, end)
        return before.strip(), calls


def arguments(text: str) -> dict[str, JsonValue]:
    """A call's arguments (what is between its braces) as values: strings quoted with `<|"|>`, numbers, `true`,
    `false`, `null`, objects in braces and lists in brackets; keys are bare (or quoted)."""
    parsed, _ = _object(text, 0, closing=None)
    return parsed


def _object(text: str, at: int, closing: str | None) -> tuple[dict[str, JsonValue], int]:
    found: dict[str, JsonValue] = {}
    while at < len(text):
        at = _skip(text, at)
        if at >= len(text) or (closing is not None and text[at] == closing):
            return found, at + 1
        colon = text.find(":", at)
        if colon < 0:
            break
        key = text[at:colon].strip()
        if key.startswith(QUOTE) and key.endswith(QUOTE):
            key = key[len(QUOTE) : -len(QUOTE)]
        found[key], at = _value(text, colon + 1)
    return found, at


def _value(text: str, at: int) -> tuple[JsonValue, int]:
    at = _skip(text, at, commas=False)
    if text.startswith(QUOTE, at):
        end = text.find(QUOTE, at + len(QUOTE))
        end = len(text) if end < 0 else end
        return text[at + len(QUOTE) : end], end + len(QUOTE)
    if at < len(text) and text[at] == "{":
        return _object(text, at + 1, closing="}")
    if at < len(text) and text[at] == "[":
        items: list[JsonValue] = []
        at += 1
        while at < len(text):
            at = _skip(text, at)
            if at >= len(text) or text[at] == "]":
                return items, at + 1
            item, at = _value(text, at)
            items.append(item)
        return items, at
    end = at
    while end < len(text) and text[end] not in ",}]":
        end += 1
    return _bare(text[at:end].strip()), end


def _bare(word: str) -> JsonValue:
    """An unquoted value: a number, true, false or null, or else the text itself."""
    if word in ("true", "false"):
        return word == "true"
    if word == "null":
        return None
    try:
        return int(word)
    except ValueError:
        pass
    try:
        return float(word)
    except ValueError:
        return word


def _skip(text: str, at: int, *, commas: bool = True) -> int:
    while at < len(text) and (text[at].isspace() or (commas and text[at] == ",")):
        at += 1
    return at


def _schema_typed(parsed: dict[str, JsonValue], tools: Sequence[ToolSpecification], name: str) -> dict[str, JsonValue]:
    """Arguments as their schema's types where the model quoted what the schema says is a number or a flag."""
    schema: Any = next((tool.input_schema.get("properties") for tool in tools if tool.name == name), None) or {}
    typed: dict[str, JsonValue] = {}
    for key, value in parsed.items():
        kind = schema.get(key, {}).get("type") if isinstance(schema.get(key), dict) else None
        if isinstance(value, str) and kind in ("integer", "number", "boolean"):
            converted = _bare(value.strip())
            typed[key] = converted if not isinstance(converted, str) else value
        else:
            typed[key] = value
    return typed


def _call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"
