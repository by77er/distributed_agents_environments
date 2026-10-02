"""Renderers for the Qwen model families: implementations of `rollout_train.recorder.Renderer`.

A profile names one for a channel (`renderer = "rollout_qwen:qwen35"`); it is called with the channel's model.
"""

from typing import cast

from rollout_train.recorder.renderers import (
    ChatTemplateRenderer,
    JsonToolCalls,
    Renderer,
    ThinkingFormat,
    Tokenizer,
    XmlFunctionCalls,
)

__all__ = ["qwen3", "qwen35", "tokenizer_of"]


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
