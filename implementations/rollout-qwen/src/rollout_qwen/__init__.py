"""Renderers for the Qwen model families: implementations of `rollout_train.recorder.Renderer`.

A run's settings name one for a channel (`channels.NAME.renderer = "rollout_qwen:qwen35"`); it is called with the
channel's model. Each says the models it renders: `qwen35` Qwen3.5's, `qwen3` Qwen3's (not Qwen3.5's, nor
Qwen3-Coder's).
"""

from rollout_train.recorder.renderers import (
    ChatTemplateRenderer,
    JsonToolCalls,
    Renderer,
    ThinkingFormat,
    Tokenizer,
    XmlFunctionCalls,
    renders,
    tokenizer_of,
)

__all__ = ["qwen3", "qwen35"]

END_OF_TEXT = "<|endoftext|>"
"""A token the models end on besides `<|im_end|>`: engines stop there whatever the stop tokens."""


@renders(r"qwen3\.5")
def qwen35(model: str | Tokenizer) -> Renderer:
    """Qwen3.5: XML function calls, and thinking the prompt opens. `model` is a checkpoint's name, or its tokenizer.
    A turn ends with `<|im_end|>`, or with the end of text, where the model stops too."""
    return ChatTemplateRenderer(
        "qwen3.5",
        tokenizer_of(model) if isinstance(model, str) else model,
        XmlFunctionCalls(),
        ThinkingFormat(open="<think>", close="</think>", prompt_opens=True, forced_close="\n</think>\n\n"),
        end="<|im_end|>",
        stops=(END_OF_TEXT,),
    )


@renders(r"qwen3(?![.\d]|-coder)")
def qwen3(model: str | Tokenizer) -> Renderer:
    """Qwen3: JSON tool calls, and thinking the model opens. `model` is a checkpoint's name, or its tokenizer.
    A turn ends with `<|im_end|>`, or with the end of text, where the model stops too."""
    return ChatTemplateRenderer(
        "qwen3",
        tokenizer_of(model) if isinstance(model, str) else model,
        JsonToolCalls(),
        ThinkingFormat(open="<think>", close="</think>", prompt_opens=False, forced_close="\n</think>\n\n"),
        end="<|im_end|>",
        stops=(END_OF_TEXT,),
    )
