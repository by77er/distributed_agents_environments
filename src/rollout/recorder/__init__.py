"""The recorder (docs/core/recorder): serves trainable model channels and records tokens, logprobs and adapters.

- `renderers`: model families' token formats, pluggable (`renderer_for("qwen3.5", tokenizer)`).
- `recorder`: `Recorder`, `Channel` and the `Engine` protocol; `RecordedTurn` is what trainers read.
"""

from rollout.recorder.recorder import Channel, Engine, Generation, RecordedEndpoint, RecordedTurn, Recorder
from rollout.recorder.renderers import (
    RENDERERS,
    ChatTemplateRenderer,
    JsonToolCalls,
    Renderer,
    ThinkingFormat,
    ToolCallFormat,
    XmlFunctionCalls,
    renderer_for,
)

__all__ = [
    "RENDERERS",
    "Channel",
    "ChatTemplateRenderer",
    "Engine",
    "Generation",
    "JsonToolCalls",
    "RecordedEndpoint",
    "RecordedTurn",
    "Recorder",
    "Renderer",
    "ThinkingFormat",
    "ToolCallFormat",
    "XmlFunctionCalls",
    "renderer_for",
]
