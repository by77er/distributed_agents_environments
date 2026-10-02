"""The recorder (docs/core/recorder): serves trainable model channels and records what trainers need.

- `renderers`: model families' token formats, pluggable (`renderer_for("qwen3.5", tokenizer)`).
- `recorder`: `Recorder` serves a run's recorded bindings; a session exports `Epoch`s (token sequences with the
  spans the policy sampled, their logprobs and weights versions).
- `compat`: the recorder over HTTP, for harnesses that bring their own loop.
"""

from rollout.recorder.recorder import Epoch, RecordedEndpoint, Recorder, Span
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
    "ChatTemplateRenderer",
    "Epoch",
    "JsonToolCalls",
    "RecordedEndpoint",
    "Recorder",
    "Renderer",
    "Span",
    "ThinkingFormat",
    "ToolCallFormat",
    "XmlFunctionCalls",
    "renderer_for",
]
