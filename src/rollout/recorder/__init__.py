"""The recorder (docs/core/recorder): serves trainable model channels and records what trainers need.

- `renderers`: model families' token formats (`qwen35`, `qwen3`): a function from a checkpoint to its renderer.
- `recorder`: `Recorder` serves a run's recorded bindings; a session exports `Epoch`s (token sequences with the
  spans the policy sampled, their logprobs and weights versions).
- `compat`: the recorder over HTTP, for harnesses that bring their own loop.
"""

from rollout.recorder.recorder import Epoch, RecordedEndpoint, Recorder, Span
from rollout.recorder.renderers import (
    ChatTemplateRenderer,
    JsonToolCalls,
    Renderer,
    ThinkingFormat,
    ToolCallFormat,
    XmlFunctionCalls,
    qwen3,
    qwen35,
)

__all__ = [
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
    "qwen3",
    "qwen35",
]
