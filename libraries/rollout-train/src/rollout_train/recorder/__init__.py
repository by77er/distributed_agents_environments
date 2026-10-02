"""The recorder (docs/core/recorder): serves trainable model channels and records what trainers need.

- `renderers`: what a model family's token format must provide, and the pieces most are built from.
- `recorder`: `Recorder` serves a run's recorded bindings; a session exports `Epoch`s (token sequences with the
  spans the policy sampled, their logprobs and weights versions).
- `compat`: the recorder over HTTP, for harnesses that bring their own loop.
"""

from rollout_train.recorder.recorder import Epoch, RecordedEndpoint, Recorder, Span
from rollout_train.recorder.renderers import (
    ChatTemplateRenderer,
    JsonToolCalls,
    Renderer,
    ThinkingFormat,
    ToolCallFormat,
    XmlFunctionCalls,
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
]
