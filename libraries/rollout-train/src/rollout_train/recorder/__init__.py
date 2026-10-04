"""The recorder: serves trainable model channels and records what trainers need
(docs/libraries/rollout-train/recorder.md).

- `renderers`: what a model family's token format must provide, and the pieces most are built from.
- `recorder`: `Recorder` serves a run's recorded bindings; a session exports `Segment`s (token sequences with the
  spans the policy sampled, their logprobs and the version of the weights: the served checkpoint's depth).
- `compat`: the recorder over HTTP (OpenAI's and Anthropic's APIs), for harnesses that bring their own loop.
"""

from rollout_train.recorder.recorder import RecordedEndpoint, Recorder, Segment, Span
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
    "JsonToolCalls",
    "RecordedEndpoint",
    "Recorder",
    "Renderer",
    "Segment",
    "Span",
    "ThinkingFormat",
    "ToolCallFormat",
    "XmlFunctionCalls",
]
