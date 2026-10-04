"""What recording a trainable channel takes (docs/libraries/rollout-train/recorder.md); the gateway
(`rollout_train.gateway`) is what records, with these:

- `renderers`: what a model family's token format must provide, and the pieces most are built from.
- `sampling`: `sample_turn`, one turn sampled with the thinking budget.
- `segments`: what a session's turns export: `Segment`s (token sequences with the spans the policy sampled, their
  logprobs and the version of the weights: the served checkpoint's depth), joined by prefix-continuation.
- `compat`: OpenAI's and Anthropic's APIs, read and answered, for harnesses that bring their own loop.
"""

from rollout_train.recorder.renderers import (
    ChatTemplateRenderer,
    JsonToolCalls,
    Renderer,
    ThinkingFormat,
    ToolCallFormat,
    XmlFunctionCalls,
)
from rollout_train.recorder.sampling import sample_turn
from rollout_train.recorder.segments import Segment, Span, segments_of

__all__ = [
    "ChatTemplateRenderer",
    "JsonToolCalls",
    "Renderer",
    "Segment",
    "Span",
    "ThinkingFormat",
    "ToolCallFormat",
    "XmlFunctionCalls",
    "sample_turn",
    "segments_of",
]
