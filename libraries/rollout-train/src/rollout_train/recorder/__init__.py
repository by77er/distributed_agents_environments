"""What recording a trainable channel takes (docs/libraries/rollout-train/recorder.md); the gateway
(`rollout_train.gateway`) is what records, with these:

- `renderers`: what a model family's token format must provide, the pieces most are built from, and the models a
  family's renderer says it renders.
- `sampling`: `sample_turn`, one turn sampled with the thinking budget.
- `segments`: what a session's turns export: `Segment`s (token sequences with the spans the policy sampled, their
  logprobs, the version of the weights (the served checkpoint's depth) and what their turns were sampled with),
  joined by prefix-continuation.
- `compat`: OpenAI's and Anthropic's APIs, read and answered, for harnesses that bring their own loop.
"""

from rollout_train.recorder.renderers import (
    ChatTemplateRenderer,
    JsonToolCalls,
    Renderer,
    ThinkingFormat,
    ToolCallFormat,
    XmlFunctionCalls,
    rendered,
    renders,
    tokenizer_of,
)
from rollout_train.recorder.sampling import sample_turn
from rollout_train.recorder.segments import BEHAVIOUR, TOKEN_LEVEL, Segment, Span, TeacherScores, segments_of

__all__ = [
    "BEHAVIOUR",
    "TOKEN_LEVEL",
    "ChatTemplateRenderer",
    "JsonToolCalls",
    "Renderer",
    "Segment",
    "Span",
    "TeacherScores",
    "ThinkingFormat",
    "ToolCallFormat",
    "XmlFunctionCalls",
    "rendered",
    "renders",
    "sample_turn",
    "segments_of",
    "tokenizer_of",
]
