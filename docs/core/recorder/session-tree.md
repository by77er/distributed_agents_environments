# Epochs: what a session exports

Status: **Working** (2026-10-02) · Code: `rollout.recorder.Recorder.export`

A session (one model slot of one run) is a series of samples. Training wants token sequences. The recorder joins
samples into sequences by one rule:

> A turn whose prompt begins with everything an earlier turn held (its prompt and what it sampled) continues that
> turn's sequence.

| What happened | What is exported |
|---|---|
| The context only grew (each prompt is the last one, the reply, and more) | One sequence for the whole conversation, with a sampled span per turn. It is trained in one pass: the prompt is run once, not once per turn. |
| The context was edited (a compaction, an observation replaced by a shorter form, a chat template that drops earlier thinking) | The edited turn begins a new sequence. |
| A prompt was repeated exactly (a client retried) | The later sample replaces the earlier one. |

Each span carries the weights version that sampled it, so a sequence that spans a weight update says so token by
token. Tokens the recorder forced (closing an over-budget thought) lie between spans: they are context, and are not
trained on.

```python
@dataclass(frozen=True)
class Epoch:
    tokens: list[int]
    spans: list[Span]          # Span(start, end, version): tokens[start:end] were sampled at that version
    logprobs: list[float]      # of the tokens inside the spans, in order
```

Whether a conversation is one sequence or many is decided by what its program sends, and by the model family's
template. The Minecraft swarm edits every turn (the last observation loses its map), so each of its turns is a
sequence; a harness that only appends gets one.
