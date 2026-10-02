# Recorder

Status: **Working** (2026-10-02) · Code: `rollout_train.recorder` · See [inference](../../inference/README.md), [episodes](../trajectories/README.md)

Serves the model slots a run binds to a trainable channel, and keeps what training needs of every sample: the tokens
the policy was shown, the tokens it sampled, their logprobs, and the weights version that sampled them.

```python
recorder = Recorder({"policy": channel})
runner = LocalRunner(recorder=recorder)        # a run's RecordedModel(channel="policy") binding is served by it
epochs = recorder.sessions(run_id)             # by model slot: what each exports
```

| Piece | What it does |
|---|---|
| `Recorder(channels, base_url=None)` | An endpoint for each recorded binding (`endpoint`), what a run's slots export (`sessions`, `export`), publishing weights to a channel (`publish`), forgetting a run (`forget`). |
| `renderers` | A model family's token format: `render` (messages and tools to a prompt), `parse` (sampled tokens to a canonical message), its stop tokens and how it delimits thinking. `renderer_for("qwen3.5", tokenizer)`; register another in `RENDERERS`. |
| [`compat`](session-api.md) | The recorder over HTTP, for harnesses that bring their own loop. |

## A sample

The endpoint renders the request's context to tokens, samples from the channel, parses the result and records the
turn. A retried effect returns the recorded result; it is not sampled twice.

**Thinking has a budget.** A first phase samples until thinking closes or the budget runs out; then the close is
forced (not sampled, so never trained on) and a second phase samples the answer. A request may cap its own output
(`max_output_tokens`): the answer's room comes first and thinking gets what is left, down to none.

**No turn is longer than the channel's limit** (`Limits.sequence`, the trainer's): a long prompt leaves less room to
think, and a prompt that leaves no room to answer is refused with `ContextOverflow`, which callers compact on.

## What a session exports

A list of [epochs](session-tree.md): token sequences with the spans the policy sampled. An append-only conversation
is one sequence, however many turns it has; a turn whose context was edited begins another.
