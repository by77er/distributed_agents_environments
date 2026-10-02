# Recorder

Code: `rollout_train.recorder` · See [`Recorder`](../../guide/reference.md#recorder),
[`Epoch`](../../guide/reference.md#epoch), [channels](channels.md), [episodes](episodes.md)

The recorder serves the model slots a run binds to a trainable channel, and keeps what training needs of every
sample: the tokens the policy was shown, the tokens it sampled, their logprobs, and the weights version that sampled
them. It is the [model endpoint](../rollout/contracts/model-endpoint.md) for recorded bindings.

```python
recorder = Recorder({"policy": channel})
runner = LocalRunner(recorder=recorder)        # a run's RecordedModel(channel="policy") binding is served by it
epochs = recorder.sessions(run_id)             # by model slot: what each exports
```

| Member | Does |
|---|---|
| `endpoint(binding)` | the model endpoint for a recorded binding. A binding that names no channel of the recorder raises `ValueError`. |
| `export(session_id)`, `sessions(run_id)` | what one session exports, and what every slot of a run exports |
| `publish(channel, adapter, path)` | serves new weights on a channel ([publishing weights](channels.md#publishing-weights)) |
| `forget(run_id)` | drops everything kept of a run: its turns, its recorded results and the keys of its harnesses |
| `base_url` | where the recorder is served over HTTP, for [harnesses that bring their own loop](harness-endpoint.md) |

A session is one model slot of one run. Its capability contract comes from its channel: the context limit is the
channel's longest turn, and the most output is the channel's thinking and answer room together
([limits](channels.md#limits)).

## A sample

The endpoint renders the request's context to tokens, samples from the channel, parses the result into canonical
content and records the turn.

- **A retried effect is not sampled twice.** A request whose `effect_id` was sampled returns the recorded result.
- **The binding decides how it samples**: the temperature and `top_p` of its `SamplingParameters`.
- **One turn, one set of weights.** The adapter and the version are read when the sample starts, and both of its
  phases use them.
- **No turn is longer than the channel's limit.** A long prompt leaves less room to think. A prompt that leaves no
  room to answer is refused with `ContextOverflow`, which callers compact on.
- **The reply** is the parsed message, a finish reason (`tool_use` when it calls tools, `stop` when it ended on a
  stop token, `length` otherwise) and usage in tokens.

### Thinking

Thinking has a budget (`Limits.thinking`). For a model family whose prompt opens the thinking block:

1. A first phase samples until the thinking closes or the budget runs out.
2. If the budget ran out, the close is forced. The forced tokens are not sampled, so they are never trained on.
3. A second phase samples the answer, unless the turn already ended.

- A request may cap its own output (`max_output_tokens`). The answer's room comes first and thinking gets what is
  left, down to none: the block is then closed before it starts.
- A model whose thinking was closed for it may go on thinking and close the block again itself. The last close
  ends the thinking: everything before it is kept as reasoning, and what follows is the answer.
- A turn whose thinking never closes is all reasoning.

For a family that opens the block itself, or has none, one phase samples with the thinking and answer room
together.

## What a session exports

Training wants token sequences; a session is a series of samples. The recorder joins samples into sequences
([`Epoch`](../../guide/reference.md#epoch)) by one rule:

> A turn whose prompt begins with everything an earlier turn held (its prompt and what it sampled) continues that
> turn's sequence.

| What happened | What is exported |
|---|---|
| The context only grew: each prompt is the last one, the reply, and more | One sequence for the whole conversation, with a sampled span per turn. It is trained in one pass: the prompt is run once, not once per turn. |
| The context was edited: a compaction, an observation replaced by a shorter form, a chat template that drops earlier thinking | The edited turn begins a new sequence. |
| A prompt was repeated exactly: a client retried | The later sample replaces the earlier one. |

- **Spans** ([`Span`](../../guide/reference.md#span)) mark the tokens the policy sampled. Each carries the weights
  version that sampled it, so a sequence that spans a weight update says so token by token.
- **Logprobs** are those of the tokens inside the spans, in order: the behaviour logprobs.
- **Forced tokens** (the close of an over-budget thought) lie between spans. They are context, and are not trained
  on.
- Whether a conversation is one sequence or many is decided by what its program sends and by the model family's
  template. A program that only appends gets one. A program that shortens each observation once it is no longer the
  current one gets a sequence per turn.

A job reads `sessions(run_id)` when a run ends and puts the sequences in the run's
[episode](episodes.md#how-an-episode-is-assembled).

## Renderers

A [`Renderer`](../../guide/reference.md#renderer) is a model family's token format. It turns canonical messages and
tool specifications into prompt tokens, says which tokens end a turn and how thinking is delimited, and parses
sampled tokens back into a canonical message: reasoning, text and tool calls. The recorder and trainers depend only
on this protocol.

- A model family is supported by a function that makes its renderer from a checkpoint's name. A profile names the
  function as `module:name` ([deploying](../../guide/deploying.md)).
- Most are a [`ChatTemplateRenderer`](../../guide/reference.md#chattemplaterenderer): the tokenizer's chat
  template, with the family's [`ToolCallFormat`](../../guide/reference.md#toolcallformat) (`XmlFunctionCalls`,
  `JsonToolCalls`) and [`ThinkingFormat`](../../guide/reference.md#thinkingformat).
- Tool call arguments that a family writes as text (`XmlFunctionCalls`) are converted to the types the tool's schema
  declares.
- `rollout_qwen` has the functions for the Qwen families: [Qwen renderers](../../implementations/rollout-qwen.md).
  `PlainRenderer` is a readable format for tests ([training](training.md#trying-it-without-a-gpu)).
