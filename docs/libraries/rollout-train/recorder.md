# Recording

Code: `rollout_train.recorder` · See [`sample_turn`](../../guide/reference.md#sample_turn),
[`Segment`](../../guide/reference.md#segment), [`segments_of`](../../guide/reference.md#segments_of),
[the gateway](gateway.md), [channels](channels.md), [episodes](episodes.md)

A run's model slots bound to a trainable channel are recorded: of every sample, training keeps the tokens the policy
was shown, the tokens it sampled, their logprobs, and the weights version that sampled them. The
[gateway](gateway.md) is what records, whether it runs in the runner's own process or as replicas of its own; a
runner's recorded slots sample through it ([a runner served by the gateway](gateway.md#a-runner-served-by-the-gateway)).
This page is what recording does with a sample, the same wherever the gateway runs:

- `sample_turn` (`rollout_train.recorder.sampling`): one turn sampled with the [thinking budget](#thinking);
- `segments_of` (`rollout_train.recorder.segments`): [what a session exports](#what-a-session-exports);
- `Renderer` (`rollout_train.recorder.renderers`): a model family's [token format](#renderers);
- `rollout_train.recorder.compat`: OpenAI's and Anthropic's APIs, read and answered, for
  [harnesses that bring their own loop](harness-endpoint.md).

A session is one model slot of one run. Its capability contract comes from its channel: the context limit is the
channel's longest turn, and the most output is the channel's thinking and answer room together
([limits](channels.md#limits)).

## A sample

The gateway renders the request's context to tokens, samples from the channel, parses the result into canonical
content and records the turn.

- **A retried effect is not sampled twice.** A request whose `effect_id` was recorded is answered with the recorded
  result.
- **The binding decides how it samples**: the temperature and `top_p` of its `SamplingParameters`.
- **One turn, one set of weights.** The checkpoint is chosen when the sample starts (where the channel's engines are
  elsewhere, the checkpoint the run says, or the newest close enough that the session's server has), and both of its
  phases sample it. Where the checkpoint is not served there after all, the server is gone, or an answer names another
  checkpoint, the turn is sampled again from the start, three times at most: no token is stamped with a version that
  did not sample it.
- **No turn is longer than the channel's limit.** A long prompt leaves less room to think. A prompt that leaves no
  room to answer is refused with `ContextOverflow`, which callers compact on.
- **The reply** is the parsed message, a finish reason (`tool_use` when it calls tools, `stop` when it ended on a
  stop token, `length` otherwise) and usage in tokens.
- **Links.** A request may say how it follows from earlier ones of its session (`SampleRequest.links`: a type and the
  earlier request's effect id). `Memory` says so of its compactions ([links between requests](gateway.md#links-between-requests)).

### Thinking

Thinking has a budget (`Limits.thinking`):

1. A first phase samples until the thinking closes or the budget runs out. For a model family that opens the block
   itself (Qwen3), rather than its prompt (Qwen3.5), the phase also has room for the opening.
2. If the budget ran out while thinking, the close is forced. The forced tokens are not sampled, so they are never
   trained on. A model that opens no block and answers at once is not closed.
3. A second phase samples the answer, unless the turn already ended.

- A request may cap its own output (`max_output_tokens`). The answer's room comes first and thinking gets what is
  left, down to none: the block is then closed before it starts.
- A model whose thinking was closed for it may go on thinking and close the block again itself. The last close
  ends the thinking: everything before it is kept as reasoning, and what follows is the answer.
- A turn whose thinking never closes is all reasoning.

For a family with no thinking block, one phase samples with the thinking and answer room together.

## What a session exports

Training wants contexts that only grew; a session is a series of samples. `segments_of` joins a session's turns into
segments ([`Segment`](../../guide/reference.md#segment)), each the tokens of a context that only grew, by one rule:

> A turn whose prompt begins with everything an earlier turn held (its prompt and what it sampled) continues that
> turn's segment.

| What happened | What is exported |
|---|---|
| The context only grew: each prompt is the last one, the reply, and more | One segment for the whole conversation, with a sampled span per turn. It is trained in one pass: the prompt is run once, not once per turn. |
| The context was edited: a compaction, an observation replaced by a shorter form, a chat template that drops earlier thinking | The edited turn begins a new segment. |
| A prompt was repeated exactly: a client retried | The later sample replaces the earlier one. |

- **Spans** ([`Span`](../../guide/reference.md#span)) mark the tokens the policy sampled. Each carries the weights
  version that sampled it, so a segment that spans a weight update says so token by token.
- **Logprobs** are those of the tokens inside the spans, in order: the behaviour logprobs.
- **Forced tokens** (the close of an over-budget thought) lie between spans. They are context, and are not trained
  on.
- Whether a conversation is one segment or many is decided by what its program sends and by the model family's
  template. A program that only appends gets one. A program that shortens each observation once it is no longer the
  current one gets a segment per turn.

A [runner](rollouts.md#a-runner) reads a run's segments from the gateway's turn store when the run ends
(`GatewayEndpoints.sessions`) and puts them in the run's [episode](episodes.md#how-an-episode-is-assembled).

## Renderers

A [`Renderer`](../../guide/reference.md#renderer) is a model family's token format. It turns canonical messages and
tool specifications into prompt tokens, says which tokens end a turn and how thinking is delimited, and parses
sampled tokens back into a canonical message: reasoning, text and tool calls. The gateway and trainers depend only on
this protocol.

- A model family is supported by a function that makes its renderer from a checkpoint's name. A profile names the
  function as `module:name` ([deploying](../../guide/deploying.md)).
- Most are a [`ChatTemplateRenderer`](../../guide/reference.md#chattemplaterenderer): the tokenizer's chat
  template, with the family's [`ToolCallFormat`](../../guide/reference.md#toolcallformat) (`XmlFunctionCalls`,
  `JsonToolCalls`) and [`ThinkingFormat`](../../guide/reference.md#thinkingformat).
- Tool call arguments that a family writes as text (`XmlFunctionCalls`) are converted to the types the tool's schema
  declares.
- `rollout_qwen` has the functions for the Qwen families ([Qwen renderers](../../implementations/rollout-qwen.md)),
  `rollout_gemma` the one for Gemma ([Gemma renderers](../../implementations/rollout-gemma.md)).
  `PlainRenderer` is a readable format for tests ([training](training.md#trying-it-without-a-gpu)).
