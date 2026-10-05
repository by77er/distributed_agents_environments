# Recording

For people who train or add a model family: what training keeps of every sample, the thinking budget, segments, and
renderers.

**Read first:** [Channels and engines](channels.md). **Next:** [The gateway](gateway.md).

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
channel's longest turn, and the most output is the channel's thinking and answer budgets together, or the context
limit where either is none ([limits](channels.md#limits)).

## A sample

The gateway renders the request's context to tokens, samples from the channel, parses the result into canonical
content and records the turn.

- **A retried effect is not sampled twice.** A request whose `effect_id` was recorded is answered with the recorded
  result.
- **The binding decides how it samples**: the temperature and `top_p` of its `SamplingParameters`.
- **One turn, one set of weights.** The [checkpoint](checkpoints.md) is chosen when the sample starts (where the
  channel's engines are elsewhere, the checkpoint the run says, or the newest close enough that the session's server
  has), and both of its phases sample it. Where the checkpoint is not served there after all, the server is gone, or an
  answer names another checkpoint, the turn is sampled again from the start, three times at most: no token is stamped
  with a version that did not sample it.
- **No turn is longer than the channel's limit.** A long prompt leaves less room to think. A prompt that leaves no
  room to answer is refused with `ContextOverflow`, which callers compact on: room for the answer budget, or with no
  answer budget, for `MINIMUM_ANSWER` (256) tokens, and for a forced close where thinking has a bound.
- **The reply** is the parsed message, a finish reason (`tool_use` when it calls tools, `stop` when it ended on a
  stop token, `length` otherwise) and usage in tokens.
- **Links.** A request may say how it follows from earlier ones of its session (`SampleRequest.links`: a type and the
  earlier request's effect id). `Memory` says so of its compactions ([links between requests](gateway.md#links-between-requests)).

### Thinking

A turn's room is what the context leaves after the prompt (the channel's longest turn), within the request's own cap
(`max_output_tokens`, from a harness's `max_tokens` or `max_output_tokens`) where it gives one. The budgets are the
channel's (`Limits.thinking`, and `Limits.answer` for the answer), or the binding's where it gives one
(`SamplingParameters.thinking_tokens` and `answer_tokens`: an eval entry's, say), which its key carries. Each is
optional, and none is the default:

| Budgets | How the turn samples |
|---|---|
| neither | One generation with all the room. Nothing is forced: a turn still thinking when the room runs out ends there, all reasoning (`length`). |
| only `answer` | Thinking takes the room the answer's leaves, then is closed by force if still open; the answer has its budget. |
| only `thinking` | Thinking takes its budget (less, where the room is short: the least answer, `MINIMUM_ANSWER`, comes first), then is closed by force if still open; the answer has whatever room is left. |
| both | Thinking takes its budget, then is closed by force if still open; the answer has its budget. |

Where a budget is set:

1. A first phase samples until the thinking closes or its bound is reached. For a model family that opens the block
   itself (Qwen3), rather than its prompt (Qwen3.5), the phase also has room for the opening.
2. If the bound was reached while thinking, the close is forced. The forced tokens are not sampled, so they are never
   trained on. A model that opens no block and answers at once is not closed.
3. A second phase samples the answer, unless the turn already ended.

- A request's cap counts every token of the reply: the answer's room comes first, then the forced close, and thinking
  gets what is left, down to none (the block is then closed before it starts).
- A model whose thinking was closed for it may go on thinking and close the block again itself. The last close
  ends the thinking: everything before it is kept as reasoning, and what follows is the answer. With no budget, the
  reply parses the same way: reasoning up to the model's own close, the answer after it.
- A turn whose thinking never closes is all reasoning.

For a family with no thinking block, one phase samples: with the thinking and answer budgets together where both are
set, else with all the room.

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
- **What it was sampled with** (`sampled_with`) is what every one of its turns was sampled with, of `token_exact`,
  `sampled_logprobs` and `honours_sampling` (`TOKEN_LEVEL`). A segment without `token_exact` and `sampled_logprobs`
  (`BEHAVIOUR`) has no importance weight: the [algorithm](training.md#the-algorithm) does not train on its group with a
  policy gradient that corrects for where tokens were sampled (a preference loss and a likelihood do), and
  a [dataset](datasets.md) of such turns is `supervised`.
- **Whether it is trained on** (`trained`): false for a segment of a slot that is not trained (a judge's, a fixed
  opponent's). Such a segment is kept in its episode, for the monitor and for what it sampled, and the
  [algorithm](training.md#the-algorithm) and [datasets](datasets.md) never train on it.
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

- A model family is supported by a function that makes its renderer from a checkpoint's name. A run's settings name the
  function as `module:name` (`channels.NAME.renderer`, [run settings](../../guide/cluster.md#run-settings)).
- Most are a [`ChatTemplateRenderer`](../../guide/reference.md#chattemplaterenderer): the tokenizer's chat
  template, with the family's [`ToolCallFormat`](../../guide/reference.md#toolcallformat) (`XmlFunctionCalls`,
  `JsonToolCalls`) and [`ThinkingFormat`](../../guide/reference.md#thinkingformat).
- Tool call arguments that a family writes as text (`XmlFunctionCalls`) are converted to the types the tool's schema
  declares.
- `rollout_qwen` has the functions for the Qwen families ([Qwen renderers](../../implementations/rollout-qwen.md)),
  `rollout_gemma` the one for Gemma ([Gemma renderers](../../implementations/rollout-gemma.md)).
  `PlainRenderer` is a readable format for tests ([training](training.md#trying-it-without-a-gpu)).
