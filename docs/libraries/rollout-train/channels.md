# Channels and engines

Code: `rollout_train.inference` · See [`Channel`](../../guide/reference.md#channel),
[`Engine`](../../guide/reference.md#engine), [`Limits`](../../guide/reference.md#limits), [recorder](recorder.md)

A **channel** is a trainable model being served, by name. Task code never sees one: a run's binding names a channel
for a model slot, and the recorder samples from it. Whoever trains publishes new weights to it; whoever deploys
decides which engines stand behind it. An **engine** is one replica serving the channel's model.

```python
channel = Channel("policy", engines=[engine_a, engine_b], renderer=renderer, limits=Limits(sequence=8000))
generation = await channel.generate(prompt, max_tokens=64, ..., adapter=channel.adapter, session="r_1/ada")
version = await channel.publish("kpqxwlmrtsnvoyzu", "/checkpoints/kpqxwlmrtsnvoyzu/weights", 3)  # by id, at depth 3
```

A deployment describes its channels in a profile: the model, its renderer, its limits, and one entry per engine
([deploying](../../guide/deploying.md)).

## A channel

- **It has one or more engines**, and the renderer of its model family ([renderers](recorder.md#renderers)).
- **A session stays with one engine.** A session's requests go to the same replica every time, where the shared
  beginnings of its prompts are cached. Sessions spread over the replicas.
- **`adapter` and `version`** say which weights sample now. The adapter is `None` for the base model. The version
  is the number the weights were published as ([publishing weights](#publishing-weights)), and is recorded with
  every sampled token.
- **`context_limit`** is the longest turn the channel takes ([limits](#limits)). It is what programs are told, in a
  slot's capability contract.

## Engines

[`Engine`](../../guide/reference.md#engine) is a protocol: tokens in; tokens, logprobs and a finish reason out. An
engine runs in this process's care, or is a client of a server elsewhere.

- **Logprobs are of the distribution sampled from**, after temperature, and every sampled token has one.
- **A stop token is part of what was sampled.** A generation that ends on one has the finish reason `stop` and
  holds the token; one that ran out of room has `length`.
- **Adapters are named.** `load_adapter` registers one, and a request names the one it samples from.
- **`max_model_len`** is the most tokens, prompt and completion together, that the engine accepts.
- **`sleep` and `wake`** free the accelerator and take it back, for a trainer that shares it.
- **`processes`** are the processes the engine started on this machine. A deployment writes them down so that the
  next one can end them if this process is killed ([deploying](../../guide/deploying.md)).

`VllmEngine` is the engine this repository gives: [vLLM engine](../../implementations/rollout-vllm.md).
`ScriptedEngine` stands for one in tests ([training](training.md#trying-it-without-a-gpu)).

## Limits

[`Limits`](../../guide/reference.md#limits) say what a turn may take, in tokens: how much thinking, how much room
for the answer after it, and the longest turn.

- The deployment's hardware decides them. An open profile sets `Limits.sequence` to what the trainer can train on
  (`Budget.segment_tokens`, [the trainer](training.md#the-trainer)), and the channel's longest turn,
  `context_limit`, is the smaller of that and what its engines accept.
- Code above the channel receives the outcome, never the numbers: a context limit in a model's capability contract,
  and a refusal when a context is full ([a sample](recorder.md#a-sample)).

## Publishing weights

`Channel.publish(adapter, path, version)` loads the adapter on every engine of the channel, samples from it from
then on, and returns the version it is served as: the number given, which the training loop gives as the checkpoint's
depth ([checkpoints](checkpoints.md)), or one more than the last when none is given. The training loop names the
adapter by the checkpoint's id. Publishing what is being served changes
nothing.

- The adapter before stays loaded, so that a turn in progress (a thought, then its answer) finishes under the
  weights it began with. The one before that is dropped.
- Every sampled span records the version it was sampled at
  ([what a session exports](recorder.md#what-a-session-exports)).
- Callers publish through the recorder (`Recorder.publish`, which a profile's platform hands on as `publish`),
  naming the channel.

## Sharing an accelerator

A trainer that shares the engines' accelerator is wrapped in [`Colocated`](training.md#the-trainer), which uses four
members of a channel:

| Member | Does |
|---|---|
| `pause()` | holds new requests back, and waits for those in flight to finish |
| `sleep()` | puts every engine to sleep |
| `wake()` | wakes every engine |
| `resume()` | lets requests go on |

## Throughput

`Channel.take()` returns what passed through since the last call, and starts counting again: requests, tokens in
and out, tokens per second over the time the channel was generating, tokens per second per stream, and mean
concurrency. An open [profile](../../guide/deploying.md)'s runner takes it in each heartbeat, with the adapter and
version the channel serves ([heartbeats](rollouts.md#heartbeats)), and the [monitor](monitor.md) shows it from there.
