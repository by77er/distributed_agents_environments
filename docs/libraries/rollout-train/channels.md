# Channels and engines

Code: `rollout_train.inference` · See [`Channel`](../../guide/reference.md#channel),
[`Engine`](../../guide/reference.md#engine), [`Limits`](../../guide/reference.md#limits), [recorder](recorder.md)

A **channel** is a trainable model being served, by name. Task code never sees one: a run's binding names a channel
for a model slot, and the gateway samples from it. Whoever trains publishes new weights to it; whoever deploys
decides which engines stand behind it. An **engine** is one replica serving the channel's model.

```python
channel = Channel("policy", engines=[engine_a, engine_b], renderer=renderer, limits=Limits(sequence=8000))
generation = await channel.generate(prompt, max_tokens=64, ..., adapter=channel.adapter, session="r_1/ada")
version = await channel.publish("kpqxwlmrtsnvoyzu", "/checkpoints/kpqxwlmrtsnvoyzu/weights", 3)  # by id, at depth 3
```

A deployment describes its channels in a profile: the model, its renderer, its limits, and one entry per engine
([deploying](../../guide/deploying.md)). A channel's engines may be on other machines: what it should serve is then
written down in the ledger, engine hosts load it into their servers, and runners ask for it by name
([what a channel should serve](#what-a-channel-should-serve), [engines elsewhere](#engines-elsewhere)).

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
`RemoteEngine` is a vLLM server on another machine, over its OpenAI-compatible API
([engines elsewhere](#engines-elsewhere)).
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
- `publish(…, full=True)` serves a full checkpoint ([full weights](checkpoints.md#full-weights-and-merges)): the
  channel pauses its engines, has each read the files into the model it holds (`Engine.load_weights`), drops the
  adapters it had loaded, and samples with no adapter from then on. `Channel.serving` names what is served, an
  adapter or full weights.
- Callers publish through a profile's platform (`Platform.publish`, which the loop is handed as `publish`), naming
  the channel.
- `Channel.loaded` names the adapters loaded: the one served, and the one before. `Channel.held` names the full
  checkpoint the engines hold, if they hold one.

## What a channel should serve

Code: `rollout_train.serving` · See [`Serving`](../../guide/reference.md#serving)

A run's channels are named within the run (`RUN/NAME`). What each should serve is written down in the ledger, so that
whatever serves it, on any machine, can follow: each time the training loop serves a checkpoint, it appends to the
run's `serving` table, under its fence and before it publishes to engines of its own, a
[`Serving`](../../guide/reference.md#serving) keyed `NAME/CHECKPOINT`:

| Field | Says |
|---|---|
| `channel` | the channel's name within the run |
| `checkpoint`, `depth`, `kind` | the checkpoint by id (none: the model the engines are started with; the run's first record, when it starts from the base model), the version its samples are stamped with, and `lora` or `full` |
| `files`, `layout` | the manifest of what the engines load (the checkpoint's weights, or what they were [resharded](checkpoints.md#resharding) into), and that layout |
| `over` | for an adapter over a full checkpoint, that checkpoint: the engines hold its weights first |
| `model`, `sequence` | the model the line began from, by name, and the longest turn the trainer can train on, which every runner applies |
| `served_by`, `max_lag` | for an eval its training run's schedule asks for, the training run's channel (`RUN/NAME`), whose engines serve its checkpoint, and 0: it plays that checkpoint and no other |

`wanted(ledger, run, channel)` is what a channel should serve now: its record of the greatest depth, as a channel
never goes back. `record_serving(ledger, run, serving, fence)` appends one; one written before changes nothing.

A `Follower` (`rollout_train.following`) keeps a process's channels serving what a run says: every two seconds it
reads `wanted`, and for a channel that serves something older reads the files from the blob store (hard links, where
the store is files on the same disk) under its directory and publishes them, named by the checkpoint's id; the files of
what is no longer loaded are deleted. An adapter over a full checkpoint has that checkpoint's weights published first.
A load that fails is tried again at the next look. With a `Presence`, it beats (kind `engines`, `follows`: the run)
with what each channel serves and how fast, each engine's address (for a server elsewhere) and why a load failed, if
one did; at once when something changed, every 15 seconds otherwise.

## Engines elsewhere

Code: `rollout_train.inference.remote` · See [`RemoteEngine`](../../guide/reference.md#remoteengine),
[`RemoteChannel`](../../guide/reference.md#remotechannel), [deploying](../../guide/deploying.md#engines-on-other-machines)

An engine elsewhere is a stock vLLM OpenAI-compatible server, or a router or proxy in front of several. The model a
request names is the checkpoint it samples from: the base model by its own name, a LoRA checkpoint by its id, loaded as
an adapter of that name.

`RemoteEngine(model, address=, connection=)` is an `Engine` over that API, serving `model` under its own name:

| Member | Request |
|---|---|
| `generate` | `POST /v1/completions`: the prompt's token ids, `model` the adapter (or `model`), `max_tokens`, `temperature`, `top_p`, `stop_token_ids`, `logprobs: 0`, `return_token_ids`, `skip_special_tokens: false`, `session_id`, `request_id`. The answer's `token_ids` are the tokens and its `token_logprobs` their logprobs, the stop token among them; an answer that names another model is refused (`Unserved`), and a model the server does not have is `NotLoaded` |
| `models()` | `GET /v1/models`: the models it has, by name, and the longest sequence it accepts (`max_model_len`) |
| `load_adapter`, `remove_adapter` | `POST /v1/load_lora_adapter`, `/v1/unload_lora_adapter`, by name, from a path on the server's machine (the server must allow it: `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`) |
| `load_weights` | refused: a vLLM server serves full weights only under the name it was started with |
| `sleep`, `wake` | `POST /sleep?level=2`, `/wake_up` (the server must allow it: `VLLM_SERVER_DEV_MODE=1`) |

`Connection` is how a client reaches servers: a bearer token read from an environment variable or a file (never written
down), a CA bundle, a client certificate. A server must give logprobs of the distribution it sampled from
(`--logprobs-mode processed_logprobs`), as `VllmEngine` does.

A `RemoteChannel` is one run's channel as a runner samples it: what the gateway samples from in place of a `Channel`
(`Sampler`: a name, a renderer, limits, a context limit, `weights(session)` and `generate`). Its servers are one URL
(a router, a proxy, a server) or a list. Every `every` seconds (2) it reads every checkpoint the run has said the
channel serves (`serving_of`) and asks each server which models it has; then:

- **Each turn asks for the checkpoint the run says**, by name: `weights(session)` gives it and its depth, the version
  the turn's tokens are stamped with. Where the session's server does not have it yet, the newest one before it that
  the server has, no more than `max_lag` checkpoints behind (`Serving.max_lag` where the run says: 0 for an eval). A
  server that answers that it does not have a model (`NotLoaded`) is asked for the one before, from the start of the
  turn, and for the newest again at the next look. A full checkpoint is never asked for.
- **A session's turns go to one server**, where there is a list: of those that answer and have a checkpoint close
  enough, the one a hash of the session and the server's address ranks first. Nothing is kept per session.
- **A turn waits** while no server has a checkpoint close enough, for five minutes at most (`NoReplica`); a server that
  does not answer is given no turn until it does.

`Routes` holds the channels of every run a runner plays whose engines are elsewhere, each made when first asked for; a
recorder with `routes` samples a binding's channel named within its run (`RUN/NAME`) from them ([recorder](recorder.md)).
`max_lag` is 1 unless a profile says otherwise: the checkpoint before, which a server serves while it loads the newest.
Every token is stamped with the depth of the checkpoint its answer names, and the trainer's importance weight corrects
for the difference.

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
