# Channels and engines

For people who train and deploy: how a channel serves the model a run trains, its engines and limits, what it should
serve, engine hosts, and engines on other machines.

**Read first:** [Checkpoints, runs and the ledger](checkpoints.md). **Next:** [Record turns for training](recorder.md).

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

A run's settings describe its channels: each one's provider, model, renderer and limits
([run settings](../../guide/cluster.md#run-settings)); the cluster config says what each provider is. What a channel
should serve is written down in the ledger, engine hosts and followers load it into their engines, and runners ask for
it by name
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
- **The most likely tokens, when asked.** `generate(…, top=K)` gives, at each sampled token, the K most likely tokens
  there and their logprobs (`Generation.top_tokens`, `top_logprobs`), of the same distribution.
- **Scoring.** `score(tokens, start=, end=, top=, adapter=)` samples nothing: it gives the logprob of each token at
  positions `start` to `end` of `tokens` given the tokens before it, and the `top` most likely tokens at each, under
  the adapter named (`Scores`: `start`, `logprobs`, `top_tokens`, `top_logprobs`). Scores are of the model's own
  distribution, at temperature 1. It is how a teacher scores a student's tokens.
- **`max_model_len`** is the most tokens, prompt and completion together, that the engine accepts.
- **`sleep` and `wake`** free the accelerator and take it back, for a trainer that shares it.
- **`processes`** are the processes the engine started on this machine.

`VllmEngine` is the engine this repository gives: [vLLM engine](../../implementations/rollout-vllm.md).
`RemoteEngine` is a vLLM server on another machine, over its OpenAI-compatible API
([engines elsewhere](#engines-elsewhere)).
`ScriptedEngine` stands for one in tests ([training](training.md#trying-it-without-a-gpu)). `TinkerEngine` neither
scores nor gives the most likely tokens: Tinker's SDK takes both, and they wait for a live test.

A channel scores as it samples: `Channel.score` on the session's engine, once a load in progress has ended, counted
in its throughput with the tokens scored as tokens in and none out; `Channel.scored(…, name=)` scores what is served
under a name, as a server elsewhere is asked.

## Limits

[`Limits`](../../guide/reference.md#limits) say what a turn may take, in tokens: how much thinking, how much room
for the answer after it, and the longest turn. Each is optional, and none is set unless a run's settings or the code that
makes the channel sets it.

| Limit | Set | None (the default) |
|---|---|---|
| `thinking` (a channel's `thinking_tokens`) | thinking past this many tokens is closed by force, and the answer follows | thinking runs until the model closes it, or until the room left after the answer's is spent |
| `answer` (`answer_tokens`) | room for the answer after the thinking | the answer has whatever room the turn has left |
| `sequence` | the longest turn, prompt and completion | the engines' own longest |

With neither budget, a turn is one generation that may fill all the context its prompt leaves: nothing is cut off or
closed by force, and a reply that reaches the end of the context ends there (`length`). How each combination samples
is in [thinking](recorder.md#thinking).

- The deployment's hardware decides them. A run's driver sets the trained channel's `Limits.sequence` to what the
  trainer can train on (`trainer.segment_tokens`, [the trainer](training.md#the-trainer)), and the channel's longest turn,
  `context_limit`, is the smaller of that and what its engines accept.
- Code above the channel receives the outcome, never the numbers: a context limit in a model's capability contract,
  and a refusal when a context is full ([a sample](recorder.md#a-sample)). The contract's most output is the thinking
  and answer budgets together, or the context limit where either is none.

## Publishing weights

`Channel.publish(adapter, path, version)` loads the adapter on every engine of the channel, samples from it from
then on, and returns the version it is served as: the number given, which the training loop gives as the checkpoint's
depth ([checkpoints](checkpoints.md)), or one more than the last when none is given. The training loop names the
adapter by the checkpoint's id. Publishing what is being served changes
nothing.

- The adapters before stay loaded, so that a turn in progress (a thought, then its answer) finishes under the
  weights it began with: `Channel.keep` in all with the one served (`max_lag + 1`, two for the default `max_lag` of
  one; a follower sets it to its run's, `keeping(n)`). Older ones are dropped.
- Every sampled span records the version it was sampled at
  ([what a session exports](recorder.md#what-a-session-exports)).
- `publish(…, full=True)` serves a full checkpoint ([full weights](checkpoints.md#full-weights-and-merges)): the
  channel pauses its engines, has each read the files into the model it holds (`Engine.load_weights`), drops the
  adapters it had loaded, and samples with no adapter from then on. `Channel.serving` names what is served, an
  adapter or full weights.
- Callers publish through what the loop is handed as `publish` (a run's driver's `Run.publish`), naming the channel:
  a channel whose engines are in the process loads it, and engine hosts elsewhere follow the run's serving record.
- `Channel.loaded` names the adapters loaded: the one served, and those before it. `Channel.held` names the full
  checkpoint the engines hold, if they hold one; `adapters()` lists both with the version each was published as.
- `Channel.sample(prompt, …, name=)` samples what is served under a name, as a server elsewhere is asked: an
  adapter, the full checkpoint held, or the model's own (`Channel.model`). A name not served here is `NotLoaded`,
  decided once a load in progress ends, so a turn caught by a full checkpoint's load is sampled again.

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
| `files`, `layout` | the manifest of what the engines load (the checkpoint's weights, or what a [bridge](checkpoints.md#bridges) made of them), and that bridge's name |
| `over` | for an adapter over a full checkpoint, that checkpoint: the engines hold its weights first |
| `model`, `sequence` | the model the line began from, by name, and the longest turn the trainer can train on, which every runner applies |
| `max_lag` | how many checkpoints behind this a sample may be: 0 for an eval, which plays its checkpoint and no other |

`wanted(ledger, run, channel)` is what a channel should serve now: the record of the greatest depth it serves, as a
channel never goes back; `serving_of(ledger, run, channel)` is every record it serves. `record_serving(ledger, run,
serving, fence)` appends one; one written before changes nothing.

Whose records a channel serves is its mode, which the run's start records in its run settings
(`channels.NAME.mode`; `source_of`):

| Mode | Serves |
|---|---|
| the trained channel, or none said | its own records |
| `follows`, with `follows` and `lag` | the followed channel's records but its newest `lag` (deepest last), under its own name: the followed channel's checkpoint `lag` records back, and the base model until the followed channel has more than `lag` |
| `fixed`, with `checkpoint` (an id) | that checkpoint, at its depth, with `max_lag` 0 |
| `fixed`, with none | its own records, which nothing writes: the base model |

So whatever follows a channel (a `Follower`, the engines of a run or a pool, Tinker's sampler) and whatever samples it
(a `RemoteChannel`) serves a following or a pinned channel with nothing more than its run and its name.

A `Follower` (`rollout_train.following`) keeps runs' channels serving what each run says: every two seconds it
reads `wanted` for each, and for a channel that serves something older reads the files from the blob store (hard
links, where the store is files on the same disk) under its directory and publishes them, named by the checkpoint's
id; the files of what is no longer loaded are deleted. An adapter over a full checkpoint has that checkpoint's weights
published first. A load that fails is tried again at the next look.

- **What it serves can change while it runs.** Given a run and its channels, it serves those. Given `bindings` (the
  `(run, channel)` pairs it serves now, asked at each look) and `opened` (a channel for a pair new to it), it serves
  whatever they say: a pair it no longer serves has its adapters removed. Each pair is a channel of its own over
  engines that may be shared, so several runs' adapters sit side by side on one engine.
- **Each channel keeps its run's window**: `max_lag + 1` adapters (from the newest `Serving.max_lag`; two when the
  record does not say), following changes to it.
- **Full weights load replica by replica.** A follower that is replica `i` of `n` loads a channel's new full
  checkpoint only once every replica before it that still beats holds it, so the others serve the checkpoint before
  meanwhile.
- **It holds its view to what a server says it has**, where an engine can say (`models()`: a vLLM server's
  `/v1/models`, which lists each adapter with its `parent`): an adapter the server lost (it started again) is loaded
  again; one no channel knows of (loaded before the follower started again) is removed, unless a channel should
  serve it now. `RemoteEngine.load_adapter` takes a name the server holds already as loaded.

With a `Presence`, it beats (kind `engines`; `runs`, and `follows` where it serves one run; its `replica`) with what
each channel serves and how fast, why a load failed, if one did, and each engine's address (for a server elsewhere)
with every adapter it holds, each with its run, channel, checkpoint and depth, so that a turn can go to an engine that
holds its checkpoint; at once when something changed, every 15 seconds otherwise.

## Engine hosts

Code: `rollout_train.inference.hosts` · See [`EngineHost`](../../guide/reference.md#enginehost),
[`HostServer`](../../guide/reference.md#hostserver)

An `EngineHost` is one replica's engines as a Ray actor (`started(name, spec, ledger_at, blobs_at, bound=…)`): its
engine (`module:name`, made with the model and its options), and a follower that keeps it serving what the runs bound
to it should. It shares nothing with whoever trains but the ledger and the blob store: no placement group, no Ray
cluster. So the same actor is a run's own replica, started by the run's job, or a replica of a long-lived pool
(`detached`) serving every run bound to it; `bind(run, channel)` and `unbind` change what it serves while it runs.

| Member | Does |
|---|---|
| `models()` | what it holds, by name, as vLLM's `/v1/models` lists it: the model (unless full weights replaced it), each adapter (its `parent` the model), each full checkpoint, with the depth each was published as |
| `generate(prompt, …, adapter=)` | samples the checkpoint named (None: the model); `NotLoaded` where it does not hold it once a load in progress ends |
| `score(tokens, start=, end=, top=, adapter=)` | scores tokens with the checkpoint named, refused as `generate` is |
| `pause`, `resume`, `sleep`, `wake` | hold its requests back, and free and take back the GPU, for a trainer that shares it |
| `bind`, `unbind`, `bound`, `follow`, `served`, `about` | what it serves, a look now, what its beats say |

`host_spec(cluster, provider, model, settings=)` says what a host of a `vllm` provider's model asks for: the kind's
engine, the model's options with the provider's `max_logprobs` (the top-k it declares), a replica's `gpus` (half of them where the run's trainer shares the provider's card,
`colocate_with`), and the custom resources `[placement.engines]` names. Ray places it by those: on Kubernetes, a GPU
share no node has free makes KubeRay's autoscaler start a GPU worker. Ray starts a host again when it dies
(`max_restarts=-1`); its engines start afresh and its follower loads what the serving records say again.

`HostServer(handle, address)` is a `CheckpointServer` over a host's handle, for a `RemoteChannel` in the same Ray
cluster: a host that does not answer is `Unreachable`. `HostPausable(handle)` is what
[`Colocated`](#share-a-gpu-between-engines-and-the-trainer) pauses and puts to sleep.

## Engines elsewhere

Code: `rollout_train.inference.remote` · See [`RemoteEngine`](../../guide/reference.md#remoteengine),
[`RemoteChannel`](../../guide/reference.md#remotechannel), [deploying](../../guide/deploying.md#engines-on-other-machines)

An engine elsewhere is a stock vLLM OpenAI-compatible server, or a router or proxy in front of several. The model a
request names is the checkpoint it samples from: the base model by its own name, a LoRA (low-rank adaptation) checkpoint
by its id, loaded as an adapter of that name.

`RemoteEngine(model, address=, connection=)` is an `Engine` over that API, serving `model` under its own name:

| Member | Request |
|---|---|
| `generate` | `POST /v1/completions`: the prompt's token ids, `model` the adapter (or `model`), `max_tokens`, `temperature`, `top_p`, `stop_token_ids`, `logprobs` (`top`, 0 unless asked), `return_token_ids`, `skip_special_tokens: false`, `session_id`, `request_id`, and `return_tokens_as_token_ids` with a `top`. The answer's `token_ids` are the tokens and its `token_logprobs` their logprobs, the stop token among them, and its `top_logprobs` the most likely tokens at each, by id (`token_id:ID`); an answer that names another model is refused (`Unserved`), and a model the server does not have is `NotLoaded` |
| `score` | `POST /v1/completions`: the token ids up to `end`, `model` as for `generate`, `max_tokens: 1`, `temperature: 0`, `prompt_logprobs` (`top`), `skip_special_tokens: false`, `session_id`, `request_id`. The answer's `prompt_logprobs` hold each position's tokens by id with their logprobs (none at the first); the one token generated is dropped. Refused as `generate` is. The server caps `top` at its `--max-logprobs` |
| `models()` | `GET /v1/models`: the models it has, by name, and the longest sequence it accepts (`max_model_len`) |
| `load_adapter`, `remove_adapter` | `POST /v1/load_lora_adapter`, `/v1/unload_lora_adapter`, by name, from a path on the server's machine (the server must allow it: `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`) |
| `load_weights` | refused: a vLLM server serves full weights only under the name it was started with |
| `sleep`, `wake` | `POST /sleep?level=2`, `/wake_up` (the server must allow it: `VLLM_SERVER_DEV_MODE=1`) |

`Connection` is how a client reaches servers: a bearer token read from an environment variable or a file (never written
down), a CA bundle, a client certificate. A server must give logprobs of the distribution it sampled from
(`--logprobs-mode processed_logprobs`), as `VllmEngine` does.

A `RemoteChannel` is one run's channel as the gateway samples it, in place of a `Channel`
(`Sampler`: a name, a renderer, limits, a context limit, `weights(session)`, `generate` and `score`). Its servers are
one URL (a router, a proxy, a server) or a list, of URLs or of any `CheckpointServer` (`models()`, `generate` and
`score` by checkpoint name: `RemoteEngine`, or an engine host's `HostServer`). It scores as it samples: on the
session's server, under the checkpoint named, with the same refusals. Every `every` seconds (2) it reads every checkpoint the run has said the
channel serves (`serving_of`) and asks each server which models it has; then:

- **Each turn asks for the checkpoint the run says**, by name: `weights(session)` gives it and its depth, the version
  the turn's tokens are stamped with. Where the session's server does not have it yet, the newest one before it that
  the server has, no more than `max_lag` checkpoints behind (`Serving.max_lag` where the run says: 0 for an eval). A
  server that answers that it does not have a model (`NotLoaded`) is asked for the one before, from the start of the
  turn, and for the newest again at the next look. A full checkpoint is asked for where a server lists it (an engine
  host does; a vLLM server never does, serving full weights only under its model's name).
- **A session's turns go to one server**, where there is a list: of those that answer and have a checkpoint close
  enough, the one a hash of the session and the server's address ranks first. Nothing is kept per session.
- **A turn waits** while no server has a checkpoint close enough, for five minutes at most (`NoReplica`); a server that
  does not answer is given no turn until it does.

`Routes` holds the channels whose engines are elsewhere of every run the gateway samples, each made when first asked
for; the gateway (`Gateway.routes`) samples a binding's channel, named within its run (`RUN/NAME`) or not, from them
([which checkpoint](gateway.md#which-checkpoint)). `max_lag` is 1 (`MAX_LAG`) unless the run says
otherwise: the checkpoint before, which a server serves while it loads the newest.
Every token is stamped with the depth of the checkpoint its answer names, and the trainer's importance weight corrects
for the difference.

## Share a GPU between engines and the trainer

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
concurrency. A run's runner takes it in each heartbeat, with the adapter and
version the channel serves ([heartbeats](rollouts.md#heartbeats)), and the [monitor](monitor.md) shows it from there.
