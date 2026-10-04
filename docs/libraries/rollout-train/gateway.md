# The gateway

Code: `rollout_train.gateway` · See [`Gateway`](../../guide/reference.md#gateway),
[`TurnStore`](../../guide/reference.md#turnstore), [`Keyring`](../../guide/reference.md#keyring),
[`GatewayEndpoints`](../../guide/reference.md#gatewayendpoints), [recording](recorder.md),
[harnesses over HTTP](harness-endpoint.md), [episodes](episodes.md)

The gateway stands between programs and harnesses on one side and the endpoints that sample a policy on the other.
Programs and harnesses speak a standard model API to it, with a key per model slot. It is how every run records its
samples: a runner's recorded slots sample through it, a gateway in the runner's own process or replicas of their own
([a runner served by the gateway](#a-runner-served-by-the-gateway)). For each request it:

1. verifies the key;
2. renders the request with the channel's renderer;
3. chooses the checkpoint to sample from;
4. samples with the [thinking budget](recorder.md#thinking);
5. records the turn;
6. replies in the request's own API.

It keeps no session. Everything a turn needs is in the request and its key, and everything it leaves is in the
ledger and the blob store, so any replica answers any request, and a replica can die at any moment. Only caches are
kept in memory: the turns' blobs a replica has read, and what each run's channel should serve and what its servers
have, asked again every few seconds.

```python
routes = Routes({"policy": Route(renderer, "Qwen/Qwen3-0.6B", ("http://router:8000",))}, ledger)
gateway = Gateway(TurnStore(ledger, blobs), Keyring.from_environment(), routes=routes)
app = create_app(gateway)               # serve with uvicorn, as many replicas as wanted
```

| Path | Does |
|---|---|
| `POST /v1/chat/completions` | OpenAI's Chat Completions |
| `POST /v1/responses` | OpenAI's Responses |
| `POST /v1/messages` | Anthropic's Messages |
| `POST /v1/samples` | a [`SampleRequest`](../../guide/reference.md#samplerequest), answered with a `SampleResult`, for programs in a runner |
| `GET /v1/models` | the channels, as models |
| `GET /healthz` | 200 while the process serves |
| `GET /readyz` | 200 when the ledger and the blob store answer; 503, saying which does not, otherwise |

What a request in the three APIs means, how reasoning goes both ways, streams and errors are described in
[harnesses over HTTP](harness-endpoint.md). The native path answers errors as
`{"error": {"type", "message"}}`, with the `context_limit` of a context too long for the model.

A reply's headers say what served it:

| Header | Says |
|---|---|
| `X-Rollout-Checkpoint` | the checkpoint that sampled it, by id (the base model's name before the first checkpoint) |
| `X-Rollout-Depth` | that checkpoint's depth: the version stamped on its tokens |
| `X-Rollout-Request-Id` | the request id it was recorded under: its `Idempotency-Key`, or one the gateway chose |
| `X-Rollout-Replayed` | `true` when the turn was recorded before, and the reply is the recorded one |

## What a request guarantees

- **A retried request is not sampled twice.** A request id (`Idempotency-Key`, or a native request's `effect_id`)
  under which a turn was recorded is answered with the recorded reply, by whichever replica it reaches. A request
  without one is a new sample, and repeating its prompt exactly replaces the earlier turn in the segments.
- **A reply is sent only once its turn is recorded.** A replica that dies before recording leaves nothing behind, and
  the client's retry is sampled again elsewhere. One that dies after recording, before replying, leaves the turn,
  and the retry is answered with it.
- **The first turn recorded under a request id is the one kept.** Two replicas that sample one request at once (a
  retry that reached another replica while the first was still sampling) both record, and the ledger keeps the first
  append. Both answer with the turn that was kept.
- **A turn samples one checkpoint**, the one chosen when it began, through both phases of its thinking. An endpoint
  that unloads it in between has the turn sampled again from the start, on the next choice.
- **The sampling parameters are the binding's**, carried in the key: what a client sends is ignored.

## Keys

A key is signed, and says by itself which session it samples for: the gateway verifies it with nothing but the
secret, and looks nothing up.

```
rk1.KID.PAYLOAD.SIGNATURE
```

`PAYLOAD` is a [`Grant`](../../guide/reference.md#grant) as JSON, base64url-encoded without padding:

| Field | Says |
|---|---|
| `run` | the run whose tables the turns go under (a training run, an eval's run) |
| `run_id` | the program's run that plays the attempt; the session is `{run_id}/{slot}` |
| `slot`, `channel` | the model slot, and the channel that serves it |
| `episode`, `attempt` | `GROUP/EPISODE` and the attempt, for an episode a run asked for |
| `fence` | `[scope, number]`: the fence the turns are appended under |
| `temperature`, `top_p` | how the binding samples |
| `thinking`, `answer` | the thinking and answer room the binding gives (`SamplingParameters.thinking_tokens`, `answer_tokens`), in place of the channel's own; absent: the channel's |
| `expires` | seconds since the epoch. A key is taken up to `leeway` (30 s) after it, for clocks that differ |

`SIGNATURE` is the HMAC-SHA256 of `rk1.KID.PAYLOAD` under the secret named `KID`. A
[`Keyring`](../../guide/reference.md#keyring) holds the secrets by id: the first signs, and every one verifies. A
secret is rotated by putting a new one first, and removing the old once the keys it signed have expired. Secrets are
read from `ROLLOUT_GATEWAY_KEYS` (`KID:SECRET` pairs separated by commas) or from a file named by
`ROLLOUT_GATEWAY_KEYS_FILE` or a profile's `[gateway] keys` (one `KID SECRET` per line). They are never written to the
ledger. A secret is 32 bytes at least. A gateway in a runner's own process given none signs with a secret it makes when
it starts: its keys are taken only while it runs.

Whoever plays an attempt mints a key per slot: [`GatewayEndpoints`](#a-runner-served-by-the-gateway) does, for a
runner. Clients send it as OpenAI's do (a bearer token) or as Anthropic's do (`x-api-key`).

A key is refused (401 in each API's shape) when it is malformed, signed by a secret the gateway does not have,
forged, expired, or names a channel the gateway does not serve. A key for one session cannot sample for another: a
native request's `session_id` must be the key's.

## The turn store

Every turn is two things ([`TurnStore`](../../guide/reference.md#turnstore)):

- **A blob** holding all of it:
  - who sampled it: the run, the episode and attempt, the program's run, the slot, the channel;
  - the checkpoint that served it, and its depth;
  - the prompt's tokens, the completion's tokens, and which were sampled and which forced;
  - the behaviour logprobs;
  - the reply: the parsed message, how it finished, usage;
  - the links its harness declared;
  - timings: when it started, how long each generation took, and the whole turn.
- **A ledger record** naming the blob, appended under the request id to the table of the program's run,
  `runs/RUN/turns/RUN_ID`, under the fence the key names. Besides the blob it says the slot, the episode and
  attempt, the checkpoint and depth, how many prompt tokens and sampled tokens the turn has, and when it was recorded.

The ledger record is what makes a turn count: a blob that no record names is never read.

**A turn stores only the tokens it adds.** A session's prompts repeat each other. Each turn's prompt begins with most
of an earlier turn's tokens (its prompt and what it sampled): all of them where the context only grew, and up to the
first edit where it was edited, such as an observation shortened once it is no longer the current one. A turn names
that earlier turn (its `parent`) and how many of its tokens it begins with (`shared`), and holds only the rest of its
prompt. A session's turns form a tree, and a [segment](recorder.md#what-a-session-exports) is a path from a root
along turns that share all of their parent's tokens.

The parent is found without reading any tokens back. Each blob holds `marks`: digests (BLAKE2b, 16 bytes) of its
tokens' prefixes, at every 512 tokens and at its end. A new prompt is compared with the marks of the session's three
latest turns. A turn read back is checked against its own last mark, and a turn that does not read back as it was
recorded raises.

The blob is a header (JSON) and arrays (little-endian 32-bit tokens; logprobs as 32-bit floats where that loses
nothing, as engines produce them, else 64-bit), compressed with LZMA2.

**What a turn costs**, measured on the shape of a Minecraft episode (`tests/rollout_train/gateway/test_storage.py`):
Qwen3.5's tokens, a system prompt and the action tools, a summary, recent turns in their remembered form, and the
current observation in full with its map. That is 5,044 prompt tokens and 224 sampled per turn, compacting when the
context passes 5,600 tokens:

| | Bytes per turn |
|---|---|
| The turn's blob | 3,164 |
| Its ledger record | 441 |
| Each turn's tokens and logprobs compressed on their own, for comparison | 4,824 |
| The same, raw | 22,857 |

Most of a blob is what is new each turn: the current observation, about 1,600 bytes. The logprobs take about 900
bytes, the reply about 700, and the sampled tokens about 400. A conversation that only grows costs less.

### Which turns count

An attempt's turns are kept apart from every other attempt's, in a table of its own program's run. Two things keep a
stale attempt's turns out of training:

- **The fence refuses them.** A turn is appended under the fence its key names, so once the fence is taken again
  (a newer attempt of the episode under a fence per episode, `runs/RUN/episodes/GROUP/EPISODE`, or a runner started
  again under its own), the stale attempt's next turn is refused with `Fenced`. The gateway answers 401: "this
  key's attempt was taken over", and records nothing.
- **The episode's record decides.** An episode's record names the program's run that played it (`run_id`), and its
  segments are assembled from that run's table alone. Turns another attempt recorded before it was taken over stay
  in their own table, which no record names.

### Segments

`TurnStore.sessions(run, run_id)` reads a program's run's turns, in the order they were recorded, and assembles each
slot's segments with `segments_of` (`rollout_train.recorder.segments`: [what a session
exports](recorder.md#what-a-session-exports)). A runner puts them in the episode when the run ends ([how an episode is
assembled](episodes.md#how-an-episode-is-assembled)). The segments are the same whether the gateway ran in the
runner's process or elsewhere, and whether or not the runner was started again while the run played.

### Links between requests

A request may say how it follows from earlier ones: a harness in `X-Rollout-Links` (a JSON list of `{"type",
"source"}`), a program in its `SampleRequest.links`, each source an earlier request's id. `Memory` declares its
compactions this way, as the Minecraft team's agents compact. Types are labels (a letter, then letters, digits and `._:-`), and every label is kept
as it was sent:

| Type | From, to |
|---|---|
| `continuation` | a request, to the one that goes on from it |
| `compaction_attempt` | the request being compacted, to the request that asks for the summary |
| `compaction` | that attempt, to the request that goes on from the summary: the attempt was accepted |
| `subagent_call`, `subagent_return` | a request, to the subagent's first request; the subagent's last, to the request it returns to |

With `accepted_only`, `sessions` trains on what a compaction attempt sampled only if a later request names it as the
source of a `compaction`. Its tokens are kept as context either way. By default every sample is trained on.

## Which checkpoint

The gateway samples a channel through its `Sampler` ([channels](channels.md)):

- **A channel whose engines serve elsewhere** (`RemoteEngine`: vLLM servers, or a router in front of them) is
  sampled per run, as a `RemoteChannel`. It reads what the run says its channel should serve
  ([what a channel should serve](channels.md#what-a-channel-should-serve)), and asks the servers for that checkpoint
  by its id, as the model's name. Where a server has not loaded it yet, it asks for the newest one before it that the
  server has, no more than `max_lag` checkpoints behind (1, or what the profile's channel or the run says). With none
  close enough, a turn waits up to its patience, then fails as the endpoint failing does: 503.
- **A channel whose engines are in the gateway's own process** samples whatever it serves.

A turn's weights are chosen once, when it begins: both phases of its thinking ask for the same checkpoint. The turn
records the checkpoint that served it (by id; the base model's name before the first checkpoint) and that
checkpoint's depth. An answer that names another checkpoint than the one asked for, or a server that unloaded it
between the phases (`Unserved`), has the turn sampled again from the start, up to three times; nothing of a failed
attempt is recorded.

## Running it

A runner whose profile names no `[gateway] url` runs a gateway in its own process, over the profile's channels and
routes, and records through it with no HTTP in between; it serves it to harnesses at the profile's `serve`
([deploying](../../guide/deploying.md#the-gateway)). Replicas of their own serve the same code:

```bash
ROLLOUT_GATEWAY_KEYS_FILE=~/.config/rollout/gateway.keys \
    uv run rollout gateway profile.toml --listen 127.0.0.1:8830     # a replica: start as many as wanted
```

`rollout gateway PROFILE` serves a replica of what the profile describes (`deployed`): its channels and their
engines, its ledger and blob store, and its `[gateway]` table ([deploying](../../guide/deploying.md#the-gateway)).
Replicas share nothing but the ledger and the blob store.

- **Behind a proxy.** A proxy in front of the replicas terminates TLS, checks its own credentials, and may serve them
  under a path of its own (`https://models.example/gw/v1`). The gateway builds no URL from a request. It trusts
  `X-Forwarded-*` headers only from `--proxied` addresses (127.0.0.1 unless given).
- **TLS of its own.** With `--certificate` and `--private-key`, a replica serves HTTPS itself.
- **Health.** A load balancer sends traffic to replicas whose `/readyz` answers 200.
- **Heartbeats.** A replica beats beside the ledger every 15 seconds ([heartbeats](rollouts.md#heartbeats)) as
  `gateway/HOST/LISTEN` (`rollout_train.gateway.beats`): kind `gateway`, its host, where it listens, its machine, and
  what each channel it samples serves and how fast. The [monitor](monitor.md#the-machines) shows the replicas alive.

## A runner served by the gateway

[`GatewayEndpoints`](../../guide/reference.md#gatewayendpoints) is a runner's recorder
([`RecordedEndpoints`](../../guide/reference.md#recordedendpoints)), over a gateway in this process or one at a URL:

```python
endpoints = GatewayEndpoints.of(gateway, "http://runner:8800")   # a gateway in this process, served there
endpoints = GatewayEndpoints("https://models.example/gw", keyring, TurnStore(ledger, blobs), routes=routes)
```

- `admit(run_id, Attempt(run, fence, episode, attempt))` says which attempt a program's run plays, before it
  starts. The [episode runner](rollouts.md#a-runner) admits each attempt under the fence of its episode, which its
  claim took, and admits an adopted run again under the fence it takes anew;
- `endpoint(binding)` serves its recorded slots. A program's sample is a call of the gateway in this process, or a
  post to `/v1/samples` under a key minted for the slot, posted again under the same effect id when the gateway
  cannot be reached or a replica fails;
- `address()` hands a harness the gateway's URL and such a key;
- `reaches(run, binding)` says whether every recorded model of a run's binding can be sampled now (a routed channel's
  servers have a checkpoint close enough), which the episode runner asks before claiming;
- `sessions(run, run_id)` reads what each slot recorded, when the run ends.

What a channel guarantees a session (its capability contract) is its channel's: in this process, or for a routed
channel, as the runner sees its servers.

**Hooks.** A program's samples reach the runner's hooks through its endpoints. A harness's go straight to the
gateway: one in the runner's process tells the runner's hooks of each (`Gateway.hooks`), and the
[monitor](monitor.md) reads the turns of an episode a gateway elsewhere recorded.

**A runner started again.** A durable run resumed by a runner started again asks again for the samples it had not
heard back from, under the same effect ids, and is answered with the recorded turns; what it had recorded before
survived in the turn store. Its episode is trained on like any other.
