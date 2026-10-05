# The gateway

For people who train, deploy or connect a harness: the service that samples channels and records every turn, its keys,
its turn store, and how to run it.

**Read first:** [Record turns for training](recorder.md). **Next:** [Train any harness over HTTP](harness-endpoint.md).

Code: `rollout_train.gateway` · See [`Gateway`](../../guide/reference.md#gateway),
[`TurnStore`](../../guide/reference.md#turnstore), [`Keyring`](../../guide/reference.md#keyring),
[`GatewayEndpoints`](../../guide/reference.md#gatewayendpoints), [recording](recorder.md),
[harnesses over HTTP](harness-endpoint.md), [episodes](episodes.md)

The gateway stands between programs and harnesses on one side and the endpoints that sample a policy on the other.
Programs and harnesses speak a standard model API to it, with a key per model slot. It is how every run records its
samples: a run's runner samples its recorded slots through a gateway in the run's driver, and harnesses reach it over
HTTP; the cluster's replicas serve the same code ([a runner served by the gateway](#a-runner-served-by-the-gateway)). For each request it:

1. verifies the key;
2. renders the request with the channel's renderer;
3. chooses the checkpoint to sample from;
4. samples with the [thinking budget](recorder.md#thinking);
5. records the turn;
6. replies in the request's own API.

It keeps no session. Everything a turn needs is in the request and its key, and everything it leaves is in the
[ledger](checkpoints.md#the-ledger) and the blob store, so any replica answers any request, and a replica can die at any
moment. Only caches are kept in memory: the turns' blobs a replica has read, and what each run's channel should serve
and what its servers have, asked again every few seconds.

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
| `POST /v1/messages/count_tokens` | how many tokens a Messages request's prompt renders to with the channel's renderer; nothing is recorded |
| `POST /v1/samples` | a [`SampleRequest`](../../guide/reference.md#samplerequest), answered with a `SampleResult`, for programs in a runner |
| `POST /v1/scores` | a [`ScoreRequest`](../../guide/reference.md#scorerequest): the logprobs the channel gives tokens it is handed ([scoring tokens](#scoring-tokens)) |
| `GET /v1/models` | the channels, as models; each whose engines are in the replica's process with its `contract` (`context_limit`, `max_output_tokens`). With `?run=RUN`, the channels that run's start names, as `RUN/NAME`, each with its contract |
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

## Scoring tokens

A client holding a run's key can ask the key's channel for the logprobs of tokens it hands over, sampling nothing: a
teacher scoring a student's tokens. `POST /v1/scores` takes a `ScoreRequest`:

```json
{"effect_id": "s1", "session_id": "r_1/teacher", "tokens": [151644, 872, ...], "start": 412, "top": 20}
```

| Field | Says |
|---|---|
| `effect_id` | the request id: a request under one that was recorded is answered with the recorded scores |
| `session_id` | the key's session |
| `tokens` | the sequence, at least two tokens |
| `start`, `end` | the positions scored, `start` to `end` (absent: to the end). `start` is at least 1: the first token has nothing before it |
| `top` | how many of the most likely tokens to give at each position (0 unless given) |

The answer gives, for each position scored, the logprob of the token there given those before it, and the most likely
tokens there with their logprobs, most likely first, under the checkpoint the session's channel serves, which the
headers name as for a sample:

```json
{"start": 412, "logprobs": [-0.31, ...], "top_tokens": [[1734, 279, ...], ...], "top_logprobs": [[-0.31, -1.9, ...], ...]}
```

- **It is recorded first, as a turn of its own use** (`use: "score"`), in the run's table of turns under the key's slot
  and fence: the tokens up to `end` are its prompt, it has no completion, and its blob holds the scores. Its ledger
  record says `use`, how many prompt tokens it has and that it sampled none; its reply's usage counts the tokens as
  tokens in. A scoring turn is never trained on: `sessions` leaves it out of the segments.
- **It is scored by one checkpoint**, chosen as a turn's is, and scored again from the start when that checkpoint stops
  being served before it ends, up to three times.
- **Refused:** a session not the key's (401); a range that does not lie within the tokens, a `top` the channel does
  not take, or a request id a sample used (400); a sequence that leaves no room for the one token vLLM generates after
  it, `ContextOverflow` with the most tokens it takes (400).

## Keys

A key is signed, and says by itself which session it samples for: the gateway verifies it with nothing but the
secret, and looks nothing up.

```text
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
| `trained` | `false` for a slot that is not trained (a judge, a fixed opponent: `RecordedModel.trained`); absent: trained |
| `expires` | seconds since the epoch. A key is taken up to `leeway` (30 s) after it, for clocks that differ |

`SIGNATURE` is the HMAC-SHA256 of `rk1.KID.PAYLOAD` under the secret named `KID`. A
[`Keyring`](../../guide/reference.md#keyring) holds the secrets by id: the first signs, and every one verifies. A
secret is rotated by putting a new one first, and removing the old once the keys it signed have expired. Secrets are
read from `ROLLOUT_GATEWAY_KEYS` (`KID:SECRET` pairs separated by commas) or from a file named by
`ROLLOUT_GATEWAY_KEYS_FILE` or the cluster config's `[gateway] keys_file` (one `KID SECRET` per line). They are never written to the
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
  - what it was sampled with (`sampled_with`): of `token_exact`, `sampled_logprobs` and `honours_sampling`, what its
    sampler can do. Every engine and server a channel samples from does all three, and a turn whose blob does not
    say is read back as sampled with all three;
  - whether its slot is trained (`trained`, from the key): false for a judge's or a fixed opponent's turn;
  - the reply: the parsed message, how it finished, usage;
  - the links its harness declared;
  - timings: when it started, how long each generation took, and the whole turn;
  - for a scoring turn, its use (`score`) and its scores ([scoring tokens](#scoring-tokens)).
- **A ledger record** naming the blob, appended under the request id to the table of the program's run,
  `runs/RUN/turns/RUN_ID`, under the fence the key names. Besides the blob it says the slot, the episode and
  attempt, the checkpoint and depth, how many prompt tokens and sampled tokens the turn has, when it was recorded,
  its use where it is not a sample, and `trained: false` for a turn of a slot that is not trained.

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

- **A channel a run's start names** with a provider the gateway's
  [`ChannelDirectory`](../../guide/reference.md#channeldirectory) knows is built the first time a key of that run is
  shown (or the runner asks whether the run's channels can be sampled): from the run settings its newest start
  records, the channel's providers, model, renderer and thinking budget, as a `RemoteChannel` over those providers'
  servers. Nothing registers it, and it serves what its mode says ([what a channel should
  serve](channels.md#what-a-channel-should-serve)): the trained channel what the run trains, a `follows` channel the
  followed channel's checkpoint `lag` records back, a `fixed` one its pinned checkpoint or the base model. The
  directory knows providers by name with the servers each is reached at (`Provided`): `ChannelDirectory.of(cluster,
  ledger)` takes every provider of the [cluster config](../../guide/cluster.md) whose servers answer vLLM's API at its
  endpoints (`vllm`, `vllm-servers`, `runpod-inference`). A run's channel takes precedence over a channel of the same
  name in the gateway's own process.
- **A routed channel** (`Routes`: a run's channel on its engine hosts, or on servers at addresses, or a router in front
  of them) is sampled per run, as a `RemoteChannel`. It reads what the run says its channel should serve
  ([what a channel should serve](channels.md#what-a-channel-should-serve)), and asks the servers for that checkpoint
  by its id, as the model's name. Where a server has not loaded it yet, it asks for the newest one before it that the
  server has, no more than `max_lag` checkpoints behind (1, or what the run's settings say). With none
  close enough, a turn waits up to its patience, then fails as the endpoint failing does: 503.
- **A channel whose engines are in the gateway's own process** (an engine that calls a hosted API, such as
  `TinkerEngine`, in a run's driver) samples whatever it serves: the base model, until the run publishes a checkpoint
  to it.

One gateway samples channels of every kind at once: a run's `TinkerEngine` channel beside a channel on its engine hosts
and one on servers at addresses, say.

A turn's weights are chosen once, when it begins: both phases of its thinking ask for the same checkpoint. The turn
records the checkpoint that served it (by id; the base model's name before the first checkpoint) and that
checkpoint's depth. An answer that names another checkpoint than the one asked for, or a server that unloaded it
between the phases (`Unserved`), has the turn sampled again from the start, up to three times; nothing of a failed
attempt is recorded.

## Running it

A run's driver runs a gateway in its own process, over the run's channels ([launching runs](launching.md#the-driver)):
its routed channels (engine hosts, servers at addresses) and its channels with engines in the process (Tinker). Its
runner records through it with no HTTP in between, and serves it to harnesses on its node, at a free port on
`127.0.0.1`; it signs keys with a secret of its own ([deploying](../../guide/deploying.md#the-gateway)). Replicas of the
cluster's gateway serve the same code:

```bash
uv run rollout gateway --cluster --listen 0.0.0.0:8900     # a replica: start as many as wanted
```

`rollout gateway --cluster` serves a replica over the cluster config's ledger and blob store, with the keys
`[gateway] keys_file` or `keys_env` names (else the environment's), and a `ChannelDirectory` of its providers
(`ChannelDirectory.of`): every channel a run's start names on a provider whose servers answer vLLM's API at its
endpoints, a judge's channel or one that follows the trained channel among them. Replicas share nothing but the ledger
and the blob store.

- **Behind a proxy.** A proxy in front of the replicas terminates TLS, checks its own credentials, and may serve them
  under a path of its own (`https://models.example/gw/v1`). The gateway builds no URL from a request. It trusts
  `X-Forwarded-*` headers only from `--proxied` addresses (127.0.0.1 unless given).
- **TLS of its own.** With `--certificate` and `--private-key`, a replica serves HTTPS itself.
- **Health.** A load balancer sends traffic to replicas whose `/readyz` answers 200.
- **Heartbeats.** A replica beats beside the ledger every 15 seconds ([heartbeats](rollouts.md#heartbeats)) as
  `gateway/HOST/LISTEN` (`rollout_train.gateway.beats`): kind `gateway`, its host, where it listens, its machine, and
  what each channel it samples serves and how fast. The [monitor](monitor.md#the-machines) shows the replicas alive.

## Claude Code and Codex

Both run against the gateway unchanged, Claude Code over Messages and Codex over Responses, with a key minted for a
run's slot. `tests/rollout_train/gateway/test_harnesses.py` replays requests recorded from Claude Code 2.1.288 and
codex-cli 0.157.1, so the formats keep working without the binaries.

**Variables.** A sandbox's lease puts each slot's address in its environment
([sandboxes](../rollout/sandboxes.md#harnesses-inside-a-sandbox)), each suffixed with the slot's name in capitals
(`OPENAI_API_KEY_AGENT_1`), and unsuffixed too when the sandbox's spec names one slot:

| Variable | Value | Read by |
|---|---|---|
| `OPENAI_BASE_URL` | the gateway's base URL, ending in `/v1` | OpenAI's clients; Codex through its provider's `base_url` |
| `OPENAI_API_KEY` | the slot's key | OpenAI's clients; Codex through its provider's `env_key` |
| `OPENAI_MODEL` | the name to send as the model | |
| `ANTHROPIC_BASE_URL` | the same URL without `/v1` (Anthropic's clients add `/v1/messages`) | Claude Code |
| `ANTHROPIC_AUTH_TOKEN` | the slot's key, sent as a bearer token | Claude Code, which takes it without the approval an interactive session asks for `ANTHROPIC_API_KEY` |
| `ANTHROPIC_MODEL` | the name to send as the model | Claude Code |

**Claude Code.** Given those variables, a home of its own, and telemetry and nonessential traffic switched off, it
connects to nothing but the gateway:

```bash
export HOME=/sandbox/home CLAUDE_CONFIG_DIR=/sandbox/claude CLAUDE_CODE_TMPDIR=/sandbox/tmp
export DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_AUTOUPDATER=1
cd /work && claude -p "Create a file named hello.txt containing the word hi." \
    --output-format stream-json --verbose \
    --tools Read,Write,Edit,Glob,Grep --allowedTools Read,Write,Edit,Glob,Grep --disallowedTools Bash \
    --permission-prompts none --restricted --strict-mcp-config
```

`--restricted` keeps its file tools in the working directory. It sends `HEAD /api/hello` first, answered 404, which
it ignores; `/context` counts tokens at `/v1/messages/count_tokens`; `/compact` asks for a summary like any other
request.

**Codex.** It reads no base URL from its environment (with only `OPENAI_BASE_URL` set it goes to OpenAI's servers):
it is given a provider in `$CODEX_HOME/config.toml`. Its plugin sync reaches GitHub and `chatgpt.com` at startup;
with it off, and analytics and feedback off, it connects to nothing but the gateway:

```toml
model = "policy"                       # OPENAI_MODEL
model_provider = "rollout"
model_context_window = 32768           # the channel's context
check_for_update_on_startup = false

[model_providers.rollout]
name = "rollout"
base_url = "http://gateway:8830/v1"    # OPENAI_BASE_URL
env_key = "OPENAI_API_KEY"
wire_api = "responses"
supports_websockets = false

[features]
plugins = false

[sandbox_workspace_write]              # commands write in the working directory only
exclude_slash_tmp = true
exclude_tmpdir_env_var = true

[analytics]
enabled = false

[feedback]
enabled = false
```

```bash
codex exec --sandbox workspace-write --cd /work --skip-git-repo-check --ephemeral --json \
    "Run the command: echo hi > hello.txt"
```

The provider can be given as flags instead (`-c model_provider=rollout -c model_providers.rollout.base_url=...`,
one per key). Codex warns that it has no metadata for the model's name and uses its defaults. A compaction is a
request like any other.

**Supported:**

- Messages: `system` given as text or blocks; `system` messages anywhere in `messages`; `thinking` blocks sent
  back, whatever their signature; `tool_use` and `tool_result` blocks, several in one message; streams; every
  `anthropic-beta` header; `count_tokens`, counted with the channel's renderer (the whole prompt, through the
  start of the reply) and not recorded.
- Responses: `instructions`, `developer` messages, `reasoning` items sent back with their content, several function
  calls and their outputs in one turn, streams.
- A reply cut off inside its thinking (the model stopped before closing it) is all reasoning: Claude Code asks for
  more, Codex ends its turn.

**Ignored:** the model's name; sampling parameters; Anthropic's `thinking`, `output_config`, `context_management`,
`safeguards` and `metadata`; Responses' `reasoning`, `include` (there is no encrypted reasoning), `tool_choice`,
`parallel_tool_calls` and `client_metadata`; caching hints (`cache_control`, `prompt_cache_key`), since the engines
cache by prefix; the block Claude Code puts first in `system` for Anthropic's billing; provider tools (`web_search`).

Every request a harness sends is a turn of its slot, its compactions' summaries too.

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

What a channel guarantees a session (its capability contract) is its channel's: in this process, for a routed
channel as the runner sees its servers. A channel
a run's start names, sampled by a gateway elsewhere, is learned from that gateway's `/v1/models?run=RUN`, which lists
the run's channels as `RUN/NAME`, each with its contract.

Each slot of a run's binding names its channel (`RecordedModel.channel`), so each slot's key routes to the channel the
run binds the slot to: `bind(program, channel, slots={"judge": "judge"})` serves `judge` from the channel `judge` and
every other slot from `channel`.

**Hooks.** A program's samples reach the runner's hooks through its endpoints. A harness's go straight to the
gateway: one in the runner's process tells the runner's hooks of each (`Gateway.hooks`), so the
[monitor](monitor.md)'s feed has both; one elsewhere does not, and the monitor reads the turns of an episode it
recorded from the ledger.

**A sample asked for again.** A sample asked for again under an effect id the gateway has recorded (a retry whose
reply never arrived) is answered with the recorded turn (`X-Rollout-Replayed: true`); nothing is sampled twice.
