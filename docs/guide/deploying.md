# Deploying

A deployment is described once, in a profile: the channels and the engines behind them, the trainer, the runner, and
where each environment's tool sets and sandbox pools live. Whoever trains gets a trainer, the checkpoints and a way to
publish checkpoints from it, while a runner it opens plays the episodes the run asks for, and never learns what stands
behind them; an environment is named in it only by its tool sets and pools. This page is the one place the profile file is
described. The code is `rollout_train.profile`, and the command is `rollout` (`rollout_train.cli`).

```toml
directory = "~/.cache/rollout/runs/first"     # the run's own: run.json, checkpoints in use, the monitor's feed
serve = "0.0.0.0:8900"                        # optional: the runner's gateway, for harnesses, over HTTP
address = "http://trainer-1:8900"             # what others reach it at, if not http://{serve}
feed_runs = 80                                # optional: episodes kept in the monitor's feed
episodes_at_once = 6                          # optional: the most episodes this machine's runner plays at once

[channels.policy]
model = "cyankiwi/Qwen3.5-9B-AWQ-4bit"
renderer = "rollout_qwen:qwen35"              # the model family's token format
engine = "rollout_vllm:VllmEngine"            # what serves it; each entry of `engines` is one replica's options
engines = [{ gpu_memory_utilization = 0.78, max_model_len = 8192, max_num_seqs = 20 }]
thinking_tokens = 1024                        # optional: thinking past this is closed by force (none: no budget)
answer_tokens = 400                           # optional: room for the answer after it (none: what the turn has left)

[trainer]
kind = "rollout_lora:LoraTrainer"             # what trains; the keys below it does not name here are its settings
channel = "policy"                            # the channel that serves what it trains
start = "diamonds"                            # optional: the checkpoint a new run trains from (by default the base model)
bookmark = "diamonds, unguided"               # optional: a bookmark the run carries to each checkpoint it makes
colocated = true                              # it shares the engines' GPU: they sleep while it steps
rank = 32
learning_rate = 5e-5
segment_tokens = 8000                        # the longest turn it can train on: its channel takes it as its limit
segments_per_step = 384

[pools.minecraft]                             # the worlds, made in this process by `worlds(directory, size=6)`
kind = "minecraft_team.worlds:worlds"         # or, for a pool on a machine of its own: minecraft = "http://worlds:8710"
size = 6

[memory]
runs_gib = 6                                  # must be available to admit runs
training_gib = 4                              # and to start a step

[evals]
suite = "words-held-out"                      # evaluate checkpoints as they are made (its eval data, say); "" for none
every = 2                                     # the checkpoint of every second step
episodes = 1                                  # episodes of each start (by default the suite's)

[gateway]                                     # optional: the gateway that records every turn (`rollout gateway`)
url = "https://models.example/gw"             # its replicas, which the runner records through (none: one in its process)
listen = "127.0.0.1:8830"                     # where a replica serves
keys = "~/.config/rollout/gateway.keys"       # the secrets keys are signed with (else ROLLOUT_GATEWAY_KEYS)
```

```bash
uv run rollout train profile.toml minecraft_team.environment:environment --groups 100 --directory RUN  # --groups-per-step 4
uv run rollout monitor RUN                     # the web page over RUN's ledger and its runs: http://localhost:8765
uv run rollout report RUN minecraft_team.environment:environment --watch   # charts; posted to DISCORD_WEBHOOK_URL if set
uv run rollout imitate profile.toml --directory RUN                  # a supervised step on solved, guided episodes
uv run rollout checkpoints --ledger RUN                                 # every checkpoint: where it came from, its bookmarks
uv run rollout checkpoints --cluster                                    # the same, over the cluster config's ledger
uv run rollout cluster check                                            # what of the cluster config does not resolve here
uv run rollout bookmark diamonds first:20 --ledger RUN               # name the checkpoint run "first" made at step 20
uv run rollout rename first "diamonds, unguided" --ledger RUN        # call a run something else (its id stays)
uv run rollout pause first --ledger URL                              # nothing new starts; what plays plays out
uv run rollout resume first --ledger URL                             # paused: goes on; stopped: launched again
uv run rollout pool minecraft_team.worlds:worlds --directory DATA --ledger URL --port 8710   # worlds on a machine of their own
uv run rollout tools FACTORY --directory DATA --port 8700            # a tool set on a machine of its own
uv run rollout engines engines.toml --run first   # load what run first serves into this machine's vLLM servers
uv run rollout runner runner.toml --run first                       # a machine that plays first's episodes, and nothing else
uv run rollout gateway profile.toml --listen 127.0.0.1:8830          # a replica of the gateway: as many as wanted
uv run rollout train profile.toml ENVIRONMENT --set trainer.learning_rate=3e-5 --set start=diamonds  # change settings
uv run rollout train profile.toml ENVIRONMENT --preset faster --settings run.toml   # a preset, then a file, then flags
uv run rollout preset save faster --from-run first --set trainer.learning_rate=1e-4 --ledger RUN   # a preset's next version
uv run rollout env check ENVIRONMENT --profile profile.toml --groups 4   # does it hold together; do its groups teach
uv run rollout suite make words-v1 --environment ENVIRONMENT --seeds 1,2,3 --ledger RUN       # an eval configuration
uv run rollout suite edit words-v1 --seeds 1,2,3,4 --ledger RUN                       # its next version
uv run rollout eval profile.toml words-v1 --checkpoint diamonds --episodes 4         # play its newest with a checkpoint
uv run rollout eval profile.toml teams-every-task --environment ENVIRONMENT     # its eval data, frozen on first use
uv run rollout launcher --ledger URL --profiles PROFILES --environment ENVIRONMENT --runs RUNS   # start runs asked for here
uv run rollout launcher --ledger URL --profiles PROFILES --environment ENVIRONMENT --runs RUNS \
    --ray http://127.0.0.1:8265 --as-job                             # the same, as a Ray job; each run a Ray job too
```

`rollout COMMAND --help` lists each command's options. An environment is named as `module:name`, like everything else a
profile or the command is told by name. `train --name NAME` names a new run (by default after its directory);
`rename` names a run again, by its name or its id; `pause` and `resume` pause a run and resume it, in place or by a
launcher ([pausing and resuming](../libraries/rollout-train/training.md#pausing-and-resuming)); `bookmark` names a checkpoint by any reference, and `checkpoints` lists
them all ([checkpoints, runs and the ledger](../libraries/rollout-train/checkpoints.md#the-command-line)). `train` plays `--groups` groups and takes a step whenever `--groups-per-step`
of them have something to train on ([training](../libraries/rollout-train/training.md#the-loop)). `suite` makes,
edits (each edit a version of its own) and lists suites, and `eval` plays one with a checkpoint, training nothing ([evals](../libraries/rollout-train/evals.md)); `env check`
checks an environment before anything trains on it, and with a profile plays a few groups and flags those that teach
nothing ([checking an environment](../libraries/rollout-train/rollouts.md#checking-an-environment)); `report` and
`imitate` are described in [reporting](../libraries/rollout-train/training.md#reporting) and
[imitation](../libraries/rollout-train/training.md#imitation); `engines` and `runner` in
[engines on other machines](#engines-on-other-machines).

## The file

A key the profile does not have is an error, so a misspelt guard is never silently no guard.

| Part | What it decides | To scale it |
|---|---|---|
| `directory` | Where the run's state is kept. `rollout train --directory` replaces it: one profile, many runs | |
| `channels` | Which model each channel serves, its token format, what serves it, and how much it may think and answer (`thinking_tokens`, `answer_tokens`: [`Limits`](reference.md#limits)' `thinking` and `answer`; either left out is no budget, and with neither a turn may fill all the context its prompt leaves, [limits](../libraries/rollout-train/channels.md#limits)). `reshard` names the bridge that makes the files its engines load from a checkpoint's (`verbatim`, `peft-from-tinker`, …): each checkpoint is then [bridged](../libraries/rollout-train/checkpoints.md#bridges) before it is served. Without it, the engines load the trainer's files as they are. A channel whose `engine` is `rollout_train.inference:RemoteEngine` has its engines on other machines: vLLM servers, each entry of `engines` one's `address` (`via`, `max_lag`, `connection`: [engines on other machines](#engines-on-other-machines)) | Add entries to `engines`, each with its own options (a device, an address): sessions spread over them, each staying with one. Or serve them on machines of their own (`rollout engines`), behind a router |
| `trainer` | What trains which channel, and its settings. The longest segment it can train on becomes that channel's longest turn. `start` is the checkpoint a new run trains from, by any [reference](../libraries/rollout-train/checkpoints.md#references) (by default the base model: the channel's `model`); a run started again goes on from its own newest checkpoint. `bookmark` names a bookmark the run moves to each checkpoint it makes | `colocated = false` when it has an accelerator of its own: engines then serve through a step |
| `ray` | A Ray cluster the run connects to (`ray = "auto"`: the one this machine is part of, or `ray://host:port`): its bridges then run as Ray tasks on that cluster ([Ray](#ray)). Without it, they run in the run's process | Add nodes to the cluster |
| `serve`, `address` | Where the gateway in the runner's own process listens for [harnesses](../libraries/rollout-train/harness-endpoint.md), and the URL others reach it at | |
| `tools` | Each tool set an environment imports by name: `module:name` of what makes it in this process, or a URL | Run `rollout tools` where the tool set should live |
| `pools` | Each sandbox pool, by the kind of sandbox it serves ([sandboxes](../libraries/rollout/sandboxes.md)): `module:name` of the provider that makes them in this process, or a table whose `kind` is that and whose other keys are its settings (`size`: how many at once), or a URL. A pool in this process is named `KIND@HOST/DIRECTORY`, keeps its leases beside the ledger and has a keeper that ends them with their claims | Run `rollout pool` where the sandboxes should live, and give its URL |
| `ledger` | Where the run's tables and the checkpoints are kept ([the ledger](../libraries/rollout-train/checkpoints.md#the-ledger)): a directory (`ledger = "path"`), or a table naming a ledger (`[ledger]` with `kind = "rollout_train.database:DatabaseLedger"` and a `url`: `sqlite:///~/…` on one machine, `postgresql://…` for several; `rollout ledger copy` moves one to the other). Without it, `directory/ledger`. Beside it are kept, as ordinary state changed in place: the registry of runs' names, bookmarks and the versions suites' names point to, the runners' heartbeats, the launches, what is wanted of each run's settings, and the sandboxes' leases | Runs that share a ledger and a blob store share one graph of checkpoints, and can start from each other's |
| `blobs` | Where episodes, each step's batch and what each step left behind are kept. Without it, files under `directory/blobs`. With `kind = "module:name"`, the store that makes, called with the table's other entries (`rollout_s3:S3BlobStore`, say). Retention deletes a released checkpoint's files an hour at the earliest after they were last put ([checkpoints](../libraries/rollout-train/checkpoints.md#checkpoints)) | Point it at an object store that the machines share: `python -m rollout_s3.copying DIRECTORY... --to s3://BUCKET/PREFIX` copies stores of files into a bucket, and `python -m rollout_train.relocating` rewrites a copy of a database ledger so that its records name the bucket, as `deploy/k3s/migrate.sh` does |
| `memory` | System memory that must be available before the runner claims another episode (`runs_gib`: short of it, it waits) and before a colocated step starts (`training_gib`: short of it, the run stops with `NotEnoughMemory`, before the step) rather than exhaust its machine | |
| `feed_runs` | How many episodes the [monitor](../libraries/rollout-train/monitor.md)'s feed keeps | |
| `evals` | A suite the run plays with the checkpoint of every `every`th step (1 unless it says otherwise), `episodes` episodes of each start (the suite's unless it says otherwise), between that step and the next, on the trained channel ([evals during training](../libraries/rollout-train/evals.md#evals-during-training)). A suite named by its name is played in the version its name points to as each step is decided (`NAME@N` names one version for good). Each is an eval of its own, a run named `NAME-eval-STEP`. Without it, or with `suite = ""`, the run evaluates nothing; a launch from the monitor says a suite or none, where its profile does not. A running run's evals can be changed ([what can change while a run goes](#what-can-change-while-a-run-goes)) | Ask for evals as launches instead, so that they run on engines of their own |
| `gateway` | The [gateway](#the-gateway) the runner records through and `rollout gateway` serves: where its replicas are reached (`url`; none: a gateway in the runner's own process) and serve (`listen`), the file of the secrets its keys are signed with (`keys`), and how long a key minted for a slot is good for (`lifetime`, 6 hours) | Give it a `url` and start replicas behind a proxy |
| `episodes_at_once` | How many episodes the run keeps work waiting for, and the places of this machine's runner: the most it plays at once, whatever groups they are of (6 unless it says otherwise), what the engines can take. An episode is claimed only while its sandboxes' pools have room for it too | Raise it with the engines' `max_num_seqs`, and the pools' sizes with the memory for their sandboxes |

## What can change while a run goes

Some of a run's settings can change between two steps without breaking it, and are taken from its next step on:
`groups_per_step`, `max_lag` (how many checkpoints behind the newest a turn of the trained channel may begin; the
channel's `max_lag` to start with), the `evals` (`evals.suite`, `evals.every`, `evals.episodes`; an edit of the suite,
too, as a new version its name points to), and the settings its trainer takes
between steps (`trainer.learning_rate`, and for `rollout_lora`'s trainers the clips, `truncate`, `tokens_per_step`,
`max_kl` and `max_gradient_norm`). The rest are fixed when it starts: the channels, their models and engines, the
trainer's kind, what its weights are and its other settings (the adapter's `rank`, the longest segment), the runner,
`episodes_at_once`. The monitor's run page changes the changeable ones and shows the fixed ones; what is wanted is kept
beside the ledger, and each step's record says the settings it used ([changing a running run's
settings](../libraries/rollout-train/training.md#changing-a-running-runs-settings)). The profile's own values are
where a run starts; a run started again starts from them and takes what is wanted at its next step.

## What a profile names

Engines, renderers, trainers, tool sets and sandbox providers are implementations, named as `module:name`. `rollout_train.profile`
imports none of them: it calls what the name resolves to, and passes it what the profile says of it. Their defaults
are their own.

| Key | Called with | Implementations in this repository |
|---|---|---|
| `engine` of a channel | The channel's `model`, and one entry of `engines` as keyword arguments. Once per entry | `rollout_vllm:VllmEngine` ([vLLM engine](../implementations/rollout-vllm.md)), `rollout_tinker:TinkerEngine` ([Tinker](../implementations/rollout-tinker.md)) |
| `renderer` of a channel | The channel's `model` | `rollout_qwen:qwen35`, `rollout_qwen:qwen3` ([Qwen renderers](../implementations/rollout-qwen.md)), `rollout_gemma:gemma4` ([Gemma renderers](../implementations/rollout-gemma.md)) |
| `kind` of the trainer | The trained channel's `model`, and every other key of `[trainer]` except `channel`, `start`, `bookmark` and `colocated` as keyword arguments | `rollout_lora:LoraTrainer` ([LoRA trainer](../implementations/rollout-lora.md)), `rollout_tinker:TinkerTrainer` ([Tinker](../implementations/rollout-tinker.md)) |
| An entry of `tools` | The run's directory | An environment's own |
| An entry of `pools` | The run's directory, and the entry's other keys as keyword arguments | `minecraft_team.worlds:worlds` ([Minecraft team](../products/minecraft-team.md#the-worlds)) |

`rollout_train.testing` has a scripted engine and a readable renderer for profiles that need no GPU
(`rollout_train.testing:scripted_engine`, `rollout_train.testing:plain_renderer`). The GPU packages are installed
with `uv sync --all-extras`.

The Tinker trainer and engine train and sample at Thinking Machines, on no GPU of this machine. They are installed
with the workspace's `tinker` extra (`uv sync --extra tinker`, or `--all-extras`), and a profile that names them runs
from the workspace like any other, with the key in `TINKER_API_KEY` or `~/.tinker/credentials.json`
([Tinker trainer and engine](../implementations/rollout-tinker.md)). One gateway hosts a Tinker channel beside
channels on this machine's engines. `environments/minecraft/profiles/tinker.toml` is the Minecraft environment's.

## Opening a profile

In code, a profile opens into a platform:

```py
async with Profile.load(Path("profile.toml")).open() as platform:
    binding = binding_for(environment, "policy", platform.tool_bindings, platform.pool_bindings)
    await train(
        environment, platform.trainer, platform.checkpoints, start=platform.origin, channel="policy",
        base=platform.profile.channels["policy"].model,
        directory=platform.profile.directory / "checkpoints", publish=platform.publish, binding=binding,
        run=platform.run.id, hooks=[platform.feed], kept=platform.bookmarked, made=platform.made,
        reshard=platform.reshard if platform.layout else None,
    )
```

Opening starts, in order: with `ray`, the connection to the Ray cluster; the run (registered the first time:
`run.json`) and the checkpoint it starts from; the blob store and the checkpoints; the engines a killed process left
behind are ended (`engine.json`); the trainer; each channel's engines; the channels, the trained one with the trainer's
longest segment as its longest turn; the monitor's feed in `directory/feed`; what the runner records through
(`platform.recorder`: the [gateway](#the-gateway) at `[gateway] url`, or one in this process, `platform.gateway`); the
tool sets; the pools, each with its keeper; the runner, and the
[episode runner](../libraries/rollout-train/rollouts.md#a-runner) over it. A colocated trainer is wrapped in
[`Colocated`](reference.md#colocated). With `serve`, the gateway in this process listens there for harnesses.
`open(training=False)` (what `rollout eval` opens) makes no trainer; the trained channel's engines still load what its
`start` is served over. `platform.eval_run(step)` registers the run of the eval of the checkpoint made at `step`
(`NAME-eval-STEP`) and adds it to the runs the episode runner plays.
A channel whose engines are on other machines starts no engine here: its turns are sampled there
(`platform.routes`), and `platform.publish` leaves it to the engine hosts to load what the loop wrote down.
`open(plays=RUNS)` (what `rollout runner` opens) is a runner and nothing else: it registers no run in its directory
and makes no trainer, and its episode runner plays those runs (by id), or every run whose channels it reaches when none
is named. `profile.engines()` makes clients of the servers of the channels whose engines are elsewhere, and
nothing else (what `rollout engines` loads checkpoints into).
Leaving the block stops all of it in reverse, also when starting fails half way. The training loop serves the
run's newest checkpoint (else the one it starts from) on its channel when it starts. `platform.layout` is the trained
channel's `reshard` (a bridge's name); `platform.reshard` runs that bridge on a checkpoint, as a Ray task when the profile
names `ray`, else in this process, its scratch files under `directory/resharding`.

`rollout train` writes the run's directory; `rollout monitor RUN` is a separate process that serves the page over it
and over every other run sharing its ledger (`rollout monitor` also takes the ledger itself: a database's URL or a
ledger's directory). `rollout train --monitor http://HOST:PORT` writes where that page serves into the run's start, so
that a monitor on another machine asks it for the run's episodes ([monitor](../libraries/rollout-train/monitor.md)).
`rollout monitor RUN --cluster` imports environments from git, with the cluster config's blob store and Ray cluster
([importing from git](../libraries/rollout-train/monitor.md#importing-from-git)).

## Runners

Opening a profile starts an [episode runner](../libraries/rollout-train/rollouts.md#a-runner) named
`HOST/DIRECTORY` (this machine's name and the run directory's), with `episodes_at_once` places and the profile's tool
sets and pools. It plays the episodes of the run in its directory, claiming them in the ledger and recording them
there, with their trajectories and events in the blob store; it claims an episode only while the pools of its
sandboxes have room for them, and leases them under the claim. Started again, it takes its fence anew, and what it
had claimed is open to be played again; the sandboxes leased under those claims are deleted by their pools
([a runner started again](../libraries/rollout-train/rollouts.md#a-runner-started-again)).

Runners on other machines share a run's work through the ledger and the blob store alone: a database ledger they
all reach (`postgresql://…`) and a blob store they all reach (an object store), with the channels, tool sets and pools
the run's plan names. `rollout runner PROFILE --run RUN` is such a runner: its profile's channels, tool sets and pools,
no trainer, and the runs it is named (repeatable; none named: every run whose channels it reaches). The run does not
know where its episodes were played.

Each runner beats every 15 seconds ([heartbeats](../libraries/rollout-train/rollouts.md#heartbeats)): its host, its
machine's memory, GPUs and disk, its engines' processes, what each channel serves and how fast, and how full its pools
are. A runner that
stops beating for 90 seconds is taken to be gone, and what it had claimed is played by others. `rollout train` also
writes into the run's start where its blob store is (its kind and settings; a store's credentials come from its
environment and are never written), so a monitor anywhere reads the run's finished episodes back, and shows its
machines and engines from the beats: from the run's machine it reads only the episodes still playing (its feed).

## Engines on other machines

A run's channel can be served by vLLM servers on machines of their own. Three roles meet through the ledger, the
heartbeats beside it and the blob store, wherever each runs:

| Role | Does | Command |
|---|---|---|
| The trainer | writes down, under the run's fence, what each of its channels should serve: the checkpoint, its depth and kind, and the files its engines load ([what a channel should serve](../libraries/rollout-train/channels.md#what-a-channel-should-serve)) | `rollout train` |
| Engine hosts | load what the run says from the blob store into the vLLM servers on their machine, as an adapter named by the checkpoint's id, and beat with what each serves | `rollout engines` |
| Episode runners | ask the channel's servers (a router in front of them, or the servers themselves) for the checkpoint the run says, by name, and record what they sample through the gateway (in their own process, or replicas of its own), so tokens and logprobs are recorded exactly as sampled | `rollout runner`, or the runner `rollout train` opens |

A profile that runs everything in one process needs none of this: its channels' engines are in that process, the loop
publishes to them directly, and no request leaves it.

### The servers

Each engine is a stock vLLM OpenAI-compatible server, started on its machine with the model, LoRA, the logprobs of the
distribution sampled from, and adapters loaded and unloaded while it runs:

```bash
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True uv run vllm serve Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000 \
    --enable-lora --max-lora-rank 32 --max-loras 2 --logprobs-mode processed_logprobs --max-model-len 8192 \
    --api-key "$ROLLOUT_ENGINES_TOKEN"            # (optional: TLS with --ssl-certfile, --ssl-keyfile, --ssl-ca-certs)
```

`--max-loras` must cover the adapters every run it serves keeps loaded: `max_lag + 1` each (2 here, for one run with
`max_lag = 1`).

A request names the checkpoint it samples from as its `model`: the base model by its own name, a LoRA checkpoint by
its id. It completes the prompt's token ids (`/v1/completions` with `return_token_ids` and `logprobs`), so the tokens
and their logprobs come back exactly as sampled, the stop token among them, and the answer names the model that
sampled it. A request carries the session (`session_id`, which a router may keep to one server) and a name of its own
(`request_id`: the effect, the attempt and the phase of the turn). Nothing else of the protocol is ours: any router
(vLLM's production stack, llm-d, SGLang's router, Dynamo), proxy or tunnel can stand in front of the servers.

A vLLM server serves full weights only under the name it was started with, so a full checkpoint (or an adapter over
one) is not served this way: a channel whose run trains every weight keeps its engines in the trainer's process, or
starts a server on the checkpoint's files as a model of its own.

### An engine host

An engine host's profile names the channel with `RemoteEngine` as its engine and the servers on its machine as its
`engines`, the shared ledger and blob store, and a directory for the checkpoints it fetches:

```toml
directory = "~/.cache/rollout/engines"        # the checkpoints it fetches, which the servers here read
[ledger]
kind = "rollout_train.database:DatabaseLedger"
url = "postgresql://trainer@db-1/rollout"
[blobs]
kind = "rollout_s3:S3BlobStore"
bucket = "rollout"

[channels.policy]
model = "Qwen/Qwen3-0.6B"                     # the name the servers serve the base model under
renderer = "rollout_qwen:qwen3"
engine = "rollout_train.inference:RemoteEngine"
engines = [{ address = "http://127.0.0.1:8000" }]   # the servers on this machine
connection = { token_env = "ROLLOUT_ENGINES_TOKEN" }
```

```bash
uv run rollout engines engines.toml --run first --name gpu-1
```

It reads what the run (`--run`, by name or id) says each of its channels (by name) should serve every two seconds, and
when a channel serves something older, fetches the checkpoint's files under its directory (hard links, from a blob
store of files on the same disk) and loads them into each server (`/v1/load_lora_adapter`, named by the checkpoint's
id). The adapters before stay loaded, the run's `max_lag + 1` in all, so that a turn begun under one finishes
under it; older ones are unloaded, and their files deleted. It beats as `--name` (by default this machine's name; kind `engines`) with the run it
follows, its machine, and for each channel what it serves, each server's address, and why a load failed, if one did.

### A channel whose engines are elsewhere

The trainer's and the runners' profiles name the same channel, with the servers a runner reaches:

```toml
directory = "~/.cache/rollout/runs/first"

[channels.policy]
model = "Qwen/Qwen3-0.6B"
renderer = "rollout_qwen:qwen3"               # the gateway renders and records with it
engine = "rollout_train.inference:RemoteEngine"
engines = [{ address = "http://gpu-1:8000" }, { address = "http://gpu-2:8000" }]
via = "https://router.example.com"            # optional: a router or proxy every request goes to instead
max_lag = 1                                   # optional: checkpoints a sample may be behind what the run says
connection = { token_env = "ROLLOUT_ENGINES_TOKEN", ca = "~/.config/rollout/ca.pem" }   # optional
```

| Key | What it says |
|---|---|
| `engines` | the servers, each by its `address`. With no router, a session's turns go to one of them, worked out from the session's id alone, among those that answer and have a checkpoint close enough |
| `via` | a URL every request goes to in place of the servers' addresses: a router, a proxy, a tunnel. Engine hosts still load at the addresses |
| `max_lag` | how many checkpoints behind what the channel should serve a sample may be, where its server does not have the newest yet (1 unless it says otherwise) |
| `connection` | how servers are reached: `token_env` or `token_file`, a bearer token read from that environment variable or file (never written to the ledger or the beats); `ca`, a CA bundle the server's certificate is verified against; `certificate` and `key`, a client certificate for servers that ask for one |

The trainer's process starts no engine for such a channel: the loop writes down each checkpoint it serves, and the
engine hosts load it. An eval its schedule asks for plays its checkpoint and no other.

### How stale a sample may be

Choosing the checkpoint is the runner's: each turn asks for the one the run last wrote down, by name. Where the server
(or the router) does not have it yet, the turn asks for the newest one before it that it has, no more than `max_lag`
checkpoints behind (1: the checkpoint before, which a server serves while it loads the newest); a server that answers
that it does not have a model is asked for the one before, and for the newest again at the next look (every two
seconds). A turn waits while no server has a checkpoint close enough, up to five minutes. Every token is stamped with
the depth of the checkpoint its answer names, and the trainer's importance weight (old / behaviour, truncated at
`truncate`) corrects for the difference, as it does for the turns of an episode that spanned a step on one machine. A
runner claims a run's episodes only while a server has a checkpoint of each of its channels close enough.

### Directly or through a router or proxy

Either layout is a matter of configuration: `engines` alone sends to the servers directly; `via` sends everything to
one URL that passes requests on. What makes either safe:

- **Requests carry what they need.** The checkpoint is the request's `model`; the session and the request's name
  travel with it. Nothing between needs to know more than vLLM's API, or keep a session to a connection.
- **Answers are checked.** The answer names the model that sampled it: the runner refuses one that names another, and
  samples the turn again, so a proxy that sends a request to the wrong model, or answers from a cache, cannot make a
  recorded version wrong. What a server has is judged from its own listing (`/v1/models`) and answers, never from the
  proxy.
- **A retry is safe.** A request sent again through a proxy may be sampled again by the server; the gateway keeps the
  answer it was given, under the turn's effect, once.
- **Authentication is the deployment's.** A bearer token (vLLM's `--api-key`, or the proxy's own) and TLS with a CA
  bundle and a client certificate, read from where `connection` says. A server reached by its address alone is known
  by the identity its certificate carries: `connection = { identity = "spiffe://rollout/pod/NAME", … }` checks that
  URI SAN in the handshake in place of the host name.

### Pods on RunPod

Pods rented on RunPod run an image of their own: `deploy/images/inference` (a stock vLLM server, the follower beside
it, Envoy in front) or `deploy/images/trainer` (the training service a run reaches with
`rollout_train.pods.RemoteTrainer`). Each is reached at a public TCP port over mutual TLS, with certificates from the
cluster's step-ca; `rollout_runpod` holds what renting and admitting them takes (RunPod's API, one-time tokens,
revocation). What is built,
the security model, and how they become provider kinds are in
[RunPod pods as inference and training providers](../research/runpod-providers.md).

### A layout

```
   trainer-1                       gpu-1, gpu-2                             runner-1 … runner-N
   rollout train trainer.toml      vllm serve … (port 8000)                 rollout runner runner.toml
     writes what policy should     rollout engines engines.toml --run first     --run first
     serve (runs/first/serving)    ──▶ loads each checkpoint by id    ◀── asks for the checkpoint by name
                                       into its server, beats              (via a router, or directly), records
                    └───────── ledger (Postgres) · beats beside it · blob store (S3) ─────────┘
```

The trainer's profile names the trainer and the channel with its servers (its own runner can play too, or
`episodes_at_once = 0` leaves the playing to the others); each GPU box runs a vLLM server and `rollout engines` over an
engine host's profile; each runner box runs `rollout runner` over a profile with the channel, the tool sets and the
pools. A channel whose engines are clients of a service elsewhere (a sampler switched to the checkpoint named) can be
served by the runner's own process: `rollout runner --run RUN` keeps the channels whose engines are in its process
following what that run says, as an engine host does.

## The gateway

`rollout gateway PROFILE` serves a replica of the [gateway](../libraries/rollout-train/gateway.md): it samples the
profile's channels for programs and harnesses that hold a signed key, and records every turn in the profile's ledger
and blob store before it replies. Replicas keep no session, so as many as wanted stand behind one proxy, and any of
them may stop at any moment.

- **Keys.** The secrets are read from `[gateway] keys`, else from the environment (`ROLLOUT_GATEWAY_KEYS`, or a file
  named by `ROLLOUT_GATEWAY_KEYS_FILE`), and never from the ledger. Whoever mints keys holds the same secrets.
- **Which checkpoint.** A channel whose engines serve elsewhere (`engine = "rollout_train.inference:RemoteEngine"`,
  with its servers or `via` a router) is sampled per run from what the run says it should serve, within its
  `max_lag` ([engines on other machines](#engines-on-other-machines)); a channel with engines of its own is started in
  the replica.
- **TLS and proxies.** A proxy in front terminates TLS and checks its own credentials; `--proxied` names the addresses
  whose `X-Forwarded-*` headers are trusted. `--certificate` and `--private-key` serve TLS from the replica itself.
- **Health.** `/healthz` answers while the process serves; `/readyz` answers 200 once the ledger and the blob store
  answer, and 503 otherwise.

A run's runner records through the gateway too ([a runner served by the
gateway](../libraries/rollout-train/gateway.md#a-runner-served-by-the-gateway)):

- **With no `url`** (the default), through a gateway in its own process, over the profile's channels and routes: the
  same code a replica runs, with no HTTP in between, recording in the profile's ledger and blob store. It tells the
  monitor's feed of the samples harnesses ask for, and listens for them at `serve`. With no keys given, it signs with
  a secret it makes when it starts.
- **With `url`**, through the replicas there, and the runner starts no engine. A `RemoteEngine` channel is sampled
  on its servers, which the replicas reach too; any other channel is hosted by the replicas, its engines started in
  their processes, and the runner asks them what it guarantees ([a gateway elsewhere that hosts
  channels](../libraries/rollout-train/gateway.md#a-gateway-elsewhere-that-hosts-channels)). A hosted channel serves
  the base model only. The replicas hold the same secrets as the runner (`keys`, or the environment), and share its
  ledger and blob store.

Either way the runner's episodes are assembled from the turns in the ledger, and an episode a runner started again
adopts trains like any other.

## Launchers

A launcher starts the training runs and evals asked for (from the monitor's page, say) on a machine that can run
them. Run one per training machine:

```bash
uv run rollout launcher --ledger "sqlite:///~/.cache/rollout/ledger.db" \
    --profiles environments/minecraft/profiles --environment minecraft_team.environment:environment \
    --runs ~/.cache/rollout/runs --at-once 1
```

| Option | What it is |
|---|---|
| `--ledger` | the database (or a ledger's directory) the launches and heartbeats are kept beside: the profiles' own |
| `--profiles` | a directory of profiles it offers: every `*.toml` there that loads, by its file's name (one that names no trainer, for evals only) |
| `--environment` | an environment it offers, as `module:name` (repeatable) |
| `--runs` | where it makes each run's directory: the run's name in letters, digits and dashes, and the end of the launch's id |
| `--at-once` | how many runs it plays at once: 1 on one GPU |
| `--ray` | a Ray cluster's job server (`http://127.0.0.1:8265`): each run is then a Ray job ([Ray](#ray)) |
| `--gpus` | with `--ray`, the accelerators each run's Ray job asks for (1) |
| `--as-job` | with `--ray`, submit the launcher itself as a Ray job, and return |
| `--name` | what it beats as besides its host (`launcher/HOST/NAME`), where a machine has several launchers: one per project environment, say |
| `--cluster` | the cluster config ([cluster](cluster.md)) whose inference providers' models it offers evals; alone, found as every `--cluster` is |

It beats like a runner, saying what it offers: each profile, with what it launches (`kinds`: `run` and `eval` for a
profile that names a trainer, `eval` alone for one that names none, whose evals play on its first channel), the base
model of that channel, the base models an eval may play with it (`models`: that model, then, with `--cluster`, each
model the cluster's inference providers serve whose kind's implementation is the profile's channel's engine and which
the channel's renderer renders, by the model's name or its `base`: `rollout_qwen:qwen3` offers no Qwen3.5 model, and a
renderer that says nothing of its models offers every one), the kinds of sandbox its pools serve (`pools`) and the
settings a launch may change, with their values in the file (the trainer's settings, `trainer.start`,
`trainer.bookmark`, `episodes_at_once`, each channel's `thinking_tokens` and `answer_tokens`, and `evals.suite`,
`evals.every` and `evals.episodes`, which the monitor's **New run** form asks for as the run's evals; a profile without
a trainer has `episodes_at_once` and the channels' budgets only); its environments; and how many runs it plays. With
`--ray` it offers every environment imported from git too ([writing an environment others can import](publishing.md)),
by `NAME@VERSION`, each with the profiles that have a pool of every kind of sandbox the version's environment declares
(its check records them in its description, `sandboxes`; a version whose description does not say goes with the
profiles that have no pools), listed in each profile's `published`, and submits a run on one as a Ray job in that
version's runtime environment. A profile with pools the environment does not use plays it too: a pool makes no
sandbox until an episode asks for one. A channel's budget left empty in the form, or set to `none`
(`--set channels.policy.thinking_tokens=none`), is no budget, whatever the profile says.
A launch (`rollout_train.launches`) names a profile, an environment, the run's name, the checkpoint it starts from, a
bookmark, `groups`, `groups_per_step`, `seed`, and the settings it changes, by dotted key (any `trainer.` key, one
every training run can change, or one the profile offers). A training run's launch says the evals it makes
(`evals.suite`: a suite, or null for none), unless its profile's `[evals]` says them; the monitor refuses one that says
neither. The launcher claims the oldest launch asked for one of its profiles that the profile launches (an eval, for a
profile without a trainer: the monitor refuses a training run of one), whose environments it plays with that profile,
while it has room (a claim is one change, so two launchers never start one launch), and starts

```bash
python -m rollout_train.cli train PROFILE ENVIRONMENT --directory RUNS/NAME-ID --name NAME --groups G \
    --groups-per-step K --seed S --set KEY=VALUE ...
```

with its output in the run's `train.log` (each setting's value as JSON: a launch that says no evals passes
`--set evals.suite=null`, so the profile's `[evals]` is not used). A launch of kind `eval` names a version of a suite (`NAME@N`), the checkpoint that plays it
(`start`) or the base model (`model`, one the launcher offers with the profile; neither: the profile's), and its
episodes a start (none: the version's), and is started as

```bash
python -m rollout_train.cli eval PROFILE SUITE@N --directory RUNS/NAME-ID --name NAME --episodes N \
    --checkpoint REF --set KEY=VALUE ...     # or, for a base model: --model MODEL
```

([evals](../libraries/rollout-train/evals.md#made-edited-and-asked-for-from-the-page)). It notes how the launch goes: `claimed`, `running` (with the process),
then `ended`, or `failed` with the end of the output. A launch asked to stop before it is claimed is `stopped` at
once; a run going is sent an interrupt and stops as on Ctrl-C (`stopping`, then `stopped`). A launch that resumes a
run (`resumes`: the run, `directory`: its own; made by `rollout resume` or the monitor's **Resume**, never by the New run
form) is started the same way in that directory, which names the run, so it goes on from the ledger
([pausing and resuming](../libraries/rollout-train/training.md#pausing-and-resuming)). Launches are ordinary
state beside the ledger: `launches.json` beside a ledger of files, the `launches` table in a database ledger's
database.

Every change of a launch's state compares and sets, in the store: it is made only if the launch is in the state its
writer expects, and may go where it is sent (`rollout_train.launches.MOVES`):

| From | To |
|---|---|
| `asked` | `claimed`, `stopped` |
| `claimed` | `running`, `stopping`, `stopped`, `ended`, `failed` |
| `running` | `stopping`, `stopped`, `ended`, `failed` |
| `stopping` | `stopped`, `ended`, `failed` |

A finished launch (`ended`, `failed`, `stopped`) goes nowhere. So a stop asked for while the launcher starts the run
stays `stopping` (the launcher's `running` is refused) and the launcher interrupts the run it started; a stop of a
launch the page saw `asked` that a launcher claimed meanwhile becomes `stopping`, not `stopped`. The runs a launcher started go on if the launcher stops. Started again under its name, it follows its Ray
jobs again; a run it started as a process of its own cannot be waited on by another process, so once that process is
gone its launch is noted `ended`, its end unseen. While no launcher of that name beats, the monitor shows its launches
going as `lost`.

## Ray

A launcher with `--ray` submits each run as a Ray job in place of starting a process: Ray places it on a node with
`--gpus` accelerators and one CPU free, queues it until there is one, and supervises it. The job's command is the
same `rollout train` (or `rollout eval`) command, run from the launcher's working directory, so every node needs the same checkout, the
same environment and the same run directories. The launcher follows the job until it ends and writes its output to
the run's `train.log`; the launch notes the job (`job`) in place of a process. Stop stops the job. `--as-job` submits
the launcher itself as a long-lived Ray job and returns; it refuses when a launcher already runs as a Ray job on this
host. A profile with `ray` connects its run to the cluster, and runs each checkpoint's bridge as a Ray task, asking for the
CPUs and memory the bridge declares ([bridges](../libraries/rollout-train/checkpoints.md#bridges)).

Ray comes with `rollout-train` (`uv sync` installs it). On a machine, start a head node, its temporary directory
on disk (Ray writes its sessions and spilled objects there, and `/tmp` may be memory), and stop it with `ray stop`:

```bash
uv run ray start --head --node-ip-address 127.0.0.1 --dashboard-host 127.0.0.1 --num-gpus 1 --temp-dir ~/.cache/ray
uv run rollout launcher --ledger "sqlite:///~/.cache/rollout/ledger.db" --profiles environments/minecraft/profiles \
    --environment minecraft_team.environment:environment --runs ~/.cache/rollout/runs --ray http://127.0.0.1:8265 --as-job
uv run ray job list --address http://127.0.0.1:8265                 # the launcher, and each run it submitted
uv run ray stop
```

Ray's workers run in the cluster's own environment: the run tells Ray not to start them through `uv run`, which would
build each a fresh environment without the extras.

## On Kubernetes

The chart `deploy/chart/rollout` runs the platform in one namespace, by role: the stores (Postgres and an S3 gateway),
a long-lived Ray cluster (KubeRay: a head, and a GPU group and a CPU group the autoscaler starts from zero), the
launchers (each with `--ray`, submitting every run it claims to that cluster as a Ray job), the gateway and the
monitors (Deployments, each behind a Service and an Ingress). Its profiles name the ledger and the blob store at their
addresses in the cluster, and every pod mounts one volume for run directories and other local state. On this machine
it runs in K3s; `deploy/k3s/README.md` installs it (its Volumes section says where the volumes are kept, how they are
kept when their claims go, and to back them up before uninstalling K3s), `deploy/k3s/migrate.sh` copies a machine's
ledger, blobs and run directories into it, and `deploy/k3s/cutover.md` moves the services over.

## Stopping

A run can be paused instead: its process stays, with its engines and GPU, and starts nothing new until it is resumed
(`rollout pause`, `rollout resume`, or the monitor's buttons). A run stopped, failed or lost is resumed by a launch into
its own directory ([pausing and resuming](../libraries/rollout-train/training.md#pausing-and-resuming)).

A run asked to stop (an interrupt, a termination, a hang-up) stops what it started: its episodes, its tool sets, its
engines, a step in progress. One that is killed outright cannot. It leaves its engines' process ids in `engine.json`
in the run's directory, and the next run in that directory ends them before starting its own
([the engine core process](../implementations/rollout-vllm.md#the-engine-core-process)).
