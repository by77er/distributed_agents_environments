# Deploying

A deployment is described once, in a profile: the channels and the engines behind them, the trainer, the runner, and
where each environment's tool set lives. Whoever trains gets a trainer, the versions and a way to publish versions
from it, while a runner it opens plays the episodes the run asks for, and never learns what stands behind them; an
environment is named in it only by its tool set. This page is the one place the profile file is
described. The code is `rollout_train.profile`, and the command is `rollout` (`rollout_train.cli`).

```toml
directory = "~/.cache/rollout/runs/first"     # the run's state: versions in use, metrics, the monitor's feed
runner = "local"                              # or "durable": runs survive this process
serve = "0.0.0.0:8900"                        # optional: the model endpoint for harnesses, over HTTP
address = "http://trainer-1:8900"             # what others reach it at, if not http://{serve}
feed_runs = 80                                # optional: episodes kept in the monitor's feed
episodes_at_once = 6                          # optional: the most episodes this machine's runner plays at once

[channels.policy]
model = "cyankiwi/Qwen3.5-9B-AWQ-4bit"
renderer = "rollout_qwen:qwen35"              # the model family's token format
engine = "rollout_vllm:VllmEngine"            # what serves it; each entry of `engines` is one replica's options
engines = [{ gpu_memory_utilization = 0.78, max_model_len = 8192, max_num_seqs = 20 }]
thinking_tokens = 1024
answer_tokens = 400

[trainer]
kind = "rollout_lora:LoraTrainer"             # what trains; the keys below it does not name here are its settings
channel = "policy"                            # the channel that serves what it trains
start = "diamonds"                            # optional: the version a new run trains from (by default the base model)
bookmark = "diamonds, unguided"               # optional: a bookmark the run carries to each version it makes
colocated = true                              # it shares the engines' GPU: they sleep while it steps
rank = 32
learning_rate = 5e-5
segment_tokens = 8000                        # the longest turn it can train on: its channel takes it as its limit
segments_per_step = 384

[tools]
minecraft = "minecraft_team.worlds:tools"    # made in this process by `tools(directory)`; or "http://worlds:8700"

[memory]
runs_gib = 6                                  # must be available to admit runs
training_gib = 4                              # and to start a step
```

```bash
uv run rollout train profile.toml minecraft_team.catalog:catalog --groups 100 --directory RUN  # --groups-per-step 4
uv run rollout monitor RUN                     # the web page over RUN's ledger and its runs: http://localhost:8765
uv run rollout report RUN minecraft_team.catalog:catalog --watch   # charts; posted to DISCORD_WEBHOOK_URL if set
uv run rollout imitate profile.toml --directory RUN                  # a supervised step on solved, guided episodes
uv run rollout versions --ledger RUN                                 # every version: where it came from, its bookmarks
uv run rollout bookmark diamonds RUN:20 --ledger RUN                 # name a version (or move the bookmark there)
uv run rollout rename RUN "diamonds, unguided" --ledger RUN          # call a run something else (its id stays)
uv run rollout tools minecraft_team.worlds:tools --directory DATA --port 8700   # a tool set on a machine of its own
```

`rollout COMMAND --help` lists each command's options. A catalog is named as `module:name`, like everything else a
profile or the command is told by name. `train --name NAME` names a new run (by default after its directory);
`rename` names a run again, by its name or its id; `bookmark` names a version by any reference, and `versions` lists
them all ([versions, runs and the ledger](../libraries/rollout-train/versions.md#the-command-line)). `train` plays `--groups` groups and takes a step whenever `--groups-per-step`
of them have something to train on ([training](../libraries/rollout-train/training.md#the-loop)); `report` and
`imitate` are described in [reporting](../libraries/rollout-train/training.md#reporting) and
[imitation](../libraries/rollout-train/training.md#imitation).

## The file

A key the profile does not have is an error, so a misspelt guard is never silently no guard.

| Part | What it decides | To scale it |
|---|---|---|
| `directory` | Where the run's state is kept. `rollout train --directory` replaces it: one profile, many runs | |
| `channels` | Which model each channel serves, its token format, what serves it, and how much it may think and answer (`thinking_tokens`, `answer_tokens`, in place of [`Limits`](reference.md#limits)' own) | Add entries to `engines`, each with its own options (a device, an address): sessions spread over them, each staying with one |
| `trainer` | What trains which channel, and its settings. The longest segment it can train on becomes that channel's longest turn. `start` is the version a new run trains from, by any [reference](../libraries/rollout-train/versions.md#references) (by default the base model: the channel's `model`); a run started again goes on from its own newest version. `bookmark` names a bookmark the run moves to each version it makes | `colocated = false` when it has an accelerator of its own: engines then serve through a step |
| `runner` | `local` runs episodes in this process; `durable` records them so that they survive it ([durable runner](../implementations/rollout-durable/README.md)) | |
| `serve`, `address` | Where the [model endpoint for harnesses](../libraries/rollout-train/harness-endpoint.md) listens, and the URL others reach it at | |
| `tools` | Each tool set an environment imports by name: `module:name` of what makes it in this process, or a URL | Run `rollout tools` where the environment's servers should live |
| `ledger` | Where the run's tables and the versions are kept ([the ledger](../libraries/rollout-train/versions.md#the-ledger)): a directory (`ledger = "path"`), or a table naming a ledger (`[ledger]` with `kind = "rollout_train.database:DatabaseLedger"` and a `url`: `sqlite:///~/…` on one machine, `postgresql://…` for several; `rollout ledger copy` moves one to the other). Without it, `directory/ledger`. The registry of runs' names and bookmarks is kept beside it | Runs that share a ledger and a blob store share one graph of versions, and can start from each other's |
| `blobs` | Where episodes, each step's batch and what each step left behind are kept. Without it, files under `directory/blobs`. With `kind = "module:name"`, the store that makes, called with the table's other entries (`rollout_s3:S3BlobStore`, say) | Point it at an object store that the machines share |
| `memory` | System memory that must be available before the runner claims another episode (`runs_gib`: short of it, it waits) and before a colocated step starts (`training_gib`: short of it, the step stops with `NotEnoughMemory`) rather than exhaust its machine | |
| `feed_runs` | How many episodes the [monitor](../libraries/rollout-train/monitor.md)'s feed keeps | |
| `episodes_at_once` | How many episodes the run keeps work waiting for, and the places of this machine's runner: the most it plays at once, whatever groups they are of (6 unless it says otherwise), what the engines and the memory for the programs' worlds can take | Raise it with the engines' `max_num_seqs` and the machine's memory |

## What a profile names

Engines, renderers, trainers and tool sets are implementations, named as `module:name`. `rollout_train.profile`
imports none of them: it calls what the name resolves to, and passes it what the profile says of it. Their defaults
are their own.

| Key | Called with | Implementations in this repository |
|---|---|---|
| `engine` of a channel | The channel's `model`, and one entry of `engines` as keyword arguments. Once per entry | `rollout_vllm:VllmEngine` ([vLLM engine](../implementations/rollout-vllm.md)) |
| `renderer` of a channel | The channel's `model` | `rollout_qwen:qwen35`, `rollout_qwen:qwen3` ([Qwen renderers](../implementations/rollout-qwen.md)), `rollout_gemma:gemma4` ([Gemma renderers](../implementations/rollout-gemma.md)) |
| `kind` of the trainer | The trained channel's `model`, and every other key of `[trainer]` except `channel`, `start`, `bookmark` and `colocated` as keyword arguments | `rollout_lora:LoraTrainer` ([LoRA trainer](../implementations/rollout-lora.md)) |
| An entry of `tools` | The run's directory | An environment's own, such as `minecraft_team.worlds:tools` ([Minecraft team](../products/minecraft-team.md)) |

`rollout_train.testing` has a scripted engine and a readable renderer for profiles that need no GPU
(`rollout_train.testing:scripted_engine`, `rollout_train.testing:plain_renderer`). The GPU packages are installed
with `uv sync --all-extras`.

## Opening a profile

In code, a profile opens into a platform:

```python fragment
async with Profile.load(Path("profile.toml")).open() as platform:
    binding = binding_for(catalog, "policy", platform.tool_bindings)
    await train(
        catalog, platform.trainer, platform.versions, start=platform.origin, channel="policy",
        directory=platform.profile.directory / "versions", publish=platform.publish, binding=binding,
        run=platform.run.id, hooks=[platform.feed], kept=platform.bookmarked, made=platform.made,
    )
```

Opening starts, in order: the trainer; each channel's engines; the channels, the trained one with the trainer's
longest segment as its longest turn; the recorder; the monitor's feed in `directory/feed`; the tool sets; the blob
store and the versions; the run (registered the first time: `run.json`) and the version it starts from; the
runner, and the [episode runner](../libraries/rollout-train/rollouts.md#a-runner) over it. A colocated trainer is wrapped in [`Colocated`](reference.md#colocated). With `serve`, the endpoint for harnesses
listens there.
Leaving the block stops all of it in reverse, also when starting fails half way. The training loop serves the
run's newest version (else the one it starts from) on its channel when it starts.

`rollout train` writes the run's directory; `rollout monitor RUN` is a separate process that serves the page over it
and over every other run sharing its ledger (`rollout monitor` also takes the ledger itself: a database's URL or a
ledger's directory). `rollout train --monitor http://HOST:PORT` writes where that page serves into the run's start, so
that a monitor on another machine asks it for the run's episodes ([monitor](../libraries/rollout-train/monitor.md)).

## Runners

Opening a profile starts an [episode runner](../libraries/rollout-train/rollouts.md#a-runner) named
`HOST/DIRECTORY` (this machine's name and the run directory's), with `episodes_at_once` places and the profile's tool
sets. It plays the episodes of the run in its directory, claiming them in the ledger and recording them
there, with their trajectories and events in the blob store. Started again, it takes its fence anew, and what it had
claimed is open to be played again.

Runners on other machines share a run's work through the ledger and the blob store alone: a database ledger they
all reach (`postgresql://…`) and a blob store they all reach (an object store), with the channels and tool sets the
run's plan names. The run does not know where its episodes were played.

## Stopping

A run asked to stop (an interrupt, a termination, a hang-up) stops what it started: its episodes, its tool sets, its
engines, a step in progress. One that is killed outright cannot. It leaves its engines' process ids in `engine.json`
in the run's directory, and the next run in that directory ends them before starting its own
([the engine core process](../implementations/rollout-vllm.md#the-engine-core-process)).
