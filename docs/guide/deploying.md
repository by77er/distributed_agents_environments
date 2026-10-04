# Deploying

A deployment is described once, in a profile: the channels and the engines behind them, the trainer, the runner, and
where each environment's tool set lives. Whoever trains gets a trainer, the checkpoints and a way to publish checkpoints
from it, while a runner it opens plays the episodes the run asks for, and never learns what stands behind them; an
environment is named in it only by its tool set. This page is the one place the profile file is
described. The code is `rollout_train.profile`, and the command is `rollout` (`rollout_train.cli`).

```toml
directory = "~/.cache/rollout/runs/first"     # the run's own: run.json, checkpoints in use, the monitor's feed
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
start = "diamonds"                            # optional: the checkpoint a new run trains from (by default the base model)
bookmark = "diamonds, unguided"               # optional: a bookmark the run carries to each checkpoint it makes
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

[evals]
suite = "words-held-out"                      # optional: evaluate checkpoints as they are made (its eval data, say)
every = 2                                     # the checkpoint of every second step
episodes = 1                                  # episodes of each start
```

```bash
uv run rollout train profile.toml minecraft_team.environment:environment --groups 100 --directory RUN  # --groups-per-step 4
uv run rollout monitor RUN                     # the web page over RUN's ledger and its runs: http://localhost:8765
uv run rollout report RUN minecraft_team.environment:environment --watch   # charts; posted to DISCORD_WEBHOOK_URL if set
uv run rollout imitate profile.toml --directory RUN                  # a supervised step on solved, guided episodes
uv run rollout checkpoints --ledger RUN                                 # every checkpoint: where it came from, its bookmarks
uv run rollout bookmark diamonds first:20 --ledger RUN               # name the checkpoint run "first" made at step 20
uv run rollout rename first "diamonds, unguided" --ledger RUN        # call a run something else (its id stays)
uv run rollout tools minecraft_team.worlds:tools --directory DATA --port 8700   # a tool set on a machine of its own
uv run rollout train profile.toml ENVIRONMENT --set trainer.learning_rate=3e-5 --set trainer.start=diamonds  # change settings
uv run rollout env check ENVIRONMENT --profile profile.toml --groups 4   # does it hold together; do its groups teach
uv run rollout suite make words-v1 --environment ENVIRONMENT --seeds 1,2,3 --ledger RUN       # a frozen list of starts
uv run rollout eval profile.toml words-v1 --checkpoint diamonds --episodes 4         # play it with a checkpoint
uv run rollout eval profile.toml teams-every-task --environment ENVIRONMENT     # its eval data, frozen on first use
uv run rollout launcher --ledger URL --profiles PROFILES --environment ENVIRONMENT --runs RUNS   # start runs asked for here
uv run rollout launcher --ledger URL --profiles PROFILES --environment ENVIRONMENT --runs RUNS \
    --ray http://127.0.0.1:8265 --as-job                             # the same, as a Ray job; each run a Ray job too
```

`rollout COMMAND --help` lists each command's options. An environment is named as `module:name`, like everything else a
profile or the command is told by name. `train --name NAME` names a new run (by default after its directory);
`rename` names a run again, by its name or its id; `bookmark` names a checkpoint by any reference, and `checkpoints` lists
them all ([checkpoints, runs and the ledger](../libraries/rollout-train/checkpoints.md#the-command-line)). `train` plays `--groups` groups and takes a step whenever `--groups-per-step`
of them have something to train on ([training](../libraries/rollout-train/training.md#the-loop)). `suite` makes and
lists suites, and `eval` plays one with a checkpoint, training nothing ([evals](../libraries/rollout-train/evals.md)); `env check`
checks an environment before anything trains on it, and with a profile plays a few groups and flags those that teach
nothing ([checking an environment](../libraries/rollout-train/rollouts.md#checking-an-environment)); `report` and
`imitate` are described in [reporting](../libraries/rollout-train/training.md#reporting) and
[imitation](../libraries/rollout-train/training.md#imitation).

## The file

A key the profile does not have is an error, so a misspelt guard is never silently no guard.

| Part | What it decides | To scale it |
|---|---|---|
| `directory` | Where the run's state is kept. `rollout train --directory` replaces it: one profile, many runs | |
| `channels` | Which model each channel serves, its token format, what serves it, and how much it may think and answer (`thinking_tokens`, `answer_tokens`, in place of [`Limits`](reference.md#limits)' own). `reshard` names the layout its engines load a checkpoint's files in (`module:name`, such as `rollout_train.resharding:verbatim`): each checkpoint is then [resharded](../libraries/rollout-train/checkpoints.md#resharding) before it is served. Without it, the engines load the trainer's files as they are | Add entries to `engines`, each with its own options (a device, an address): sessions spread over them, each staying with one |
| `trainer` | What trains which channel, and its settings. The longest segment it can train on becomes that channel's longest turn. `start` is the checkpoint a new run trains from, by any [reference](../libraries/rollout-train/checkpoints.md#references) (by default the base model: the channel's `model`); a run started again goes on from its own newest checkpoint. `bookmark` names a bookmark the run moves to each checkpoint it makes | `colocated = false` when it has an accelerator of its own: engines then serve through a step |
| `ray` | A Ray cluster the run connects to (`ray = "auto"`: the one this machine is part of, or `ray://host:port`): its reshards then run as Ray tasks on that cluster ([Ray](#ray)). Without it, they run in the run's process | Add nodes to the cluster |
| `runner` | `local` runs episodes in this process; `durable` records them so that they survive it ([durable runner](../implementations/rollout-durable/README.md)) | |
| `serve`, `address` | Where the [model endpoint for harnesses](../libraries/rollout-train/harness-endpoint.md) listens, and the URL others reach it at | |
| `tools` | Each tool set an environment imports by name: `module:name` of what makes it in this process, or a URL | Run `rollout tools` where the environment's servers should live |
| `ledger` | Where the run's tables and the checkpoints are kept ([the ledger](../libraries/rollout-train/checkpoints.md#the-ledger)): a directory (`ledger = "path"`), or a table naming a ledger (`[ledger]` with `kind = "rollout_train.database:DatabaseLedger"` and a `url`: `sqlite:///~/…` on one machine, `postgresql://…` for several; `rollout ledger copy` moves one to the other). Without it, `directory/ledger`. Beside it are kept, as ordinary state changed in place: the registry of runs' names and bookmarks, the runners' heartbeats, and the launches | Runs that share a ledger and a blob store share one graph of checkpoints, and can start from each other's |
| `blobs` | Where episodes, each step's batch and what each step left behind are kept. Without it, files under `directory/blobs`. With `kind = "module:name"`, the store that makes, called with the table's other entries (`rollout_s3:S3BlobStore`, say) | Point it at an object store that the machines share |
| `memory` | System memory that must be available before the runner claims another episode (`runs_gib`: short of it, it waits) and before a colocated step starts (`training_gib`: short of it, the run stops with `NotEnoughMemory`, before the step) rather than exhaust its machine | |
| `feed_runs` | How many episodes the [monitor](../libraries/rollout-train/monitor.md)'s feed keeps | |
| `evals` | A suite the run plays with the checkpoint of every `every`th step (1 unless it says otherwise), `episodes` episodes of each start (1), between that step and the next, on the trained channel ([evals during training](../libraries/rollout-train/evals.md#evals-during-training)). Each is an eval of its own, a run named `NAME-eval-STEP`. Without it, the run evaluates nothing | Ask for evals as launches instead, so that they run on engines of their own |
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

```py
async with Profile.load(Path("profile.toml")).open() as platform:
    binding = binding_for(environment, "policy", platform.tool_bindings)
    await train(
        environment, platform.trainer, platform.checkpoints, start=platform.origin, channel="policy",
        base=platform.profile.channels["policy"].model,
        directory=platform.profile.directory / "checkpoints", publish=platform.publish, binding=binding,
        run=platform.run.id, hooks=[platform.feed], kept=platform.bookmarked, made=platform.made,
        reshard=platform.reshard if platform.layout else None,
    )
```

Opening starts, in order: the run (registered the first time: `run.json`) and the checkpoint it starts from; with
`ray`, the connection to the Ray cluster; the engines a killed process left behind are ended (`engine.json`); the trainer; each channel's engines; the channels,
the trained one with the trainer's longest segment as its longest turn; the recorder; the monitor's feed in
`directory/feed`; the tool sets; the blob store and the checkpoints; the runner, and the
[episode runner](../libraries/rollout-train/rollouts.md#a-runner) over it. A colocated trainer is wrapped in [`Colocated`](reference.md#colocated). With `serve`, the endpoint for harnesses
listens there.
`open(training=False)` (what `rollout eval` opens) makes no trainer; the trained channel's engines still load what its
`start` is served over. `platform.eval_run(step)` registers the run of the eval of the checkpoint made at `step`
(`NAME-eval-STEP`) and adds it to the runs the episode runner plays.
Leaving the block stops all of it in reverse, also when starting fails half way. The training loop serves the
run's newest checkpoint (else the one it starts from) on its channel when it starts. `platform.layout` is the trained
channel's `reshard`; `platform.reshard` reshards a checkpoint into it, as a Ray task when the profile names `ray`, else in
this process, its scratch files under `directory/resharding`.

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

Each runner beats every 15 seconds ([heartbeats](../libraries/rollout-train/rollouts.md#heartbeats)): its host, its
machine's memory, GPUs and disk, its engines' processes, and what each channel serves and how fast. A runner that
stops beating for 90 seconds is taken to be gone, and what it had claimed is played by others. `rollout train` also
writes into the run's start where its blob store is (its kind and settings; a store's credentials come from its
environment and are never written), so a monitor anywhere reads the run's finished episodes back, and shows its
machines and engines from the beats: from the run's machine it reads only the episodes still playing (its feed).

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
| `--profiles` | a directory of profiles it offers: every `*.toml` there that loads and names a trainer, by its file's name |
| `--environment` | an environment it offers, as `module:name` (repeatable) |
| `--runs` | where it makes each run's directory: the run's name in letters, digits and dashes, and the end of the launch's id |
| `--at-once` | how many runs it plays at once: 1 on one GPU |
| `--ray` | a Ray cluster's job server (`http://127.0.0.1:8265`): each run is then a Ray job ([Ray](#ray)) |
| `--gpus` | with `--ray`, the accelerators each run's Ray job asks for (1) |
| `--as-job` | with `--ray`, submit the launcher itself as a Ray job, and return |

It beats like a runner, saying what it offers: each profile, with the base model it trains and the settings a launch
may change, with their values in the file (the trainer's settings, `trainer.start`, `trainer.bookmark`,
`episodes_at_once`, each channel's `thinking_tokens` and `answer_tokens`, and `evals.suite`, `evals.every` and
`evals.episodes`); its environments; and how many runs it plays.
A launch (`rollout_train.launches`) names a profile, an environment, the run's name, the checkpoint it starts from, a
bookmark, `groups`, `groups_per_step`, `seed`, and the settings it changes, by dotted key (any `trainer.` key, or one
the profile offers). The launcher claims the oldest launch asked for one of its profiles while it has room (a claim
is one change, so two launchers never start one launch), and starts

```bash
python -m rollout_train.cli train PROFILE ENVIRONMENT --directory RUNS/NAME-ID --name NAME --groups G \
    --groups-per-step K --seed S --set KEY=VALUE ...
```

with its output in the run's `train.log`. A launch of kind `eval` names a suite, the checkpoint that plays it
(`start`) and its episodes a start, and is started as

```bash
python -m rollout_train.cli eval PROFILE SUITE --directory RUNS/NAME-ID --name NAME --episodes N \
    --checkpoint REF --set KEY=VALUE ...
```

([evals](../libraries/rollout-train/evals.md#asked-for-from-the-page)). It notes how the launch goes: `claimed`, `running` (with the process),
then `ended`, or `failed` with the end of the output. A launch asked to stop before it is claimed is `stopped` at
once; a run going is sent an interrupt and stops as on Ctrl-C (`stopping`, then `stopped`). Launches are ordinary
state beside the ledger: `launches.json` beside a ledger of files, the `launches` table in a database ledger's
database. The runs a launcher started go on if the launcher stops. Started again under its name, it follows its Ray
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
host. A profile with `ray` connects its run to the cluster, and runs each checkpoint's reshard as a Ray task of one CPU
([resharding](../libraries/rollout-train/checkpoints.md#resharding)).

Ray is the `ray` extra (`uv sync --all-extras` installs it). On a machine, start a head node, its temporary directory
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

## Stopping

A run asked to stop (an interrupt, a termination, a hang-up) stops what it started: its episodes, its tool sets, its
engines, a step in progress. One that is killed outright cannot. It leaves its engines' process ids in `engine.json`
in the run's directory, and the next run in that directory ends them before starting its own
([the engine core process](../implementations/rollout-vllm.md#the-engine-core-process)).
