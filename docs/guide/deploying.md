# Deploying

A deployment is described once, in a profile: the channels and the engines behind them, the trainer, the runner, and
where each environment's tool set lives. Whoever trains gets jobs and a trainer from it and never learns what stands
behind them; an environment is named in it only by its tool set. This page is the one place the profile file is
described. The code is `rollout_train.profile`, and the command is `rollout` (`rollout_train.cli`).

```toml
directory = "~/.cache/rollout/runs/first"     # the run's state: adapters, the job's log, metrics, the monitor's feed
runner = "local"                              # or "durable": runs survive this process
serve = "0.0.0.0:8900"                        # optional: rollout jobs and the model endpoint for harnesses, over HTTP
address = "http://trainer-1:8900"             # what others reach it at, if not http://{serve}
feed_runs = 80                                # optional: episodes kept in the monitor's feed

[channels.policy]
model = "cyankiwi/Qwen3.5-9B-AWQ-4bit"
renderer = "rollout_qwen:qwen35"              # the model family's token format
engine = "rollout_vllm:VllmEngine"            # what serves it; each entry of `engines` is one replica's options
engines = [{ gpu_memory_utilization = 0.78, max_model_len = 8192, max_num_seqs = 20 }]
thinking_tokens = 1024
answer_tokens = 400

[trainer]
kind = "rollout_lora:LoraTrainer"             # what trains; the keys below it does not name here are its settings
channel = "policy"                            # the channel whose policy it trains
colocated = true                              # it shares the engines' GPU: they sleep while it steps
rank = 32
learning_rate = 5e-5
sequence_tokens = 8000                        # the longest turn it can train on: its channel takes it as its limit
sequences_per_step = 384

[tools]
minecraft = "minecraft_swarm.worlds:tools"    # made in this process by `tools(directory)`; or "http://worlds:8700"

[memory]
runs_gib = 6                                  # must be available to admit runs
training_gib = 4                              # and to start a step
```

```bash
uv run rollout train profile.toml minecraft_swarm.catalog:catalog --groups 100 --directory RUN
uv run rollout monitor RUN/feed                # the web page over the run: http://localhost:8765
uv run rollout report RUN minecraft_swarm.catalog:catalog --watch
uv run rollout tools minecraft_swarm.worlds:tools --directory DATA --port 8700   # a tool set on a machine of its own
```

`rollout COMMAND --help` lists each command's options. A catalog is named as `module:name`, like everything else a
profile or the command is told by name.

## The file

A key the profile does not have is an error, so a misspelt guard is never silently no guard.

| Part | What it decides | To scale it |
|---|---|---|
| `directory` | Where the run's state is kept. `rollout train --directory` replaces it: one profile, many runs | |
| `channels` | Which model each policy is, its token format, what serves it, and how much it may think and answer (`thinking_tokens`, `answer_tokens`, in place of [`Limits`](reference.md#limits)' own) | Add entries to `engines`, each with its own options (a device, an address): sessions spread over them, each staying with one |
| `trainer` | What trains which channel, and its settings. The longest sequence it can train on becomes that channel's longest turn | `colocated = false` when it has an accelerator of its own: engines then serve through a step |
| `runner` | `local` runs episodes in this process; `durable` records them so that they survive it ([durable runner](../implementations/rollout-durable/README.md)) | |
| `serve`, `address` | Where the rollout service and the [model endpoint for harnesses](../libraries/rollout-train/harness-endpoint.md) listen, and the URL others reach them at | A training loop elsewhere connects with `RolloutClient(url)` |
| `tools` | Each tool set an environment imports by name: `module:name` of what makes it in this process, or a URL | Run `rollout tools` where the environment's servers should live |
| `memory` | System memory that must be available before runs are admitted (`runs_gib`) and before a colocated step starts (`training_gib`); short of it the run stops with `NotEnoughMemory` rather than exhaust its machine | |
| `feed_runs` | How many episodes the [monitor](../libraries/rollout-train/monitor.md)'s feed keeps | |

## What a profile names

Engines, renderers, trainers and tool sets are implementations, named as `module:name`. `rollout_train.profile`
imports none of them: it calls what the name resolves to, and passes it what the profile says of it. Their defaults
are their own.

| Key | Called with | Implementations in this repository |
|---|---|---|
| `engine` of a channel | The channel's `model`, and one entry of `engines` as keyword arguments. Once per entry | `rollout_vllm:VllmEngine` ([vLLM engine](../implementations/rollout-vllm.md)) |
| `renderer` of a channel | The channel's `model` | `rollout_qwen:qwen35`, `rollout_qwen:qwen3` ([Qwen renderers](../implementations/rollout-qwen.md)) |
| `kind` of the trainer | The trained channel's `model`, the run's directory, and every other key of `[trainer]` except `channel` and `colocated` as keyword arguments | `rollout_lora:LoraTrainer` ([LoRA trainer](../implementations/rollout-lora.md)) |
| An entry of `tools` | The run's directory | An environment's own, such as `minecraft_swarm.worlds:tools` ([Minecraft swarm](../products/minecraft-swarm.md)) |

`rollout_train.testing` has a scripted engine and a readable renderer for profiles that need no GPU
(`rollout_train.testing:scripted_engine`, `rollout_train.testing:plain_renderer`). The three GPU packages are
installed with `uv sync --all-extras`.

## Opening a profile

In code, a profile opens into a platform:

```python fragment
async with Profile.load(Path("profile.toml")).open() as platform:
    binding = binding_for(catalog, "policy", platform.tool_bindings)
    await train(platform.jobs, catalog, platform.trainer, platform.store, channel="policy", binding=binding)
```

Opening starts, in order: the trainer; each channel's engines; the channels, the trained one with the trainer's
longest sequence as its longest turn and the trainer's latest adapter published to it; the recorder; the monitor's
feed in `directory/feed`; the tool sets; the runner; the rollout jobs. A colocated trainer is wrapped in
[`Colocated`](reference.md#colocated). With `serve`, the rollout service and the endpoint for harnesses listen there.
Leaving the block stops all of it in reverse, also when starting fails half way.

`rollout train` writes the feed; `rollout monitor RUN/feed` is a separate process that serves the page over it.

## Stopping

A run asked to stop (an interrupt, a termination, a hang-up) stops what it started: its episodes, its tool sets, its
engines, a step in progress. One that is killed outright cannot. It leaves its engines' process ids in `engine.json`
in the run's directory, and the next run in that directory ends them before starting its own
([the engine core process](../implementations/rollout-vllm.md#the-engine-core-process)).
