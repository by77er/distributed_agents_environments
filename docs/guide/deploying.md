# Deploying

A deployment is described once, in a profile: the channels and the engines behind them, the trainer, the runner, and
where each environment's tool set lives. Whoever trains gets jobs and a trainer from it and never learns what stands
behind them; an environment is named in it only by its tool set.

```toml
directory = "~/.cache/rollout/runs/first"     # the run's state: adapters, the job's log, metrics, the monitor's feed
runner = "local"                              # or "durable": runs survive this process
serve = "0.0.0.0:8900"                        # optional: rollout jobs and the model endpoint for harnesses, over HTTP
address = "http://trainer-1:8900"             # what others reach it at, if not http://{serve}

[channels.policy]
model = "cyankiwi/Qwen3.5-9B-AWQ-4bit"
renderer = "rollout_qwen:qwen35"   # the model family's token format
engine = "rollout_vllm:VllmEngine"     # what serves it; each entry of `engines` is one replica's options
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
rollout train profile.toml minecraft_swarm.catalog:catalog --groups 100 --directory RUN
rollout monitor RUN/feed                       # the web page over the run: http://localhost:8765
rollout report RUN minecraft_swarm.catalog:catalog --watch
rollout tools minecraft_swarm.worlds:tools --directory DATA --port 8700   # a tool set on a machine of its own
```

| Part | What it decides | To scale it |
|---|---|---|
| `channels` | Which model each policy is, its token format, what serves it, and how much it may think and answer | Add entries to `engines`, each with its own options (a device, an address): sessions spread over them, each staying with one |
| `trainer` | What trains which channel, and its settings. The longest sequence it can train on becomes that channel's longest turn | `colocated = false` when it has an accelerator of its own: engines then serve through a step |
| `runner` | `local` runs episodes in this process; `durable` records them so that they survive it | |
| `serve`, `address` | Where the rollout service and the [model endpoint for harnesses](../core/recorder/session-api.md) listen, and the URL others reach them at | A training loop elsewhere connects with `RolloutClient(url)` |
| `tools` | Each tool set an environment imports by name: `module:name` of what makes it in this process, or a URL | Run `rollout tools` where the environment's servers should live |
| `memory` | What must be free before runs are admitted and before a step starts; a run stops rather than exhaust its machine | |

Engines, renderers and trainers are named as `module:name`, and what the profile says of each is passed to it: an
engine is called with the model and its entry of `engines`, a renderer with the model, a trainer with the model, the
directory and its settings. Their defaults are their own (`VllmEngine`, `LoraSettings` in the
[reference](reference.md)). A key the profile does not have is an error.

In code, a profile opens into a platform:

```python fragment
async with Profile.load(Path("profile.toml")).open() as platform:
    binding = binding_for(catalog, "policy", platform.tool_bindings)
    await train(platform.jobs, catalog, platform.trainer, platform.store, channel="policy", binding=binding)
```

A run asked to stop (an interrupt, a termination, a hang-up) stops what it started: its episodes, its engines, a
step in progress. One that is killed outright leaves its engines' process ids in `engine.json`; the next run in that
directory ends them before starting its own.
