# Deploying

A deployment is described once, in a profile: the channels and the engines behind them, the trainer, the runner, and
where each environment's tool set lives. Whoever trains gets jobs and a trainer from it and never learns what stands
behind them; an environment is named in it only by its tool set.

```toml
directory = "~/.cache/rollout/runs/first"     # the run's state: adapters, the job's log, metrics, the monitor's feed
runner = "local"                              # or "durable": runs survive this process
serve = "127.0.0.1:8900"                      # optional: rollout jobs and the model endpoint for harnesses, over HTTP

[channels.policy]
model = "cyankiwi/Qwen3.5-9B-AWQ-4bit"
renderer = "qwen3.5"
thinking_tokens = 1024
answer_tokens = 400
engines = [{ gpu_share = 0.78, max_model_len = 8192, concurrency = 32 }]   # one entry per replica

[trainer]
rank = 32
learning_rate = 5e-5
sequence_tokens = 8000                        # the longest turn it can train on: the channels take it as their limit
sequences_per_step = 384
colocated = true                              # it shares the engines' GPU: they sleep while it steps

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
| `channels` | Which model each policy is, its token format, how much it may think and answer, and its replicas | Add entries to `engines`: sessions spread over them, each staying with one |
| `trainer` | The adapter's rank, the learning rate, and what a step can afford. Its longest sequence becomes the channels' longest turn | `colocated = false` when it has an accelerator of its own: engines then serve through a step |
| `runner` | `local` runs episodes in this process; `durable` records them so that they survive it | |
| `serve` | An address for the rollout service and the [model endpoint for harnesses](../core/recorder/session-api.md) | A training loop elsewhere connects with `RolloutClient(url)` |
| `tools` | Each tool set an environment imports by name: a factory called in this process, or a URL | Run `rollout tools` where the environment's servers should live |
| `memory` | What must be free before runs are admitted and before a step starts; a run stops rather than exhaust its machine | |

In code, a profile opens into a platform:

```python fragment
async with Profile.load(Path("profile.toml")).open() as platform:
    await platform.train(catalog, groups=100)      # or use platform.jobs and platform.trainer directly
```

A run asked to stop (an interrupt, a termination, a hang-up) stops what it started: its episodes, its engines, a
step in progress. One that is killed outright leaves its engines' process ids in `engine.json`; the next run in that
directory ends them before starting its own.
