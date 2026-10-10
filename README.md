# rollout

rollout is a reinforcement learning platform for language-model agents. It trains them on long, multi-turn,
multi-agent episodes, such as four bots sharing a Minecraft world, a gridworld or a math problem. Episodes keep playing
while the policy trains, and each new checkpoint is served to the agents during the run. The same code runs on a
single 16 GB GPU and on a Kubernetes cluster.

## An example turn

This is what an agent sees on turn 270 of `iron-underground-40m`, a task in which two agents have 40 minutes of game
time to end up holding as much iron as they can:

```text
Time left: 17.4 of 40 minutes of game time; 211 of 480 turns.

You are ada, at (-37, 14, 85) (overworld, dripstone_caves, night, no sky above); health 18/20, food 15/20.
Inventory: 21 cobblestone, 6 raw_iron, 1 stone_pickaxe, 9 torch. Holding stone_pickaxe.
Notable in sight: iron_ore at (-40, 12, 88), 4 away (3 in sight; also (-41, 12, 88), (-41, 11, 89)).
Teammates in sight: ben at (-33, 15, 80).
Hostile: zombie (id 412) 11 away.
Team chat, oldest first:
- ben, 2 turns ago: furnace is up by the entrance, bring me raw iron
- you, 1 turn ago: 6 so far, more ore here
```

ada replies with a tool call, `mine {"x": -40, "y": 12, "z": 88}`. The world stays frozen while every agent thinks,
then runs while the actions happen. When the time is up, every agent on the team gets the reward `log(1 + iron)`.
The count has no cap, and the same task exists with budgets of 5, 10, 20, 40, 80 and 160 minutes, so results show
how the amount grows with time. These tasks are described in [Minecraft horizons](docs/products/minecraft-horizons.md).

## Features

- **One gateway for every agent loop.** Agents sample through a gateway that speaks OpenAI Chat Completions, OpenAI
  Responses and Anthropic Messages, and records every token and logprob. Claude Code, Codex or your own loop can
  produce training data unchanged.
- **Asynchronous, off-policy training.** Episodes keep playing while the trainer steps. Each turn records the
  checkpoint that sampled it, `max_lag` bounds how stale a turn can be (you can change it mid-run), and importance
  weights correct for the rest. Whole multi-turn segments train together.
- **Training and inference backends.** Train with [Tinker](https://thinkingmachines.ai/tinker/), or with a LoRA or
  full-weight trainer on your own GPUs, and serve on [vLLM](https://github.com/vllm-project/vllm). Each checkpoint is
  converted for each inference backend.
- **Shared sandboxes.** Sandboxes such as a Minecraft server or a computer are leased by claim, so four agents on four
  machines play in the same world.
- **Recovery from crashes.** Every process coordinates through a fenced, append-only ledger in Postgres. If a runner,
  server or trainer is killed mid-episode, no work is lost or counted twice. Runs pause and resume.
- **Evals.** Versioned suites of environments run on a schedule or against any checkpoint, with a score for each
  environment.

On one machine, the `minecraft-one-gpu` preset puts the 4-bit Qwen3.5-9B on vLLM and a rank-32 LoRA trainer on the
same 16 GB card. The engines sleep while the trainer steps. On a cluster, the included Helm chart runs the same
processes under [KubeRay](https://github.com/ray-project/kuberay).

## Writing an environment

```python
from rollout.contracts import Message
from rollout.harness import End, Observation, RunContext, Task


class Arithmetic(Task):
    def __init__(self, parameters: dict[str, str]) -> None:
        self.question, self.answer = parameters["question"], parameters["answer"]

    async def start(self, run: RunContext) -> Observation:
        return Observation(self.question)

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        return End(reward=1.0 if reply.text.strip() == self.answer else 0.0)
```

This task asks a question and rewards an exact answer. To check an environment, train on it and watch:

```sh
uv run rollout env check minecraft_team.environment:environment
uv run rollout preset load deploy/chart/rollout/files/presets --cluster
uv run rollout train minecraft_team.environment:environment --preset minecraft-one-gpu
uv run rollout monitor --cluster   # runs, every transcript, the checkpoint tree, evals, machines
```

## How it fits together

```
 environments ──► episode runners ──► gateway ──► inference (vLLM, Tinker, pods)
                        │                │               ▲
                        ▼                ▼               │ follows serving records
                   ┌──────────── ledger (Postgres) ──────┴───┐   blob store (S3)
                   └── training loop ──► training provider ──► checkpoints ──► bridges
```

1. Runners play episodes, and the gateway records each turn they sample.
2. The training loop scores groups of episodes and has the training provider take a step.
3. The checkpoint is converted for each inference backend, and the loop records what should be served.
4. Inference servers follow that record.

No process holds state that another depends on; they coordinate only through the ledger and the blob store. Read
more in [the architecture](docs/architecture/overview.md).

## Getting started

rollout is a [uv](https://docs.astral.sh/uv/) workspace for Python 3.13.

```sh
uv sync                  # everything that needs no GPU
uv sync --all-extras     # plus vLLM, the trainers, Tinker and the renderers (Linux, NVIDIA GPU)
uv run pytest            # the tests, including every example in the docs
```

Once the extras are installed, keep syncing with `--all-extras`: a plain `uv sync` uninstalls them.

- [Start here](docs/start/README.md): the ideas, in five minutes.
- [Run a first episode](docs/guide/getting-started.md): a task played against a scripted model.
- [Deploy](docs/deploy/README.md): on one machine, or on Kubernetes ([K3s with a GPU](deploy/k3s/README.md)).
- [All documentation](docs/README.md) and the [API reference](docs/guide/reference.md).

| Directory | Contents |
|---|---|
| `libraries/` | `rollout` (the API environments are written against) and `rollout-train` (the ledger, gateway, training loop, evals and monitor) |
| `implementations/` | Backends: vLLM, LoRA and full-weight trainers, Tinker, Qwen and Gemma renderers, S3, RunPod pods, verifiers |
| `environments/` | A Minecraft team, Minecraft horizons, a gridworld where agents spread out to press plates, and open-ended answers scored by a judge |
| `deploy/` | The platform image, the Helm chart, a K3s setup for one GPU, and local services |

To check your changes, run `uv run ruff check && uv run ruff format --check && uv run pyright`. After changing public
names or docstrings, regenerate the API reference with `uv run python scripts/generate_reference.py`.
