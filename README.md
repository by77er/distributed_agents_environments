# rollout

**Reinforcement learning for agents, from one GPU to a cluster.**

rollout trains language-model agents on long, multi-turn, multi-agent episodes: a team of four playing Minecraft,
agents spreading out across a gridworld, or a math problem. You write an environment once. rollout plays it at scale,
records every token the policy sampled, trains on the result, and serves each new checkpoint back to the agents while
they play.

## Why rollout

- **Any harness trains as it is.** Agents reach the model through one gateway that speaks the OpenAI Chat Completions
  and Responses APIs and the Anthropic Messages API, and records each turn token for token. So Claude Code, Codex or
  your own agent loop can generate training data without any changes.
- **Mix and match backends.** Train on [Tinker](https://thinkingmachines.ai/tinker/) or on your own GPU, and serve
  the result on local [vLLM](https://github.com/vllm-project/vllm). Each checkpoint is converted to the format each
  inference backend loads, and each turn records what it was sampled with (exact tokens, behaviour logprobs), so
  training never learns from turns it can't weight correctly.
- **Asynchronous, off-policy by design.** Episodes keep playing while the policy trains. Each turn is pinned to the
  checkpoint that sampled it, staleness is bounded by a `max_lag` you can change mid-run, and importance weights
  correct for the rest. Multi-turn conversations train as whole segments, not one reply at a time.
- **Environments with real worlds in them.** An environment bundles its training and eval data, curriculum and
  rewards. Sandboxes (a Minecraft server, a computer) are leased by claim, so every agent in an episode shares the
  same world, even when they run on different machines.
- **Crash-safe without a coordinator.** Every process coordinates through a fenced, append-only ledger, and its
  guarantees are tested under real concurrency. Any runner, server or training job can die or be replaced mid-episode
  without losing or double-counting work, and runs pause and resume.
- **Evals are first-class.** Versioned suites of environments run on a schedule during training or on any checkpoint
  by hand, with per-environment scores and each checkpoint's eval history.
- **One workstation or a cluster, same code.** It runs on a single 16 GB GPU, and on Kubernetes with
  [KubeRay](https://github.com/ray-project/kuberay) from the included Helm chart.

## A taste

An environment's task, run against a scripted model:

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

Then check an environment, train on it, and watch:

```sh
uv run rollout env check minecraft_team.environment:environment
uv run rollout preset load deploy/chart/rollout/files/presets --cluster
uv run rollout train minecraft_team.environment:environment --preset minecraft-one-gpu
uv run rollout monitor --cluster
```

The monitor shows runs, every episode's transcript, the checkpoint tree, evals, environments and the machines doing
the work, and it can start new runs from a form.

## How it fits together

```
 environments ──► episode runners ──► gateway ──► inference (vLLM, Tinker, pods)
                        │                │               ▲
                        ▼                ▼               │ follows serving records
                   ┌──────────── ledger (Postgres) ──────┴───┐   blob store (S3)
                   └── training loop ──► training provider ──► checkpoints ──► bridges
```

Runners play episodes and sample through the gateway, which records each turn. The training loop scores groups of
episodes, has a training provider take a step, converts the checkpoint for each inference backend, and writes down
what should be served. Inference servers follow that record. No process holds state another one depends on: they
coordinate through the ledger and the blob store. [The architecture](docs/architecture/overview.md) has the details.

## Getting started

rollout is a [uv](https://docs.astral.sh/uv/) workspace for Python 3.13.

```sh
uv sync                  # everything that needs no GPU
uv sync --all-extras     # with vLLM, the trainers, Tinker and the model families' renderers (Linux, NVIDIA GPU)
uv run pytest            # the tests, including every example in the docs
```

- [Start here](docs/start/README.md): what rollout does and the ideas it is built on, in five minutes
- [Run a first episode](docs/guide/getting-started.md): write a task and play one episode against a scripted model
- [Deploy the platform](docs/deploy/README.md): on one machine, or on Kubernetes with the Helm chart
  ([K3s with a GPU](deploy/k3s/README.md))
- [All documentation](docs/README.md), by what you want to do, and the [API reference](docs/guide/reference.md)

## Repository

| Directory | Contents |
|---|---|
| `libraries/` | `rollout`, the API environments are written against, and `rollout-train`: the ledger, gateway, training loop, evals and monitor |
| `implementations/` | Backends: vLLM, LoRA and full-weight trainers, Tinker, Qwen and Gemma renderers, S3, RunPod pods, verifiers environments |
| `environments/` | Environments to train on: a Minecraft team, a gridworld where agents spread out to press plates, and open-ended answers a judge scores |
| `deploy/` | The platform image, the Helm chart, a K3s setup for one GPU, and local services |
| `docs/`, `tests/`, `scripts/` | Documentation, tests and the reference generator |

Development checks: `uv run ruff check && uv run ruff format --check && uv run pyright`, and
`uv run python scripts/generate_reference.py` after changing public names or docstrings.
