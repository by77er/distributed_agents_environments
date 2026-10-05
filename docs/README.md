# rollout documentation

rollout trains language-model agents with reinforcement learning, on episodes as long and as crowded as a team of
four playing Minecraft. These pages are for everyone who works with it: people who write environments, people who
train and evaluate models, and people who deploy and run the platform.

New here? Read [Start here](start/README.md) first. It takes five minutes and explains the main ideas.

## Find what you need

Each section has an index page that lists its pages in the order to read them.

- **[Write an environment](guide/README.md).** Write a task, give the model tools, test it with a scripted model,
  and publish it in a git repository so that a cluster can import it.
- **[Train a model](train/README.md).** How episodes are played and recorded, the training loop and its objectives,
  checkpoints, datasets and supervised steps.
- **[Evaluate a model](evaluate/README.md).** Suites of environments, evals of any checkpoint or base model, and
  evals on a schedule during training.
- **[Deploy the platform](deploy/README.md).** What runs where, choosing a setup, installing on Kubernetes with the
  Helm chart, the stores, GPUs, access and troubleshooting.
- **[Operate the platform](operate/README.md).** Watch runs in the monitor, pause and resume them, name
  checkpoints, check a cluster config and back up the stores.
- **[Extend the platform](extend/README.md).** Add an inference provider, a trainer, an objective or a model
  family's renderer, against the contracts the libraries define.

## Look something up

- [Reference](reference/README.md): the API reference, the command line, run settings, the cluster config, the
  objective presets and the glossary.
- [Design notes](research/README.md): designs, proposals and measurements, each marked with its status.

## Packages

The repository is a [uv](https://docs.astral.sh/uv/) workspace for Python 3.13. Its packages are libraries (what code
is written against), implementations (one backend for an interface a library defines) and environments (something to
train on).

| Directory | Import | What it is | Implements |
|---|---|---|---|
| `libraries/rollout` | `rollout` (`rollout.harness`, `rollout.contracts`, `rollout.local`, `rollout.environment`, `rollout.testing`) | What environments are written against: tasks, agents, programs, tools, conversations, and a runner in this process | |
| `libraries/rollout-train` | `rollout_train` (and `.rollouts`, `.inference`, `.recorder`, `.monitor`, `.profile`, `.cli`, `.testing`) | Reinforcement learning on `rollout`: episode runners and episodes, the training loop, channels, the gateway, the cluster config | |
| `implementations/rollout-vllm` | `rollout_vllm` | vLLM as an engine | `Engine` |
| `implementations/rollout-lora` | `rollout_lora` | A trainer for 4-bit checkpoints with LoRA (low-rank adaptation), and a full-weight trainer | `Trainer` |
| `implementations/rollout-tinker` | `rollout_tinker` | A trainer and an engine at Thinking Machines (Tinker); the `tinker` extra | `Trainer`, `Engine` |
| `implementations/rollout-qwen` | `rollout_qwen` | The Qwen families' token formats | `Renderer` |
| `implementations/rollout-gemma` | `rollout_gemma` | Gemma 4's token format | `Renderer` |
| `implementations/rollout-openai` | `rollout_openai` | The OpenAI Responses API as a model endpoint | `ModelEndpoint` |
| `implementations/rollout-s3` | `rollout_s3` | Blobs in S3 or an S3-compatible store | `Blobs` |
| `implementations/rollout-runpod` | `rollout_runpod` | GPU pods on RunPod (the pods API), and certificates for them from step-ca | |
| `implementations/rollout-verifiers` | `rollout_verifiers` | Prime Intellect's verifiers environments, played through the gateway | `Environment` |
| `environments/minecraft` | `minecraft_team` | One to four agents in a Minecraft world; depends on `rollout` only | `Environment` |
| `environments/gridworld` | `gridworld` | Two to four agents on a grid level, with plates, doors, a gate and a lever; depends on `rollout` only | `Environment` |
| `environments/judging` | `judging` | Open-ended requests answered by the policy and scored by a judge against a rubric; depends on `rollout` only | `Environment` |

`uv sync` installs every package that needs no GPU. `uv sync --all-extras` adds `rollout-vllm`, `rollout-lora`,
`rollout-qwen` and `rollout-gemma` (`--extra gemma` installs that one alone), and `rollout-tinker` (`--extra tinker`).
`implementations/rollout-verifiers` is a project of its own, with its own lock, outside the workspace.

## Conventions

- **Pages describe the code as it is.** A page's `Code:` line names the code it describes. Pages under
  [Design notes](research/README.md) say at their top whether what they describe is built, in progress or proposed.
- **Each page says where it fits.** It opens with what it covers and who it is for, and links to what to read first
  and what to read next.
- **Define once.** Every type and every fact has one defining page; others link to it. Types and protocols are
  listed in the generated [API reference](guide/reference.md), and types that cross layers are described under
  [Contracts](libraries/rollout/contracts/README.md).
- **Full type names.** `Observation`, not `Obs`.
- **Runnable examples.** In the pages under `docs/guide/`, a block tagged exactly `python` runs in the test suite
  (`tests/test_docs.py`); a block tagged `py` shows a shape and is not run.
- **For AI coding assistants:** `llms.txt` at the repository's root indexes these pages.
