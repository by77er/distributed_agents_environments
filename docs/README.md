# Documentation

`rollout` runs agents in environments, and trains them. Start with the [developer guide](guide/README.md) to build
with it, or with [three ways in](guide/perspectives.md) to see it from where you stand: building an environment,
designing training, or deploying.

The documentation follows the repository's split into libraries, implementations, products and environments.

## Guide

How to build with it. Runnable examples are part of the test suite.

| Page | What it covers |
|---|---|
| [Guide index](guide/README.md) | What is available, the concepts, where each name lives |
| [Three ways in](guide/perspectives.md) | The three surfaces: environments, training, deployment |
| [Getting started](guide/getting-started.md) | Install, a first task, one episode against a scripted model |
| [Tasks](guide/tasks.md), [Tools](guide/tools.md), [Agents](guide/agents.md) | Environments, action spaces, what the model sees |
| [Conversations](guide/conversations.md), [Content](guide/content.md) | Messages and waiting; canonical content, blobs, digests |
| [Runs and events](guide/runs-and-events.md) | What a run records: effects, identifiers, events; the local runner |
| [Models](guide/models.md), [Testing](guide/testing.md) | Third-party models through the Responses API (`rollout-openai`); scripted models |
| [Deploying](guide/deploying.md) | The profile file and the `rollout` command |
| [API reference](guide/reference.md) | Every public name, generated from the source |

## Architecture

| Page | What it covers |
|---|---|
| [Overview](architecture/overview.md) | Layers, protocols and their implementations, who sees what, a turn under each runner |
| [Glossary](architecture/glossary.md) | Every term, defined once |

## Libraries

Interfaces, and what runs with no implementation beyond this process.

| Package | Pages |
|---|---|
| `rollout` | [Harness](libraries/rollout/README.md): the loop, programs and runners. [Sandboxes](libraries/rollout/sandboxes.md), [determinism](libraries/rollout/determinism.md), [hooks](libraries/rollout/hooks.md), [memory](libraries/rollout/memory.md). [Contracts](libraries/rollout/contracts/README.md): [identifiers](libraries/rollout/contracts/identifiers.md), [canonical content](libraries/rollout/contracts/canonical-content.md), [run events](libraries/rollout/contracts/run-events.md), [effects](libraries/rollout/contracts/effects.md), the [model endpoint](libraries/rollout/contracts/model-endpoint.md) |
| `rollout-train` | [Rollouts](libraries/rollout-train/rollouts.md), [checkpoints, runs and the ledger](libraries/rollout-train/checkpoints.md), [episodes](libraries/rollout-train/episodes.md), [training](libraries/rollout-train/training.md), [channels](libraries/rollout-train/channels.md), the [recorder](libraries/rollout-train/recorder.md) and its [endpoint for harnesses](libraries/rollout-train/harness-endpoint.md), the [monitor](libraries/rollout-train/monitor.md) |

## Implementations

Each implements one interface a library defines.

| Page | What it covers |
|---|---|
| [Durable runner](implementations/rollout-durable/README.md) | Runs that survive their process, on DBOS. [Evicting idle runs](implementations/rollout-durable/eviction.md); [several runners](implementations/rollout-durable/runners.md) on one Postgres, and the database stores use |
| [vLLM engine](implementations/rollout-vllm.md) | Options (speculative decoding among them), adapters by name, sleep and wake, the engine core process, measurements |
| [LoRA trainer](implementations/rollout-lora.md) | Settings, a fresh process per step, the memory bound, the step, metrics, measurements |
| [Tinker trainer and engine](implementations/rollout-tinker.md) | Training and sampling at Thinking Machines: installing (a project of its own), the key, the objective as Tinker's losses, what a checkpoint holds, serving its adapters here, costs |
| [Qwen renderers](implementations/rollout-qwen.md) | The token formats of Qwen3.5 and Qwen3 |
| [Gemma renderers](implementations/rollout-gemma.md) | The token format of Gemma 4 |
| [Computers](implementations/rollout-computers.md) | Environment backends and the tools that act on them |
| [verifiers environments](implementations/rollout-verifiers.md) | Prime Intellect's verifiers environments as environments here, their harnesses reaching the recorder |
| [Models](guide/models.md), [Content](guide/content.md#media-and-blobs) | `rollout-openai` and `rollout-s3` are described in the guide |

## Products and environments

| Page | What it covers |
|---|---|
| [Project assistant](products/project-assistant.md) | A long-lived conversational agent about one code repository |
| [Agent sessions](products/agent-sessions.md) | Independent agents with their own computers, which create and message each other |
| [Minecraft team](products/minecraft-team.md) | One to four agents in a Minecraft world: an environment to train on |

## Development

| Page | What it covers |
|---|---|
| [Local services](development/local-services.md) | Postgres and an S3-compatible store for a development machine |

## Packages

| Directory | Import | What it is | Implements |
|---|---|---|---|
| `libraries/rollout` | `rollout` (`rollout.harness`, `rollout.contracts`, `rollout.local`, `rollout.environment`, `rollout.testing`) | What environments are written against: tasks, agents, programs, tools, conversations, and a runner in this process | |
| `libraries/rollout-train` | `rollout_train` (and `.rollouts`, `.inference`, `.recorder`, `.monitor`, `.profile`, `.cli`, `.testing`) | Reinforcement learning on `rollout`: episode runners and episodes, the training loop, channels, the recorder, profiles | |
| `implementations/rollout-durable` | `rollout_durable` | A runner whose runs survive their process, on DBOS; a database for stores | `Runner` |
| `implementations/rollout-vllm` | `rollout_vllm` | vLLM as an engine | `Engine` |
| `implementations/rollout-lora` | `rollout_lora` | A trainer for 4-bit checkpoints with LoRA | `Trainer` |
| `implementations/rollout-tinker` | `rollout_tinker` | A trainer and an engine at Thinking Machines (Tinker); a uv project of its own | `Trainer`, `Engine` |
| `implementations/rollout-qwen` | `rollout_qwen` | The Qwen families' token formats | `Renderer` |
| `implementations/rollout-gemma` | `rollout_gemma` | Gemma 4's token format | `Renderer` |
| `implementations/rollout-computers` | `rollout_computers` | Computers for tasks, and the tools that act on them | `EnvironmentService` |
| `implementations/rollout-openai` | `rollout_openai` | The OpenAI Responses API as a model endpoint | `ModelEndpoint` |
| `implementations/rollout-s3` | `rollout_s3` | Blobs in S3 or an S3-compatible store | `Blobs` |
| `implementations/rollout-verifiers` | `rollout_verifiers` | Prime Intellect's verifiers environments, played through the recorder | `Environment` |
| `products/project-assistant` | `project_assistant` | A conversational agent about one repository | |
| `products/agent-sessions` | `agent_sessions` (and `agent_sessions.coordination`) | Agents with their own computers, and their coordination | |
| `environments/minecraft` | `minecraft_team` | One to four agents in a Minecraft world; depends on `rollout` only | `Environment` |

`uv sync` installs every package that needs no GPU. `uv sync --all-extras` adds `rollout-vllm`, `rollout-lora`,
`rollout-qwen` and `rollout-gemma` (`--extra gemma`: that one alone). `implementations/rollout-verifiers` is a project of
its own, with its own lock, outside the workspace.

## Research

Proposals: what could be built, and the records it would need.

| Page | What it covers |
|---|---|
| [The policy graph](research/policy-dag.md) | The graph of checkpoints with what trains, serves and evaluates them: distillation (on and off policy), trainers and their queues, the way from written weights to served ones, evaluations |
| [Curricula](research/curricula.md) | Building training curricula and frozen evaluation suites from a run's data |
| [Thinking Machines' API](research/thinking-machines.md) | Training and sampling through Tinker as a trainer and an engine of their own |
| [Prime Intellect's verifiers](research/prime-compat.md) | verifiers environments run here: what maps, a spike on a Hub environment, how stable the API is, and exporting ours |

## Conventions

- **Pages describe the code as it is.** A page's `Code:` line names the code it describes.
- **Define once.** Every type and every fact has one defining page; others link to it. Types and protocols are listed
  in the generated [API reference](guide/reference.md), and types that cross layers are described in
  `libraries/rollout/contracts/`.
- **Full type names.** `Observation`, not `Obs`.
