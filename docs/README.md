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
| [Runs and events](guide/runs-and-events.md), [Testing](guide/testing.md) | What a run records; scripted models |
| [Models](guide/models.md) | Third-party models through the Responses API (`rollout-openai`) |
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
| `rollout` | [Harness](libraries/rollout/README.md): the loop, programs and runners. [Determinism](libraries/rollout/determinism.md), [hooks](libraries/rollout/hooks.md), [memory](libraries/rollout/memory.md). [Contracts](libraries/rollout/contracts/README.md): [identifiers](libraries/rollout/contracts/identifiers.md), [canonical content](libraries/rollout/contracts/canonical-content.md), [run events](libraries/rollout/contracts/run-events.md), [effects](libraries/rollout/contracts/effects.md), the [model endpoint](libraries/rollout/contracts/model-endpoint.md) |
| `rollout-train` | [Rollouts](libraries/rollout-train/rollouts.md), [episodes](libraries/rollout-train/episodes.md), [training](libraries/rollout-train/training.md), [policies, versions and the ledger](libraries/rollout-train/policies.md), [channels](libraries/rollout-train/channels.md), the [recorder](libraries/rollout-train/recorder.md) and its [endpoint for harnesses](libraries/rollout-train/harness-endpoint.md), the [monitor](libraries/rollout-train/monitor.md) |

## Implementations

Each implements one interface a library defines.

| Page | What it covers |
|---|---|
| [Durable runner](implementations/rollout-durable/README.md) | Runs that survive their process, on DBOS. [Evicting idle runs](implementations/rollout-durable/eviction.md); [several runners](implementations/rollout-durable/runners.md) on one Postgres, and the database stores use |
| [vLLM engine](implementations/rollout-vllm.md) | Options, sleep and wake, adapters, the engine core process, measurements |
| [LoRA trainer](implementations/rollout-lora.md) | Settings, a process per step, the memory bound, the step, metrics, measurements |
| [Qwen renderers](implementations/rollout-qwen.md) | The token formats of Qwen3.5 and Qwen3 |
| [Gemma renderers](implementations/rollout-gemma.md) | The token format of Gemma 4 |
| [Computers](implementations/rollout-computers.md) | Environment backends and the tools that act on them |
| [Models](guide/models.md), [Content](guide/content.md#media-and-blobs) | `rollout-openai` and `rollout-s3` are described in the guide |

## Products and environments

| Page | What it covers |
|---|---|
| [Project assistant](products/project-assistant.md) | A long-lived conversational agent about one code repository |
| [Agent sessions](products/agent-sessions.md) | Independent agents with their own computers, which create and message each other |
| [Minecraft swarm](products/minecraft-swarm.md) | Four agents in a Minecraft world: an environment to train on |

## Development

| Page | What it covers |
|---|---|
| [Local services](development/local-services.md) | Postgres and an S3-compatible store for a development machine |

## Packages

| Directory | Import | What it is | Implements |
|---|---|---|---|
| `libraries/rollout` | `rollout` (`rollout.harness`, `rollout.contracts`, `rollout.local`, `rollout.catalog`, `rollout.testing`) | What environments are written against: tasks, agents, programs, tools, conversations, and a runner in this process | |
| `libraries/rollout-train` | `rollout_train` (and `.rollouts`, `.inference`, `.recorder`, `.monitor`, `.profile`, `.cli`, `.testing`) | Reinforcement learning on `rollout`: jobs and episodes, the training loop, channels, the recorder, profiles | |
| `implementations/rollout-durable` | `rollout_durable` | A runner whose runs survive their process, on DBOS; a database for stores | `Runner` |
| `implementations/rollout-vllm` | `rollout_vllm` | vLLM as an engine | `Engine` |
| `implementations/rollout-lora` | `rollout_lora` | A trainer for 4-bit checkpoints with LoRA | `Trainer` |
| `implementations/rollout-qwen` | `rollout_qwen` | The Qwen families' token formats | `Renderer` |
| `implementations/rollout-gemma` | `rollout_gemma` | Gemma 4's token format | `Renderer` |
| `implementations/rollout-computers` | `rollout_computers` | Computers for tasks, and the tools that act on them | `EnvironmentService` |
| `implementations/rollout-openai` | `rollout_openai` | The OpenAI Responses API as a model endpoint | `ModelEndpoint` |
| `implementations/rollout-s3` | `rollout_s3` | Blobs in S3 or an S3-compatible store | `Blobs` |
| `products/project-assistant` | `project_assistant` | A conversational agent about one repository | |
| `products/agent-sessions` | `agent_sessions` (and `agent_sessions.coordination`) | Agents with their own computers, and their coordination | |
| `environments/minecraft` | `minecraft_swarm` | One to four agents in a Minecraft world; depends on `rollout` only | `Catalog` |

`uv sync` installs every package that needs no GPU. `uv sync --all-extras` adds `rollout-vllm`, `rollout-lora`,
`rollout-qwen` and `rollout-gemma` (`--extra gemma`: that one alone).

## Conventions

- **Pages describe the code as it is.** A page's `Code:` line names the code it describes.
- **Define once.** Every type and every fact has one defining page; others link to it. Types and protocols are listed
  in the generated [API reference](guide/reference.md), and types that cross layers are described in
  `libraries/rollout/contracts/`.
- **Full type names.** `Observation`, not `Obs`.
