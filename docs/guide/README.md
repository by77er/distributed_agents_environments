# Developer guide

How to build with `rollout`: write tasks, tools and agents, run episodes, test them, and train on them. The
[documentation index](../README.md) lists everything else.

Every block tagged exactly `python` in this guide runs as part of the test suite (`tests/test_docs.py`). A block
tagged `python fragment` shows a shape and is not run. Every profile in a `toml` block is loaded by the tests. The
[API reference](reference.md) is generated from the source, and a test fails when it is out of date.

## Pages

| Page | Read it to |
|---|---|
| [Three ways in](perspectives.md) | see the system from where you stand: building an environment, designing training, deploying |
| [Getting started](getting-started.md) | install, write a first task, run one episode |
| [Tasks](tasks.md) | define an environment: hooks, observations, rewards, endings, extra model slots |
| [Tools](tools.md) | give the model an action space with `@tool` methods and imported tool sets |
| [Agents](agents.md) | control what the model sees and how it acts |
| [Conversations](conversations.md) | wait for messages, and handle messages that arrive mid-turn |
| [Content](content.md) | build and read messages, tool calls, tool results; blobs; digests |
| [Runs and events](runs-and-events.md) | understand what a run records: effects, identifiers, events; start runs with the local runner |
| [Models](models.md) | run against a real model through the Responses API |
| [Testing](testing.md) | test tasks and agents with a scripted model |
| [Deploying](deploying.md) | describe engines, the trainer, the runner and tool sets in a profile; the `rollout` command |
| [API reference](reference.md) | look up any public name |

Beyond the guide:

| For | Read |
|---|---|
| Runs that survive their process | [Durable runner](../implementations/rollout-durable/README.md) |
| Computers for tasks: shell and file tools on an environment | [Computers](../implementations/rollout-computers.md) |
| Long episodes in a context that fits any model | [Memory](../libraries/rollout/memory.md) |
| Training: rollout jobs, episodes, the training loop, policies, recorded channels | [Rollouts](../libraries/rollout-train/rollouts.md), [episodes](../libraries/rollout-train/episodes.md), [training](../libraries/rollout-train/training.md), [policies](../libraries/rollout-train/policies.md), [channels](../libraries/rollout-train/channels.md), [recorder](../libraries/rollout-train/recorder.md) |
| What a profile names: an engine, a trainer, a model family's renderer | [vLLM engine](../implementations/rollout-vllm.md), [LoRA trainer](../implementations/rollout-lora.md), [Qwen renderers](../implementations/rollout-qwen.md), [Gemma renderers](../implementations/rollout-gemma.md) |

## Concepts in one place

The [glossary](../architecture/glossary.md) defines every term; these are the ones the guide uses most.

| Term | Meaning |
|---|---|
| **Run** | One execution of a program: for an agent, one episode. Identified by `run_id` (`r_{ulid}`). |
| **Task** | The environment the agent acts in: first observation, a response to each reply, tools, scoring. |
| **Agent** | The policy side: what the model sees each turn, and how its output becomes one reply. |
| **Observation** | What the model is shown next, plus the reward for the reply it answers and whether the episode ended. |
| **Reply** | The assistant `Message` the agent returns each turn. |
| **Turn** | One reply and the observation that answers it. `run.turn` counts replies. |
| **Model slot** | A named model a task uses. The agent acts through `policy`; tasks may declare others. |
| **Model endpoint** | What serves a model slot: a recorder, an API adapter, or a scripted endpoint in tests. |
| **Effect** | An operation that leaves task or agent code, such as a model sample. It has a stable `effect_id`. |
| **Run event** | A typed record of something that happened in a run, in a gapless sequence. |
| **Catalog** | What an environment offers to train on: rows, easiest first, and how to draw a start of one. |
| **Group** | One start of one row, played as several episodes that are compared with each other. |
| **Step** | One update of the policy by the trainer, over several groups. |
| **Episode** | A finished run as training sees it: labels, outcome, result, and each model slot's trajectory of segments. |
| **Channel** | A trainable policy being served, by name. A binding names one for a model slot; training publishes weights to it. |

## Where things live

Each row is a module of one package; the [documentation index](../README.md#packages) lists the packages.

| Import from | Package | For |
|---|---|---|
| `rollout.harness` | `rollout` | `Task`, `Agent`, `tool`, `Observation`, `End`, `WaitFor`, `RunContext`, `rollout`, `Program`, `Memory`, conversation types |
| `rollout.contracts` | `rollout` | `Message`, content blocks, `ToolSpecification`, `ToolResult`, identifiers, digests, events |
| `rollout.local` | `rollout` | `LocalRunner`, `LocalRunContext`: runs in this process |
| `rollout.catalog` | `rollout` | `Catalog`, `Row`, `binding_for` |
| `rollout.testing` | `rollout` | `ScriptedModelEndpoint`, `local_run`, `events_of`, `payload`, `tool_call_reply` |
| `rollout_train.rollouts` | `rollout-train` | `Jobs`, `Job`, `Ticket`, `Episode`, `RolloutJobs`; `RolloutClient` in `rollout_train.rollouts.service` |
| `rollout_train` | `rollout-train` | `train`, `Grpo`, `Curriculum`, `Trainer`, `Colocated`, `Policies`, `FileLedger` |
| `rollout_train.inference`, `rollout_train.recorder` | `rollout-train` | `Channel`, `Engine`, `Limits`; `Recorder`, `Segment`, `Renderer` |
| `rollout_train.profile`, `rollout_train.monitor` | `rollout-train` | `Profile`, `Platform`; `RunFeed` |
| `rollout_train.testing` | `rollout-train` | `ScriptedEngine`, `PlainRenderer`, `plain_channel` |
| `rollout_durable` | `rollout-durable` | `DurableRunner` |
| `rollout_computers`, `rollout_computers.tools` | `rollout-computers` | `NamespaceEnvironments`, `LocalEnvironments`; `ComputerTools` |
| `rollout_openai` | `rollout-openai` | `ResponsesEndpoint`, `codex_provider`, `ApiKey` |
| `rollout_s3` | `rollout-s3` | `S3BlobStore` |
| `rollout_vllm` | `rollout-vllm` | `VllmEngine` |
| `rollout_lora` | `rollout-lora` | `LoraTrainer`, `LoraSettings` |
| `rollout_qwen` | `rollout-qwen` | `qwen35`, `qwen3` |
| `rollout_gemma` | `rollout-gemma` | `gemma4` |

## Conventions

- Hooks and tools are `async` methods. Everything a task or agent does outside its own code goes through `run`.
- Contract types are immutable pydantic models: construct new values rather than modifying existing ones.
- Type names are never abbreviated: `Observation`, not `Obs`.
- For AI coding assistants: [`llms.txt`](../../llms.txt) at the repository root indexes the documentation.
