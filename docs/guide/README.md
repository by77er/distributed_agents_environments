# Developer guide

How to build with `rollout`: write tasks, tools and agents, run episodes, test them, and train on them. The
[documentation index](../README.md) lists everything else.

Every Python example in this guide runs as part of the test suite (`tests/test_docs.py`), and the
[API reference](reference.md) is generated from the source, so both match the code.

## What you can use

| For | Start with |
|---|---|
| Building an environment, training on one, or deploying both | [Three ways in](perspectives.md) |
| Tasks, agents, `@tool` methods, imported tools, observations, rewards | [Getting started](getting-started.md) |
| Runs: start, send, cancel, event streams, conversations, message delivery by mode | [Runs and events](runs-and-events.md) |
| Runs that survive their process | [durability](../durability/README.md) |
| Trainable models: recorded channels, engines, rollout jobs, episodes, the training loop | [recorder](../core/recorder/README.md), [rollouts](../core/rollouts/README.md), [training](../core/training.md) |
| Third-party models through the Responses API; a scripted endpoint for tests | [Models](models.md), [Testing](testing.md) |

## Pages

| Page | Read it to |
|---|---|
| [Three ways in](perspectives.md) | see the system from where you stand: building an environment, designing training, deploying |
| [Getting started](getting-started.md) | install, write a first task, run one episode |
| [Tasks](tasks.md) | define an environment: hooks, observations, rewards, endings, extra model slots |
| [Tools](tools.md) | give the model an action space with `@tool` methods |
| [Agents](agents.md) | control what the model sees and how it acts |
| [Conversations](conversations.md) | wait for messages, and handle messages that arrive mid-turn |
| [Content](content.md) | build and read messages, tool calls, tool results; digests |
| [Runs and events](runs-and-events.md) | understand what a run records: effects, identifiers, events, failures |
| [Models](models.md) | run against a real model through the Responses API |
| [Testing](testing.md) | test tasks and agents with a scripted model |
| [Deploying](deploying.md) | describe engines, the trainer, the runner and tool sets in a profile |
| [API reference](reference.md) | look up any public name |

## Concepts in one place

| Term | Meaning |
|---|---|
| **Run** | One execution of a program. For an agent, one episode. Identified by `run_id` (`r_{ulid}`). |
| **Task** | The environment the agent acts in: first observation, a response to each reply, tools, scoring. |
| **Agent** | The policy side: what the model sees each turn, and how its output becomes one reply. |
| **Observation** | What the model is shown next, plus the reward for the reply it answers and whether the episode ended. |
| **Reply** | The assistant `Message` the agent returns each turn. |
| **Turn** | One reply and the observation that answers it. `run.turn` counts replies. |
| **Model slot** | A named model a task uses. The agent acts through `policy`; tasks may declare others. |
| **Model endpoint** | What serves a model slot: a recorder, an API adapter, or a scripted endpoint in tests. |
| **Channel** | A trainable policy being served, by name. A binding names one for a model slot; training publishes weights to it. |
| **Episode** | A finished run as training sees it: labels, outcome, result, and each slot's token sequences with logprobs. |
| **Catalog** | What an environment offers to train on: rows, easiest first, and how to draw a start of one. |
| **Effect** | An operation that leaves task or agent code, such as a model sample. It has a stable `effect_id`. |
| **Run event** | A typed record of something that happened in a run, in a gapless sequence. |

## Where things live

| Import from | For |
|---|---|
| `rollout.core.harness` | `Task`, `Agent`, `tool`, `Observation`, `End`, `WaitFor`, `RunContext`, `rollout`, conversation types |
| `rollout.core.contracts` | `Message`, content blocks, `ToolSpecification`, `ToolResult`, identifiers, digests, events |
| `rollout.core.local` | `LocalRunContext`: runs an episode in process |
| `rollout.core.testing` | `ScriptedModelEndpoint`, `local_run`, `events_of`, `payload`, `tool_call_reply` |
| `rollout.rollouts` | `Jobs`, `Job`, `Ticket`, `Episode`, `Catalog`, `Row`, `RolloutJobs`; `RolloutClient` in `rollout.rollouts.service` |
| `rollout.training` | `train`, `Grpo`, `Curriculum`, `Trainer`, `LoraTrainer`, `Colocated` |
| `rollout.inference`, `rollout.recorder` | `Channel`, `Engine`, `Limits`; `Recorder`, `Epoch`, renderers |
| `rollout.profile` | `Profile`, `Platform` |

## Conventions

- Hooks and tools are `async` methods. Everything a task or agent does outside its own code goes through `run`.
- Contract types are immutable pydantic models: construct new values rather than modifying existing ones.
- Type names are never abbreviated: `Observation`, not `Obs`.
- For AI coding assistants: [`llms.txt`](../../llms.txt) at the repository root indexes this guide.
