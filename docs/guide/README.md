# Developer guide

How to build with `rollout`: write tasks, tools and agents, run episodes, and test them. This guide describes the
code as it exists; the [design docs](../README.md) describe the whole system, including parts not built yet.

Every Python example in this guide runs as part of the test suite (`tests/test_docs.py`), and the
[API reference](reference.md) is generated from the source, so both match the code.

## What you can use today

The project is in milestone M0 of the [development plan](../development/plan.md): the core loop, without a GPU. Next
comes P1, a project assistant on a third-party model, then the durable runner (M2), then RL (M1).

| Available now | Not yet |
|---|---|
| Tasks, agents, `@tool` methods, observations, rewards | The Responses API adapter on a Codex login (P1) |
| `LocalRunner`: start, send, cancel, event streams, conversations | Imported tools (`imports`, `ToolBinding`) |
| Message delivery by mode: queue, steer, interrupt | The durable runner (M2) |
| Effect identities, argument digests, run events, `run.emit` | The recorder, rollout jobs, samples, weight updates (M1) |
| A scripted model endpoint for tests | |

Until a model adapter exists, the model is a `ScriptedModelEndpoint` from `rollout.core.testing`, or any object
that implements the `ModelEndpoint` protocol.

## Pages

| Page | Read it to |
|---|---|
| [Getting started](getting-started.md) | install, write a first task, run one episode |
| [Tasks](tasks.md) | define an environment: hooks, observations, rewards, endings, extra model slots |
| [Tools](tools.md) | give the model an action space with `@tool` methods |
| [Agents](agents.md) | control what the model sees and how it acts |
| [Conversations](conversations.md) | wait for messages, and handle messages that arrive mid-turn |
| [Content](content.md) | build and read messages, tool calls, tool results; digests |
| [Runs and events](runs-and-events.md) | understand what a run records: effects, identifiers, events, failures |
| [Testing](testing.md) | test tasks and agents with a scripted model |
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
| **Effect** | An operation that leaves task or agent code, such as a model sample. It has a stable `effect_id`. |
| **Run event** | A typed record of something that happened in a run, in a gapless sequence. |

## Where things live

| Import from | For |
|---|---|
| `rollout.core.harness` | `Task`, `Agent`, `tool`, `Observation`, `End`, `WaitFor`, `RunContext`, `rollout`, conversation types |
| `rollout.core.contracts` | `Message`, content blocks, `ToolSpecification`, `ToolResult`, identifiers, digests, events |
| `rollout.core.local` | `LocalRunContext`: runs an episode in process |
| `rollout.core.testing` | `ScriptedModelEndpoint`, `local_run`, `events_of`, `payload`, `tool_call_reply` |

## Conventions

- Hooks and tools are `async` methods. Everything a task or agent does outside its own code goes through `run`.
- Contract types are immutable pydantic models: construct new values rather than modifying existing ones.
- Type names are never abbreviated: `Observation`, not `Obs`.
- For AI coding assistants: [`llms.txt`](../../llms.txt) at the repository root indexes this guide.
