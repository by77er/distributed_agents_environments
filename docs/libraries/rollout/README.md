# The harness

For environment authors who need more than a task: the loop, programs, run specifications, runners, and how failures end
a run.

**Read first:** [Write a task](../../guide/tasks.md). **Next:** [Sandboxes](sandboxes.md).

Code: `rollout.harness` · See [Write an environment](../../guide/README.md), [contracts](contracts/README.md)

The harness is the loop that runs one [episode](../rollout-train/episodes.md) and the interfaces around it. The loop
drives a **task** (the environment the agent acts in) with an **agent** (the policy side). A **runner** executes runs of
it.

Writing tasks, tools and agents is covered by [Write an environment](../../guide/README.md). This page covers the rest:
the loop, programs, run specifications and runners. The fields and signatures of every type named here are in the
[API reference](../../guide/reference.md#rolloutharness).

| Page | Covers |
|---|---|
| [guide: tasks](../../guide/tasks.md) | `Task`, `Observation`, `End`, rewards, `RunContext` |
| [guide: tools](../../guide/tools.md) | `@tool` methods, imported tools, `ToolSet` |
| [guide: agents](../../guide/agents.md) | `Agent`, `History`, `ContextHints`, `Model` |
| [guide: runs and events](../../guide/runs-and-events.md) | effects, identifiers, run events |
| [sandboxes.md](sandboxes.md) | `SandboxSpec`, `Sandbox`, pools and providers: what a program runs against, leased for the run |
| [hooks.md](hooks.md) | `RunHooks`: watching every event and model sample of a runner |
| [memory.md](memory.md) | `Memory` and `CompactingAgent`: a context that fits any model |
| [determinism.md](determinism.md) | time, randomness and effect identity in task code, and the rules that keep them reliable |
| [contracts](contracts/README.md) | the types that cross layers, and what they guarantee |

## The loop

`rollout(task, agent, run)` runs one episode (`rollout.harness.loop`). The order of a task's hooks is in
[tasks](../../guide/tasks.md#the-parts-of-a-task). The loop guarantees:

- Every observation that `start` and `respond` return is checked against the
  [validation rules](../../guide/tasks.md#validation) before it is recorded. A violation raises `InvalidObservation`.
- `run.record` appends a turn to the history and records an `observation.recorded` event. `run.turn` counts
  recorded replies.
- When `max_turns` replies have been made and the episode has not ended, the loop records `End(truncated=True)`.
- `agent.act` must return an ASSISTANT message; anything else raises `TypeError`.
- `score` runs only when the loop ends with an observation whose `end` is set. After an exception it does not run.
- `teardown` runs on every path, including failure and cancellation.
- `record` is the one member of `RunContext` that only the loop uses.

## Program

A run executes a [`Program`](../../guide/reference.md#program). The loop is one program,
[`AgentProgram`](../../guide/reference.md#agentprogram); any other subclass implements `main` itself. A runner asks
a program what it needs before it starts the run:

| `Program` method | Default | `AgentProgram(task, agent)` |
|---|---|---|
| `model_slots()` | one slot, `policy` | the task's `models` |
| `context_hints()` | `ContextHints()` | the task's `context_hints` |
| `imports()` | none | the task's `imports` |
| `sandboxes()` | none | the task's `sandboxes` ([sandboxes](sandboxes.md)) |
| `tool_specifications()` | none | the specifications of the task's `@tool` methods |
| `await main(run)` | raises `NotImplementedError` | `await rollout(task, agent, run)` |

A run names its program with a [`ProgramReference`](../../guide/reference.md#programreference): the class as
`module:QualifiedName`, and JSON parameters. A runner in any process can create the program from it.

| Function | Use it to |
|---|---|
| `agent_program(task, agent)` | refer to the loop for a task class and an agent class. It registers both classes. |
| `register(cls)` | make a class resolvable by name in this process when it cannot be imported (one defined in a script) |
| `resolve(name)` | get the class a name refers to: registered, or imported |
| `instantiate(reference)` | create the program. A task, an agent or a program is constructed with its parameters, or with no arguments when they are `None`. |
| `with_row(reference, row)` | get the same program for another row of parameters. For the loop, the row replaces the task's parameters. |
| `bind(reference, channel)` | get a `RunBinding` that serves each model slot from a recorded [channel](../rollout-train/channels.md) (`channel`, unless `slots` names another for the slot; recorded as trained or not, as the slot declares), each import from the tool set registered under the import's name unless `tools` says otherwise, and each kind of sandbox from the pool registered under the kind's name unless `pools` says otherwise |

## Run specifications

A [`RunSpecification`](../../guide/reference.md#runspecification) is a program and a binding. The binding decides
what task and agent code must not: which model serves a slot, how it samples, where a tool set or a pool lives.

| Type | Says |
|---|---|
| [`RunBinding`](../../guide/reference.md#runbinding) | how each model slot and each import is served, and which pool serves each kind of sandbox |
| [`ModelBinding`](../../guide/reference.md#modelbinding) | exactly one of `direct` and `recorded` |
| [`DirectModel`](../../guide/reference.md#directmodel) | a provider's API. `provider` is the key of an endpoint factory registered with the runner. Nothing is recorded. |
| [`RecordedModel`](../../guide/reference.md#recordedmodel) | a channel served through the [gateway](../rollout-train/gateway.md), which records every sample; `trained` says whether its turns may be trained on (the slot's `ModelSlot.trained`) |
| [`SamplingParameters`](../../guide/reference.md#samplingparameters) | how a bound model samples. It belongs to bindings; task and agent code cannot set it. |
| [`ToolBinding`](../../guide/reference.md#toolbinding) | exactly one of `local` (a tool set registered with the runner) and `url` (a tool set served over HTTP, [tools](../../guide/tools.md#serving-a-tool-set-over-http)) |
| [`PoolBinding`](../../guide/reference.md#poolbinding) | exactly one of `local` (a pool registered with the runner) and `url` (a pool served over HTTP, [sandboxes](sandboxes.md#over-http)) |

## Runner

[`Runner`](../../guide/reference.md#runner) and [`RunHandle`](../../guide/reference.md#runhandle) are protocols.
Code written against them holds any implementation:

| Runner | Code | Behaviour |
|---|---|---|
| `LocalRunner(...)` | `rollout.local` | Runs each program as a task on the current asyncio loop. Nothing persists: a process crash loses its runs. |

It takes `providers` (endpoint factories for direct bindings, by provider name), `tool_sets` (for local tool
bindings, by name), `pools` (for local pool bindings, by name), `blobs`, `gateway` (serves recorded bindings: a
`RecordedEndpoints`, such as the [gateway](../rollout-train/gateway.md)'s endpoints) and `hooks`.

What the protocols guarantee:

- `launch` is called once, before anything else that starts or reaches a run. `close` releases what the runner
  holds, and the runner cannot be used afterwards.
- `start` begins a run and returns its handle. The run's `labels` are recorded in its `run.created` event. Its
  sandboxes are acquired under its `lease` (by default its `run_id`) before the program starts
  ([sandboxes](sandboxes.md#the-runner)).
- `run(run_id)` returns a run's handle, and raises `KeyError` for a run the runner does not know.
- A handle's `events(from_seq=...)` yields every event from `from_seq`, then new ones as they are recorded, and
  ends with the run. `recorded_events()` returns what is recorded so far. `outcome` is `None` while the run is live.
- `cancel` records `run.cancel_requested`, lets `teardown` run, and returns once the run has ended. The
  `LocalRunner` cancels the run's task at once.
- When a run ends, its runner releases its sandboxes.

A `LocalRunHandle` also has `context`, the run's `LocalRunContext`.

## Failures

| What happens | Result |
|---|---|
| A hook returns an observation that breaks the validation rules | `run.failed` with class `invalid_observation`; `RunOutcome.failure_class` is `RunFailureClass.INVALID_OBSERVATION` |
| Any other exception leaves the program: a hook raised (`setup` included), the agent returned a message that is not from the assistant, an endpoint error went unhandled | `run.failed` with class `task_error` and detail `ExceptionType: message`; `RunFailureClass.TASK_ERROR` |
| A sandbox cannot be acquired: its pool stays full for `ACQUIRE_SECONDS`, refuses the key (`LeaseRefused`), or has lost its sandbox (`SandboxLost`) | `run.failed` with class `task_error`, before the program starts ([sandboxes](sandboxes.md#the-runner)) |
| A `@tool` body raises or times out, its arguments do not validate, or the model calls a tool that does not exist | no failure: the model receives an error result ([tools](../../guide/tools.md#tool-errors-are-observations)) |
| An imported tool set raises | no failure: the effect completes as `failed` and the model receives an error result |
| A binding leaves a model slot, an import or a kind of sandbox unserved | `LocalRunner.start` raises `ValueError` |
| The process crashes | its runs are lost; an episode runner's episodes are open again in the [ledger](../rollout-train/checkpoints.md#the-ledger), and are played again |

`teardown` has run by the time a run fails or is cancelled.
