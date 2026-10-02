# The harness

Code: `rollout.harness` · See [developer guide](../../guide/README.md), [contracts](contracts/README.md)

The harness is the loop that runs one episode and the interfaces around it. The loop drives a **task** (the
environment the agent acts in) with an **agent** (the policy side). A **runner** executes runs of it.

Writing tasks, tools, agents and conversations is covered by the developer guide. This page covers the rest: the
loop, programs, run specifications, runners and how messages are routed. The fields and signatures of every type
named here are in the [API reference](../../guide/reference.md#rolloutharness).

| Page | Covers |
|---|---|
| [guide: tasks](../../guide/tasks.md) | `Task`, `Observation`, `End`, `WaitFor`, rewards, `RunContext` |
| [guide: tools](../../guide/tools.md) | `@tool` methods, imported tools, `ToolSet` |
| [guide: agents](../../guide/agents.md) | `Agent`, `History`, `ContextHints`, `Model` |
| [guide: conversations](../../guide/conversations.md) | `Envelope`, `Address`, priorities, delivery modes |
| [guide: runs and events](../../guide/runs-and-events.md) | effects, identifiers, run events |
| [hooks.md](hooks.md) | `RunHooks`: watching every event and model sample of a runner |
| [memory.md](memory.md) | `Memory` and `CompactingAgent`: a context that fits any model |
| [determinism.md](determinism.md) | rules for code that runs under the durable runner |
| [contracts](contracts/README.md) | the types that cross layers, and what they guarantee |

## The loop

`rollout(task, agent, run)` runs one episode (`rollout.harness.loop`). The order of a task's hooks is in
[tasks](../../guide/tasks.md#anatomy). The loop guarantees:

- Every observation that `start`, `respond`, `resume` and `steer` return is checked against the
  [validation rules](../../guide/tasks.md#validation) before it is recorded. A violation raises `InvalidObservation`.
- `run.record` appends a turn to the history and records an `observation.recorded` event. A `WaitFor` records only
  the reply it answers. `run.turn` counts recorded replies.
- A `WaitFor` suspends the run until a message of its kind arrives or its timeout passes. A message goes to
  `Task.resume`; a timeout continues with the wait's `on_timeout`.
- When `max_turns` replies have been made and the episode has not ended, the loop records `End(truncated=True)`.
- The agent acts inside `run.interruptible`. An interruption goes to `Task.resume`, and the reply that was cancelled
  is not recorded.
- `agent.act` must return an ASSISTANT message; anything else raises `TypeError`.
- Steering messages are taken after `respond`, and only when its observation does not end the episode.
- `score` runs only when the loop ends with an observation whose `end` is set. After an exception it does not run.
- `teardown` runs on every path, including failure and cancellation. The one exception is a cancellation with the
  message `rollout: unload`, which the durable runner uses to take a waiting run out of memory without ending it
  ([evicting idle runs](../../implementations/rollout-durable/eviction.md)).
- `record`, `wait_for_message`, `take_steering_messages` and `interruptible` are the members of `RunContext` that
  only the loop uses.

## Program

A run executes a [`Program`](../../guide/reference.md#program). The loop is one program,
[`AgentProgram`](../../guide/reference.md#agentprogram); any other subclass implements `main` itself. A runner asks
a program what it needs before it starts the run:

| `Program` method | Default | `AgentProgram(task, agent)` |
|---|---|---|
| `model_slots()` | one slot, `policy` | the task's `models` |
| `context_hints()` | `ContextHints()` | the task's `context_hints` |
| `imports()` | none | the task's `imports` |
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
| `bind(reference, channel)` | get a `RunBinding` that serves every model slot from one recorded channel, and each import from the tool set registered under the import's name unless `tools` says otherwise |

## Run specifications

A [`RunSpecification`](../../guide/reference.md#runspecification) is a program and a binding. The binding decides
what task and agent code must not: which model serves a slot, how it samples, where a tool set lives.

| Type | Says |
|---|---|
| [`RunBinding`](../../guide/reference.md#runbinding) | how each model slot and each import is served, and how priorities map to delivery modes ([conversations](../../guide/conversations.md#priority-and-delivery-mode)) |
| [`ModelBinding`](../../guide/reference.md#modelbinding) | exactly one of `direct` and `recorded` |
| [`DirectModel`](../../guide/reference.md#directmodel) | a provider's API. `provider` is the key of an endpoint factory registered with the runner. Nothing is recorded. |
| [`RecordedModel`](../../guide/reference.md#recordedmodel) | a channel served through the [recorder](../rollout-train/recorder.md) |
| [`SamplingParameters`](../../guide/reference.md#samplingparameters) | how a bound model samples. It belongs to bindings; task and agent code cannot set it. |
| [`ToolBinding`](../../guide/reference.md#toolbinding) | exactly one of `local` (a tool set registered with the runner) and `url` (a tool set served over HTTP, [tools](../../guide/tools.md#serving-a-tool-set-over-http)) |
| [`Deployment`](../../guide/reference.md#deployment) | a name, `{namespace}/{name}`, and the specification that a conversation addressed to it runs |

## Runner

[`Runner`](../../guide/reference.md#runner) and [`RunHandle`](../../guide/reference.md#runhandle) are protocols.
Code written against them holds either implementation:

| Runner | Code | Behaviour |
|---|---|---|
| `LocalRunner(...)` | `rollout.local` | Runs each program as a task on the current asyncio loop. Nothing persists: a process crash loses its runs. |
| `DurableRunner(directory, ...)` | `rollout_durable` | Runs each program inside a DBOS workflow in the runner's process. Effects are recorded steps, so a run resumes after a crash ([durability](../../implementations/rollout-durable/README.md)). |

Both take `providers` (endpoint factories for direct bindings, by provider name), `tool_sets` (for local tool
bindings, by name), `environments` (an `EnvironmentService`), `blobs`, `recorder` (serves recorded bindings) and
`hooks`.

What the protocols guarantee:

- `launch` is called once, before anything else that starts or reaches a run. `close` releases what the runner
  holds, and the runner cannot be used afterwards.
- `start` begins a run and returns its handle. The run's `labels` are recorded in its `run.created` event.
- `deploy` registers or replaces a deployment. A conversation's next run uses the current one.
- `run(run_id)` returns a run's handle, and raises `KeyError` for a run the runner does not know.
  `conversation_of` gives the conversation a run serves; `conversation_runs` gives a conversation's runs, oldest
  first.
- A handle's `events(from_seq=...)` yields every event from `from_seq`, then new ones as they are recorded, and
  ends with the run. `recorded_events()` returns what is recorded so far. `outcome` is `None` while the run is live.
- `cancel` records `run.cancel_requested`, lets `teardown` run, and returns once the run has ended. The
  `LocalRunner` cancels the run's task at once; the `DurableRunner` stops the run at its next effect, wait or turn
  boundary.
- When a run ends, its runner destroys the environments the run still owns.

A `LocalRunHandle` also has `context`, the run's `LocalRunContext`.

## Sending messages

Both runners inherit `send` from [`MessageRouter`](../../guide/reference.md#messagerouter). A runner supplies only
the transport: where its runs are, how a message reaches one, and where delivered messages are remembered. Using
`send` is covered in [conversations](../../guide/conversations.md#sending-and-replying). The router guarantees:

- **One identity per message.** The `message_id` is the caller's `idempotency_key`, or a new `m_{ulid}`. The
  runner writes it and the `sender` on the envelope.
- **Claim after delivery.** A message is remembered as delivered only once it is delivered. A send that fails
  before that can be retried with the same key. A retry of a delivered message returns its `message_id` and
  delivers nothing. This holds for run addresses and conversation addresses alike.
- **A conversation address** reaches the conversation's live run, and starts a run of the deployment's current
  specification when none is live. The `reply_to` of the message that starts a conversation becomes its `origin`.
  An address that names no deployed agent raises `ValueError`.
- **A run address** reaches that run, and raises [`RunNotLive`](../../guide/reference.md#runnotlive) when the run
  has ended or is unknown.
- **An external address** raises `ValueError`: a runner delivers to runs and conversations.
- **The delivery mode** comes from the receiving run's `DeliveryPolicy`, after the sender's priority is capped.
- **Hand-over.** Messages that a conversation's finished run never consumed go to its next run, which is started
  unless one is live. They are queued for it whatever the delivery policy says, in the order they were sent.
- **Exclusion.** A message is checked, delivered and claimed under one hold per address, among everything that
  sends to it.

## Failures

| What happens | Result |
|---|---|
| A hook returns an observation that breaks the validation rules | `run.failed` with class `invalid_observation`; `RunOutcome.failure_class` is `RunFailureClass.INVALID_OBSERVATION` |
| Any other exception leaves the program: a hook raised (`setup` included), the agent returned a message that is not from the assistant, an endpoint error went unhandled | `run.failed` with class `task_error` and detail `ExceptionType: message`; `RunFailureClass.TASK_ERROR` |
| A `@tool` body raises or times out, its arguments do not validate, or the model calls a tool that does not exist | no failure: the model receives an error result ([tools](../../guide/tools.md#errors-are-observations)) |
| An imported tool set raises | no failure: the effect completes as `failed` and the model receives an error result |
| A binding leaves a model slot or an import unserved | `LocalRunner.start` raises `ValueError` |
| The process crashes | `LocalRunner`: its runs are lost. `DurableRunner`: its runs resume by [replay](determinism.md) |

`teardown` has run by the time a run fails or is cancelled.
