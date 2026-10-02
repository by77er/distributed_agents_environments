# Harness

Status: **Working** (2026-10-02) · Code: `rollout.harness`

The harness is the framework-owned rollout loop and the interfaces around it. The loop drives a **task** (the
environment the agent acts in) with an **agent** (the policy side). A **runner** executes runs of it.

Writing tasks, tools, agents and conversations is covered by the [developer guide](../../guide/README.md). This page
is the reference for the rest: the loop itself, programs, run specifications and runners.

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

## The loop

`rollout(task, agent, run)` runs one episode. This is `rollout.harness.loop`, without its validation calls:

```python
async def rollout(task: Task, agent: Agent, run: RunContext) -> None:
    unloading = False
    try:
        await task.setup(run)
        observation = await task.start(run)
        run.record(observation)
        while True:
            if isinstance(observation, WaitFor):
                envelope = await run.wait_for_message(observation)
                observation = observation.on_timeout if envelope is None else await task.resume(run, envelope)
                run.record(observation)
                continue
            if observation.end is not None:
                break
            if task.max_turns is not None and run.turn >= task.max_turns:
                observation = End(truncated=True)
                run.record(observation)
                break
            try:
                reply = await run.interruptible(agent.act(run, run.history, task.tools_for_turn(run)))
            except Interrupted as interruption:
                observation = await task.resume(run, interruption.envelope)
                run.record(observation)
                continue
            observation = await task.respond(run, reply)
            if isinstance(observation, Observation) and observation.end is None:
                steering = await run.take_steering_messages()
                if steering:
                    observation = await task.steer(run, steering, observation)
            run.record(observation, reply=reply)
        episode_reward = await task.score(run)
        if episode_reward is not None:
            run.reward(episode_reward)
    except asyncio.CancelledError as cancelled:
        unloading = cancelled.args == ("rollout: unload",)
        raise
    finally:
        if not unloading:
            await task.teardown(run)
```

- Every observation that `start`, `respond`, `resume` and `steer` return is checked against the
  [validation rules](../../guide/tasks.md#validation) before it is recorded. A violation raises `InvalidObservation`.
- `run.record` appends a turn to the history and records an `observation.recorded` event. A `WaitFor` records only
  the reply it answers. `run.turn` counts recorded replies.
- `agent.act` must return an ASSISTANT message; anything else raises `TypeError`.
- `score` runs only when the loop ends with an observation whose `end` is set. After an exception it does not run.
- `teardown` runs on every path, including failure and cancellation. The one exception is a cancellation with the
  message `rollout: unload`, which the durable runner uses to take a waiting run out of memory without ending it
  ([evicting idle runs](../../durability/eviction.md)).
- `record`, `wait_for_message`, `take_steering_messages` and `interruptible` are the members of `RunContext` that
  only the loop uses.

## Program

A run executes a `Program`. The loop is one program, `AgentProgram`; any other subclass implements `main` itself.

| `Program` method | Default | `AgentProgram(task, agent)` |
|---|---|---|
| `model_slots()` | `{"policy": ModelSlot()}` | the task's `models` |
| `context_hints()` | `ContextHints()` | the task's `context_hints` |
| `imports()` | `[]` | the task's `imports` |
| `tool_specifications()` | `[]` | the specifications of the task's `@tool` methods |
| `await main(run)` | raises `NotImplementedError` | `await rollout(task, agent, run)` |

A run names its program with a `ProgramReference`, so that a runner in any process can create it:

| Field | Type | Meaning |
|---|---|---|
| `program` | `str` | `module:QualifiedName` of a `Program` class |
| `parameters` | JSON | passed to the program's constructor; for `AgentProgram`: `task`, `agent`, `task_parameters`, `agent_configuration` |
| `code_reference` | `str \| None` | `{package}@{content_hash}`, carried with the specification |

| Function | Does |
|---|---|
| `agent_program(task, agent=Agent, *, task_parameters=None, agent_configuration=None)` | a reference to the loop for a task class and an agent class; registers both classes |
| `register(cls)` | makes a class resolvable by name in this process even if it cannot be imported (defined in a script, say); returns the name |
| `resolve(name)` | the class a name refers to: registered, or imported |
| `instantiate(reference)` | creates the program; a task or agent is constructed with its parameters, or with no arguments when they are `None` |
| `with_row(reference, row)` | the same program for another row of parameters (for the loop: the task's parameters) |
| `bind(reference, channel, *, tools=None)` | a `RunBinding` that serves every model slot from one recorded channel, and each import from the tool set registered under the import's name, or as `tools` says |

## Run specifications

| Type | Field | Type | Meaning |
|---|---|---|---|
| `RunSpecification` | `program` | `ProgramReference` | what to run |
| | `binding` | `RunBinding` | how its slots and imports are served |
| `RunBinding` | `models` | `Mapping[str, ModelBinding]` | model slot → how it is served |
| | `imports` | `Mapping[str, ToolBinding]` | import name → how the tool set is served; default `{}` |
| | `delivery` | `DeliveryPolicy` | priority → delivery mode ([conversations](../../guide/conversations.md#priority-and-delivery-mode)) |
| `ModelBinding` | `direct` | `DirectModel \| None` | a provider's API: `provider` (the key of an endpoint factory registered with the runner), `model`, `sampling` |
| | `recorded` | `RecordedModel \| None` | a channel served through the [recorder](../recorder/README.md): `channel`, `sampling` |
| `ToolBinding` | `local` | `str \| None` | the name of a tool set registered with the runner, in its process |
| | `url` | `str \| None` | a tool set served over HTTP ([tools](../../guide/tools.md#serving-a-tool-set-over-http)) |
| `Deployment` | `name` | `str` | `{namespace}/{name}`, e.g. `acme/support-bot` |
| | `specification` | `RunSpecification` | what a conversation addressed to the deployment runs |

A `ModelBinding` sets one of `direct` and `recorded`; a `ToolBinding` sets one of `local` and `url`.
`SamplingParameters` has `temperature` (1.0), `top_p` (1.0), `top_k`, `max_output_tokens`, `stop`, `seed` and
`reasoning_effort`. It belongs to bindings; task and agent code cannot set it.

## Runner

```python
class Runner(Protocol):
    async def start(
        self,
        specification: RunSpecification,
        *,
        run_id: str | None = None,
        conversation: ConversationKey | None = None,
        labels: Mapping[str, str] | None = None,
    ) -> RunHandle: ...

    async def send(
        self,
        to: Address,
        envelope: Envelope,
        *,
        priority: Priority = Priority.NORMAL,
        idempotency_key: str | None = None,
    ) -> str: ...            # the message_id; a message to a conversation starts its run when none is live

    async def cancel(self, run_id: str, *, reason: str) -> None: ...


class RunHandle(Protocol):
    @property
    def run_id(self) -> str: ...
    async def result(self) -> RunOutcome: ...
    def events(self, *, from_seq: int = 0) -> AsyncIterator[RunEvent]: ...   # from from_seq, until the run ends
```

`RunOutcome` has `status` (`RunStatus.COMPLETED`, `FAILED` or `CANCELLED`), `failure_class` and `detail`.

| Runner | Code | Behaviour |
|---|---|---|
| `LocalRunner(...)` | `rollout.local` | Runs each program as a task on the current asyncio loop. Nothing persists: a process crash loses its runs. |
| `DurableRunner(directory, ...)` | `rollout_durable` | Runs each program inside a DBOS workflow in the runner's process. Effects are recorded steps, so a run resumes after a crash ([durability](../../durability/README.md)). Needs `await launch()` before use and `await close()` after. |

Both take `providers` (endpoint factories for direct bindings, by provider name), `tool_sets` (for local tool
bindings, by name), `environments` (an `EnvironmentService`), `blobs`, `recorder` (serves recorded bindings) and
`hooks`. Beyond the protocol, both have:

| Member | Does |
|---|---|
| `send(..., sender=None)` | names the sender, which `DeliveryPolicy.max_priority_by_sender` caps |
| `deploy(deployment)` | registers or replaces a deployment; a conversation's next run uses the current one |
| `run(run_id)` | the run's handle |
| `conversation_of(run_id)` | the `ConversationKey` the run serves, or `None` |
| `conversation_runs(deployment, key)` | the handles of a conversation's runs, oldest first |

Their handles add `done`, `outcome` (the `RunOutcome`, or `None` while the run is live) and `recorded_events()`.
A `LocalRunHandle` also has `context`, the run's `LocalRunContext`.

`cancel` records `run.cancel_requested`, lets `teardown` run, and returns once the run has ended. The `LocalRunner`
cancels the run's task at once; the `DurableRunner` stops the run at its next effect, wait or turn boundary. When a
run ends, its runner destroys the environments the run still owns.

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
