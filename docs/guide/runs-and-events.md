# Runs and events

A run records what happened as a sequence of typed **run events**. Consumers such as rollout jobs, client event
streams, the monitor and tests read these events, never a runner's internals. This page covers what a run records
and how to read it.

## The local run context

`LocalRunContext` implements the [run context](tasks.md#the-run-context) in process: it owns the history, assigns
effect identities, records events, and delivers messages. Nothing persists; a process crash loses the run. The
`LocalRunner` creates one per run; tests create one with `local_run` ([testing](testing.md)).

```python fragment
LocalRunContext(
    run_id: str,                                 # r_{ulid}; see rollout.contracts.new_run_id
    endpoints: Mapping[str, ModelEndpoint],      # one endpoint per model slot
    *,
    context_hints: ContextHints | None = None,   # the task's hints, for the agent
    tool_sets: Mapping[str, ToolSet] | None = None,          # import name → tool set; becomes run.tools
    environment_service: EnvironmentService | None = None,   # the backend behind run.environments
    blobs: Blobs | None = None,                  # run.blobs
    conversation: ConversationKey | None = None,
    generation: int = 0,                         # the middle part of every effect_id
    on_event: Callable[[RunEvent], None] | None = None,   # called for each event as it is recorded
    retain_events: bool = True,                  # keep every event in `events`
)
```

Beyond the run context's own members it has:

| Attribute or method | Holds |
|---|---|
| `events` | every `RunEvent`, in order |
| `rewards` | every `run.reward(...)` assignment, as `RewardAssignment(slot, value, key)`, including the episode reward from `score` |
| `excluded_from_training` | the reason given to `run.exclude_from_training`, or `None` |
| `deliver(envelope, mode)` | delivers a message ([conversations](conversations.md)) |

## Effects and their identity

An **effect** is an operation that leaves task or agent code:

| Kind | Requested through |
|---|---|
| `model.sample` | `run.models[slot].sample(...)` |
| `tool.call` | a call to an imported tool (`run.tools.call`, which the default `respond` uses) |
| `environment.lifecycle` | `run.environments.create(...)` and `Environment.destroy()` |
| `environment.call` | `Environment.execute`, `put` and `get` |
| `output.emit` | `run.emit(...)` |

Each effect gets an identity that is the same every time the run's code is run again, which is what makes retries
safe under the durable runner ([determinism](../core/harness/determinism.md)):

| Identifier | Format | Example |
|---|---|---|
| `run_id` | `r_{ulid}` | `r_01M3KEZ1MEYZHX9V33XZH65STF` |
| `effect_id` | `{run_id}:{generation}:{ordinal}`: the k-th effect the run requested; runners use generation 0 | `r_01M3…STF:0:0` |
| `session_id` | `{run_id}/{model_slot}` | `r_01M3…STF/policy` |
| `arguments_digest` | SHA-256 of the canonical JSON of the effect's arguments | `b09f2be8…` |

Both reach whatever performs the effect, so a receiver that deduplicates performs each `effect_id` once and can
tell a repeat from a different call. `EffectIdentity.parse` and `SessionIdentity.parse` split identifiers into their
parts; no other identifier should be parsed.

## Events

Each `RunEvent` has `run_id`, `seq` (gapless from 0), `type`, `schema_version`, `recorded_at` and a JSON
`payload`. One single-turn episode with a score produces:

```python
import asyncio

from rollout.contracts import Message, RunEventType
from rollout.harness import Agent, End, Observation, RunContext, Task, rollout
from rollout.testing import events_of, local_run, payload


class Addition(Task):
    async def start(self, run: RunContext) -> Observation:
        return Observation("What is 2 + 2?")

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        return End(reward=1.0 if reply.text.strip() == "4" else 0.0)

    async def score(self, run: RunContext) -> float | None:
        return 1.0


async def main() -> None:
    task = Addition()
    run, endpoint = local_run(task, replies=["4"])
    await rollout(task, Agent(), run)

    assert [event.type for event in run.events] == [
        RunEventType.OBSERVATION_RECORDED,   # the start observation
        RunEventType.EFFECT_REQUESTED,       # the policy sample
        RunEventType.EFFECT_COMPLETED,
        RunEventType.OBSERVATION_RECORDED,   # End(reward=1.0), bound to the reply
        RunEventType.REWARD_ASSIGNED,        # score() → run.reward(1.0)
    ]
    sample = endpoint.requests[0]
    final = payload(events_of(run, RunEventType.OBSERVATION_RECORDED)[-1])
    assert final["reply_effect_id"] == sample.effect_id
    assert final["reward"] == 1.0 and final["end"] == "terminated"


asyncio.run(main())
```

A run started by a runner also has lifecycle events: `run.created` at `seq = 0`, `tools.resolved` when it has
tools, and one terminal event. A bare `LocalRunContext`, as above, has none of them.

## The local runner

`LocalRunner` runs programs as asyncio tasks in the current process. A run is described by a `RunSpecification`: a
program, named so that another process could re-create it, and a binding that says which endpoint serves each model
slot. Direct model bindings name a provider, and the runner creates endpoints with the factory registered for it.

```python
from rollout.harness import DirectModel, ModelBinding, RunBinding, RunSpecification, RunStatus, agent_program
from rollout.local import LocalRunner
from rollout.testing import ScriptedModelEndpoint


async def run_with_runner() -> None:
    endpoint = ScriptedModelEndpoint(["4"])
    runner = LocalRunner(providers={"scripted": lambda model: endpoint})
    specification = RunSpecification(
        program=agent_program(Addition),                  # the task loop for Addition with the default Agent
        binding=RunBinding(models={"policy": ModelBinding(direct=DirectModel(provider="scripted", model="script"))}),
    )
    handle = await runner.start(specification, labels={"suite": "smoke"})

    types = [event.type async for event in handle.events()]   # streams until the run ends
    assert types[0] is RunEventType.RUN_CREATED and types[-1] is RunEventType.RUN_COMPLETED
    assert (await handle.result()).status is RunStatus.COMPLETED


asyncio.run(run_with_runner())
```

The handle's `context` is the run's `LocalRunContext`. Sending messages, cancelling, deployments, bindings and the
`DurableRunner` are in the [harness reference](../core/harness/README.md#runner).

## Event types

| Type | When | Payload |
|---|---|---|
| `run.created` | a runner starts the run | `specification`, `conversation`, `labels` |
| `tools.resolved` | the run starts with tools | `specifications`: the `@tool` methods, then the imported tools |
| `observation.recorded` | a hook's observation is recorded | `messages`, `reward`, `end`, `reply_effect_id` (`null` for observations that answer no reply), `info`, `digest` |
| `effect.requested` | an effect starts | `effect_id`, `kind`, `arguments_digest`, `payload` (the arguments) |
| `effect.completed` | an effect finishes | `effect_id`, `status` (`ok`, `failed`, `outcome_unknown`), `payload`, `error_class` unless the status is `ok` |
| `reward.assigned` | `run.reward(...)`, including `score` | `slot`, `value`, `key` |
| `training.excluded` | `run.exclude_from_training(reason)` | `reason` |
| `output.emitted` | `run.emit(...)`, after its effect | `kind`, `payload`, `to` when given, `effect_id` |
| `run.suspended` | the run starts waiting for a message | `waiting_for`: `kind`, `timeout` in seconds |
| `message.received` | a message is delivered | `envelope`, `mode` |
| `turn.interrupted` | an interrupt cancels the agent's reply | `reply_effect_id` of the cancelled sample, if one was in flight |
| `run.cancel_requested` | `runner.cancel(run_id, reason=...)` | `reason`, `by` |
| `run.completed` | the program returned | `outcome` |
| `run.failed` | the program raised | `class`, `detail` ([failures](../core/harness/README.md#failures)) |
| `run.cancelled` | the run stopped after a cancellation | |

A `model.sample` effect's arguments are the `session_id`, the context digest, the spec hashes of the offered tools,
`max_output_tokens` and `tool_choice`. Its completion payload is the `SampleResult`: message, finish reason, usage.
Tokens and logprobs never appear in run events; they live in the recorder.

The events are specified in [run events](../contracts/run-events.md).
