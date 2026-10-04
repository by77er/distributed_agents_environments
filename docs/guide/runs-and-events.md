# Runs and events

A run records what happened as a sequence of typed **run events**. Consumers such as episode runners, client event
streams, the monitor and tests read these events, never a runner's internals. This page covers what a run records
and how to read it.

## The local run context

`LocalRunContext` implements the [run context](tasks.md#the-run-context) in process: it owns the history, assigns
effect identities and records events. Nothing persists; a process crash loses the run. The
`LocalRunner` creates one per run; tests create one with `local_run` ([testing](testing.md)).

```py
LocalRunContext(
    run_id: str,                                 # r_{ulid}; see rollout.contracts.new_run_id
    endpoints: Mapping[str, ModelEndpoint],      # one endpoint per model slot
    *,
    context_hints: ContextHints | None = None,   # the task's hints, for the agent
    tool_sets: Mapping[str, ToolSet] | None = None,   # import name → tool set; becomes run.tools
    blobs: Blobs | None = None,                  # run.blobs
    on_event: Callable[[RunEvent], None] | None = None,   # called for each event as it is recorded
)
```

Beyond the run context's own members it has:

| Attribute or method | Holds |
|---|---|
| `events` | every `RunEvent`, in order |
| `rewards` | every `run.reward(...)` assignment, as `RewardAssignment(slot, value, key)`, including the episode reward from `score` |
| `excluded_from_training` | the reason given to `run.exclude_from_training`, or `None` |

## Effects and their identity

An **effect** is an operation that leaves task or agent code:

| Kind | Requested through |
|---|---|
| `model.sample` | `run.models[slot].sample(...)` |
| `tool.call` | a call to an imported tool (`run.tools.call`, which the default `respond` uses) |
| `output.emit` | `run.emit(...)` |

Each effect gets an identity, which receivers use to perform it once: the gateway answers a sample asked for again
under its `effect_id` with the turn it recorded ([determinism](../libraries/rollout/determinism.md)):

| Identifier | Format | Example |
|---|---|---|
| `run_id` | `r_{ulid}` | `r_01M3KEZ1MEYZHX9V33XZH65STF` |
| `effect_id` | `{run_id}:{generation}:{ordinal}`: the k-th effect the run requested; the generation is 0 for every run | `r_01M3…STF:0:0` |
| `session_id` | `{run_id}/{model_slot}` | `r_01M3…STF/policy` |
| `arguments_digest` | SHA-256 of the canonical JSON of the effect's arguments | `b09f2be8…` |

Both reach whatever performs the effect, so a receiver that deduplicates performs each `effect_id` once and can
tell a repeat from a different call. `EffectIdentity.parse` and `SessionIdentity.parse` split
identifiers into their parts; no other identifier should be parsed
([identifiers](../libraries/rollout/contracts/identifiers.md)).

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
tools, `sandboxes.acquired` when its program declares sandboxes, and one terminal event. A bare `LocalRunContext`, as above, has none of them.

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

The handle's `context` is the run's `LocalRunContext`. Cancelling and bindings are in the
[harness reference](../libraries/rollout/README.md#runner).

## Event types

| Group | Types |
|---|---|
| Lifecycle | `run.created`, `tools.resolved`, `sandboxes.acquired`, `run.cancel_requested`, and one terminal event: `run.completed`, `run.failed` ([failures](../libraries/rollout/README.md#failures)) or `run.cancelled` |
| Episode | `observation.recorded`, `reward.assigned` (`run.reward`, and `score`), `training.excluded`, `output.emitted` (`run.emit`) |
| Effects | `effect.requested`, `effect.completed` with status `ok` or `failed` |

A sample's completion carries the canonical reply, the finish reason and the usage. Tokens and logprobs never appear
in run events; they live in the gateway's turn store. Each type's payload and the order of events are specified in
[run events](../libraries/rollout/contracts/run-events.md).
