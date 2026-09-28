# Runs and events

A run records what happened as a sequence of typed **run events**. Consumers such as the trajectory assembler,
client event streams and tests read these events, never a runner's internals. This page covers what a run records
today and how to read it.

## The local run context

`LocalRunContext` implements the run context in process: it owns the history, assigns effect identities, emits
events, and delivers messages. Nothing persists; a process crash loses the run.

```python fragment
LocalRunContext(
    run_id: str,                                 # r_{ulid}; see rollout.core.contracts.new_run_id
    task: Task,                                  # its declared model slots and context hints
    endpoints: Mapping[str, ModelEndpoint],      # one endpoint per model slot the task declares
    *,
    conversation: ConversationKey | None = None,
    generation: int = 0,
    on_event: Callable[[RunEvent], None] | None = None,   # called for each event as it is recorded
)
```

| Attribute or method | Holds |
|---|---|
| `events` | every `RunEvent`, in order |
| `rewards` | every `run.reward(...)` assignment, including the episode reward from `score` |
| `excluded_from_training` | the reason given to `run.exclude_from_training`, or `None` |
| `history` | the episode's turns |
| `deliver(envelope, mode)` | delivers a message ([conversations](conversations.md)) |

`local_run(task, replies)` from `rollout.core.testing` builds one with a scripted endpoint bound to every slot.

## Effects and their identity

An **effect** is an operation that leaves task or agent code. Today the model sample is the one effect the core
performs; tool calls to imported tools, environment operations, messages and timers join it later.

Each effect gets an identity that is stable across re-executions, which is what makes retries safe under the
durable runner:

| Identifier | Format | Example |
|---|---|---|
| `run_id` | `r_{ulid}` | `r_01M3KEZ1MEYZHX9V33XZH65STF` |
| `effect_id` | `{run_id}:{generation}:{ordinal}`, the k-th effect of the generation | `r_01M3…STF:0:0` |
| `session_id` | `{run_id}/{model_slot}` | `r_01M3…STF/policy` |
| `arguments_digest` | SHA-256 of the canonical JSON of the effect's arguments | `b09f2be8…` |

A receiver that sees a known `effect_id` with a different digest rejects it. `EffectIdentity.parse` and
`SessionIdentity.parse` split identifiers into their parts; no other identifier should be parsed.

## Events

Each `RunEvent` has `run_id`, `seq` (gapless from 0), `type`, `schema_version`, `recorded_at` and a JSON
`payload`. One single-turn episode with a score produces:

```python
import asyncio

from rollout.core.contracts import Message, RunEventType
from rollout.core.harness import Agent, End, Observation, RunContext, Task, rollout
from rollout.core.testing import events_of, local_run, payload


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

> **Known gap:** the contract says `seq = 0` is `run.created`, and runs end with `run.completed`, `run.failed` or
> `run.cancelled`. Those lifecycle events are the runner's job and arrive with the `LocalRunner` (M0 task 5). A bare
> `LocalRunContext` starts with the first observation.

### Events emitted today

| Type | When | Payload |
|---|---|---|
| `observation.recorded` | a hook's observation is recorded | `messages`, `reward`, `end`, `reply_effect_id` (`null` for observations that answer no reply), `info`, `digest` |
| `effect.requested` | an effect starts | `effect_id`, `kind`, `arguments_digest`, `payload` (the arguments) |
| `effect.completed` | an effect finishes | `effect_id`, `status` (`ok`, `failed`), `payload`, `error_class` when it failed or was cancelled |
| `reward.assigned` | `run.reward(...)`, including `score` | `slot`, `value`, `key` |
| `training.excluded` | `run.exclude_from_training(reason)` | `reason` |
| `run.suspended` | the run waits for a message | `waiting_for`: `kind`, `timeout` in seconds |
| `message.received` | a message is delivered | `envelope`, `mode` |
| `turn.interrupted` | an interrupt cancels the agent's reply | `reply_effect_id` of the cancelled sample, if one was in flight |

A `model.sample` effect's arguments are the `session_id`, the context digest, the spec hashes of the offered tools,
`max_output_tokens` and `tool_choice`. Its completion payload is the `SampleResult`: message, finish reason, usage.
Tokens and logprobs never appear in run events; they live in the recorder.

The full catalog, including the events that later layers emit, is in [run events](../contracts/run-events.md).

## Failures

`rollout()` raises when an episode cannot finish. `teardown` has run by then.

| Raised | Cause | The runner (M0 task 5) records |
|---|---|---|
| `InvalidObservation` | a hook returned an observation that breaks the [validation rules](tasks.md#validation) | `run.failed` with class `INVALID_OBSERVATION` |
| the hook's exception | a task hook raised | `run.failed` with class `TASK_ERROR` |
| `TypeError` | the agent returned a message that is not from the assistant | `run.failed` with class `TASK_ERROR` |
| errors from the endpoint | e.g. `ContextOverflow` that the agent did not handle | `run.failed` |

Exceptions inside `@tool` bodies do not fail the run: they become error results for the model ([tools](tools.md)).
`Interrupted` never escapes `rollout()`; the loop handles it.
