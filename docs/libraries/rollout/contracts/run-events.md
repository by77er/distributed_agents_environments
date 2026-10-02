# Run events

Code: `rollout.contracts.events` · See [`RunEvent`](../../../guide/reference.md#runevent),
[`RunEventType`](../../../guide/reference.md#runeventtype), [guide: runs and events](../../../guide/runs-and-events.md)

Run events are the typed record of what happened in a run. Every runner produces the same stream: the `LocalRunner`
keeps it in memory, the `DurableRunner` in its store
([durability](../../../implementations/rollout-durable/README.md#state)). Rollout jobs, hooks, the monitor, clients
and tests read these events, never a runner's internals.

## The stream

- **`seq`** numbers a run's events from 0 without gaps. A reader that has seen `seq` n asks for the events from
  n + 1 (`RunHandle.events(from_seq=...)`).
- **`recorded_at`** is `run.now()` when the event was recorded. Under the `LocalRunner` that is the wall clock.
  Under the `DurableRunner` it is the time of the run's latest recorded input, so a replay records the same events
  with the same times, and the store keeps the first copy of each ([determinism](../determinism.md#replay)).
- **`schema_version`** is `RUN_EVENT_SCHEMA_VERSION`.
- **`payload`** is JSON. Its keys depend on the type and are listed below.
- **The catalog is closed.** `RunEventType` lists every type.

## Catalog

### Lifecycle

| Type | Recorded when | Payload |
|---|---|---|
| `run.created` | a runner starts the run. It is the run's first event. | `specification`, `conversation` (or `null`), `labels` |
| `tools.resolved` | the run starts with tools. It follows `run.created`. | `specifications`: the program's `@tool` methods, then the imported tools |
| `run.suspended` | the loop starts to wait for a message and none of the wanted kind is held | `waiting_for`: `kind`, and `timeout` in seconds or `null` |
| `run.cancel_requested` | a cancellation reaches the run | `reason`, `by` |
| `run.completed` | the program returned | `outcome` |
| `run.failed` | the program raised | `class` (a [`RunFailureClass`](../../../guide/reference.md#runfailureclass)), `detail` ([failures](../README.md#failures)) |
| `run.cancelled` | the run stopped after a cancellation | none |

### Episode

| Type | Recorded when | Payload |
|---|---|---|
| `observation.recorded` | the loop records an observation | `messages`, `reward`, `end`, `reply_effect_id` (the sample whose reply it answers, or `null`), `info`, `digest` (of the messages, the reward and the ending) |
| `reward.assigned` | `run.reward(...)`, which the loop also calls with what `score` returns | `slot`, `value`, `key` |
| `training.excluded` | `run.exclude_from_training(reason)` | `reason` |
| `output.emitted` | `run.emit(...)`, after its effect has completed | `kind`, `payload`, `to` when given, `effect_id` |

### Effects

| Type | Recorded when | Payload |
|---|---|---|
| `effect.requested` | an effect starts | `effect_id`, `kind`, `arguments_digest`, `payload` (the arguments, [effects](effects.md#identity)) |
| `effect.completed` | the effect finishes, however it finishes | `effect_id`, `status`, `payload` (the result, or the error's message), and `error_class` unless the status is `ok` |

A model sample completes with the sample result: the canonical reply, the finish reason and the usage. Tokens and
logprobs never appear in run events; a recorder keeps them ([recorder](../../rollout-train/recorder.md)).

### Messages

| Type | Recorded when | Payload |
|---|---|---|
| `message.received` | a message is delivered to the run | `envelope`, `mode` |
| `turn.interrupted` | a message delivered with mode `interrupt` cancels the agent's reply | `reply_effect_id`: the sample that was cancelled, or `null` when none was in flight |

## Order

- `run.created` is at `seq` 0. A run context made without a runner, as tests make one, records no lifecycle events.
- An `effect.requested` comes before the `effect.completed` with the same `effect_id`. Effects in flight together
  interleave.
- A run started by a runner ends with exactly one of `run.completed`, `run.failed` and `run.cancelled`
  (`TERMINAL_EVENT_TYPES`). It is the run's last event, and it decides the status of the
  [`RunOutcome`](../../../guide/reference.md#runoutcome).
- `teardown` has run before the terminal event is recorded.

## What is read from them

| Reader | Reads |
|---|---|
| A rollout job | `run.created` (labels), rewards on observations and `reward.assigned`, `output.emitted` of kind `result`, `training.excluded`, and the terminal event: together an [episode](../../rollout-train/episodes.md#how-an-episode-is-assembled) |
| [Hooks](../hooks.md) | every event, as it is recorded |
| A client of a conversation | `output.emitted`: what the agent said, and to whom |
| The eviction of idle runs | `run.suspended`: since when a run has waited |
