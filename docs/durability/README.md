# Durability

Status: **Working** (2026-10-02) · Code: `rollout.durable`

`DurableRunner` implements the `Runner` protocol on DBOS, a durable-workflow library over SQLite or Postgres. A run
survives the death of its process: restarted, it continues where it stopped without repeating the effects it already
performed. Task, agent and program code is the same as under the `LocalRunner`
([runs and events](../guide/runs-and-events.md)); it must follow the
[determinism rules](../core/harness/determinism.md). The runner needs the `durable` extra.

```python
runner = DurableRunner(Path("state"), providers=providers, tool_sets=tool_sets, environments=environments)
runner.deploy(deployment)            # conversations of this deployment start runs of its specification
await runner.launch()                # starts DBOS, which resumes the runs a crash left unfinished
handle = await runner.start(specification)
outcome = await handle.result()
await runner.close()                 # unloads resident runs; they stay pending and resume on the next launch
```

One `DurableRunner` can be active in a process, because DBOS is a process-wide singleton. It takes the `LocalRunner`'s
keyword arguments (`providers`, `tool_sets`, `environments`, `blobs`, `recorder`, `hooks`) and these:

| Parameter | Default | Meaning |
|---|---|---|
| `directory` | required | local files; on SQLite it holds both databases |
| `application` | `"rollout"` | the DBOS application name |
| `evict_after`, `eviction_interval` | | unloading runs that wait: [evicting idle runs](eviction.md) |
| `database`, `runner_id`, `heartbeat_interval`, `takeover_after` | | sharing a Postgres database, and taking over a dead runner's runs: [several runners](runners.md) |

Besides `start`, `send` and `cancel`, the runner has `deploy(deployment)`, `run(run_id)`,
`conversation_runs(deployment, key)` and `conversation_of(run_id)`. Runs are returned as `DurableRunHandle`, which
reads the store and so stays valid across processes and restarts: `run_id`, `outcome`, `done`, `await result()`,
`events(from_seq=0)` (follows the run until its terminal event) and `recorded_events()`.

## State

DBOS's system database holds the journal: workflow inputs, step results and messages (`dbos.sqlite`, or the `dbos`
schema in Postgres). The run store (`RunStore`) holds these tables (`runs.sqlite`, or the same Postgres database):

| Table | Columns | Purpose |
|---|---|---|
| `runs` | `run_id`, `specification`, `conversation`, `status`, `outcome`, `evicted`, `wake_at`, `evictions`, `last_activity` | one row per run; `status` is `running` until the run ends |
| `events` | `run_id`, `seq`, `event` | the [run events](../contracts/run-events.md): a projection for consumers, not the journal. Inserts ignore an existing `(run_id, seq)`, so a replay never duplicates an event |
| `conversations` | `address`, `conversation_key`, `live_run_id` | each conversation's live run |
| `conversation_runs` | `address`, `run_id`, `position` | every run of a conversation, in order |
| `messages` | `message_id`, `address` | the claim table: messages delivered to conversations |
| `attempts` | `effect_id` | attempt markers of guarded effects |
| `runners` | `runner_id`, `heartbeat_at` | heartbeats of runners sharing the database |

## Runs and replay

A run is the DBOS workflow `rollout.run`, with the `run_id` as its workflow id. The workflow instantiates the program
and runs it with a `DurableRunContext`, inside the runner's process and with its privileges.

On recovery DBOS executes the workflow again from the start. Every step and receive it already performed returns its
recorded result, so the program takes the same path and regenerates the same effect identities and events. `run.now()`
is the time of the latest recorded input (the start, a step's completion, a message's sending, a wait's timeout), so
a replay sees the same clock.

A run ends as `completed`, `cancelled` or `failed` (`invalid_observation`, or `task_error` for any other exception).
The runner then destroys the environments the run still owns.

## Effects

Each effect is one DBOS step. The context assigns the `effect_id` and the argument digest
([effects](../contracts/effects.md)), records `effect.requested`, performs the effect inside the step and records
`effect.completed`.

DBOS records a step's result after the step ran, so a step in flight at a crash runs again on recovery. Receivers that
deduplicate by `effect_id` absorb the repeat, and a model call in flight is sampled again. Effects that cannot be
deduplicated are **guarded**: the step first inserts its `effect_id` into `attempts`. If the marker is already there,
an earlier attempt was interrupted, and the step raises `OutcomeUnknown` instead of performing the effect; the effect
completes with status `outcome_unknown`. Guarded effects are `Environment.execute` and calls to imported tools that
are side-effecting and do not deduplicate ([tools](../guide/tools.md#after-a-crash)).

## Messages

`send` returns the `message_id`: the `idempotency_key` when given, otherwise a new one. Priorities and delivery modes
are those of [conversations](../guide/conversations.md).

To a conversation, `send` works under a lock per conversation (in the process on SQLite, in the database on Postgres):

1. A `message_id` already in `messages` is a retry; nothing more happens.
2. The conversation's live run is found, or a run of its deployment is started.
3. The message goes to the run through DBOS `send`, with `{run_id}:{message_id}` as DBOS's idempotency key.
4. The `message_id` is claimed in `messages`. After a crash before this step a retry delivers again, which DBOS
   deduplicates while the same run is live.

To a run address, `send` delivers if the run is running and raises `RunNotLive` otherwise.

A run receives on two DBOS topics. `inbox` carries every message and cancellation request. The run reads it when it
waits (`WaitFor`), at turn boundaries (steering) and before each reply; reads are recorded, so a replay sees the same
messages. `interrupt` carries only a signal, for interrupting messages and cancellations. A reply races that signal
through `DBOS.asyncio_wait`, which records which finished first.

A wait records `run.suspended` and blocks in a DBOS receive with the wait's timeout; without a timeout it repeats
24-hour receives. When a conversation's run ends, the messages it never consumed are queued to the conversation's
next run, which is started if none is live.

## Cancellation

Cancellation is cooperative. `cancel(run_id, reason=…)` sends a request to the run's inbox and a signal on
`interrupt`, then waits for the run to end. The run records `run.cancel_requested`, raises `RunCancelled` at its next
effect, wait or turn boundary, runs `teardown` and ends as `cancelled`. Cancelling a run that has ended does nothing.

## Limits

- A message sent to a conversation between its run's last inbox read and the run being marked finished is claimed but
  never consumed.
- Unconsumed messages are handed to the next run by the process the run finished in. If that process dies after the
  workflow is recorded and before the hand-over, they are not delivered.
