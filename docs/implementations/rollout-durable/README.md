# Durable runner

Code: `rollout_durable`

`DurableRunner` implements the [`Runner`](../../guide/reference.md#runner) protocol on DBOS, a
durable-workflow library over SQLite or Postgres. A run survives the death of its process: restarted, it continues
where it stopped without repeating the effects it already performed. Task, agent and program code is the same as
under the `LocalRunner` ([runs and events](../../guide/runs-and-events.md)); it must follow the
[determinism rules](../../libraries/rollout/determinism.md). The package is `rollout-durable`
(`implementations/rollout-durable`), which `uv sync` at the repository root installs.

```python
from rollout_durable import DurableRunner

runner = DurableRunner(Path("state"), providers=providers, tool_sets=tool_sets, environments=environments)
runner.deploy(deployment)            # conversations of this deployment start runs of its specification
await runner.launch()                # starts DBOS, which resumes the runs a crash left unfinished
handle = await runner.start(specification)
outcome = await handle.result()
await runner.close()                 # unloads resident runs; they stay pending and resume on the next launch
```

One `DurableRunner` can be active in a process, because DBOS is a process-wide singleton. It takes the `LocalRunner`'s
keyword arguments (`providers`, `tool_sets`, `environments`, `blobs`, `recorder`, `hooks`) and the ones below. Their
defaults are in the [reference](../../guide/reference.md#durablerunner).

| Parameter | Meaning |
|---|---|
| `directory` | Local files; on SQLite it holds both databases |
| `application` | The DBOS application name |
| `evict_after`, `eviction_interval` | Unloading runs that wait: [evicting idle runs](eviction.md) |
| `database`, `runner_id`, `heartbeat_interval`, `takeover_after` | Sharing a Postgres database, and taking over a dead runner's runs: [several runners](runners.md) |

Its methods and guarantees are those of the [`Runner` protocol](../../libraries/rollout/README.md#runner). Its
handles, `DurableRunHandle`, read the store, so a handle stays valid across processes and restarts.

A [profile](../../guide/deploying.md) whose `runner` is `durable` runs its episodes on one, with its state under
`runs/` in the run's directory.

## State

DBOS's system database holds the journal: workflow inputs, step results and messages (`dbos.sqlite`, or the `dbos`
schema in Postgres). The run store (`RunStore`) holds these tables (`runs.sqlite`, or the same Postgres database):

| Table | Columns | Purpose |
|---|---|---|
| `runs` | `run_id`, `specification`, `conversation`, `status`, `outcome`, `evicted`, `wake_at`, `evictions`, `last_activity` | One row per run; `status` is `running` until the run ends |
| `events` | `run_id`, `seq`, `event` | The [run events](../../libraries/rollout/contracts/run-events.md): a projection for consumers, not the journal. Inserts ignore an existing `(run_id, seq)`, so a replay never duplicates an event |
| `conversations` | `address`, `conversation_key`, `live_run_id` | Each conversation's live run |
| `conversation_runs` | `address`, `run_id`, `position` | Every run of a conversation, in order |
| `messages` | `message_id`, `address` | The claim table: messages that were delivered |
| `attempts` | `effect_id` | Attempt markers of guarded effects |
| `runners` | `runner_id`, `heartbeat_at` | Heartbeats of runners sharing the database |

`DurableRunner.database` is the [`Database`](runners.md#the-database) these tables are in. A product may keep its own
tables in it.

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
([effects](../../libraries/rollout/contracts/effects.md)), records `effect.requested`, performs the effect inside the
step and records `effect.completed`.

DBOS records a step's result after the step ran, so a step in flight at a crash runs again on recovery. Receivers that
deduplicate by `effect_id` absorb the repeat, and a model call in flight is sampled again. Effects that cannot be
deduplicated are **guarded**: the step first inserts its `effect_id` into `attempts`. If the marker is already there,
an earlier attempt was interrupted, and the step raises `OutcomeUnknown` instead of performing the effect; the effect
completes with status `outcome_unknown`. Guarded effects are `Environment.execute`, and calls to imported tools and
operations on sandboxes that are side-effecting and do not deduplicate ([tools](../../guide/tools.md#after-a-crash)).

A run acquires its [sandboxes](../../libraries/rollout/sandboxes.md#the-runner) each time it is executed, before its
program, under the lease it was started with: recovered or woken, it gets the same sandboxes back while their leases
hold. It releases them when its program ends, and not when it is unloaded. A replay that takes another way than the
first execution (a sandbox refused on resuming, say) records its terminal event after every event stored, so the
run's stream still ends. The runner says `resumes = True`: an episode runner over it leaves its runs to be resumed
when it closes, and adopts them when it starts again.

## Messages

`send` is the harness's [`MessageRouter.send`](../../libraries/rollout/README.md#sending-messages), the same code
under both runners: under a lock on the address, a claimed `message_id` is a retry, otherwise the message is delivered
to the run and then claimed. The durable runner supplies the transport:

| Part of `send` | In the durable runner |
|---|---|
| The lock on an address | `Database.lock`: in the process on SQLite, in the database on Postgres |
| Claims | The `messages` table |
| Delivery | DBOS `send` to the run, with `{run_id}:{message_id}` as DBOS's idempotency key, under the run's own lock. It records the run's `last_activity` and wakes the run if it is evicted |

After a crash between delivery and claim, a retry delivers again, which DBOS deduplicates while the same run is live.

A run receives on two DBOS topics. `inbox` carries every message and cancellation request. The run reads it when it
waits (`WaitFor`), at turn boundaries (steering) and before each reply; reads are recorded, so a replay sees the same
messages. `interrupt` carries only a signal, for interrupting messages and cancellations. A reply races that signal
through `DBOS.asyncio_wait`, which records which finished first.

A wait records `run.suspended` and blocks in a DBOS receive with the wait's timeout; without a timeout it repeats
receives of `LONG_WAIT_SECONDS`. When a conversation's run ends, the process it ran in hands the messages it never
consumed to the conversation's next run (`MessageRouter`'s hand-over).

## Cancellation

Cancellation is cooperative. `cancel(run_id, reason=…)` sends a request to the run's inbox and a signal on
`interrupt`, wakes the run if it is [evicted](eviction.md), then waits for the run to end. The run records `run.cancel_requested`, raises `RunCancelled` at its next
effect, wait or turn boundary, runs `teardown` and ends as `cancelled`. Cancelling a run that has ended does nothing.

## Limits

- A message sent to a conversation between its run's last inbox read and the run being marked finished is claimed but
  never consumed.
- If the process a conversation's run finished in dies after the workflow is recorded and before the hand-over, the
  messages the run never consumed are not delivered.
