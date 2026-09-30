# Durability

Status: **Proposed** · Layer: durability (optional) · See [ADR-0017](../decisions/0017-dbos-substrate.md), [ADR-0018](../decisions/0018-effects-at-least-once.md), [research](../research/substrate-dbos.md)

## Purpose

`DurableRunner` implements the core [`Runner`](../core/harness/README.md#runner) protocol so that runs survive
process and machine failures, suspend for days without holding compute, and resume on new code at defined points.
It is built on **DBOS** (MIT, a durable-workflow library over Postgres or SQLite). Task, agent and program code is
unchanged; it only has to follow the [determinism rules](../core/harness/determinism.md).

## Implementation status

`rollout.durable.DurableRunner` exists in a first, **trusted-mode** form (2026-09-28). What it does and what it does
not do yet:

| Design element | Status |
|---|---|
| Runs as DBOS workflows (workflow id = `run_id`), on SQLite (one runner) or Postgres (several) | built |
| Effects as recorded steps with deterministic `effect_id`s; replay by re-execution | built: model samples, imported tool calls, outputs |
| `run.now()` = time of the latest recorded input | built |
| Durable `WaitFor` (a recorded receive with a durable timeout) | built |
| Delivery modes: queue and steer read at turn boundaries; interrupt races the reply through `DBOS.asyncio_wait` | built |
| Cooperative cancellation through the inbox; `teardown` runs | built |
| Run events as a projection, idempotent on `(run_id, seq)` | built |
| Recovery after `kill -9` on the next launch, no duplicated effects except the one in flight | built and tested |
| The pump / sandboxed task host split over `HarnessHost` | **not yet**: the program runs inside the workflow, in the runner's process, so only T0 and T1 code may run durably |
| Suspension without compute | built: runs idle past a threshold are evicted and woken by a message or their deadline ([evicting idle runs](eviction.md)); one activation per message with exported state is still to come |
| Generations and state export | not yet |
| Conversation activations on a partitioned queue | not yet: a conversation's run is one long workflow; sends are serialized per conversation by a database lock, across runners |
| Attempt markers for side effects without deduplication | built: a guarded effect in flight at a crash completes as `outcome_unknown` instead of running again (environment commands, tools without deduplication) |
| Several runners on one database; recovery of a dead runner's runs | built: heartbeats and takeover through DBOS's queue ([several runners](runners.md)); a reaper and poison-run quarantine are not yet |
| Postgres | built ([several runners](runners.md)) |

A known window: a message sent to a conversation at the moment its run finishes can be left unconsumed. The
activation queue closes it.

## Design: the pump and the task host

```
 DBOS workflow "pump" (trusted, generic)                    task host (sandboxed; see task-host.md)
 ┌─────────────────────────────────────────┐   HarnessHost  ┌──────────────────────────────────────┐
 │ start task host with the run's inputs   │ ◀────────────▶ │ deterministic asyncio loop running   │
 │ loop:                                   │  (local IPC,   │ Program / Task / Agent code          │
 │   authorize each effect request         │   JSON only)   │ returns effect requests; receives    │
 │   run it as a DBOS step, in the order   │                │ completions in recorded order        │
 │   the task host requested it            │                └──────────────────────────────────────┘
 │   record completion order               │
 │   feed completions back                 │──▶ recorder · environment layer · tool bindings · blob store
 └─────────────────────────────────────────┘
```

- **The pump is a pure forwarder**, the same for every program. On recovery DBOS re-runs it; recorded steps return
  their stored completions, which the pump re-feeds to a fresh task host. Because the task host is a pure function of
  its inputs, it regenerates the same requests. That is the whole replay mechanism; there is no second log.
- **Steps start in the task host's deterministic request order**, so DBOS step identifiers — and therefore
  `effect_id`s — are deterministic even for concurrent tool calls.
- **Completion order is recorded** (`DBOS.asyncio_wait`), so on replay the task host sees completions in the same
  order.
- **Only JSON crosses the boundary.** The task host never holds database credentials.
- **Two version axes.** The pump's DBOS application version changes rarely (DBOS patching). Program code is pinned
  per run by `code_reference` and routed to task hosts serving that version.

```python
@DBOS.workflow()
async def pump(run_id: str, specification: RunSpecification, generation: int = 0,
               state: StateReference | None = None) -> RunOutcome:
    host = await task_hosts.attach(run_id, specification.program.code_reference, state)
    decision = await host.start(specification)
    pending: dict[asyncio.Future[EffectCompletion], EffectRequest] = {}
    while decision.outcome is None:
        for request in authorize(run_id, decision.effects):       # ownership, schemas, sizes, budgets
            pending[asyncio.ensure_future(perform(run_id, generation, request))] = request
        done, _ = await DBOS.asyncio_wait(list(pending), return_when=asyncio.FIRST_COMPLETED)
        completions = [future.result() for future in done]
        for future in done:
            del pending[future]
        decision = await host.step(completions)
        if decision.hand_over is not None:                        # new generation (replay budget, deploy drain)
            await start_generation(run_id, specification, generation + 1, decision.hand_over)
            return RunOutcome.handed_over(generation)
    return decision.outcome

@DBOS.step(retries_allowed=True, max_attempts=5)
async def perform(run_id: str, generation: int, request: EffectRequest) -> EffectCompletion:
    request = request.with_identity(effect_id=f"{run_id}:{generation}:{request.ordinal}")
    return await EFFECT_EXECUTORS[request.kind].perform(request)  # recorder, environments, tool bindings
```

## Run kinds

| Kind | DBOS shape | On crash | Used for |
|---|---|---|---|
| **Durable** | `pump` workflow; one step per effect; generations | resumes | long episodes, coding agents, swarm members, workflows |
| **Best-effort** | a workflow with a single step that drives the whole episode | the episode is re-run (resampled) | short synchronous RL episodes (~6 rows per episode) |
| **Conversation activation** | `pump` workflow started from a queue partitioned by conversation key | resumes; ends at `WaitFor` | conversations |

## Effects

- `effect_id = {run_id}:{generation}:{ordinal}`, with the argument digest ([effects](../contracts/effects.md)).
- DBOS runs a step and records its result afterwards, so a crash can re-execute a step. Receivers that
  deduplicate absorb this. For tools that cannot deduplicate, `perform` first commits an **attempt marker**
  (`INSERT … ON CONFLICT DO NOTHING` on `effect_id`); a re-execution that finds one completes with
  `OUTCOME_UNKNOWN` instead of calling again.

## Generations and suspension

- A run hands over to a new generation at a resumable point when its replay budget (turns, recorded bytes, wall
  time) is exceeded or a deploy drains its version. The task host exports state; the pump starts generation *n+1*
  with it and ends.
- `WaitFor` ends the current activation: the task host exports state, the pump records `run.suspended` and returns.
  Nothing stays resident. The next matching message starts a new activation that resumes from that state.

## Conversations

Per-conversation lanes, priorities and delivery modes ([conversations](../core/harness/conversations.md)) map onto
DBOS as follows.

- **Deliver** is one database transaction: insert the envelope into the conversation's mailbox (deduplicated by
  `message_id`); if the conversation is idle, mark it scheduled and `dbos.enqueue_workflow(...)` an activation on
  the `conversations` queue with `queue_partition_key = conversation key` and `partition_concurrency = 1`. Signal-
  with-start and "one activation per conversation" follow.
- **Parking** locks the conversation row, re-checks the mailbox, and only then marks it idle, so no message is lost
  between the last check and the end of an activation.
- **Delivery modes during an activation**: `QUEUE` messages stay in the mailbox until the next `WaitFor`; `STEER`
  messages are read by the pump at the next turn boundary; `INTERRUPT` messages are signalled to the running
  activation (`DBOS.send`), which cancels the in-flight model step's request to the recorder and feeds the message to
  `Task.resume`.
- Envelopes addressed to runs rather than conversations use the same mailbox table keyed by `run_id`.

## Cancellation and cleanup

DBOS does not run further steps after a workflow is cancelled or times out, so:

1. **Cancellation is cooperative by default.** `Runner.cancel` sends a control message; the pump delivers
   `run.cancel_requested` at the next turn boundary and the task's `teardown` runs normally. Run deadlines are
   computed from recorded time, not DBOS workflow timeouts.
2. **Hard cancellation** (`DBOS.cancel_workflow`) is for runaway runs only.
3. **A reaper** destroys resources still owned by runs that ended without cleanup (P12).

## Recovery

- Executors have stable identifiers (`executor_id`) and heartbeat. A **recovery controller** declares an executor
  dead when its heartbeat is stale and its process is gone, then re-enqueues its pending workflows (DBOS's own
  re-enqueue statement). DBOS's ownership token fences the dead executor's writes; any step it had already started
  may run twice and is absorbed by receivers. DBOS Conductor is not used.
- **Poison runs**: a run that crashes its task host repeatedly during recovery is quarantined
  (`run.failed{POISONED}`) before it reaches another host.

## Run events and retention

DBOS's step records are the durability journal and are deleted by retention (24–72 hours). The typed
[run events](../contracts/run-events.md) consumers need are written by the pump as a projection (DBOS streams or our
own table) and archived. Inline step payloads are capped at about 4–8 KiB; larger values go to blob storage.

## Deployment

| Profile | System database | Executors |
|---|---|---|
| Local | SQLite | one process |
| Cluster | one Postgres | a pool of executor processes |
| Fleet | one Postgres per cell ([platform](../platform/cells.md)) | executor pools per cell |

## Alternatives and switching cost

Restate is the designated alternative ([research](../research/substrate-restate.md)); our own runtime is the
fallback. The substrate-specific code is the pump, conversation scheduling, the recovery controller and the platform
adapter. Task hosts, programs, the recorder and the RL plane do not depend on DBOS.

## Open questions and spikes

- Retention at our write rate (DBOS deletes rows rather than dropping partitions), recovery storms, `NOTIFY`
  pressure from messaging, and fencing under network partitions: spikes S-D1 to S-D9 in the
  [research report](../research/substrate-dbos.md#10-risks-open-questions-spike-tests).
