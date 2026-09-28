# DBOS as the durable execution substrate

Status: **Draft** · 2026-09-27 · Question: how well does DBOS (DBOS Transact for Python, plus DBOS Conductor where
relevant) serve as the durable execution substrate, including the proposed virtual-actor layer (charter item 5) and
durable agent identities (charter item 6)?

Conventions: **Fact** statements carry a source (docs, blog, or source file at commit
[`c179fe2`](https://github.com/dbos-inc/dbos-transact-py/tree/c179fe2), 2026-09-25, which is `main` just after
release 3.1.0). **Analysis** and **Estimate** are ours. **Unverified** marks claims we could not confirm.

---

## 1 Verdict

- **Viable, with a specific shape.** DBOS is a library over the per-cell Postgres we already chose (ADR-0002). It
  gives us workflows, step memoization, fencing (`owner_xid`), durable queues with per-key serialization, messages,
  streams, delayed starts, patching and fork, all under MIT. That covers most of the run store and runtime.
- **Use it in "pump mode", not native mode.** A small, generic, trusted DBOS workflow drives our sandboxed replay
  driver (the task host) and turns each effect the driver requests into a DBOS step. This keeps tenant code away
  from database credentials, keeps our deterministic concurrency, and decouples task-code versions from DBOS
  application versions. Native mode (task code *is* the DBOS workflow) is acceptable only if Q10 = in-house.
- **The virtual-actor layer is feasible** as an entity table, an atomic SQL deliver function
  (`dbos.enqueue_workflow` exists in SQL), and activation workflows on a partitioned queue. It is also the
  answer to idle runs being resident and pinned to one version.
- **Build our own recovery controller.** Conductor is not required. Hosted Conductor's per-checkpoint pricing does
  not fit our volume, and self-hosting it needs an enterprise licence.
- **Gaps we must engineer around:** cancellation and timeouts do not run cleanup steps. There is no
  persist-before-dispatch (use attempt markers). Retention is delete-based (no partitioning). Recovery reads one
  database transaction per recorded step. `send` still issues a `NOTIFY` on the write path.
- **Scale:** about 7.5k step checkpoints/s per cell is roughly 15–25 % of DBOS's own single-server measurement.
  Retention and vacuum churn and WAL volume from payloads will bind before the insert rate. Spikes S-D1 to S-D9
  decide it.

---

## 2 What it is

### Model

**Fact.** A DBOS application decorates functions as `@DBOS.workflow()` and `@DBOS.step()`. Workflow inputs, every
step's output (or error) and the workflow's final output are checkpointed in a Postgres (or SQLite) **system
database**. On recovery, the workflow function is re-executed from the start. Each step whose output is recorded
returns it without running, and execution continues live from the first step with no checkpoint
([architecture](https://docs.dbos.dev/architecture)). The guarantees: "Workflows always run to completion, resuming
from the last completed step; steps execute at least once but never re-execute after completion"
([workflow tutorial](https://docs.dbos.dev/python/tutorials/workflow-tutorial)).

**Fact (source).** A step runs *first* and is recorded *after*. `invoke_step` checks for a recorded output (one
transaction with two `SELECT`s, which also raises if the workflow is `CANCELLED`). It then runs the body. After the
body returns, it inserts into `operation_outputs` in a second transaction that first locks the workflow's
`workflow_status` row `FOR NO KEY UPDATE` and checks `owner_xid` ([`_core.py`
invoke_step](https://github.com/dbos-inc/dbos-transact-py/blob/c179fe2/dbos/_core.py),
[`_sys_db.py` `_record_operation_result_txn`, `_check_owner_txn`](https://github.com/dbos-inc/dbos-transact-py/blob/c179fe2/dbos/_sys_db.py)).
A step's identity is its `function_id`, a per-workflow counter assigned in call order. It is readable inside the
step as `DBOS.step_id`, with `DBOS.workflow_id` and `DBOS.step_status.current_attempt`
([`_dbos.py`](https://github.com/dbos-inc/dbos-transact-py/blob/c179fe2/dbos/_dbos.py)).

### Architecture and storage

**Fact (source,
[`system_database.py`](https://github.com/dbos-inc/dbos-transact-py/blob/c179fe2/dbos/_schemas/system_database.py)).**
Tables in schema `dbos`:

| Table | Holds |
|---|---|
| `workflow_status` | one row per workflow: status (`DELAYED`, `ENQUEUED`, `PENDING`, `SUCCESS`, `ERROR`, `CANCELLED`, `MAX_RECOVERY_ATTEMPTS_EXCEEDED`), `executor_id`, `application_version`, `application_name`, queue name / partition key / priority / deduplication id, deadline, `owner_xid` (current execution's ownership token), `parent_workflow_id`, `forked_from`, attributes (JSONB, GIN-indexed) |
| `workflow_input`, `workflow_output` | inputs and outputs, split out of `workflow_status` in 3.0 |
| `operation_outputs` | one row per step, send, recv, sleep, event, stream write, child start and `get_result` (PK `(workflow_uuid, function_id)`) |
| `notifications` | `send`/`recv` messages (FK to destination workflow, `ON DELETE CASCADE`; `message_uuid` = idempotency key) |
| `workflow_events`, `workflow_events_history` | `set_event` key/value (latest + immutable history) |
| `streams` | `write_stream` entries `(workflow_uuid, key, offset)` |
| `queues`, `workflow_schedules`, `application_versions` | database-backed queue configuration, cron schedules, version registry |

Background threads per process: a queue manager (one polling thread per queue), a notification listener
(`LISTEN` on three channels), a notifier that flushes coalesced `pg_notify` batches (every 10 ms by default),
a workflow-timeout sweeper, a scheduler and a retention sweeper ([`_dbos.py`
launch](https://github.com/dbos-inc/dbos-transact-py/blob/c179fe2/dbos/_dbos.py)). Async workflows run as coroutines
on one background event loop. System-database calls are synchronous SQLAlchemy/psycopg called through
`asyncio.to_thread`, with a connection pool of 20 by default (`sys_db_pool_size`) ([`_dbos_config.py`](https://github.com/dbos-inc/dbos-transact-py/blob/c179fe2/dbos/_dbos_config.py)).
Waits (`recv_async`, `get_event_async`, `sleep_async`) park the coroutine, not a thread
([#652 resolution](https://github.com/dbos-inc/dbos-transact-py/issues/652), `test_high_async_concurrency`).

### License, maturity, SDKs

- **Fact.** `dbos` 3.1.0 on PyPI (2026-09-23), MIT, Python ≥ 3.10. 3.2.0 alphas are in flight
  ([PyPI](https://pypi.org/project/dbos/)). 3.0.0 (2026-09-15) removed `@DBOS.transaction`, in-memory queues,
  decorator schedules and the admin server, and split inputs and outputs into their own tables. 3.0 processes
  cannot share a system database with 2.x processes ([upgrading](https://docs.dbos.dev/python/upgrading),
  [3.0.0 release](https://github.com/dbos-inc/dbos-transact-py/releases/tag/3.0.0)).
- **Fact.** SDKs exist for Python, TypeScript, Java and Go. DBOS Go 1.0 is "production-ready" (August 2026). Several
  applications in different languages can share one system database, isolated by `application_name`
  ([August 2026 update](https://www.dbos.dev/blog/new-in-dbos-august-2026),
  [dbos-transact-golang](https://github.com/dbos-inc/dbos-transact-golang), MIT).
- **Fact.** SQL entry points `dbos.enqueue_workflow(...)` and `dbos.send_message(...)` are installed by migrations,
  so any Postgres client can enqueue or send in its own transaction ([`_migration.py`](https://github.com/dbos-inc/dbos-transact-py/blob/c179fe2/dbos/_migration.py)).
- **Fact (maturity signal).** Correctness bugs in recovery and ownership were found and fixed through mid-2026:
  Postgres restart during recovery stranding workflows ([#716](https://github.com/dbos-inc/dbos-transact-py/issues/716)),
  stale pending lists executing work after ownership was cleared
  ([#742](https://github.com/dbos-inc/dbos-transact-py/issues/742)), a draining executor re-claiming recovered work
  ([#804](https://github.com/dbos-inc/dbos-transact-py/issues/804)), and cancel racing completion
  ([#767](https://github.com/dbos-inc/dbos-transact-py/issues/767)). The ownership-token scheme (`owner_xid`) and
  "Improve Workflow Conflict Resolution" (2026-09-24) are weeks old. **Analysis:** the project is active and responsive
  (1.6k stars, 6 open issues), but the fencing code we would rely on is young.

### Deployment and Conductor

- **Fact.** Without Conductor, a restarting process recovers `PENDING` workflows tagged with *its own*
  `executor_id` and *its own* application version. In a distributed deployment, "some coordination is required,
  either automatically through services like DBOS Conductor or manually"
  ([architecture](https://docs.dbos.dev/architecture), [workflow recovery](https://docs.dbos.dev/production/workflow-recovery)).
  The internal `recover_pending_workflows(executor_ids)` re-enqueues a dead executor's `PENDING` rows (status →
  `ENQUEUED`, `owner_xid` → `NULL`) so that exactly one live executor claims each one
  ([`_recovery.py`](https://github.com/dbos-inc/dbos-transact-py/blob/c179fe2/dbos/_recovery.py),
  `reenqueue_for_recovery`).
- **Fact.** Conductor is a proprietary service. Executors hold an outbound websocket to it; it sends `RECOVERY`,
  `CANCEL`, `RESUME`, `FORK_WORKFLOW`, `DELETE`, `EXPORT_WORKFLOW`, `RETENTION` and similar commands
  ([`_conductor/protocol.py`](https://github.com/dbos-inc/dbos-transact-py/blob/c179fe2/dbos/_conductor/protocol.py)).
  By default it waits 60 s after a disconnect before declaring an executor dead
  ([workflow recovery](https://docs.dbos.dev/production/workflow-recovery)). Self-hosting needs a licence key: free
  for development, paid for production, no source access
  ([hosting Conductor](https://docs.dbos.dev/production/hosting-conductor),
  [Conductor licence](https://www.dbos.dev/conductor-license)). Hosted tiers: Pro $99/month with 1M checkpoints,
  then $50 per million; Teams $499/month with 10M, then $40 per million; self-hosted Conductor only on Enterprise
  (custom pricing) ([pricing](https://www.dbos.dev/dbos-pricing)).
- **Estimate.** At our design point (~90k checkpoints/s ≈ 230B/month), list-price hosted Conductor would cost
  about $9M/month. It is not an option. Recovery is ~300 lines of our own (§8).

### Agent-framework integrations

**Fact.** Pydantic AI's `DBOSAgent` wraps `Agent.run` as a workflow and model requests and MCP calls as steps. It runs
tools in parallel by default. Streaming events are buffered inside workflows, not delivered live
([Pydantic docs](https://pydantic.dev/docs/ai/integrations/durable_execution/dbos/)). The OpenAI Agents SDK
integration provides `DBOSRunner` ([docs](https://docs.dbos.dev/integrations/openai-agents)). **Analysis:** these show
the native-mode mapping works for single agents. None addresses tenant isolation, trajectory capture or
hundreds of thousands of idle agents.

---

## 3 Concept mapping

### 3.1 Table

| Our concept | DBOS primitive | Fit | Gap and what we add |
|---|---|---|---|
| **Run** | Workflow; `workflow_id = run_id` (`r_{cell_id}_{ulid}`) | good | The run log becomes DBOS rows, not our event schema; RL-facing events are a projection (§3.5). Whether the ID validator accepts our format is **unverified** (believed yes). |
| **Effect** | `@DBOS.step()` (async); retries via `max_attempts`, `backoff_rate`, `should_retry`; `timeout_seconds` for async steps | good | Execute-then-record, not persist-then-dispatch (P5). The in-flight effect is invisible in the database until it completes. |
| **Effect identity** | `f"{DBOS.workflow_id}:{DBOS.step_id}"`; deterministic across re-execution | good | DBOS assigns IDs in *call order*. Concurrent sub-coroutines that each call several steps get non-deterministic IDs ([#688](https://github.com/dbos-inc/dbos-transact-py/issues/688); docs: "Steps must be started in deterministic order"). Native mode must restrict `run.gather`; pump mode is unaffected (§3.2). |
| **Signals / inbox** | `DBOS.send` / `recv(topic, timeout)`; `send_bulk`; per-topic FIFO; exactly-once from workflows; `idempotency_key` from outside | good | Destination must exist (FK). Messages to a finished workflow are retained unconsumed. A waiting `recv` keeps the workflow resident. |
| **Signal-with-start, run keys** | `SetWorkflowID(key)` with reuse policy `return-existing`, then `send`; or one SQL transaction calling `dbos.enqueue_workflow(workflow_id => …)` + `dbos.send_message(…)` | good | We add the run-key → run_id table (or derive the ID as a hash of the key). |
| **Timers** | `DBOS.sleep_async` (durable wake time, but the coroutine stays resident); delayed enqueue (`delay_seconds` → `DELAYED` status); database-backed cron schedules | good | Long waits should be delayed enqueues of a continuation, not resident sleeps. |
| **Child runs** | `DBOS.start_workflow_async` / `enqueue` inside a workflow (recorded as a step with `child_workflow_id`); `handle.get_result()` recorded | good | Cancelling or timing out a parent cascades to children. |
| **Cancellation + teardown** | `DBOS.cancel_workflow` → `CANCELLED`, `owner_xid = NULL`; the run is pre-empted at its next step start; `preemptible=True` async steps poll every 1 s | **poor** | **Cleanup does not run durably.** After cancel or timeout every step raises `DBOSWorkflowCancelledError`, including steps in `finally` ([source](https://github.com/dbos-inc/dbos-transact-py/blob/c179fe2/dbos/_sys_db.py) `_check_operation_execution_txn`, `cancel_timed_out_workflows`). We add cooperative cancellation plus a reaper (§3.3). |
| **Versioning** | Application version (default: hash of workflow source); executors dequeue and recover only their own version; `set_latest_application_version`; `enable_patching` + `DBOS.patch()` / `deprecate_patch()` (sets the version to the constant `PATCHING_ENABLED`) | good | Native mode pins runs to executor deployments. Pump mode makes the DBOS version nearly constant, with task code pinned by `code_reference` (ours). |
| **Snapshots / continuation** | none; recovery always re-executes from the start, one read transaction per recorded step (no prefetch found in source) | **partial** | Bound replay with *generations* (continue-as-new built on delayed enqueue) and actor activations (§3.4). |
| **Outputs / streams (`run.emit`)** | `DBOS.write_stream` / `read_stream(offset)` / `close_stream`; exactly-once from workflows, at-least-once from steps; notifications coalesced off the write path | good | Connectors read by offset through `DBOSClient`. |
| **Token streaming** | none suitable (a database row per token) | n/a | Ephemeral side channel from the recorder or model step (pub/sub or SSE), keyed by `effect_id`. |
| **Queues / admission** | Database-backed queues: global and per-worker concurrency, rate limit, priority, deduplication, **partitioned** (`partition_concurrency`, `partition_worker_concurrency`, `partition_limiter`), `listen_queues`, polling every 1 s by default | good | Job-level buffer admission stays in the rollout controller. Deduplication cannot be combined with a partition key (Python raises). |
| **Recovery / fencing** | `executor_id` + self-recovery at launch; `recover_pending_workflows(executor_ids)`; `owner_xid` fences every checkpoint write; `max_recovery_attempts` (default 100) → dead letter | good core, **no failure detector** | We add a heartbeat-based recovery controller (§8). |
| **Cells** | One system database (or schema: `dbos_system_schema`) per cell | good | Cross-cell calls go through the target cell's Control API (keeps N4). |
| **Observability** | Built-in OpenTelemetry spans for workflows and steps (legacy or semconv attributes); `list_workflows`, `list_workflow_steps`, `get_metrics`; Conductor UI (paid) | good | Dashboards over SQL and OpenTelemetry; no Conductor needed. |

### 3.2 Two ways to host task code

**Native mode.** Task and agent code runs *as* the DBOS workflow. `RunContext` handles call `@DBOS.step`
functions. Simple, and it matches Pydantic AI and the OpenAI Agents SDK integrations:

```python
@DBOS.step(retries_allowed=True, max_attempts=5, backoff_rate=2.0)
async def model_step(session_id: str, request: ModelRequest) -> Message:
    effect_id = f"{DBOS.workflow_id}:{DBOS.step_id}"               # identical on every re-execution
    return await recorder.sample(session_id, request, idempotency_key=effect_id)

@DBOS.step(retries_allowed=True, max_attempts=3)
async def environment_step(call: EnvironmentCall) -> EnvironmentResult:
    effect_id = f"{DBOS.workflow_id}:{DBOS.step_id}"
    return await environment_client.call(call, request_id=effect_id)   # envd three-state deduplication

class NativeRunContext(RunContext):
    async def sample(self, slot: str, request: ModelRequest) -> Message:
        return await model_step(f"{self.run_id}/{slot}", request)

@DBOS.workflow()
async def rollout_workflow(specification: RunSpecification) -> RunOutcome:
    task = load_task(specification.task)        # code must be importable in this executor: version = deployment
    agent = load_agent(specification.agent)
    run = NativeRunContext(run_id=DBOS.workflow_id, specification=specification)
    await rollout(task, agent, run)             # the normative loop from harness/README.md, unchanged
    return run.outcome()
```

Consequences (analysis): task code shares the process with database credentials (§7). Determinism is by convention
only; DBOS has no sandbox, so non-determinism surfaces at recovery as `DBOSUnexpectedStepError`. `run.gather` over
multi-effect coroutines (for example `run_tools` whose tool bodies call `execute` twice) gets non-deterministic step
IDs, so tool calls must run one at a time or each as a child workflow (about 4 extra writes per call). Every task-code
deploy is a new DBOS application version.

**Pump mode (recommended).** A generic trusted workflow drives our task host, the no-network gVisor sandbox with our
replay driver ([durability.md](../core/harness/determinism.md)), over the existing `HarnessHost` socket. The
sandbox is a pure function of the inputs it is fed. So the pump does **not** record the sandbox's decisions: on
recovery DBOS re-executes the pump, the pump re-feeds each recorded completion, and the sandbox regenerates the same
effect requests. Our driver's deterministic scheduler orders requests, and the pump starts steps in exactly that
order, so DBOS step IDs are deterministic even for concurrent tool calls.

```python
@dataclass(frozen=True)
class EffectRequest:
    ordinal: int                 # position in the driver's deterministic request order
    kind: str                    # "model.request" | "environment.call" | "environment.lifecycle" | "tool.request" | ...
    payload: dict[str, Any]      # JSON only; nothing from the sandbox is ever unpickled here
    retry_class: str
    arguments_digest: str        # for our own divergence detection

@dataclass(frozen=True)
class EffectCompletion:
    ordinal: int
    effect_id: str
    status: str                  # OK | FAILED | OUTCOME_UNKNOWN
    payload: dict[str, Any]

@DBOS.workflow()
async def run_pump(run_id: str, specification: RunSpecification, generation: int = 0,
                   snapshot: SnapshotReference | None = None) -> RunOutcome:
    host = await task_hosts.attach(run_id, specification.task.code_reference, snapshot)   # sandbox, by code version
    decision: HarnessDecision = await host.start(specification)                            # deterministic
    pending: dict[asyncio.Future[EffectCompletion], EffectRequest] = {}
    while decision.outcome is None:
        authorize(run_id, decision.effects)                   # ownership checks: task code is untrusted
        for request in decision.effects:                      # started in driver order → deterministic step IDs
            pending[asyncio.ensure_future(perform_effect(run_id, request))] = request
        if decision.cancel_checkpoint:                        # the driver asks at turn boundaries
            pending[asyncio.ensure_future(poll_control(run_id))] = CONTROL_REQUEST
        done, _ = await DBOS.asyncio_wait(list(pending), return_when=asyncio.FIRST_COMPLETED)  # durable completion order
        completions = [future.result() for future in done]
        for future in done:
            del pending[future]
        decision = await host.step(completions)
        if decision.continue_as_new is not None:              # bound replay: new generation from a snapshot
            await start_next_generation(run_id, specification, generation + 1, decision.continue_as_new)
            return RunOutcome.continued(generation)
    return decision.outcome

@DBOS.step(retries_allowed=True, max_attempts=5)
async def perform_effect(run_id: str, request: EffectRequest) -> EffectCompletion:
    effect_id = f"{run_id}:{DBOS.step_id}"
    executor = EFFECT_EXECUTORS[request.kind]                 # recorder, envlet, Environment Manager, tool router
    return await executor.perform(effect_id, request)
```

`DBOS.asyncio_wait` checkpoints which futures finished, so the sandbox sees completions in the recorded order on
replay. That is the same contract as our driver ("completions are delivered in the order the log recorded them").
It costs one row per wait. Replay cost per recorded step is one database read plus sandbox CPU.

**Analysis: why pump mode.** (1) Tenant code never touches database credentials (§7). (2) Determinism stays
*enforced*, and `run.gather` keeps its semantics. (3) The DBOS application version is the pump's version, which changes
rarely. Task code versions are routed by `code_reference` to sandbox pools, exactly as in the current design. (4) The
pump can be written in Go with DBOS Go 1.0, which keeps ADR-0011 (Runtime in Go; B4 is Go ↔ Python). Feature parity of
the Go SDK (partitioned queues, `asyncio_wait` equivalent, patching) is **unverified**.

### 3.3 Effects, attempt markers and teardown

**Persist-before-dispatch is lost; we recover it where it matters.** DBOS can re-execute a step whose body ran but
whose checkpoint did not commit. For receivers that deduplicate (recorder, envd, Environment Manager, tool-router
agent and human bindings) that is fine, because `effect_id` is stable. For `side_effecting` or `unknown` tools with no
receiver deduplication, the step claims an **attempt marker** in its own committed transaction *before* dispatch:

```python
NON_DEDUPLICATING = {"side_effecting", "unknown"}

@DBOS.step(retries_allowed=False)                    # never auto-retry a non-deduplicating dispatch
async def tool_step(run_id: str, request: EffectRequest) -> EffectCompletion:
    effect_id = f"{run_id}:{DBOS.step_id}"
    if request.retry_class in NON_DEDUPLICATING and not request.receiver_deduplicates:
        first_attempt = await attempt_markers.claim(effect_id)
        #   INSERT INTO effect_attempt (effect_id, claimed_at, executor_id) VALUES (...)
        #   ON CONFLICT (effect_id) DO NOTHING RETURNING effect_id        -- committed before dispatch
        if not first_attempt:
            return EffectCompletion.outcome_unknown(request.ordinal, effect_id)   # → tool.outcome_unknown
    return await tool_router.call(request, idempotency_key=effect_id)
```

The cost is one extra write per non-deduplicating tool call only. Markers are retained for 24 h
([delivery-semantics](../architecture/delivery-semantics.md#receiver-deduplication)). A zombie executor (§6) can dispatch
at most the steps it had already started. Its checkpoint then fails the `owner_xid` check and the execution parks.
ADR-0003 already accepts this.

**Teardown.** DBOS cancellation pre-empts the next step, so `Task.teardown` cannot run durable environment destroys
after a DBOS cancel or timeout. Design:

1. **Cooperative cancellation is the default.** `CancelRun` sends a `control` message (`DBOS.send`, topic
   `"control"`). The pump races the next turn's effects against `poll_control` at turn boundaries. The driver receives
   `run.cancel_requested` and runs `teardown` normally. Run deadlines are enforced the same way: the pump computes
   `now >= deadline` from recorded time, and we do not use a DBOS workflow timeout.
2. **Hard cancellation** (`DBOS.cancel_workflow`) is reserved for runaway runs.
3. **Reaper.** A per-cell loop lists `CANCELLED` and `ERROR` runs (and `MAX_RECOVERY_ATTEMPTS_EXCEEDED`) and destroys
   what they still own through the Environment Manager (`owner = run_id`). This implements P11's "the runtime destroys
   whatever the run still owns" outside the run.

### 3.4 The virtual-actor layer on DBOS

DBOS has no actors. The layer is ours, about 2–3k lines. It hosts long-lived, mostly idle entities (durable agent
identities, conversations) at zero resident cost.

**Schema** (in the cell's Postgres, schema `actors`, beside `dbos`):

```sql
CREATE TABLE entity (
  entity_key      text PRIMARY KEY,          -- a_{cell_id}_{ulid} (identity) or {identity_key}/{run_key} (conversation)
  kind            text NOT NULL,             -- 'identity' | 'conversation'
  tenant          text NOT NULL,
  deployment      text NOT NULL,             -- name → code references + default binding
  status          text NOT NULL,             -- IDLE | SCHEDULED | RUNNING | DISABLED | ARCHIVED
  state_version   bigint NOT NULL DEFAULT 0, -- compare-and-set token; fences zombie activations
  state_schema    integer NOT NULL,          -- for upcasters
  state           jsonb NOT NULL,            -- small, versioned records; large values are blob references
  continuation    jsonb,                     -- WaitFor + loop position + optional snapshot reference
  activation_seq  bigint NOT NULL DEFAULT 0,
  updated_at      timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE entity_mailbox (
  entity_key      text NOT NULL REFERENCES entity,
  seq             bigint GENERATED ALWAYS AS IDENTITY,
  kind            text NOT NULL,             -- 'user_message' | 'approval' | 'timeout' | 'reward' | 'agent_message' ...
  message         jsonb NOT NULL,
  idempotency_key text NOT NULL UNIQUE,
  consumed_by     text,                      -- activation workflow id
  created_at      timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (entity_key, seq)
);
CREATE TABLE entity_history (                -- canonical-content turns; partitioned by time; archived to object storage
  entity_key text, turn bigint, reply jsonb, observation jsonb, reply_effect_id text, created_at timestamptz,
  PRIMARY KEY (entity_key, turn, created_at)
) PARTITION BY RANGE (created_at);
CREATE TABLE entity_state_version (          -- previous versions, for audit and rollback; pruned
  entity_key text, state_version bigint, state_schema integer, state jsonb, activation_id text,
  PRIMARY KEY (entity_key, state_version)
);
```

**Deliver** is one transaction. It is atomic signal-with-start without DBOS's `send` (so no `NOTIFY`). It schedules at
most one activation per entity:

```python
async def deliver(connection: AsyncConnection, entity_key: str, kind: str, message: dict[str, Any],
                  idempotency_key: str) -> None:
    async with connection.transaction():
        entity = await connection.fetchrow(
            "SELECT status, activation_seq FROM actors.entity WHERE entity_key = $1 FOR UPDATE", entity_key)
        if entity is None or entity["status"] in ("DISABLED", "ARCHIVED"):
            raise EntityUnavailable(entity_key)
        inserted = await connection.fetchval(
            "INSERT INTO actors.entity_mailbox (entity_key, kind, message, idempotency_key) "
            "VALUES ($1, $2, $3, $4) ON CONFLICT (idempotency_key) DO NOTHING RETURNING seq",
            entity_key, kind, message, idempotency_key)
        if inserted is None or entity["status"] != "IDLE":
            return                                             # duplicate, or an activation will drain it
        activation_seq = entity["activation_seq"] + 1
        await connection.execute(
            "UPDATE actors.entity SET status = 'SCHEDULED', activation_seq = $2 WHERE entity_key = $1",
            entity_key, activation_seq)
        await connection.execute(                              # SQL enqueue; app_version NULL → latest version runs it
            "SELECT dbos.enqueue_workflow(workflow_name => 'activate_entity', queue_name => 'entities', "
            "positional_args => ARRAY[$1::json], workflow_id => $2, queue_partition_key => $3)",
            json.dumps(entity_key), f"{entity_key}@{activation_seq}", entity_key)
```

The `entities` queue is registered with `partition_concurrency=1` and `partition_worker_concurrency=1`. That is a second
guarantee of one activation per key cluster-wide. It also selects DBOS's batched dequeue path
(`start_queued_partitioned_workflows`, one sweep across up to 8,192 partitions). Deduplication is not used because it
is incompatible with partition keys.

**Activation workflow**, running the Program layer (charter item 1):

```python
@dataclass(frozen=True)
class WaitFor:                                   # new return value of Task.respond (charter item 5)
    signal: str                                  # e.g. "user_message"
    timeout: timedelta | None = None

@DBOS.workflow()
async def activate_entity(entity_key: str) -> None:
    snapshot: EntitySnapshot = await load_entity(entity_key)             # step: state, continuation, version, deployment
    program: Program = programs.instantiate(snapshot)                    # upcasts state_schema → current
    await mark_running(entity_key, snapshot.state_version)               # step (CAS)
    while True:
        batch: list[MailboxMessage] = await claim_messages(entity_key, DBOS.workflow_id, limit=16)   # step
        if not batch:
            if await park(entity_key, snapshot.state_version):           # step; False → a message raced in
                if snapshot.continuation and snapshot.continuation.timeout_at:
                    await schedule_timeout(entity_key, snapshot)         # delayed enqueue of a 'timeout' delivery
                return                                                   # nothing resident any more
            continue
        for message in batch:
            result: ProgramResult = await program.resume(message)        # turn loop; effects are steps of this workflow
            snapshot = await commit_state(entity_key, snapshot.state_version, program.export_state(),
                                          result.continuation, result.history_rows)   # step: CAS, history append
            if result.ended:
                await finish_entity(entity_key)                          # conversations end; identities never do
                return
```

`park` locks the entity row first, then checks the mailbox in a *new statement*. Under read committed, that statement
sees any message whose `deliver` transaction held the lock, so there is no lost wake-up:

```sql
SELECT 1 FROM actors.entity WHERE entity_key = $1 AND state_version = $2 FOR UPDATE;   -- waits for deliver
SELECT EXISTS (SELECT 1 FROM actors.entity_mailbox WHERE entity_key = $1 AND consumed_by IS NULL);
UPDATE actors.entity SET status = 'IDLE' WHERE entity_key = $1;                           -- only if no message
```

**Suspension rules.** `Task.respond` returning `WaitFor("user_message", timeout=…)` ends the activation at a turn
boundary. The loop state at a turn boundary is derivable from history plus exported task and agent state. So the
continuation is JSON: turn number, pending `WaitFor`, and exported state. A pickled snapshot is kept only as an
optimization tagged with `code_reference` and discarded when the code changes. Short waits such as approvals,
`@tool(requires_approval=True)`, stay in-activation as `DBOS.recv_async(topic="approval", timeout_seconds=…)`, resident
but bounded.

**Versioned state.** `state_schema` plus registered upcasters (`@upcaster("support-agent", from_schema=3)`) migrate
state on load. Each activation is enqueued with no application version, so it runs on the latest deployed version.
Upgrades therefore happen at turn boundaries without `DBOS.patch`. Only a mid-activation deploy needs patching, and
activations are short.

**History storage.** Per-turn canonical content lives in `entity_history`, which we own, is time-partitioned and is
archived to object storage. It does not live in DBOS `operation_outputs`, whose rows are deleted by retention after
the activation finishes. That keeps the history schema ours (the reason ADR-0004 gave for building).

### 3.5 Run log as seen by RL tooling

The trajectory assembler needs `observation.recorded`, `reward.assigned` and endings joined by `effect_id` (B13).
In DBOS these are not first-class. Analysis: short episodes return them in the workflow output (`RunOutcome` carries
rewards keyed by reply `effect_id`, observations' `info`, and the ending), which is zero extra writes. Long runs
append them with `DBOS.write_stream(run_id, "events", …)` (two rows per event) or to `entity_history`. The Control API
event stream (B13) becomes a projection over these.

---

## 4 Use cases

### 4.1 Synchronous RL

**Mechanics.** The rollout controller keeps admission (`in-flight + unacknowledged < buffer_samples`) as designed. For
each admitted row it enqueues onto the cell's `rollouts` queue with `worker_concurrency` sized to executor capacity:

```python
client = DBOSClient(system_database_url=cell.system_database_url)
for run_id, row in admitted:
    client.enqueue(EnqueueOptions(workflow_name="run_pump", queue_name="rollouts", workflow_id=run_id,
                                  priority=row.priority, attributes={"job_id": job_id, **row.labels}),
                   run_id, build_run_specification(job, row))
```

The controller waits for the batch (`client.retrieve_workflow(run_id).get_result()` or an attributes query). The
trainer steps, then `Publish` runs between batches, so no episode spans a weight change. Short episodes use the
best-effort tier (§4.2) and a crash simply resamples. Cost is about 6 row writes per episode regardless of length.

### 4.2 Asynchronous RL

- **In-flight weight updates.** These are invisible to DBOS. Abort-before-update and resubmission happen inside the
  recorder (ADR-0008), which is inside one `model_step`. The step just takes longer, so its `timeout_seconds` must
  cover a resubmission. `effect_id` stays the same, and a retried step after a crash hits the recorder's cached result
  (P6).
- **High-rate short episodes.** At 30k turns/s, per-step durability costs 2 transactions and 1 checkpoint row per
  effect (§5). **Per-step durability can be relaxed**, because a DBOS step may contain arbitrary work:

```python
@DBOS.step(retries_allowed=False)
async def episode_step(run_id: str, specification: RunSpecification) -> RunOutcome:
    attempt = uuid.uuid4().hex[:8]                       # new on every (re)execution: fresh samples, no stale cache hits
    run = InProcessRunContext(run_id=run_id, effect_prefix=f"{run_id}~{attempt}")   # effect_id = {prefix}:{n}
    await drive_in_sandbox(run, specification)           # the same driver, no per-effect checkpoints
    return run.outcome()                                 # rewards, observations' info, ending → workflow output

@DBOS.workflow(max_recovery_attempts=3)
async def best_effort_run(run_id: str, specification: RunSpecification) -> RunOutcome:
    return await episode_step(run_id, specification)     # crash → recovery re-runs the whole episode = resample
```

  That is the `best_effort` tier of [delivery-semantics](../durability/README.md#run-kinds):
  about 6 writes per episode. `recovery_attempts` gives the crash rate by episode length, which is the sampling-bias
  monitor. The recorder session for an attempt should be `{run_id}~{attempt}/{slot}`, a deviation from the normative
  `session_id` format that needs deciding. A **middle tier** is possible: a `chunk_step` that runs *K* turns and
  returns the exported loop state, so durability happens every *K* turns (our snapshot concept expressed as a step).
- **Buffer and staleness** remain in the controller and the trainer. DBOS queues add nothing algorithmic.

### 4.3 Swarms

**1,000 agents messaging (one cell).**

- *Resident runs* (active swarm, turn-by-turn): each agent is a `run_pump` workflow. Messages use `DBOS.send_bulk`,
  one transaction for N messages, exactly-once from a workflow step, idempotent per destination. Receivers
  `recv_async(topic="agent_message")`. Costs per message: 1 notification row plus, in total, 1 checkpoint row for the
  whole bulk send and 2 checkpoint rows at the receiver (the `recv` timeout and its consume). Every inserted
  notification fires the `dbos_notifications_trigger` `pg_notify` inside the sender's commit (source: migration still
  creates it, and only the events and streams triggers were dropped). Every executor's listener receives every
  notification and filters in memory.
- *Mostly idle swarms* use entity mailboxes (`deliver`), with no `NOTIFY`, an activation per burst, and queue-polling
  latency (§5).

**Fan-out / fan-in and spawn.**

```python
async def fan_out(run: RunContext, rows: list[dict[str, Any]]) -> list[RunOutcome]:
    children = [await DBOS.start_workflow_async(run_pump, child_run_id(run, index), child_specification(row))
                for index, row in enumerate(rows)]            # deterministic start order
    return [await child.get_result() for child in children]  # each result is checkpointed in the parent
```

Each child costs about 4 row writes to start and finish, plus 2 checkpoint rows in the parent (start and result). A
1,000-way fan-out is about 6k rows, a few hundred milliseconds of cell write capacity. `run.spawn` into another cell
goes through that cell's Control API from a step, with `idempotency_key = effect_id`.

**Rule unchanged:** a swarm lives in one cell, and cross-cell `send` uses the Control API, not a remote `DBOSClient`.
Holding another cell's database credentials would break N4.

### 4.4 Durable agent identities

**Definition (proposed).** A *durable agent identity* is a long-lived, addressable principal that owns state and
resources across many runs:

```python
@dataclass(frozen=True)
class AgentIdentity:
    identity_id: str                           # a_{cell_id}_{ulid}: embeds the home cell, never reused
    address: str                               # agent://{tenant}/{name}; global alias → identity_id
    tenant: str
    deployment: str                            # Program class + code references + default RunBinding
    policy_binding: RunBinding                 # model slots → channels (trainable or not), imports, security classes
    permissions: IdentityPermissions           # tool allowlist, spawn and send scopes, approval rules, budgets, quotas
    credential_grants: list[CredentialGrantReference]   # broker references only (P8); never secrets
    owned_resources: list[ResourceReference]   # environments, snapshots, volumes; P11 owner = identity_id

@dataclass(frozen=True)
class IdentityState:                           # entity.state, versioned by state_schema
    profile: dict[str, Any]
    memory_namespaces: list[str]               # run.memory lives in actors.entity_memory (versioned key-value, CAS)
    open_conversations: dict[str, str]         # run_key → conversation entity key
```

- **Address and mailbox.** `agent://acme/support-bot` resolves to `identity_id`. Messages go through `deliver`. The
  identity entity routes each message by run key (for example a Slack thread) to a **conversation entity**
  `{identity_id}/{run_key}`, creating it on first contact (signal-with-start).
- **Runs relate to identities** as sessions. A conversation entity is a run whose activations are serialized per
  conversation. Different conversations of one identity proceed in parallel on different partition keys.
  Identity-wide mutations (memory, owned resources) use compare-and-set on per-key versions, or go through the
  identity entity's own mailbox when they must be serialized.
- **Lifecycle:** `PROVISIONED → IDLE ⇄ SCHEDULED/RUNNING → DISABLED → ARCHIVED → DELETED`. Deletion
  crypto-shreds the per-tenant or per-identity key (the custom serializer encrypts payloads, §7). Owned environments
  hibernate between conversations and are destroyed by the reaper when their owner is archived.
- **Upgraded across deploys.** Every activation runs the latest version. State and continuations are versioned JSON
  with upcasters. Nothing is resident while idle, so nothing is pinned to an old executor. Rollback uses
  `set_latest_application_version` plus the retained `entity_state_version` rows.
- **Joining swarms.** A swarm run addresses identities by address. Their conversation entities receive `agent_message`
  deliveries, and the swarm run waits on replies through its own mailbox or `recv`.
- **Being trained.** Model slots bind to recorded channels. The session ID is `{conversation_key}/{slot}`, so each
  model call's `effect_id` joins recorder spans with rewards that arrive later as `reward` mailbox messages, recorded
  as `reward.assigned{reply_effect_id}` in `entity_history`. The assembler emits samples per conversation segment.

**Scale (estimate).** 300k identities with about 25k per cell mostly idle means 25k `entity` rows per cell, not 25k
coroutines. Activity is bounded by message rate, not population.

---

## 5 Scale and performance

### 5.1 Evidence (facts)

| Source | Result |
|---|---|
| [DBOS benchmark](https://www.dbos.dev/blog/benchmarking-workflow-execution-scalability-on-postgres) (2026-04-23), RDS db.m7i.24xlarge (96 vCPU, 384 GB, 120k IOPS io2) | 144k simple writes/s; **43k no-step workflows/s** (2 writes each; the limit was a single WAL-flushing process); queued workflows 12.1k/s on one queue, **30.6k/s** with many partitions (limit: lock contention on queue heads). Load from many async Python clients. |
| [Production checklist](https://docs.dbos.dev/production/checklist) | "processing 1000 actions per second … requires 4 Postgres vCPUs"; a single database sustains ">40K workflows or steps per second"; each workflow is "two-three database writes" plus one per step; session-mode poolers only (LISTEN/NOTIFY). |
| [LISTEN/NOTIFY post](https://www.dbos.dev/blog/postgres-listen-notify-scalability) (2026-07-24) | Stream writes went from 2.9k/s to 60k/s at 15–100 ms latency by buffering notifications off the write path; buffered notifications are lost on crash (a polling fallback covers them). |
| [Recall.ai](https://www.recall.ai/blog/postgres-listen-notify-does-not-scale) | `NOTIFY` takes a global lock at commit that serializes notifying commits; this caused outages under many concurrent writers. |
| [August 2026 update](https://www.dbos.dev/blog/new-in-dbos-august-2026) | "10x better partitioned queue performance" (no numbers published). |
| `test_high_async_concurrency` | Thousands of concurrent async waits on a 32-thread pool. |

### 5.2 Per-operation cost (from source)

| Operation | Transactions | Rows written |
|---|---|---|
| Step (live) | 2: read (status + output lookup), write (row lock on `workflow_status` + insert) | 1 insert into `operation_outputs` (heap + 3 B-tree entries) + 1 tuple lock (WAL record) |
| Step (replay) | 1 read | 0 |
| Workflow start (direct) / enqueue + dequeue | 1 / 2 | `workflow_status` + `workflow_input`; dequeue updates status (non-HOT: indexed columns change) |
| Workflow completion | 1 | status update + `workflow_output` |
| `send` from a workflow | 1 | notification row(s) + 1 checkpoint; trigger `NOTIFY` per row on the write path |
| `recv` | 2–3 | timeout checkpoint + consume update + checkpoint |
| `write_stream` from a workflow | 1 | stream row + checkpoint; `NOTIFY` coalesced |
| `DBOS.asyncio_wait` | 1 | 1 checkpoint |

Payloads are stored as `TEXT`. The default serializer is base64-encoded pickle (+33 % size); a JSON serializer is
available. Values above about 2 KB are compressed by TOAST (Postgres behaviour). We found no DBOS-level size limit
(**unverified absence**). `pg_notify` payloads carry only keys.

### 5.3 Estimates at the design point

Assumptions: 30k turns/s fleet, 12 cells, about 2.5 effects per turn in durable pump mode (1 model call, about 1.5
environment or tool calls), plus about 0.3 `asyncio_wait` checkpoints per turn.

| Quantity | Fleet | Per cell |
|---|---|---|
| Checkpoint rows | ~84k/s | **~7k/s** (peak 2–3× → 15–20k/s) |
| Transactions | ~170k/s | ~14k/s (about half are read-only, with no WAL flush) |
| WAL (payload ~3 KB avg after offload, + indexes + full-page writes) | — | ~30–60 MB/s (**estimate**) |
| Rows per run | RL episode (20 turns): ~60–70; 500-turn agent run: ~1,400 | — |
| Rows per day before retention | ~7B | ~600M (~1–2 TB/day) |
| Postgres CPU by DBOS's rule of thumb | — | ~28 vCPU steady; size 64–96 vCPU for peaks (**extrapolation**) |
| Executor processes | — | 20–50 (Python; per-process step throughput **unverified**, guess 1–3k steps/s) |
| Connections | — | 50 × (20 pool + 1 listener) ≈ 1k |

**Analysis: where it breaks, in order of likelihood.**

1. **Retention churn.** `operation_outputs` is not partitioned (PK `(workflow_uuid, function_id)`). Retention *deletes*
   rows (batched, with explicit `VACUUM`, optimized in 3.0), so steady state deletes about as many rows per second as
   it inserts. That doubles heap and index churn and makes autovacuum the limit. Mitigation: short retention (24–72 h),
   archival of long-lived runs with `export_workflow`, and a spike (S-D2). If it fails, we patch DBOS to partition by
   time, which is a fork risk.
2. **WAL volume from payloads.** Keep inline step outputs small (offload above about 4–8 KB to blob storage in the
   serializer). Our current 64 KiB inline limit is too high for this substrate.
3. **Recovery storms.** Replay reads one transaction per recorded step with no prefetch. An executor with 2k resident
   runs of about 300 recorded steps each needs about 600k reads, roughly 1–2 minutes of elevated load (**estimate**).
   Generations (continue-as-new) and actor activations bound this.
4. **`NOTIFY` on `send`.** Chatty swarms can reach thousands of notifying commits per second per cell, near the regime
   Recall.ai describes. Use `send_bulk`, entity mailboxes, or set `use_listen_notify=False`. The last is not viable:
   the polling listener issues one query per registered waiter per interval.
5. **Queue-polling latency.** Every executor polls every queue at `polling_interval_sec` (default 1 s, jittered). For
   activations, use 50–100 ms on the `entities` queue: 30 executors × 10–20 Hz = 300–600 dequeue sweeps/s. Measured
   need at the design point: at most 2.5k activations/s per cell, well under the benchmark's 30.6k/s.
6. **Resident idle runs** in native mode: 25k coroutines per cell is feasible for memory (about 0.2–2 MB each,
   **estimate**), but they are pinned to an executor and a version, and each `recv` re-polls the database every 60 s
   (about 400 queries/s per cell). The actor layer removes this.

---

## 6 Failure semantics

| Failure | What DBOS does (fact) | Our handling (design) |
|---|---|---|
| Executor process crash | `PENDING` rows keep its `executor_id`. A restart with the same ID recovers them at launch. Otherwise they wait for `recover_pending_workflows` | Stable IDs (StatefulSet pod names) plus our recovery controller (§8). In-flight steps re-execute: receivers deduplicate on `effect_id`, and non-deduplicating tools yield `outcome_unknown` through attempt markers |
| Zombie executor (declared dead, still running) | Re-enqueue sets `owner_xid = NULL`. The zombie's next checkpoint fails `_check_owner_txn` → `DBOSWorkflowConflictIDError`, and the execution parks and waits for the new owner's outcome | Duplicate dispatches are bounded by the zombie's in-flight steps and absorbed by receivers. Entity state writes also compare-and-set on `state_version` |
| Postgres failover | `db_retry` retries database calls. The listener reconnects, and missed notifications are caught by a 60 s fallback poll | Cell stall for the failover duration (as in ADR-0002). `recv` wake-ups may lag up to 60 s after failover (**tunable only by patching** `_notification_fallback_polling_interval`) |
| Notifier crash with a buffered batch | Event and stream notifications are lost; readers fall back to polling | Acceptable for `run.emit` |
| Step raises | Recorded as the step's error after retries; the workflow sees the exception | Tool exception → `tool_result{is_error}`; infrastructure failure → `*.failed` completion to the driver |
| Workflow code raises | `ERROR` with the serialized exception | `run.failed{TASK_ERROR}`, then the reaper tears down |
| Non-determinism | `DBOSUnexpectedStepError` on a step-name mismatch at the same ID. Recovery retries up to `max_recovery_attempts` (100) → `MAX_RECOVERY_ATTEMPTS_EXCEEDED` | Pump mode adds our own digest check (`arguments_digest`) → `NON_DETERMINISM`, quarantine; set `max_recovery_attempts` low (e.g. 5) |
| Cancel or timeout | `CANCELLED`; next step raises; no durable cleanup; children cascade | Cooperative cancel + reaper (§3.3) |
| No live executor for a version | Rows stay `ENQUEUED`/`PENDING` indefinitely | Controller alarms on versions with pending rows and no live executor. Pump mode makes this rare |
| Queue deleted | "Pending workflows on it are unrecoverable" | Never delete queues with pending rows |
| Message to a missing workflow | FK violation → `DBOSNonExistentWorkflowError` | Signal-with-start (§3.1) or `deliver` |

---

## 7 Security and multi-tenancy

**Facts.**
- An executor needs a read-write connection to the system database, and DDL too unless `run_migrations=False`.
  `application_name` separates applications logically, not as a security boundary. There is no row-level security.
- The default serializer is **pickle**. Any party that can write a step output, workflow input or message that a
  trusted executor deserializes can execute code in that executor. A custom `serializer` is supported
  ([`_serialization.py`](https://github.com/dbos-inc/dbos-transact-py/blob/c179fe2/dbos/_serialization.py)).
- Conductor's websocket gives Conductor cancel, resume, fork, delete and export authority over every connected
  executor and, unless `conductor_metadata_only_mode=True`, access to inputs and outputs.

**The credentials-in-process problem.** In native mode, task code runs in a process that holds the cell's
system-database credentials. Any task code can read every tenant's runs in the cell, forge checkpoints for any
workflow, cancel or fork others, or plant pickles. The no-network determinism sandbox of ADR-0013 cannot be applied to
a process that must talk to Postgres and to effect receivers.

**If Q10 = in-house only.** Native mode is tolerable with hardening:
- a JSON (or msgpack) serializer, never pickle, with per-tenant envelope encryption of payloads (also enables
  crypto-shredding);
- a DML-only executor role on schema `dbos`, a separate migration role, and a read-only role for observability;
- effect executors (recorder, envlet, tool router) authorize by `run_id` and attachment tokens, which the process
  obtains from the Environment Manager and never from task code;
- no Conductor, or self-hosted with metadata-only mode;
- **residual risk:** a buggy or compromised dependency in task code compromises the cell, and determinism is not
  enforced.

**If Q10 = external tenants.** Use pump mode (§3.2):

```
tenant task code ──(HarnessHost over a Unix socket; JSON; no network; gVisor)──▶ pump (trusted, DBOS, DB credentials)
                                                                                   │ authorize(run_id, effect)
                                                                                   ▼
                                                           recorder · envlet · Environment Manager · tool router
```

- The sandbox holds nothing but its run's state. The pump validates each `EffectRequest` against the pinned tool set,
  environment ownership and attachments, and budgets and quotas before creating a step. That is the runtime's existing
  job ([effects](../contracts/effects.md#rules)).
- Only JSON crosses into the pump. Snapshots are pickled and unpickled *inside* the sandbox only (restricted unpickler,
  as designed).
- Tenants can be pinned to executor pools with `listen_queues` (per-tenant or per-trust-tier queues) when noisy
  neighbours or compliance require it. Separate cells per high-trust tenant remain the strong option.
- The actor layer's `deliver` runs in the Control API with a role that can only call `deliver` and
  `dbos.enqueue_workflow`/`dbos.send_message`.

---

## 8 Operations

**Per cell:**
- managed Postgres (HA), schemas `dbos` + `actors` + `effects` (attempt markers);
- migrations run by a job (`dbos migrate`, or print migrations and apply them), with executors on
  `run_migrations=False`;
- the executor fleet as a StatefulSet (stable `executor_id` = pod name; `application_version` = pump version);
- task-host sandbox pools per `code_reference`;
- the recovery controller, the reaper, retention and archival;
- OpenTelemetry export.

**Recovery controller (replaces Conductor).** Each executor upserts
`executor_heartbeat(executor_id, application_version, last_seen)` every 5 s. The controller, a leader-elected per-cell
loop:
1. Treat an executor as dead when its heartbeat is older than 15 s *and* Kubernetes reports the pod gone or replaced
   (boot nonce changed).
2. Run the same statement as `reenqueue_for_recovery` for its `PENDING` rows:
   `UPDATE dbos.workflow_status SET status='ENQUEUED', owner_xid=NULL, started_at_epoch_ms=NULL,
   queue_name=COALESCE(queue_name, '<internal queue>') WHERE status='PENDING' AND executor_id = ANY($1)`.
   Using `DBOS._recover_pending_workflows` from a live executor is equivalent. The internal queue name is
   `INTERNAL_QUEUE_NAME` in source.
3. Alarm on `ENQUEUED`/`PENDING` rows whose `application_version` has no live heartbeat.

Takeover latency is about 20 s plus the queue poll, against Conductor's 60 s default.

**Deploys.**
- *Pump mode:* task-code deploys are new sandbox pools, with no DBOS version change. Pump changes use
  `enable_patching` (constant version, `DBOS.patch`) or blue-green with the old StatefulSet kept until
  `PENDING = 0` for its version.
- *Native mode:* one StatefulSet per live task version, drained by version. This is operationally heavy with long
  runs.

**Retention.** Configure DBOS retention (completed-at cutoff plus row threshold). `export_workflow` archives
long-lived or audited runs to object storage first. Monitor table and index bloat and autovacuum lag.

**Observability.** DBOS OpenTelemetry spans (`otel_attribute_format="semconv"`), SQL dashboards over
`workflow_status` (counts by status, queue, version, executor), and `get_metrics`. Conductor is optional for its UI
(Enterprise for self-hosting).

**Conductor or not.** Not required. Reconsider only for the operator UI, and then self-hosted with metadata-only mode.

---

## 9 What we still build

Rough sizes (lines of code, **estimates**):

| Component | Size | Notes |
|---|---|---|
| Pump workflow + effect steps + HarnessHost bridge + authorization | 2–3k | Python, or Go on DBOS Go (verify parity) |
| Task host (sandbox, replay driver, snapshots) | as already planned | unchanged; loses its `Load` path (replay is re-feeding) |
| Attempt markers + effect-executor adapters (recorder, envlet, Environment Manager, tool router) | ~0.6k | |
| Cooperative cancellation + reaper | ~0.5k | |
| Virtual-actor layer (schema, `deliver`, activation, `park`, timeouts, upcasters, history) | 2–3k | |
| Identity service (registry, addresses, permissions, owned resources, memory API) | 2–3k | |
| Recovery controller + heartbeats | ~0.5–0.8k | |
| Serializer (JSON/msgpack, per-tenant encryption, blob offload) | ~0.4k | |
| Control API integration (`DBOSClient` / SQL functions), run-event projection for B13 | ~1.5k | |
| Retention/archival, metrics, dashboards | ~1k | |
| **Total** | **~11–15k** | against a custom Run Store + Runtime whose size we have not estimated (**unverified comparison**) |

What we drop from the current design: the Go Runtime's partition leases, mailboxes and dispatch (DBOS queues,
executors and `owner_xid` replace them); the Run Store's event log, timers and inbox tables; the timer relay.

---

## 10 Risks, open questions, spike tests

**Risks.**
1. **Delete-based retention at 7k checkpoints/s per cell** (§5.3.1). If it fails, we fork DBOS's schema.
2. **Young fencing and recovery code** (`owner_xid`, conflict resolution, 3.0 schema split, all 2026). Chaos tests are
   needed before trusting it for side-effecting tools.
3. **Cancellation semantics** conflict with "teardown always runs". Cooperative cancellation must be airtight.
4. **Vendor direction.** Features we rely on (SQL enqueue, partitioned batched dequeue, patching) are recent and
   move quickly across majors. Revenue comes from Conductor, but the library is MIT, so the fork option is real.
5. **Native mode's determinism rules** (step start order) are easy to violate in agent code. Pump mode avoids this.
6. **Python executor throughput** (GIL, synchronous database driver through threads) may need more processes than
   planned.

**Open questions.**
- Does the Go SDK support everything pump mode needs (partitioned queues, durable wait-any, patching, custom
  serializer)? It decides whether ADR-0011 survives unchanged.
- Can our `run_id` format be used verbatim as `workflow_id`?
- Recorder `session_id` for best-effort re-attempts: `{run_id}~{attempt}/{slot}`, or a new `run_id` per attempt?
- Should `run.emit` and the RL event projection use DBOS streams (2 rows per event) or our own table?
- Is the 60 s `recv` fallback poll acceptable after Postgres failovers, or do we patch it?

**Spike tests (measurable).**

| # | Test | Pass criterion |
|---|---|---|
| S-D1 | Per-cell throughput: 25k simulated runs in pump mode, 7.5k then 20k checkpoints/s, payloads 2 / 8 / 32 KB, target instance class | p99 checkpoint commit ≤ 20 ms at 7.5k/s; find the knee; record CPU, WAL MB/s, IOPS |
| S-D2 | Retention soak: 48 h at 7.5k/s with 24 h retention | table and index size reach steady state; p99 within 20 % of the no-retention baseline; autovacuum never more than 1 h behind |
| S-D3 | Recovery storm: kill -9 an executor holding 2k runs × 300 recorded steps | all runs progressing again ≤ 2 min after the controller fires; database CPU < 80 % |
| S-D4 | Actor activations: 25k keys, 2.5k deliveries/s, polling 50 and 100 ms, chaos (kill -9, network partition) | p99 deliver → activation start ≤ 250 ms; **zero** double activations and **zero** lost wake-ups over 10M deliveries |
| S-D5 | `NOTIFY` pressure: `send` 0.5k–10k/s with 30 listeners alongside a checkpoint load | commit p99 of unrelated checkpoints degrades < 2× at the expected swarm rate; find the knee |
| S-D6 | Cancellation: 10k runs cancelled or deadline-expired mid-effect | 100 % of owned environments destroyed within 60 s; no teardown skipped |
| S-D7 | Zombie fencing: partition an executor, recover elsewhere, heal | 0 duplicate checkpoints; duplicate dispatches ≤ in-flight steps at partition time; all absorbed by receivers |
| S-D8 | Per-process capacity: steps/s per Python executor; resident idle `recv` workflows per GB | ≥ 1k steps/s per process; ≥ 5k idle workflows per GB (targets to confirm) |
| S-D9 | Pump replay: recovery time for a 500-turn run (reads + sandbox CPU); generation hand-off cost | ≤ 2 s per run; continue-as-new ≤ 10 rows |

---

## 11 Sources

- DBOS Transact Python source, commit `c179fe2` (2026-09-25):
  [`_core.py`](https://github.com/dbos-inc/dbos-transact-py/blob/c179fe2/dbos/_core.py),
  [`_sys_db.py`](https://github.com/dbos-inc/dbos-transact-py/blob/c179fe2/dbos/_sys_db.py),
  [`_sys_db_postgres.py`](https://github.com/dbos-inc/dbos-transact-py/blob/c179fe2/dbos/_sys_db_postgres.py),
  [`_queue.py`](https://github.com/dbos-inc/dbos-transact-py/blob/c179fe2/dbos/_queue.py),
  [`_recovery.py`](https://github.com/dbos-inc/dbos-transact-py/blob/c179fe2/dbos/_recovery.py),
  [`_dbos.py`](https://github.com/dbos-inc/dbos-transact-py/blob/c179fe2/dbos/_dbos.py),
  [`_dbos_config.py`](https://github.com/dbos-inc/dbos-transact-py/blob/c179fe2/dbos/_dbos_config.py),
  [`_migration.py`](https://github.com/dbos-inc/dbos-transact-py/blob/c179fe2/dbos/_migration.py),
  [`_serialization.py`](https://github.com/dbos-inc/dbos-transact-py/blob/c179fe2/dbos/_serialization.py),
  [`_schemas/system_database.py`](https://github.com/dbos-inc/dbos-transact-py/blob/c179fe2/dbos/_schemas/system_database.py),
  [`_conductor/protocol.py`](https://github.com/dbos-inc/dbos-transact-py/blob/c179fe2/dbos/_conductor/protocol.py),
  [`tests/test_concurrency.py`](https://github.com/dbos-inc/dbos-transact-py/blob/c179fe2/tests/test_concurrency.py)
- Releases: [PyPI `dbos`](https://pypi.org/project/dbos/),
  [3.0.0](https://github.com/dbos-inc/dbos-transact-py/releases/tag/3.0.0),
  [3.1.0](https://github.com/dbos-inc/dbos-transact-py/releases/tag/3.1.0),
  [upgrading to 3.0](https://docs.dbos.dev/python/upgrading)
- Docs: [architecture](https://docs.dbos.dev/architecture),
  [production checklist](https://docs.dbos.dev/production/checklist),
  [workflow recovery](https://docs.dbos.dev/production/workflow-recovery),
  [Kubernetes](https://docs.dbos.dev/production/hosting-with-kubernetes),
  [self-hosting Conductor](https://docs.dbos.dev/production/hosting-conductor),
  [queues](https://docs.dbos.dev/python/tutorials/queue-tutorial),
  [workflow communication](https://docs.dbos.dev/python/tutorials/workflow-communication),
  [workflows](https://docs.dbos.dev/python/tutorials/workflow-tutorial),
  [upgrading workflows](https://docs.dbos.dev/python/tutorials/upgrading-workflows),
  [OpenAI Agents SDK integration](https://docs.dbos.dev/integrations/openai-agents)
- Blog and commercial: [benchmark](https://www.dbos.dev/blog/benchmarking-workflow-execution-scalability-on-postgres),
  [LISTEN/NOTIFY scalability](https://www.dbos.dev/blog/postgres-listen-notify-scalability),
  [August 2026 update](https://www.dbos.dev/blog/new-in-dbos-august-2026),
  [pricing](https://www.dbos.dev/dbos-pricing), [Conductor licence](https://www.dbos.dev/conductor-license)
- Issues: [#652](https://github.com/dbos-inc/dbos-transact-py/issues/652),
  [#688](https://github.com/dbos-inc/dbos-transact-py/issues/688),
  [#582](https://github.com/dbos-inc/dbos-transact-py/issues/582),
  [#619](https://github.com/dbos-inc/dbos-transact-py/issues/619),
  [#716](https://github.com/dbos-inc/dbos-transact-py/issues/716),
  [#742](https://github.com/dbos-inc/dbos-transact-py/issues/742),
  [#767](https://github.com/dbos-inc/dbos-transact-py/issues/767),
  [#804](https://github.com/dbos-inc/dbos-transact-py/issues/804),
  [#806](https://github.com/dbos-inc/dbos-transact-py/issues/806)
- Third party: [Recall.ai, "Postgres LISTEN/NOTIFY does not scale"](https://www.recall.ai/blog/postgres-listen-notify-does-not-scale),
  [Pydantic AI + DBOS](https://pydantic.dev/docs/ai/integrations/durable_execution/dbos/),
  [DBOS Go](https://github.com/dbos-inc/dbos-transact-golang),
  [Hacker News thread on executor elasticity without Conductor](https://news.ycombinator.com/item?id=47970487)
