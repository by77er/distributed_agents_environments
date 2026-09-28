# Restate as the durable execution substrate

Status: **Draft** · 2026-09-27 · Question: how well does Restate (services, virtual objects, workflows; the Restate
server and its Python SDK) serve as the durable execution substrate, and what does the concrete design look like?

Conventions: **Fact** statements carry a source link. **Analysis** and **Design** are ours. Anything not verified
against a primary source is marked *(unverified)*. Versions checked on 2026-09-27.

---

## 1. Verdict

**Restate is a strong fit, and a better fit than DBOS for swarms and durable identities.** It fits best as the
durability layer under a platform-owned shim, not as the place where task code is deployed directly.
Virtual objects give us actors with a mailbox, a per-key lock and K/V state out of the box. Workflows map 1:1 to runs.
Suspended invocations hold no worker resources. A per-cell Restate cluster replaces the Run Store, leases, the
inbox, timers and much of the runtime. Replit already runs "tens of Restate cells" in exactly this shape.
The costs:
- `ctx.run` is at-least-once, so our receivers must deduplicate. They already do.
- There is no in-code patching, so long runs must be split into segments.
- There is no authorization between services, so tenant code (Q10) must never speak the Restate protocol.
- Flow control scopes co-locate everything in a scope on one partition, so tenant quotas stay ours.
- The server is BSL. That is fine for our use as long as tenants never touch Restate APIs.

The Python SDK only reached 1.0 in June 2026. Density and replay cost are unmeasured, so spikes S-R1…S-R4 gate
adoption.

---

## 2. What it is

### 2.1 Programming model (facts)

- **Three service types.** *Basic services* have stateless handlers with unlimited concurrency. *Virtual objects*
  are keyed entities with isolated K/V state and at most one *exclusive* (writing) handler per key at a time;
  *shared* handlers run concurrently and can only read. *Workflows* run their `run` handler exactly once per workflow
  ID; concurrent shared handlers resolve durable promises and query state
  ([Services](https://docs.restate.dev/foundations/services)). Calls to a virtual object key run serially in arrival
  order. Two sends A then B from one invocation are guaranteed to run A before B
  ([service communication](https://docs.restate.dev/develop/python/service-communication)).
- **Durable execution by journal and replay.** Every context operation (`ctx.run`, state access, calls, sleeps,
  awakeables) is recorded in a per-invocation journal. After a failure the SDK re-runs the handler and resolves
  operations from the journal. A divergence is a journal mismatch, `RT0016`
  ([request lifecycle](https://docs.restate.dev/guides/request-lifecycle),
  [versioning](https://docs.restate.dev/services/versioning)).
- **Python context API** (SDK 1.0.5):
  - `ctx.run_typed(name, action, RunOptions(max_attempts, max_duration, …))`
  - `ctx.sleep`
  - `ctx.service_call/send`, `ctx.object_call/send`, `ctx.workflow_call/send` with `idempotency_key` and
    `send_delay`
  - `ctx.awakeable()` / `resolve_awakeable`
  - `ctx.promise(name)` (workflows only)
  - `ctx.signal(name)` / `resolve_signal(invocation_id, …)` (protocol v7, preview)
  - `ctx.cancel_invocation`, `ctx.attach_invocation`
  - `ctx.random()`, `ctx.uuid()`, `ctx.time()`, all deterministic
  - combinators `restate.gather`, `select`, `as_completed`, `wait_completed`
  - `ctx.scope(scope).…` for flow control

  Sources: [context.py](https://github.com/restatedev/sdk-python/blob/main/python/restate/context.py),
  [\_\_init\_\_.py](https://github.com/restatedev/sdk-python/blob/main/python/restate/__init__.py).
- **Per-service options**: `inactivity_timeout`, `abort_timeout`, `journal_retention`, `idempotency_retention`,
  `enable_lazy_state`, `ingress_private`, `invocation_retry_policy`
  ([object.py](https://github.com/restatedev/sdk-python/blob/main/python/restate/object.py)). The default retry
  policy is 50 ms initial, ×2, 60 s max interval, 70 attempts, then **pause**
  ([configuration](https://docs.restate.dev/services/configuration)).

### 2.2 Architecture and storage (facts)

Source: [architecture reference](https://docs.restate.dev/references/architecture) unless noted.

- **One Rust binary** with four roles: metadata server (built-in Raft), HTTP ingress, log server, worker.
- **Bifrost** is a segmented, replicated log, one per partition. Each log has a single sequencer that sits with the
  partition leader. An append commits on quorum acknowledgement from a *nodeset* of log servers. Reconfiguration seals
  the active segment and publishes a new one via a metadata compare-and-swap
  ([clusters](https://docs.restate.dev/server/clusters)).
- **Partition processors** tail the log. The leader invokes handlers and materializes journals, idempotency records,
  virtual-object state, timers and inboxes into embedded RocksDB. Followers apply the same log so they can fail over
  fast. The processor state is a *cache*; the log is the source of truth.
- **Snapshots and trimming.** Snapshots of the partition store go to object storage, and the log is trimmed behind
  them. S3 (and compatible stores), GCS and Azure Blob are supported
  ([snapshots](https://docs.restate.dev/server/snapshots)).
- **Metadata** lives in the replicated Raft store (default), etcd, or S3 only
  ([metadata](https://docs.restate.dev/server/metadata)).
- **Routing.** Object keys, workflow IDs or idempotency keys hash to a partition. Cross-partition calls go into the
  origin's log and are delivered exactly once by a *shuffler*.
- **Partition count is fixed at provisioning** (`default-num-partitions`, default 24, max 65,535; "ignored for
  provisioned clusters") ([server config](https://docs.restate.dev/references/server-config)). Splitting is future
  work.
- **Replication** can be location-aware, e.g. `{zone: 2, node: 3}`. Partition replication (processor replicas) is
  separate from log replication (durability)
  ([clusters](https://docs.restate.dev/server/clusters)).
- **Fencing.** Every attempt carries an epoch, and the processor ignores events from superseded attempts or leaders.

### 2.3 Invocation model (facts, several verified in source)

- **Push, not pull.** The server opens a stream to a registered *deployment* (HTTP endpoint or Lambda) and pushes the
  invocation to it. There are no worker polling loops. With HTTP/2 the stream is bidirectional; request/response
  targets (HTTP/1.1, Lambda) re-invoke after every blocking step
  ([request lifecycle](https://docs.restate.dev/guides/request-lifecycle)). Since v1.7 the invoker keeps an HTTP/2
  connection pool with at most 128 streams per connection by default
  ([v1.7.0 notes](https://github.com/restatedev/restate/blob/main/release-notes/v1.7.0.md)).
- **`ctx.run` executes the closure first and journals the result afterwards.**
  - The SDK sends a `RunCommand` without waiting for it to be stored, runs the closure, then proposes the completion.
    The future resolves only once the runtime acknowledges the result was stored
    ([protocol.proto](https://github.com/restatedev/restate/blob/main/service-protocol/dev/restate/service/protocol.proto),
    [server_context.py](https://github.com/restatedev/sdk-python/blob/main/python/restate/server_context.py)).
  - A crash between the side effect and storing its result re-executes the closure. A public reproduction applied a
    non-deduplicating request twice in 30 of 30 trials on server 1.7.12 / SDK 1.0.5
    ([docs-restate#410](https://github.com/restatedev/docs-restate/issues/410)).
  - So **`ctx.run` is at-least-once**, and exactly-once only with a deduplicating receiver.
- **When an invocation suspends.**
  - In HTTP/2 bidirectional mode the server keeps the stream open. If no journal activity happens for
    `inactivity_timeout` (default 1 min), it closes the request stream and the SDK suspends at its await point.
  - `abort_timeout` (default 10 min) then bounds how long the SDK may still take to finish a running `ctx.run`
    before the attempt is aborted.
  - An inactivity timeout of zero forces request/response mode, where the SDK suspends at every await.
  - The server does not yet use the SDK's `AwaitingOn` hint to suspend early (there is a `todo` in the code).
  - Sources: [error handling](https://docs.restate.dev/guides/error-handling),
    [service_protocol_runner_v4.rs](https://github.com/restatedev/restate/blob/main/crates/invoker-impl/src/invocation_task/service_protocol_runner_v4.rs).
- **A suspended invocation holds no deployment resources.** It is only journal and status rows in the partition
  store, re-pushed with the full journal when a completion arrives
  ([key concepts](https://docs.restate.dev/foundations/key-concepts)). While *running* — including while waiting
  inside a `ctx.run` — it holds an HTTP/2 stream, a coroutine in the SDK process and an invoker concurrency slot.
- **Invoker limits.**
  - `concurrent-invocations-limit`: 1,000 per partition processor in 1.7. In the upcoming 1.8 it becomes 24,000 per
    node ([unreleased notes](https://github.com/restatedev/restate/blob/main/release-notes/unreleased/increase-invoker-concurrency-limit-default.md)).
  - Bounded memory pool: 1.5 GiB per node, up to 32 MiB per invocation, 32 KiB initial reservation. Journal replay
    and state loading draw on this budget ([v1.7.0](https://github.com/restatedev/restate/blob/main/release-notes/v1.7.0.md)).
- **Size limits.** A single journal entry (state value, `ctx.run` result, payload) may be at most `message-size-limit`,
  32 MiB by default; larger entries fail with `RT0003`
  ([v1.6.0](https://github.com/restatedev/restate/blob/main/release-notes/v1.6.0.md)). The ingress rejects bodies
  over the limit with 413 (v1.7.0). **No limit on journal length** is documented. The whole journal is re-sent on
  every re-invocation.
- **K/V state.**
  - Eager state (the default for objects and workflows) ships the entire object state in the start message.
  - `eager-state-size-limit` caps it; anything beyond the cap is fetched lazily. `enable_lazy_state` makes all state
    lazy.
  - Streaming of large state is an open issue ([#4344](https://github.com/restatedev/restate/issues/4344)).
- **Signals and waits.**
  - Awakeables are one-shot, ID-addressed completions that can be resolved over HTTP.
  - Durable promises are per-workflow, named, and can be resolved before anyone awaits them.
  - Signals (protocol v7, preview) are named completions addressed by invocation ID
    ([external events](https://docs.restate.dev/develop/python/external-events),
    [Python SDK 1.0.0](https://docs.restate.dev/changelog/python-sdk)).
- **Timers**: `ctx.sleep` and delayed sends (`send_delay`).
- **Idempotency**: `Idempotency-Key` on ingress and `idempotency_key` on calls. Retention is configurable (default
  24 h, v1.7). With `controlled-idempotent-sharding` (default on new clusters) the ingress adds a random key to
  calls that lack one ([v1.7.0](https://github.com/restatedev/restate/blob/main/release-notes/v1.7.0.md)).
- **Cancellation and kill**
  ([managing invocations](https://docs.restate.dev/services/invocation/managing-invocations)):
  - *Cancel* surfaces a `TerminalError` at the next await. The handler may run compensations; a `ctx.run` already
    executing finishes first. Cancel propagates to leaves of the call tree, then back up.
  - One-way and delayed sends are detached and are not cancelled.
  - *Kill* stops the tree immediately, without compensation.
  - Pause/resume exists, including "resume on a different deployment".

### 2.4 Versioning and deployments (facts)

- **Immutable deployments.** A deployment is a registered endpoint. New invocations go to the latest deployment of a
  service. Existing invocations, and every retry, stay pinned to the deployment where they started
  ([versioning](https://docs.restate.dev/services/versioning)).
- **No in-code patching.** There is no `patched()` / `get_version()` in the Python SDK (source grep). The docs advise
  against long-running handlers and suggest breaking work into chunks with delayed calls.
- **Draining.** The Kubernetes operator's `RestateDeployment` creates one ReplicaSet per version and keeps old ones
  until Restate reports no invocations pinned to them. In ReplicaSet mode it keeps at least one replica per draining
  version ([operator README](https://github.com/restatedev/restate-operator)).
- **Stuck invocations.** Pinned invocations with a bug can be paused and resumed on a new deployment, at the risk of
  `RT0016`.
- **Server upgrades can strand old invocations.**
  - The next release drops service protocol ≤ v3 and fails invocations still pinned to it
    ([unreleased notes](https://github.com/restatedev/restate/blob/main/release-notes/unreleased/drop-service-protocol-v3.md)).
  - v1.6 already rejected new invocations for deprecated SDKs.

### 2.5 Flow control (facts)

- **Scopes and limit keys** (v1.7, experimental).
  - A *scope* namespaces invocation identity and carries concurrency limits from a cluster-wide rule book.
  - *Limit keys* (up to two levels, e.g. `tenant1/user42`) subdivide a scope's limits
    ([flow control](https://docs.restate.dev/services/flow-control)).
- **A scope is also a sharding key.** For scoped objects and workflows, the partition key is derived from the scope,
  not from the object key ([identifiers.rs](https://github.com/restatedev/restate/blob/main/crates/types/src/identifiers.rs)).
  So **everything in one scope lands on one partition**.
- **Status.** vqueues become always-on in 1.8 (unreleased). Rate limits, priorities and backlog limits are on the
  roadmap ([OSS roadmap](https://docs.restate.dev/roadmap/oss)).

### 2.6 License (facts)

- **Server: Business Source License 1.1**
  ([LICENSE](https://github.com/restatedev/restate/blob/main/LICENSE)).
  - The *Additional Use Grant* permits any use except a "Public Restate Platform Service": a managed service that lets
    third parties "register their own service deployments and invoke them" through Restate's APIs.
  - Explicitly permitted:
    - production use invoking services written by the licensee;
    - internal managed platforms where only the licensee can reach Restate's APIs;
    - "any publicly exposed workflow platform that presents an abstraction layer such as … [a] proprietary API that
      consumes Restate functionality", provided that consumers do not specify, add or invoke workflows through
      Restate's APIs directly.
  - Each version converts to **Apache-2.0 four years after its release**.
- **Python SDK, shared core and operator: MIT** (GitHub license metadata).

### 2.7 Maturity and versions (facts)

- **Server** v1.7.12 (2026-09-22). The release train runs 1.6 → 1.7 with frequent patch releases; 1.8 is in
  preparation. About 4.5k GitHub stars
  ([releases](https://github.com/restatedev/restate/releases)).
- **Python SDK** v1.0.5 (2026-09-02); 1.0.0 shipped on 2026-06-23. Before that the 0.x line ran for about two years.
  Integrations ship in `restate.ext`: OpenAI Agents SDK, Pydantic AI, Google ADK, LangChain, tracing (Logfire / OTel)
  ([sdk-python](https://github.com/restatedev/sdk-python)).
- **Operator** v3.1.0 (2026-09-22).
- **Company.** Founded by creators of Apache Flink; $7M seed in 2024
  ([funding](https://startupnews.fyi/2024/06/12/restate-raises-7m-for-its-lightweight-workflows-as-code-platform/)).
  Vendor-reported production users include Replit, which runs "tens of Restate cells" with "up to 25,000 durable
  actions per second per cell" at peak ([Restate vs Temporal](https://restate.dev/vs/temporal)).
- **Test infrastructure**: Jepsen and e2e repositories exist
  ([restatedev/jepsen](https://github.com/restatedev/jepsen)). We found no published Jepsen report *(unverified)*.

### 2.8 Performance evidence (facts)

- **Restate 1.2 on a 3-node, 3-way replicated cluster**
  ([1.2 announcement](https://restate.dev/blog/announcing-restate-1.2),
  [first-principles post](https://restate.dev/blog/building-a-modern-durable-execution-engine-from-first-principles/)):
  - peak about 17,000 workflows/s ≈ 84,000–94,286 actions/s on c6id.8xlarge nodes;
  - 9-step workflow: p50 116 ms, p99 163 ms;
  - 1-step: p50 16 ms, p99 40 ms under load;
  - single step median about 3 ms at low load.
  - Not disclosed: partition count, payload sizes, storage type. Payloads were tiny *(inference from the test
    description)*.
- The vendor claims "<10 ms per durable action" p99 overhead ([vs Temporal](https://restate.dev/vs/temporal)).
- **No published per-partition throughput and no Python SDK throughput numbers.**

### 2.9 Agent-specific features and observability (facts)

- **AI patterns and integrations** ([integration guide](https://docs.restate.dev/ai/sdk-integrations/integration-guide),
  [sessions](https://docs.restate.dev/ai/patterns/sessions)):
  - durable sessions as virtual objects;
  - multi-agent handoffs, human-in-the-loop via awakeables, interrupt-and-regenerate via cancellation;
  - model calls wrapped in `ctx.run`.
- The docs require agent SDKs to **disable streaming and native `asyncio.gather` parallelism** inside durable code.
- **Token streaming is not supported** (roadmap: "native streams"). Python SSE pub/sub is "planned"
  ([streaming](https://docs.restate.dev/ai/patterns/streaming-responses)).
- **Observability** ([introspection](https://docs.restate.dev/services/introspection),
  [SQL reference](https://docs.restate.dev/references/sql-introspection)):
  - SQL introspection (DataFusion over the partition stores, optionally rate-limited) with tables `sys_invocation`,
    `sys_journal`, `sys_journal_events`, `state`, `sys_inbox`, `sys_promise`, `sys_idempotency`, `sys_deployment`,
    and `sys_vqueues` / `sys_scheduler` / `sys_user_limits`;
  - `sys_locks` replaces `sys_keyed_service_status` in 1.8;
  - a bundled UI (1.0), Prometheus metrics with Grafana dashboards, OpenTelemetry traces per invocation.
- **Journal value encryption.** A `JournalValueCodec` (Python, preview) encrypts values but not keys, names or failure
  messages ([entry_codec.py](https://github.com/restatedev/sdk-python/blob/main/python/restate/entry_codec.py),
  [service security](https://docs.restate.dev/services/security)).

---

## 3. Concept mapping

Design choices in this table are ours. §4 justifies them.

| Our concept | On Restate | Notes / gaps |
|---|---|---|
| **Run** | `Run` **workflow**, key `{run_id}.{segment}` (segment `0` for almost all runs); `main` hosts one `Program` (e.g. `AgentProgram`) | Exactly-once per key; attach/output by key. A `RunDirectory` virtual object, key `run_id`, holds the durable record across segments |
| **Effect** | `ctx.run_typed(...)` whose closure calls the recorder / Environment Manager / envlet / tool router. `spawn` → `ctx.workflow_call/send`; `send` → `ctx.object_send` | Closure runs **before** it is journaled (at-least-once). P5 "persist before effect" holds only for Restate-internal effects (calls, sends, timers), which are journaled before being acted on |
| **Effect identity** | `effect_id = f"{run_id}:{segment}:{ordinal}"`, ordinal = deterministic counter in the shim; forwarded as the idempotency key | Restate does not expose the journal index to Python; our counter is equivalent under determinism. Receivers' three-state dedupe (absent / in progress → join / done) is **mandatory**, not an optimization |
| Non-deduplicating tools | Attempt marker: `ctx.run` records a per-attempt nonce before dispatch; if a later attempt replays a marker with a foreign nonce, the dispatch closure returns `outcome_unknown` instead of calling | One extra journal entry per such call (sketch in §4.1) |
| **Signals / inbox** | `RunDirectory.post` (exclusive, key `run_id`) assigns a sequence number and resolves the workflow promise `message:{seq}`; the run awaits promises in order. Approvals: awakeable or promise `approval:{effect_id}` | Workflow promises are one-shot and named, so ordering comes from `RunDirectory` |
| **Signal-with-start / run keys** | `RunKey` virtual object keyed by the external key (e.g. a Slack thread): starts a run if none is live, else posts to it. Virtual objects need no creation, so this is atomic per key | Natural fit |
| **Timers** | `ctx.sleep`, `restate.select(message=…, timeout=ctx.sleep(…))`, delayed sends for cron-like triggers | Cron is on Restate's roadmap; today use a self-rescheduling object |
| **Child runs** | `ctx.workflow_call(run_main, key=child_key, …)` → call future; `restate.gather` for fan-in; `workflow_send` + completion callback for detached children | Cancellation propagates to called children, not to sent ones |
| **Cancellation + teardown** | `RunDirectory.cancel` → `ctx.cancel_invocation(segment_invocation_id)` → `TerminalError` at next await → loop's `finally: teardown` runs as saga compensation; shim destroys owned environments | A `ctx.run` in flight completes first (e.g. a model sample). Python SDK 1.0.4 fixed "cancellation may cancel also compensation calls" |
| **Versioning** | Restate deployments version **the shim**; task code is pinned by `code_reference` in the journaled run input and loaded by the shim; long runs cross deploys at **segment boundaries** | No `patched()`. Segments replace it |
| **Snapshots / continuation** | Segment boundary = our resumable point: pickled task and agent state + history reference in blob storage, recorded in `RunDirectory`; the next segment is a new workflow on the latest deployment | Also bounds journal size and replay cost |
| **Outputs / streams** (`run.emit`) | `ctx.run` to our event sink (dedupe on `effect_id`); connectors consume from it. Final `RunResult` (rewards, ending, observations digest) returned by the workflow and posted to the assembler | The Restate journal is not our log schema; we export |
| **Token streaming** | Out of band: recorder publishes deltas to a pub/sub channel keyed by `(run_id, effect_id)`; `ctx.run` returns only the final message | Non-durable by design, as specified |
| **Queues / admission** | Rollout Controller buffer admission stays ours. Restate queues exclusive calls per object key; flow-control scopes as a secondary guard only | Scopes co-locate on one partition (§2.5), so they cannot express per-tenant caps at scale |
| **Recovery / fencing** | Partition leader epochs + attempt epochs; failover via followers; completion re-push | Replaces leases, `FENCED`, `SEQ_CONFLICT`, takeover scans |
| **Cells** | One Restate cluster per cell (3–6 nodes, `{zone: 2, node: 3}`), its own snapshot prefix; cross-cell `send` via our relay to the other cell's ingress with an idempotency key | Matches ADR-0001; Replit runs the same pattern |
| **Observability** | SQL introspection + UI for operators; OTel traces; our own run events for users / RL | SQL is for debugging, not bulk export |

---

## 4. Design on Restate

### 4.0 Choosing the handler type for the loop

**Analysis.**
- A *basic service* has no durable address other than an idempotency key, and no promises.
- A *virtual object* per run gives segments and a mailbox. But its exclusive lock is held for the whole loop, so every
  message, status write or cancel request queues behind a run that may go for minutes.
- A **workflow** gives:
  - one exactly-once `main` per key;
  - shared handlers that don't take the lock (status, promise resolution);
  - durable promises for approvals and messages;
  - attach/output by key;
  - its own retention.

So: **run = workflow**. The long-lived, mailbox-shaped parts go on virtual objects: `RunDirectory` (the run's durable
record), `RunKey` and `AgentIdentity`.

**Three placement decisions that shape everything else.**
1. **Restate hosts a platform-owned shim, not task packages.** The `Run` workflow is our code. It resolves the task
   and agent by `code_reference` from the journaled input, so task code versions do not multiply Restate deployments.
   Several package versions can run at once under one service name.
2. **Effects run in the deployment pods.** The push model means the closure runs where the handler runs. Deployment
   pods therefore need egress to the recorder, Environment Manager, envlets and tool router, which is exactly what our
   runtime workers have today.
3. **Runs are bounded by segments.** A segment ends at the run's end, or at a resumable point once a budget is
   exceeded (turns, journal bytes, wall time), or when a deploy drain is requested. This caps replay cost (§5) and is
   our replacement for patching.

### 4.1 The loop, effects and effect identity (Q10 = in-house code, in-process)

```python
from datetime import timedelta
import uuid
import restate
from restate import RunOptions, TerminalError, InvocationRetryPolicy

run_workflow = restate.Workflow(
    "Run",
    inactivity_timeout=timedelta(minutes=2),     # idle waits suspend after ~2 min
    abort_timeout=timedelta(minutes=15),         # longest model turn / tool call we allow
    journal_retention=timedelta(hours=1),        # our event export is the durable record
    invocation_retry_policy=InvocationRetryPolicy(max_attempts=20, on_max_attempts="pause"),
)


@run_workflow.main()
async def main(ctx: restate.WorkflowContext, request: RunSegmentRequest) -> RunSegmentResult:
    run = RestateRunContext(ctx, request)
    program: Program = await run.load_program(request.specification)   # AgentProgram(task, agent) or any Program
    try:
        outcome = await program.main(run)                                 # the normative loop for AgentProgram
    except SegmentBoundary as boundary:                                   # raised at a resumable point
        return await run.hand_over(boundary.snapshot)                     # next segment on the latest deployment
    return await run.finish(outcome)


@run_workflow.handler(kind="shared")
async def deliver(ctx: restate.WorkflowSharedContext, envelope: MessageEnvelope) -> None:
    await ctx.promise(f"message:{envelope.sequence}").resolve(envelope)


class AgentProgram(Program):
    async def main(self, run: RunContext) -> RunOutcome:
        return await rollout(self.task, self.agent, run)                  # harness README, unchanged


class RestateRunContext(RunContext):
    def __init__(self, ctx: restate.WorkflowContext, request: RunSegmentRequest) -> None:
        self._ctx = ctx
        self.run_id = request.run_id
        self._segment = request.segment
        self._ordinal = request.effect_ordinal_start      # carried across segments in the snapshot
        self._attempt_nonce = uuid.uuid4().hex            # deliberately NOT journaled: identifies this attempt
        self._clients = PlatformClients.for_cell()        # recorder, environment manager, envlet, tool router
        self._message_sequence = request.message_sequence_start
        self.random = ctx.random()

    def _next_effect_id(self) -> str:
        self._ordinal += 1
        return f"{self.run_id}:{self._segment}:{self._ordinal}"

    async def _perform(self, kind: str, request: EffectRequest, retry_class: RetryClass) -> EffectResult:
        effect_id = self._next_effect_id()
        name = f"{kind}:{effect_id}:{request.argument_digest()}"   # divergence shows up as RT0016

        if retry_class in (RetryClass.PURE, RetryClass.IDEMPOTENT, RetryClass.SIDE_EFFECTING_DEDUPLICATED):
            async def dispatch() -> EffectResult:
                return await self._clients.dispatch(effect_id, request)       # receiver dedupes on effect_id
            return await self._ctx.run_typed(name, dispatch, RunOptions(max_attempts=5))

        # Side-effecting, receiver cannot deduplicate: attempt marker, then at-most-once-or-flagged.
        marker = await self._ctx.run_typed(f"mark:{effect_id}", lambda: self._attempt_nonce)

        async def dispatch_once() -> EffectResult:
            if marker != self._attempt_nonce:        # an earlier attempt passed the marker and may have dispatched
                return EffectResult.outcome_unknown(effect_id)
            return await self._clients.dispatch(effect_id, request)
        return await self._ctx.run_typed(name, dispatch_once, RunOptions(max_attempts=1))

    async def signal(self, kind: str, *, timeout: timedelta | None = None) -> Signal:
        self._message_sequence += 1
        message = self._ctx.promise(f"message:{self._message_sequence}", type_hint=MessageEnvelope).value()
        if timeout is None:
            return (await message).to_signal()
        match await restate.select(message=message, timeout=self._ctx.sleep(timeout)):
            case ["message", envelope]:
                return envelope.to_signal()
            case ["timeout", _]:
                raise SignalTimeout(kind)

    async def gather(self, *awaitables: Awaitable[T]) -> list[T]:
        # Effects are journaled futures, so completion order is recorded; never use asyncio.gather.
        return await restate.gather(*awaitables)
```

**Effect identity (design).** `effect_id` is deterministic because the shim's ordinal counter advances in code order,
and replay re-runs that code. It is forwarded as `request_id` / `Idempotency-Key` / `_meta.idempotency_key`, exactly
as in [effects](../contracts/effects.md). Two properties of Restate make receiver deduplication **load-bearing**:
- **Crash between effect and journal.** An effect that ran but whose result was never stored runs again on replay
  (fact, §2.3).
- **Zombie attempts.** An aborted or fenced attempt's closure may still be running inside the old process when the
  new attempt dispatches the same `effect_id`. This is the "in progress → join" case.

The attempt marker gives us `tool.outcome_unknown` with the same semantics as today: flag, never silently retry.

**What changes from the current design.**
- The task host no longer needs its own replay driver: the Restate SDK replays the handler.
- History is rebuilt from effect results during replay.
- Pickled snapshots are needed only at segment boundaries.
- `run.now()` becomes the time recorded by `ctx.time()` or the latest input.
- Determinism is detected (`RT0016`) but not enforced. For in-house code we add our lint rules and the argument
  digest in the step name.

### 4.2 Untrusted task code (Q10 = external tenants)

**Fact.** Restate has no authorization between services. Any handler can call any service or object handler through
its context, and deployments are fully trusted
([service security](https://docs.restate.dev/services/security),
[server security](https://docs.restate.dev/server/security)). A tenant module running as a handler could therefore
message another tenant's `AgentIdentity`, or hand out its own journal.

**Design.**
- **The shim is the trust boundary.** The `Run` workflow becomes a trusted executor that drives a sandboxed task host
  (gVisor, no network, one local socket), like today's `HarnessHost`. The shim validates every effect: schemas, sizes,
  environment ownership, permitted imports and peers.
- **Replay is lockstep.** A new attempt starts the task code fresh in a sandbox and feeds each request its journaled
  result. No second log is needed.

```python
@run_workflow.main()
async def main(ctx: restate.WorkflowContext, request: RunSegmentRequest) -> RunSegmentResult:
    executor = RestateEffectExecutor(ctx, request)                  # the _perform logic of §4.1 plus validation
    async with sandbox_pool.lease(request.specification.task.code_reference, tenant=request.tenant) as host:
        session = await host.start(request.specification, snapshot=request.snapshot)   # local IPC, not journaled
        step: SandboxStep = await session.step(RunStarted())
        while not step.finished:
            validated = [executor.validate(effect) for effect in step.effect_requests]   # ownership, schema, peers
            completions = await restate.gather(*(executor.perform(effect) for effect in validated))
            step = await session.step(EffectCompletions(completions))
    return await executor.finish(step.outcome)
```

- **Pool shape.** Sandbox hosts are pooled per pod and per code reference; each hosts many runs. The extra hop is local
  IPC, well under a millisecond *(unverified)*. Tenants never see Restate endpoints, so the BSL grant's "abstraction
  layer" condition holds (§8).
- **In-house code (Q10 in-house).** Run task code in-process and use gVisor only for the whole pod. The shim still
  validates environment ownership, because task code must not be able to address other runs' environments.

### 4.3 Use case: synchronous reinforcement learning

**Mechanics (design).**
- The Rollout Controller admits rows under its buffer bound and calls the cell Control API. The Control API issues
  `POST /restate/send/Run/{run_id}.0/main` with the row as input. The workflow key makes the create idempotent.
- The run executes: model samples through the recorder in `ctx.run`, environment calls through envlets.
- It ends with a `ctx.run` posting `RunResult` (rewards bound to reply `effect_id`s, ending, observation digests) to
  the trajectory assembler's intake, deduplicated by `run_id`. The assembler joins it with the recorder sessions by
  `effect_id` (B13, B14).
- `journal_retention` stays short (1 h). The run record lives in our export, not in Restate.
- Between trainer steps nothing special happens: `publish` goes to the Policy Registry and never touches Restate.
- **Crash handling.** Retries are automatic. For RL we set `max_attempts` low and `on_max_attempts="pause"`; a small
  controller kills paused RL runs and the Rollout Controller resamples (the `run.crashed` path).

### 4.4 Use case: asynchronous reinforcement learning

- **In-flight weight updates are invisible to Restate.** Abort-and-resubmit happens inside the recorder, below the
  model-endpoint contract (ADR-0008). The `ctx.run` holding the sample just takes longer.
- **How long a turn may take.** A sample that outlives `inactivity_timeout` (2 min) is fine: the request stream
  closes, the SDK finishes the closure within `abort_timeout`, the result is stored, and the invocation suspends and
  resumes with one replay. That is inferred from the invoker code and must be confirmed by spike S-R4.
- **Duplicate samples.** If the attempt is aborted (> 15 min) or the pod dies, the retried `ctx.run` carries the same
  `effect_id`, and the recorder joins the in-progress sample or returns the cached one. Without that, retries would
  create orphan branches in the session tree.
- **High-rate short episodes: can journaling be relaxed?** Yes, in two ways (design, validate with S-R9).
  1. **Journal-light mode for `best_effort` runs.** Skip `ctx.run` for effects whose receivers deduplicate (recorder,
     envd, Environment Manager). Call them directly with the deterministic `effect_id`. On retry, Restate re-invokes
     the handler, the code re-issues the same effect IDs, and the receivers' caches act as the journal.
     - Per turn this costs no log appends; per run only start + result.
     - Recovery replays through network round trips instead of local journal reads.
     - It is only as durable as the receivers' 24 h caches, which suits `best_effort` exactly.
     - Non-deduplicating tools still take the marker path.
  2. **Bypass Restate** for trivially short, environment-free episodes (e.g. Wordle): the Rollout Controller calls a
     plain task-host service, and a crash means resample. That is the current `best_effort` semantics, and P10-style
     removability shows the substrate is optional there.

### 4.5 Use case: swarms (1,000 agents, fan-out / fan-in, spawn, messaging)

```python
class SwarmCoordinator(Task):
    async def respond(self, run: RunContext, reply: Message) -> Observation:
        plan = parse_plan(reply)
        children = [
            run.spawn(RunSpecification(task=worker_task(item), agent=self.worker_agent, binding=run.binding))
            for item in plan.items                                   # up to ~1,000
        ]
        results: list[ChildRunResult] = await run.gather(*(child.result() for child in children))
        return Observation(summarize(results))
```

**Mechanics on Restate.**
- **Spawn and fan-in.** `run.spawn` becomes
  `ctx.workflow_call(run_main, key=f"{child_run_id}.0", arg=RunSegmentRequest(...))`. Each call is one journal command,
  and the call is idempotent by key. `child.result()` awaits the call future. The parent suspends while it waits, and
  each child completion re-pushes and replays the parent. A 1,000-way gather can therefore replay the parent up to
  1,000 times unless completions batch *(unverified; the 1.0 "cooperative suspension" change targets exactly this
  fan-out thrash)*.
  - For very wide fans, children post results to a `SwarmBarrier` virtual object, and the parent awaits one promise
    resolved when the count is reached. That is one wake instead of N.
- **Messaging.** `run.send(run_id, message)` becomes `ctx.object_send(post, key=run_id, arg=message)` on
  `RunDirectory`. `RunDirectory` assigns the sequence number and calls the target's `deliver` (§4.1). Order per
  sender → target key is guaranteed by Restate's per-object ordering (fact, §2.1). Cross-partition delivery goes
  exactly once through the shuffler.
- **Broadcast.** Topic-style broadcast (1 → 1,000) goes through a `Topic` virtual object that stores messages;
  subscribers poll them at turn boundaries. Otherwise every broadcast is 1,000 invocations.
- **Placement.** A swarm and its environments stay in one cell (ADR-0001). Inside the cell, keys hash across
  partitions.
  - Do **not** give a swarm a flow-control scope: that pins all 1,000 agents to one partition.
  - Rough load for 1,000 agents at 1 turn/10 s with one message per turn: 100 messages/s ≈ 300 invocations/s ≈
    1–2k log records/s. That is trivial for a cell.
- **Shared environments.** Unchanged: attachment tokens held by the shim, ownership checked by the shim.
- **Deadlocks.** Request/response calls between exclusive object handlers can deadlock (fact,
  [service communication](https://docs.restate.dev/develop/python/service-communication)). Our objects only `send`
  to each other or call shared handlers.

### 4.6 Use case: durable agent identities

**Definition (design).** A *durable agent identity* is a long-lived, addressable agent entity. It persists across runs
and conversations, joins swarms and is trained, while consuming no compute when idle. Concretely:

| Aspect | Definition | On Restate |
|---|---|---|
| **Address** | `agent://{tenant}/{name}`, resolved to a home cell by the Global Router | `AgentIdentity` virtual object, key `{tenant}.{name}` in the home cell |
| **Profile** | deployment reference (name → task/agent `code_reference`s + default `RunBinding`) | state key `profile` (versioned record) |
| **Policy binding** | model channel(s) the identity samples from; trainable flag | state key `binding`; samples labeled `identity={tenant}.{name}` |
| **Memory** | namespaced durable K/V (`run.memory`) plus optional retrieval store | state keys `memory/{namespace}/{key}` (lazy state); large values as `BlobReference` |
| **Mailbox** | ordered inbound messages from users, connectors, other agents | the object's own per-key inbox (exclusive `receive` calls, ordered) plus a `pending` queue in state while a run is active |
| **Credentials / permissions** | credential **grants** (broker `secret_ref`s, never secrets), allowed imports, allowed peer identities, budgets | state key `grants`; enforced by the shim and egress proxy (P8) |
| **Owned resources** | environments (e.g. a `FULL_SNAPSHOT` workspace), snapshots, sessions | state key `resources`; the identity is the `OwnerReference` (P11) |
| **Lifecycle** | `created → idle ⇄ running → retired → deleted` | status in state; deletion = `clear_all` + destroy resources + delete tenant key |

**Runs versus identities.** A run is one episode (answer a message, do a task). It is a `Run` workflow started *by*
the identity, and it reports back with `on_run_finished`. The identity holds its lock only for milliseconds per
message, never for a run's duration. Idle identities are rows in RocksDB with no invocation, stream or process. So
300k mostly-idle identities cost storage only.

```python
agent_identity = restate.VirtualObject("AgentIdentity", enable_lazy_state=True)


@agent_identity.handler()
async def receive(ctx: restate.ObjectContext, envelope: MessageEnvelope) -> None:
    identity = await IdentityState.load(ctx)                      # versioned records; migrate on read
    identity.check_sender_permitted(envelope.sender)
    active: ActiveRun | None = identity.active_run
    if active is not None:
        ctx.object_send(post, key=active.run_id, arg=envelope)    # RunDirectory.post: the run sees it next turn
        return
    run_id = f"{identity.cell}-{ctx.uuid()}"
    specification = identity.profile.run_specification_for(envelope)   # RunSpecification with the identity's binding
    handle = ctx.workflow_send(
        run_main, key=f"{run_id}.0",
        arg=RunSegmentRequest.first(run_id, specification, identity=ctx.key(), trigger=envelope),
    )
    identity.active_run = ActiveRun(run_id=run_id, invocation_id=await handle.invocation_id())
    identity.save(ctx)


@agent_identity.handler()
async def on_run_finished(ctx: restate.ObjectContext, result: RunResult) -> None:
    identity = await IdentityState.load(ctx)
    identity.record_outcome(result)                               # summaries into memory, budget accounting
    identity.active_run = None
    next_envelope = identity.pop_pending()
    identity.save(ctx)
    if next_envelope is not None:
        ctx.object_send(receive, key=ctx.key(), arg=next_envelope)


@agent_identity.handler()
async def put_memory(ctx: restate.ObjectContext, write: MemoryWrite) -> None:
    ctx.set(f"memory/{write.namespace}/{write.key}", write.value)


@agent_identity.handler(kind="shared")
async def get_memory(ctx: restate.ObjectSharedContext, read: MemoryRead) -> MemoryValue | None:
    return await ctx.get(f"memory/{read.namespace}/{read.key}", type_hint=MemoryValue)
```

**Upgrades across deploys.** Each `receive` is a short invocation, so new messages always run the latest
`AgentIdentity` code. State records carry a schema version and are migrated on read. A run in flight finishes on its
pinned deployment; if it is long, it hands over at its next segment boundary. Nothing stays pinned for months.
This matters because server upgrades can fail invocations pinned to old protocol versions (§2.4).

**Run keys.** A `RunKey` virtual object (key e.g. `slack.T1.C2.1712345`) is a thin variant: `receive` either posts to
the live run or starts one. That is signal-with-start with no race, because the key's handlers are serial.

**Training.** Identities are trained like any harness: their runs use recorded channels, and samples carry the
identity label, so a trainer can filter or weight by identity. Weight updates reach idle identities at their next run
through the channel.

---

## 5. Scale and performance

**Evidence (facts)**:
- about 85–94k actions/s on a 3-node, 3-way replicated cluster with p99 ≤ 163 ms for 9-step workflows (§2.8);
- Replit peaks at 25k actions/s per cell across tens of cells.

There are no per-partition or Python-SDK numbers.

**Estimates at the design point (analysis).** The design point is 12+ cells of about 25k runs each, 30k turns/s
fleet-wide, and mostly idle runs.

| Quantity | Estimate | Basis |
|---|---|---|
| Log records per turn | ~4–6 | model `ctx.run` (command + completion) + ~1.5 environment/tool effects × 2; events exported at run end; the v7 lighter run-ack reduces wire traffic, not records |
| Records/s per cell | ~10–15k average, ~30–40k peak | 2.5k turns/s per cell × 4–6, 2–3× headroom |
| Records/s fleet | ~120–180k | vs the ~90k "events" of the Postgres design: Restate journals each effect as two records |
| Bytes journaled per cell | ~25–50 MB/s before replication | 10–20 KB per turn (a 500-token reply ≈ 2 KB, tool outputs ≤ 64 KB each; average assumed 4–8 KB) *(unverified mix)* |
| Nodes per cell | 3 minimum, **5–6 recommended** | 3× log replication + 2 processor replicas + RocksDB compaction on payloads 100–1000× the benchmark's; confirm with S-R1 |
| Partitions per cell | 64–128, fixed at provisioning | ~100–250 records/s per partition on average; hot keys, not the average, set the limit |
| Running invocations per cell | up to ~25k (RL: most runs are inside a model `ctx.run`) | 1.7: 1,000 per partition processor → 64+ partitions is ample; 1.8: 24k per node |
| Invoker memory | raise from 1.5 GiB to 8–16 GiB per node | 25k × 32 KiB initial ≈ 0.8 GiB, plus replay bursts after pod loss |
| HTTP/2 streams to deployments | ~25k per cell → ~200 connections at 128 streams each | default pool settings |
| Deployment pods | ~15–30 per cell *(unverified)* | assumes 1–2k live invocations per Python process; spike S-R2 |
| Idle identities / suspended runs | storage only; 300k × ~1 MB journal+state ≈ 300 GB fleet, ~25 GB per cell before replication | snapshots to object storage bound recovery |

**Where it breaks (analysis).**
1. **Replay cost on long runs.** Every resume re-sends and re-runs the whole journal. A 2,000-turn run at 20 KB per
   turn has a 40 MB journal. That exceeds the default 32 MiB per-invocation budget, and each idle wake costs O(n), so
   the run costs O(n²) overall. **Segments are mandatory** (e.g. ≤ 200 turns or ≤ 4 MB per segment).
2. **Scoped hot spots.** A scope maps to one partition, so a big tenant or swarm with a scope is capped by one
   partition's throughput. That throughput is unknown; plausibly a few thousand records/s *(unverified)*.
3. **Wide fan-in** without a barrier object replays the parent once per child completion.
4. **Python SDK density** is unmeasured. A single asyncio loop per process, plus a Rust VM per invocation, sets the
   ceiling on concurrent turns per core.
5. **Fixed partition count.** Resharding means a new cell and draining the old one.
6. **Pod loss burst.** Every invocation on a lost pod replays at once on the survivors. The memory pool bounds server
   memory but not SDK CPU.
7. **N5 (≤ 50 ms p99 per step)** is met by the vendor's numbers for small payloads (p99 40 ms for one step under
   load). It must be re-measured with 10–60 KB entries.

---

## 6. Failure semantics

| Failure | Behavior on Restate | Our guarantee afterwards |
|---|---|---|
| Deployment pod crash | Stream breaks; Restate retries with backoff on any pod of the pinned deployment and replays the journal | Journaled effects not repeated; the effect in flight is re-dispatched with the same `effect_id` → receiver dedupe / join; non-dedupe tools → `outcome_unknown` via marker |
| Partition leader (node) loss | Follower promoted with a new epoch; log segment sealed and reconfigured; running attempts re-dispatched; stale-epoch events ignored ([architecture](https://docs.restate.dev/references/architecture)) | Same as above; unavailability window per partition to be measured (S-R5) |
| Loss of log quorum for a nodeset | Appends for affected partitions stall | The cell stalls, the others don't (N4) |
| Metadata quorum loss | Reconfiguration blocked; data path continues until a reconfiguration is needed | Operational alert; not a data-loss event |
| Snapshot store unavailable | Trimming stops, log grows; new replicas cannot bootstrap | Alert on log size |
| Non-determinism | `RT0016` → retries → pause (v7 lets the SDK mark it `FAIL` / `PAUSE` directly) | Map to `NON_DETERMINISM`, quarantine = paused invocation |
| `ctx.run` longer than inactivity + abort timeout | Attempt aborted and retried; closure may still be running (zombie) | Receiver joins the in-progress execution |
| Cancellation | `TerminalError` at next await; teardown runs as compensation; running `ctx.run` finishes first; detached sends not cancelled | Matches `run.cancel_requested` → wind-down → `run.cancelled`; shim backstop destroys owned environments |
| Max attempts exhausted | Invocation **paused** (default) or killed | Controller policy: resample (RL) or alert (durable runs) |
| Server upgrade drops an old protocol version | Invocations pinned to it fail | Avoided by segments and short identity handlers |

**Delivery semantics compared with [delivery-semantics](../architecture/delivery-semantics.md) (analysis).**
- *Fenced append* and *terminal-once* are provided by Restate's epochs.
- *Persist-then-dispatch* is **not** provided for `ctx.run`: a closure runs before its command is durable. It still
  holds for Restate-native effects (calls, sends, timers).
- We keep "effectively-once for dedupe-capable receivers, at-most-once-or-flagged for the rest". The only change is
  that deduplication and the marker are now *required*, where before they covered a rare takeover race.

---

## 7. Security and multi-tenancy

- **No built-in authentication or authorization on the server** (fact,
  [server security](https://docs.restate.dev/server/security)). Ingress (8080) must sit behind our Control API. Admin
  (9070) and fabric (5122) must be network-isolated. Headers that reach the ingress are **persisted in the journal**,
  so the Control API strips credentials first.
- **Deployments are fully trusted** (fact, §4.2). Request-identity signing lets deployments verify that calls come from
  our cluster, and `ingress_private` hides internal services such as `RunDirectory`. That protects deployments from
  outsiders, not tenants from each other.
- **Q10 = in-house.** One trust zone. Task code in deployment pods has network egress to platform services and holds
  no guest credentials (the broker still injects at egress). A buggy task can address other objects, which is
  acceptable in-house and caught by shim validation where it matters (environment ownership).
- **Q10 = tenants.** The trusted shim plus gVisor task hosts from §4.2. Tenant code has no network and no Restate
  access, and acts only through validated effects. That is identical to trust zone Z1t today. Optionally, cells or
  deployment pools per tenant tier add blast-radius isolation.
- **Data at rest.** Journals and state hold model messages and tool outputs. A per-tenant `JournalValueCodec` (Python,
  preview) can encrypt values with tenant keys; the envelope carries a key ID and decode resolves the tenant key.
  Deleting a tenant key then crypto-shreds journals, state and snapshots in object storage. State keys, handler names
  and failure messages stay plaintext (fact), so no personal data goes into keys or error strings.
- **Admin surface.** The SQL introspection and `/query` endpoint expose all state. Operator-only, rate-limited (v1.7.1
  option).

---

## 8. Operations (including license)

- **Deployment per cell.**
  - One `RestateCluster` (operator v3, MIT) on the cell's Kubernetes cluster: StatefulSet on local NVMe, replication
    `{zone: 2, node: 3}`, snapshots to the regional bucket (S3 / GCS / Azure), metadata in the built-in Raft store.
  - No Postgres is needed for the run store. The Environment Manager's small schema still needs one.
  - Service deployments are `RestateDeployment`s, one per **shim** version. Old ReplicaSets live until drained; with
    segments, the drain takes minutes to hours.
- **Multi-cloud.** Works on EKS, GKE and AKS with S3, GCS and Azure Blob. Nothing depends on a single cloud (N2).
  Object-store metadata is "only tested and supported on Amazon S3", so we use the Raft store everywhere.
- **Upgrades.**
  - Frequent patch releases. Some migrations are one-way: vqueues in 1.8, protocol v7 once enabled.
  - Rollback is supported only within documented windows (e.g. 1.8 → ≥ 1.7.3).
  - We need a canary cell and an automated upgrade pipeline. Features we would use are still experimental in 1.7:
    signals, scopes and v7 acks. We should target 1.8+ and avoid features flagged experimental for the critical path.
- **Retention.** Short `journal_retention` for RL; workflow retention sized for result retrieval; purge APIs. Our
  exported run records are the long-term archive.
- **Monitoring.**
  - Prometheus metrics with vendor Grafana dashboards; OTel traces per invocation.
  - SQL for incident debugging (stuck objects: `sys_locks`; queues: `sys_vqueues`).
  - Alert on paused invocations, log growth (trimming stalled), invoker memory exhaustion and leadership churn.
- **License implications (analysis; not legal advice).**
  - Our platform is (a) internal, or (b) a public product whose users submit runs, tasks and agents through **our**
    Control API and SDK. Both are explicitly permitted by the Additional Use Grant.
  - What is *not* permitted: letting third parties register their own Restate deployments or call Restate ingress /
    admin directly. The tenant-code design in §4.2 already forbids that for security reasons.
  - Risks:
    - the licensor can change terms for *future* versions (each version's grant and its 4-year Apache conversion are
      fixed);
    - forking the server is possible but constrained by the same grant.
  - Commercial options: Restate Cloud, BYOC (AWS / GCP) and Enterprise support. Terms and pricing were not researched.

---

## 9. What we still build

| Component | Size (rough) | Notes |
|---|---|---|
| `RestateRunContext` + `Program` / `AgentProgram` host: effect dispatch, ordinals, attempt markers, `gather`, signals, segments, snapshot hand-over | 3–4k LOC Python | replaces the replay driver and `HarnessHost` for in-house code |
| Trusted shim ↔ sandboxed task host protocol, sandbox pool (Q10 tenants) | 3–5k LOC + gVisor operations | reuses the task host driver from ADR-0013 in "live only" mode |
| `RunDirectory`, `RunKey`, `AgentIdentity`, `SwarmBarrier`, `Topic`, `Cron` objects | 2–3k LOC Python | |
| Control API adapter (create, signal, cancel, get, watch) over Restate ingress / admin; cross-cell relay | 2–3k LOC Go | idempotency keys end to end |
| Run event export (`run.emit`, rewards, endings, observations) to our event store; connectors read it | 1.5–2.5k LOC + a store (e.g. object storage + a queue) | the Restate journal is not our schema |
| Token streaming side channel in the recorder (pub/sub) | ~1k LOC | |
| Admission, per-tenant quotas and budgets | 1–2k LOC | flow-control scopes cannot express per-tenant caps without hot spots |
| Journal codec with per-tenant keys | ~0.5k LOC | Python codec is preview |
| Cell templates (RestateCluster, deployments, alerts, runbooks, upgrade canary) | infrastructure-as-code | |

**Removed from our build list**: the Run Store schema (events, leases, inbox, timers), partition leasing, fencing,
takeover scans, timer relay, intra-cell `send` / `spawn` transactions, and most of the Go runtime. Receivers' idempotency
(envd, recorder, Environment Manager) stays and becomes more important. **Net estimate**: about 15–25k LOC of glue
instead of about 30–50k for a custom runtime plus run store *(unverified estimate)*, in exchange for a BSL dependency
and less control over the log schema.

---

## 10. Risks, open questions and spike tests

**Risks.**
1. The Python SDK 1.x is young (1.0 in June 2026). Density and performance are unknown.
2. `ctx.run` is at-least-once. Any receiver without deduplication is a correctness hole unless it goes through the
   marker path.
3. Long runs rely on our segmentation, since there is no patching and journal replay is O(n).
4. Critical features (signals, scopes, v7) are experimental in 1.7 and default in the unreleased 1.8.
5. The partition count is fixed forever per cell.
6. Vendor and license concentration: BSL, a single company.
7. Push-only execution: no sticky workers yet (roadmap), so no per-run caches survive suspension.

**Open questions.**
- Are state writes made by a running exclusive invocation visible to concurrent shared handlers before the invocation
  ends? This affects `RunDirectory.status`.
- Does a 1,000-way `gather` batch its completions into fewer re-invocations under protocol v7?
- Does the 1.8 timeline land before our build?
- What is the single-partition throughput with realistic payloads?
- Is there vendor support for self-hosted clusters at 12+ cells?

**Spike tests (each with a pass criterion).**

| # | Spike | Pass criterion |
|---|---|---|
| S-R1 | Cell throughput: 5-node cluster, 64 partitions, `{zone: 2, node: 3}`; 25k `Run` workflows, 1 turn / 10 s, per turn one 2 KB and two 10 KB `ctx.run`s; 24 h | ≥ 40k records/s sustained; `ctx.run` persist p99 ≤ 50 ms (N5); log trimming keeps up; no leadership churn |
| S-R2 | Python SDK density: live invocations per process, each awaiting a 10 s `ctx.run` | ≥ 2k live invocations per process, ≥ 500 steps/s per core, ≤ 100 KB overhead per invocation |
| S-R3 | Replay cost: resume invocations with 200 / 500 / 2,000 turns × 20 KB | Resume of a 200-turn segment ≤ 200 ms p99; confirm behavior above the 32 MiB budget |
| S-R4 | Suspension: 25k workflows per cell awaiting promises; model `ctx.run` longer than `inactivity_timeout` | Wake p99 ≤ 100 ms after resolve; storage per suspended run measured; **zero** re-executions when a `ctx.run` crosses the inactivity timeout but not the abort timeout |
| S-R5 | Failover: kill a leader node and a deployment pod under S-R1 load | Per-partition unavailability ≤ 10 s; duplicate executions observed only at dedupe receivers, all absorbed |
| S-R6 | Swarm: 1,000 agents with mailboxes (unscoped vs scoped); spawn 1,000 children + gather vs barrier | ≥ 1k messages/s per swarm, p99 delivery ≤ 200 ms; gather re-invocations counted; scoped variant shows the single-partition ceiling |
| S-R7 | Deploy: roll shim versions with 10k in-flight runs using segments | All new segments on the new version within one segment length; old ReplicaSets drained; no `RT0016` |
| S-R8 | Duplicate-effect test (as in docs-restate#410): SIGKILL between effect and journal, for a dedupe receiver and for a marker-path tool | Dedupe: exactly one execution in 100/100 trials; marker path: 0 double applications, `outcome_unknown` reported |
| S-R9 | Journal-light mode for `best_effort` RL | ≥ 3× lower log records per turn; crash recovery reproduces identical samples from receiver caches |
| S-R10 | Upgrade path: 1.7.x → 1.8 with vqueues migration on a loaded cell | Migration time and rollback verified within the documented window |

---

## 11. Sources

**Primary: code and releases (checked 2026-09-27)**
- [restatedev/restate releases](https://github.com/restatedev/restate/releases) — v1.7.12, 2026-09-22
- [Restate LICENSE (BSL 1.1 with Additional Use Grant)](https://github.com/restatedev/restate/blob/main/LICENSE)
- [Release notes v1.6.0](https://github.com/restatedev/restate/blob/main/release-notes/v1.6.0.md) — journal entry limits, pause/resume, deployments
- [Release notes v1.7.0](https://github.com/restatedev/restate/blob/main/release-notes/v1.7.0.md) — flow control, protocol v7, invoker memory, HTTP/2 pool, retention
- [Release notes v1.7.1–v1.7.9](https://github.com/restatedev/restate/tree/main/release-notes)
- [Unreleased: vqueues enabled by default](https://github.com/restatedev/restate/blob/main/release-notes/unreleased/vqueues-enabled-by-default.md)
- [Unreleased: invoker concurrency 24,000 per node](https://github.com/restatedev/restate/blob/main/release-notes/unreleased/increase-invoker-concurrency-limit-default.md)
- [Unreleased: drop service protocol ≤ v3](https://github.com/restatedev/restate/blob/main/release-notes/unreleased/drop-service-protocol-v3.md)
- [Unreleased: `sys_keyed_service_status` → `sys_locks`](https://github.com/restatedev/restate/blob/main/release-notes/unreleased/remove-sys-keyed-service-status-table.md)
- [Service protocol definition](https://github.com/restatedev/restate/blob/main/service-protocol/dev/restate/service/protocol.proto)
- [Invoker: inactivity / abort handling](https://github.com/restatedev/restate/blob/main/crates/invoker-impl/src/invocation_task/service_protocol_runner_v4.rs)
- [Partition keys and scopes](https://github.com/restatedev/restate/blob/main/crates/types/src/identifiers.rs)
- [Issue #4344: stream state entries](https://github.com/restatedev/restate/issues/4344)
- [Issue #4801: default invoker memory limits](https://github.com/restatedev/restate/issues/4801)
- [restatedev/sdk-python](https://github.com/restatedev/sdk-python) — v1.0.5, MIT; [context.py](https://github.com/restatedev/sdk-python/blob/main/python/restate/context.py), [server_context.py](https://github.com/restatedev/sdk-python/blob/main/python/restate/server_context.py), [object.py](https://github.com/restatedev/sdk-python/blob/main/python/restate/object.py), [entry_codec.py](https://github.com/restatedev/sdk-python/blob/main/python/restate/entry_codec.py)
- [Python SDK changelog](https://docs.restate.dev/changelog/python-sdk)
- [restatedev/restate-operator](https://github.com/restatedev/restate-operator) — v3.1.0, MIT
- [docs-restate#410: `ctx.run` re-executes after a crash](https://github.com/restatedev/docs-restate/issues/410)
- [restatedev/ai-examples](https://github.com/restatedev/ai-examples) — agent loop, interrupt/regenerate, sessions

**Documentation (docs.restate.dev)**
- [Architecture](https://docs.restate.dev/references/architecture)
- [Key concepts](https://docs.restate.dev/foundations/key-concepts)
- [Services](https://docs.restate.dev/foundations/services)
- [Invocations](https://docs.restate.dev/foundations/invocations)
- [Request lifecycle](https://docs.restate.dev/guides/request-lifecycle)
- [Error handling / timeouts](https://docs.restate.dev/guides/error-handling)
- [Managing invocations](https://docs.restate.dev/services/invocation/managing-invocations)
- [Versioning](https://docs.restate.dev/services/versioning)
- [Flow control](https://docs.restate.dev/services/flow-control)
- [Service configuration](https://docs.restate.dev/services/configuration)
- [Service communication (Python)](https://docs.restate.dev/develop/python/service-communication)
- [External events (Python)](https://docs.restate.dev/develop/python/external-events)
- [Clusters](https://docs.restate.dev/server/clusters)
- [Snapshots](https://docs.restate.dev/server/snapshots)
- [Metadata](https://docs.restate.dev/server/metadata)
- [Server configuration reference](https://docs.restate.dev/references/server-config)
- [Server security](https://docs.restate.dev/server/security)
- [Service security](https://docs.restate.dev/services/security)
- [Introspection](https://docs.restate.dev/services/introspection)
- [SQL reference](https://docs.restate.dev/references/sql-introspection)
- [Kubernetes (server)](https://docs.restate.dev/server/deploy/kubernetes)
- [Kubernetes (services)](https://docs.restate.dev/services/deploy/kubernetes)
- [AI integration guide](https://docs.restate.dev/ai/sdk-integrations/integration-guide)
- [Durable sessions](https://docs.restate.dev/ai/patterns/sessions)
- [Streaming responses](https://docs.restate.dev/ai/patterns/streaming-responses)
- [OSS roadmap](https://docs.restate.dev/roadmap/oss)

**Vendor and third-party**
- [Building a modern durable execution engine from first principles](https://restate.dev/blog/building-a-modern-durable-execution-engine-from-first-principles/) (Feb 2025; benchmarks)
- [Restate 1.2 announcement](https://restate.dev/blog/announcing-restate-1.2) (benchmarks, 3-node cluster)
- [Distributed Restate — a first look](https://restate.dev/blog/distributed-restate-a-first-look/)
- [Restate vs Temporal](https://restate.dev/vs/temporal) (Replit: tens of cells, 25k actions/s per cell)
- [restate.dev](https://restate.dev/) (deployment forms, customers)
- [Restate raises $7M](https://startupnews.fyi/2024/06/12/restate-raises-7m-for-its-lightweight-workflows-as-code-platform/)
