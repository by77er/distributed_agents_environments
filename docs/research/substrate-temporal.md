# Temporal as the durable execution substrate

Status: **Draft** · 2026-09-27 · Question: how well does Temporal (open-source server and Temporal Cloud, Python SDK)
serve as the substrate for runs, effects, swarms and durable agent identities, and what does the concrete design look
like? Re-evaluates the rejection in [ADR-0004](../decisions/0004-durable-actor-runtime.md).

Conventions: **[fact]** statements are cited; **[analysis]** statements are our reasoning; **[unverified]** marks
numbers or behavior not confirmed by a primary source or by a test of our own.

## 1 Verdict

Temporal is the most mature and complete fit for our *semantics*. It covers our run log, leases and fencing, timers,
signals, child runs, cancellation, versioning with pinning and patching, and idle runs that are not resident in memory.
It does so with better recovery and upgrade stories than DBOS, and without a virtual-actor layer: entity workflows
cover durable identities directly. It is a poor fit for our *cost and rate profile*. Each model turn costs about
**12 history events, 2–3 billable actions and about 8–11 persistence transactions** [analysis]. That is roughly 4–5× the write
load of our own run-store design. At the design point it comes to about 66k–96k actions/s, which is **$4–6M/month
at the lowest published Cloud tier** [analysis on cited prices]. Self-hosted, it needs Cassandra-class persistence
per cell rather than one Postgres. Untrusted tenant code needs our own out-of-process workflow runner, because the
Python sandbox is explicitly not a security boundary. **Recommendation:** Temporal is a credible substrate
only if spikes T1, T2 and T8 (section 10) pass. Its strongest role is durable identities, swarms and long
interactive runs. High-rate RL episodes need the "compact" mode described in section 4.2.

## 2 What it is

### 2.1 Model

A **workflow** is deterministic code whose every interaction with the world is a command recorded in an append-only
**event history**. Commands include scheduling an activity, starting a timer, starting a child workflow and signalling
another workflow. On recovery, a worker replays the code against the history. **Activities** are ordinary
non-deterministic functions executed by activity workers with timeouts, retries and heartbeats.
Workflows receive **signals** (asynchronous, durable), **updates** (synchronous, validated, durable when accepted) and
**queries** (read-only) ([sending messages](https://docs.temporal.io/sending-messages)). This is the same model as
our ADR-0013: replay of `async` Python against a log. ADR-0013's own alternatives section says so.

### 2.2 Architecture [fact]

Four services ([server architecture](https://docs.temporal.io/temporal-service/temporal-server)):

- **Frontend**: stateless gRPC gateway (rate limiting, authorization, routing).
- **History**: owns workflow state. Executions are hashed by namespace and workflow ID onto a fixed number of
  **history shards**. Each shard is owned by one history process, coordinated by Ringpop membership. The shard's
  `RangeID` is "a monotonically increasing generation number used for fencing"
  ([history-service design](https://github.com/temporalio/temporal/blob/main/docs/architecture/history-service.md)).
  The shard count "is fixed at cluster creation and cannot be changed later". Deployments have run "anywhere from 1 to
  128K History Shards", with a starting point of one history process per 500 shards.
- **Matching**: task queues (partitioned) that match workflow and activity tasks to polling workers.
- **Worker**: internal system workflows (replication, archival, batch operations).

### 2.3 Storage [fact]

Persistence: Cassandra 3.11/4.0/5.0.4+, PostgreSQL 13–16, MySQL 5.7/8.0, and SQLite for development only. Visibility
(search): Elasticsearch/OpenSearch or SQL ([persistence](https://docs.temporal.io/temporal-service/persistence)).
Mutable state is persisted "in a single row" per execution, because Cassandra is the most important backend
([history-service design](https://github.com/temporalio/temporal/blob/main/docs/architecture/history-service.md)).
Temporal staff: "We recommend Cassandra for huge loads", and "Many users run with PostgreSQL in production"
([forum](https://community.temporal.io/t/postgresql-good-option-for-persistence-in-production/6153)). In July 2026
Quo (formerly OpenPhone) described outgrowing an Aurora Postgres writer and moving to Cassandra
([Quo](https://www.quo.com/blog/postgres-to-cassandra/)). Temporal Cloud uses a proprietary persistence layer with
database sharding, a write-ahead log that aggregates updates, and tiering of closed histories to object storage
([Temporal blog, 2024-03](https://temporal.io/blog/higher-throughput-and-lower-latency-temporal-clouds-custom-persistence-layer)).
That layer is **not** available to self-hosters.

### 2.4 License, versions, maturity [fact]

- Server and Python SDK are **MIT** licensed (GitHub license metadata for `temporalio/temporal` and
  `temporalio/sdk-python`). Temporal "originated as a fork of Uber's Cadence" (repository README).
- Server **v1.32.0** (2026-09-11); patch lines v1.31.3 and v1.30.7 (2026-09-18). Python SDK **1.33.0** (2026-09-15)
  ([server releases](https://github.com/temporalio/temporal/releases),
  [Python SDK releases](https://github.com/temporalio/sdk-python/releases)).
- Recent changes that matter to us:
  - Worker Deployment (versioning) APIs are GA in v1.31, and the legacy versioning APIs are removed in v1.33
    ([v1.31.0](https://github.com/temporalio/temporal/releases/tag/v1.31.0),
    [v1.32.0](https://github.com/temporalio/temporal/releases/tag/v1.32.0)).
  - Task-queue priority and fairness are GA in v1.31.
  - Eager workflow start is on by default since v1.29.
  - Eager activity execution is on by default and Standalone Activities are GA in v1.32.
  - Serverless workers are pre-release in v1.31.
  - Workflow Streams (a durable publish/subscribe channel hosted by a workflow) are in public preview.
- Adoption: in February 2026 Temporal reported "9.1 trillion lifetime action executions on Temporal Cloud", and
  said the platform "routinely handles spikes of 150,000+ actions per second". It names OpenAI among production
  users ([Series D post](https://temporal.io/blog/temporal-raises-usd300m-series-d-at-a-usd5b-valuation)).
  Temporal Cloud markets "automatic scaling to 300k Actions/second or more" ([temporal.io/cloud](https://temporal.io/cloud)).

### 2.5 Deployment [fact]

- **Self-hosted**: Helm charts, and a Kubernetes **Temporal Worker Controller** (MIT, v1.11.0 on 2026-09-15) that
  manages worker deployment versions ([worker controller](https://github.com/temporalio/temporal-worker-controller)).
  Multi-cluster replication is **experimental**. Failover is asynchronous and can roll back progress or re-run
  activities ([multi-cluster replication](https://docs.temporal.io/self-hosted-guide/multi-cluster-replication)).
- **Temporal Cloud**: AWS and GCP regions only, **no Azure**
  ([regions](https://docs.temporal.io/cloud/regions)). The SLA is 99.9%, or 99.99% for replicated
  "High Availability" namespaces (RPO under 1 minute, RTO 20 minutes)
  ([high availability](https://docs.temporal.io/cloud/high-availability)). Replicated namespaces bill actions and
  storage at 2× ([pricing](https://temporal.io/pricing.md)). The latency SLO is p99 200 ms per region. Measured p99
  in August 2026: start workflow 78 ms, signal 91 ms, signal-with-start 109 ms
  ([operating envelope](https://docs.temporal.io/cloud/operating-envelope)).

### 2.6 Python SDK [fact]

- **Sandbox.** Each workflow run re-imports its workflow file in a fresh sandbox, and proxies block known
  non-deterministic calls. The README says: "The sandbox is built to catch many non-deterministic and state sharing
  issues, but it is not secure … a simple call like `setattr(temporalio.common, "__my_key", "my value")` will leak
  across sandbox runs" ([sdk-python README](https://github.com/temporalio/sdk-python)). Non-passthrough imports cost
  CPU and memory on every run ([sandbox docs](https://docs.temporal.io/develop/python/python-sdk-sandbox)).
- **Worker cache.** Defaults: `max_cached_workflows=1000`, a 500-thread workflow task executor, and a 10 s sticky
  queue schedule-to-start timeout. Heartbeats are throttled to 0.8 × the heartbeat timeout, with a 30 s default and
  a 60 s maximum ([worker source](https://github.com/temporalio/sdk-python/blob/main/temporalio/worker/_worker.py)).
  A workflow evicted from the cache costs nothing on the worker. Its next task replays the history on any worker.
- **Custom runners.** `WorkflowRunner` / `WorkflowInstance.activate(activation) -> completion` is a pluggable
  interface. The sandbox itself is one implementation
  ([source](https://github.com/temporalio/sdk-python/blob/main/temporalio/worker/_workflow_instance.py)).
- **Versioning.** Worker Deployments with `PINNED` or `AUTO_UPGRADE` behavior, `workflow.patched()`, and
  experimental **upgrade-on-continue-as-new**: `workflow.info().is_target_worker_deployment_version_changed()` plus
  `continue_as_new(initial_versioning_behavior=ContinueAsNewVersioningBehavior.AUTO_UPGRADE)`. For multi-week
  workflows the docs recommend "PINNED + upgrade on Continue-as-New"
  ([worker versioning](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning)).
- **Cancellation.** A cancel request is raised as `asyncio.CancelledError` in the main workflow task, so `finally`
  blocks can run cleanup activities. It "is a state not an event … only the first" request is delivered. SDK 1.31
  added a fix, still disabled by default, for cancellation "being lost when another event becomes ready in the same
  workflow task" ([1.31.0](https://github.com/temporalio/sdk-python/releases/tag/1.31.0)). Termination and execution
  timeouts do not run workflow code at all.
- **Local activities** run in the workflow worker process. Their results are recorded as markers only when the
  enclosing workflow task completes. They are at-least-once and never delay persistence of other events, but signals
  are not processed until they finish ([local activities](https://docs.temporal.io/local-activity)). All local
  activities within one workflow task count as one action ([actions](https://docs.temporal.io/cloud/actions)).
- **Large payloads.** Built-in external storage (claim check to S3, public preview) offloads payloads above a
  threshold, 256 KiB by default ([external storage](https://docs.temporal.io/develop/python/data-handling/external-storage)).
- **Agent integrations.**
  - OpenAI Agents SDK (`temporalio.contrib.openai_agents`, reported GA in March 2026): model calls, tools, MCP and
    sandbox operations run as activities, and streaming goes through Workflow Streams
    ([README](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/openai_agents)).
  - Pydantic AI `TemporalDurability`: model requests, tool calls and MCP run as activities
    ([Pydantic AI](https://pydantic.dev/docs/ai/integrations/durable_execution/temporal/)).
  - Also LangGraph, Google ADK, Strands and Deep Agents plugins ([docs.temporal.io/ai](https://docs.temporal.io/ai)).

### 2.7 Limits that shape the design [fact]

| Limit | Value | Source |
|---|---|---|
| History per execution | warn 10,240 events or 10 MB; error 51,200 events or 50 MB | [defaults](https://docs.temporal.io/self-hosted-guide/defaults), [Cloud limits](https://docs.temporal.io/cloud/limits) |
| Continue-as-new suggested (self-hosted defaults) | 4,096 events or 4 MiB | [dynamic config](https://github.com/temporalio/temporal/blob/main/common/dynamicconfig/constants.go) |
| Payload (blob) | warn 256 KB, error 2 MB; gRPC message and history transaction 4 MB | [defaults](https://docs.temporal.io/self-hosted-guide/defaults) |
| Pending activities / child workflows / signals / cancels per execution | 2,000 each (≤ 500 recommended) | same |
| Signals received per execution | 10,000 (`history.maximumSignalsPerExecution`) | same, [Cloud limits](https://docs.temporal.io/cloud/limits) |
| Updates per execution | 10 in flight, 2,000 total | same |
| Sustained signals into one workflow | guidance ≤ 5/s; theoretical ≈ 1 / database latency (≈ 20/s) | [docs issue #3665](https://github.com/temporalio/documentation/issues/3665) |
| Cloud actions/s per namespace | 500 default; on-demand grows with 7-day usage; provisioned 2–12 TRUs × 500 = up to **6,000 APS** self-service, more by support ticket | [capacity modes](https://docs.temporal.io/cloud/capacity-modes) |
| Cloud visibility API | 30 calls/s per namespace | [Cloud limits](https://docs.temporal.io/cloud/limits) |
| Cloud reuse of one workflow ID | 1 new execution per second (burst allowed) | same |
| Identifier length | 1,000 bytes | same |

## 3 Concept mapping

| Our concept | Temporal mechanism | Fit and caveats |
|---|---|---|
| **Run** (`run_id`, log, code pin) | Workflow execution chain: workflow ID = `run_id`; each continue-as-new segment has its own Temporal run ID | Good. The history is the log; it holds effect requests and completions but not our domain events (observations, rewards). Those go into activity inputs and a final outcome activity (4.1). |
| **Effect** (`model.request`, `environment.*`, `tool.request`) | Activity on a per-kind task queue served by trusted effect workers; local activity only in compact mode | Good. `ActivityTaskScheduled` is committed before dispatch, which gives P5 natively. Local activities break P5 (they run before the marker is committed). |
| **Effect identity** | `{workflow_id}:{run_id}:{activity_id}` from `activity.info()`; identical across retries; server-authoritative | Good, and better than ours for multi-tenancy: task code cannot forge it. A reset gets a new run ID and therefore fresh effects, which is correct for "redo". |
| **Retry classes** | `RetryPolicy` per activity; `side_effecting` without dedupe → `maximum_attempts=1`, and a timeout maps to `tool.outcome_unknown` | Good. |
| **Signals / inbox** | Signals (fire-and-forget), updates (approvals, with validators), queries (read) | Good, within ≤ 5/s sustained per workflow and 10,000 per execution. |
| **Run keys + signal-with-start** | Workflow ID = run key; `client.start_workflow(..., start_signal=...)`; update-with-start (not atomic); `WorkflowIDConflictPolicy.USE_EXISTING` | Native. The same ID can start at most one new execution per second in Cloud. |
| **Timers** | `workflow.sleep`, `workflow.wait_condition(timeout=...)` | Native; one action per timer. |
| **Child runs / spawn** | Child workflows with `ParentClosePolicy`; `start_child_workflow` | Native. Two actions per child start in Cloud. Children do not survive a parent's continue-as-new unless `ABANDON`. |
| **Cancellation + teardown** | Workflow cancel → `CancelledError` → `finally: teardown` activities | Good. Termination and run timeouts skip cleanup, so we keep an ownership reaper (P11). SDK cancellation-loss fix is still behind a flag. |
| **Versioning** | Worker Deployments (`PINNED`), `workflow.patched()` = `run.patched()`, upgrade-on-continue-as-new | Better than ours: routing, draining and ramping are built in. |
| **Snapshots / continuation** | Continue-as-new at a turn boundary with pickled task and agent state (blob reference) as input | Our snapshot design becomes *mandatory* (not optional) for runs past about 300 turns. |
| **Outputs** (`run.emit`) | Activity to a connector, keyed by effect identity | Fine. |
| **Token streaming** | Not Temporal. Recorder or model activity → ephemeral pub/sub keyed by effect identity | Workflow Streams exist (preview), but every flush is a signal, which is too costly per token. |
| **Queues / admission** | Worker slot limits, task-queue rate limits, priority (1–5) and **fairness keys** per tenant or job ([fairness](https://docs.temporal.io/develop/task-queue-priority-fairness)) | Partial. There is no bounded queue of *workflow starts*, so the rollout controller keeps admission. Fairness is a paid Cloud feature. |
| **Recovery / fencing** | Shard `RangeID` fencing; workflow-task and activity task tokens; workflow-task timeout (10 s) and sticky timeout reschedule on death | Native. There is nothing to build (no leases, no partition ownership). |
| **Cells** | One Temporal cluster per cell (self-hosted), or one or more namespaces per cell (Cloud) | Good with self-hosting. In Cloud, the blast radius is Temporal's cells, not ours. |
| **Observability** | Web UI, history export, SDK and server metrics, OpenTelemetry interceptors, search attributes | Strong for operators. The Cloud visibility API (30/s) is not a product query path. |

**ADR-0004 re-evaluated [analysis].** The ADR rejected Temporal for three reasons. We re-check each:

1. **"Our log schema is our product, read by RL tooling."** This no longer holds. Since ADR-0007 and ADR-0015,
   training data comes from recorder session trees joined with rewards and endings. The assembler needs only a
   small outcome record per run, which one activity can publish.
2. **"History limits conflict with long agent runs."** This is solved with continue-as-new at turn boundaries.
   ADR-0013 already designed snapshot and restore; Temporal now suggests the right moment and upgrades code across it.
3. **"Per-step overhead conflicts with short RL episodes."** This **still holds**, quantified in section 5, and is the
   main reason for caution.

The authoring model is now the same as ours (ADR-0013). Temporal therefore removes most of the runtime and run store
we planned to build (ADR-0002, 0003 and 0004), in exchange for per-step cost and a heavier persistence tier.

## 4 Use cases

### 4.0 Core design on Temporal

**The loop as a workflow.** One workflow type, `AgentRun`, hosts the normative loop. The `Program` layer (charter
item 1) is the same shape with `Program.main` as the body. `RunContext` is a facade over `temporalio.workflow`, and
task and agent code never import `temporalio`. The only change to the loop is that continue-as-new must skip `teardown`.

```python
from __future__ import annotations

from datetime import timedelta
from typing import NoReturn

from temporalio import workflow
from temporalio.common import VersioningBehavior
from temporalio.workflow import ContinueAsNewVersioningBehavior

with workflow.unsafe.imports_passed_through():   # side-effect-free, content-addressed modules
    from platform.sdk import (Agent, ApprovalDecision, Continuation, End, Observation, RunOutcome,
                              RunSpecification, Signal, Task)
    from platform.sdk.loader import load_code_package
    from platform.temporal.context import TemporalRunContext


@workflow.defn(name="AgentRun", versioning_behavior=VersioningBehavior.PINNED)
class AgentRunWorkflow:
    def __init__(self) -> None:
        self.inbox: dict[str, list[Signal]] = {}
        self.approvals: dict[str, ApprovalDecision] = {}

    @workflow.signal(name="deliver")
    def deliver(self, signal: Signal) -> None:
        self.inbox.setdefault(signal.kind, []).append(signal)

    @workflow.update(name="approve")
    def approve(self, decision: ApprovalDecision) -> None:        # @tool(requires_approval=True)
        self.approvals[decision.call_id] = decision

    @approve.validator
    def validate_approval(self, decision: ApprovalDecision) -> None:
        if decision.call_id in self.approvals:
            raise ValueError("already decided")

    @workflow.run
    async def run(self, specification: RunSpecification,
                  continuation: Continuation | None = None) -> RunOutcome:
        package = load_code_package(specification.task.code_reference)     # pinned by content hash
        run = TemporalRunContext(specification, continuation, self.inbox, self.approvals)
        task, agent = await run.create_or_restore(package)                # restore = unpickle snapshot blob
        return await rollout(task, agent, run)


async def rollout(task: Task, agent: Agent, run: TemporalRunContext) -> RunOutcome:
    continuing = False
    try:
        if run.continuation is None:
            await task.setup(run)
            observation = await task.start(run)
            run.history.record_start(observation)
        else:
            observation = run.continuation.last_observation
        while observation.end is None:
            if run.should_continue_as_new():          # turn boundary: server suggestion or turn budget
                continuing = True
                await run.continue_as_new(task, agent, observation)       # NoReturn
            if task.max_turns is not None and run.turn >= task.max_turns:
                observation = End(truncated=True)
                run.history.record_end(observation)
                break
            reply = await agent.act(run, run.history, task.tools_for_turn(run))
            observation = await task.respond(run, reply)
            run.history.record(reply, observation)
        episode_reward = await task.score(run)
        if episode_reward is not None:
            run.reward(episode_reward)
        return run.outcome(observation)
    finally:
        if not continuing:                     # success, task error, or cancellation
            await run.shielded(task.teardown(run))
            await run.shielded(run.destroy_owned_environments())           # P11
            await run.shielded(run.publish_outcome())                      # rewards + ending → trajectory pipeline


class TemporalRunContext:                       # excerpt
    def should_continue_as_new(self) -> bool:
        information = workflow.info()
        return information.is_continue_as_new_suggested() or self.turns_in_segment >= 200

    async def continue_as_new(self, task: Task, agent: Agent, observation: Observation) -> NoReturn:
        snapshot_reference = await self.effects.put_blob(pickle_state(task, agent))   # may exceed 2 MB
        history_reference = await self.effects.put_blob(self.history.serialize())
        await workflow.wait_condition(workflow.all_handlers_finished)
        upgrade = workflow.info().is_target_worker_deployment_version_changed()
        workflow.continue_as_new(
            args=[self.specification,
                  Continuation(snapshot_reference, history_reference, self.turn, observation)],
            initial_versioning_behavior=(ContinueAsNewVersioningBehavior.AUTO_UPGRADE if upgrade else None),
        )
```

**Effects as activities.** Handles turn into `workflow.execute_activity` calls on per-kind task queues. The activity
workers are trusted (Go per ADR-0011, or async Python). They derive the effect identity from `activity.info()` and
check authorization against the server-supplied workflow ID, never against arguments from task code.

```python
class TemporalModel(Model):
    async def sample(self, messages: list[Message], *, tools: list[ToolSpecification] = (),
                     max_output_tokens: int | None = None,
                     tool_choice: ToolChoice | None = None) -> Message:
        delta = self.session.delta(messages)           # ContextDelta: never ship the whole context again
        reply: ModelReply = await workflow.execute_activity(
            "model.request",
            ModelRequest(slot=self.slot, context_delta=delta, tools=list(tools),
                         max_output_tokens=max_output_tokens, tool_choice=tool_choice,
                         observation_rewards=self.pending_rewards()),   # rewards also land in history
            task_queue="effects.model",
            start_to_close_timeout=timedelta(minutes=30),               # covers abort-and-resubmit
            retry_policy=RetryPolicy(initial_interval=timedelta(seconds=1), maximum_attempts=0,
                                     non_retryable_error_types=["InvalidRequest", "ContextOverflow"]),
            result_type=ModelReply,
        )
        self.session.advance(delta, reply)
        return reply.message


@activity.defn(name="environment.call")
async def environment_call(request: EnvironmentCallRequest) -> EnvironmentCallResult:
    information = activity.info()
    effect_id = f"{information.workflow_id}:{information.workflow_run_id}:{information.activity_id}"
    attachment = await environment_manager.attachment_for(request.environment_id,
                                                          run_id=information.workflow_id)
    if attachment is None:                                   # task code can forge IDs, not workflow_id
        raise ApplicationError("environment not owned or attached", type="Forbidden", non_retryable=True)
    return await envlet_client.call(attachment, request, idempotency_key=effect_id,
                                    deadline=activity_deadline(information))
```

Use **local activities** only for pure, short work (a deterministic blob hash, a cache read), and for compact mode
(4.2). They run before their marker is committed, so they cannot satisfy P5 for side effects.

**Per-cell deployment.**

- *Self-hosted (recommended if Temporal is chosen):* one Temporal cluster per cell, inside the cell's Kubernetes
  cluster, with its own persistence. Per cell:
  - 4,096 history shards, because the count cannot change later.
  - A namespace `cell-NN` for runs and identities.
  - Visibility on a separate SQL instance or OpenSearch.
  - No multi-cluster replication, since runs never migrate (cells.md).
- *Temporal Cloud:* one namespace per cell, or two if a cell exceeds the per-namespace APS ceiling (section 5), in the
  region nearest the cell. N4 then depends on Temporal's own cell boundaries, and Azure cells are not possible.

Task queues per namespace:

| Task queue | Served by | Notes |
|---|---|---|
| `runs` | Workflow workers (task hosts), versioned by Worker Deployment (build ID = platform SDK version) | Task code is *not* a deployment: it is loaded by content hash inside the run, so thousands of task packages do not create thousands of deployment versions (Cloud allows 100 versions × 100 deployments) |
| `identities` | Workflow workers for identity and swarm-coordinator workflows | `AUTO_UPGRADE` + patching |
| `effects.model` | Model effect workers → recorder or direct adapter | Fairness key = job or tenant |
| `effects.environment` | Environment effect workers → Environment Manager, envlet | Ownership check per call |
| `effects.tool` | Tool router | Retry class → `RetryPolicy` |
| `effects.platform` | Blobs, `run.memory`, outcome publishing, connectors (`run.emit`) | |

**Isolation of task code under both answers to Q10.**

- *In-house only.* Task code runs in Temporal's sandbox runner, which enforces determinism and nothing more. The
  workflow worker pods run under gVisor, with egress only to the frontend and no credentials beyond a
  namespace-scoped worker identity (mTLS certificate or API key). The residual risk is insider or bug: any code in a
  worker can use the process's client to signal, cancel or terminate any workflow in the namespace. Temporal
  authorization is per namespace and API, not per workflow [fact:
  [Principal attribution](https://github.com/temporalio/temporal/releases/tag/v1.31.0) records *who*, but does not
  restrict *which workflow*].
- *External tenants.* Untrusted code must not share a process with the Temporal connection. We build an
  **isolated runner**: the trusted worker keeps the Core connection, and each activation is forwarded to a gVisor task
  host over a local socket. That host runs the SDK's own sandboxed `WorkflowInstance` (pure Python over protobuf
  activations), and the trusted side **validates the returned commands** before completing the task. This is our
  `HarnessHost` boundary (B4) re-implemented against Temporal's activation protocol. The runner interfaces are
  internal SDK APIs, so we must track SDK releases [analysis].

```python
class IsolatedTaskHostRunner(WorkflowRunner):
    def __init__(self, pool: TaskHostPool, validator: CommandValidator) -> None:
        self.pool, self.validator = pool, validator

    def prepare_workflow(self, definition: workflow._Definition) -> None:
        self.pool.register(definition.name)

    def create_instance(self, details: WorkflowInstanceDetails) -> WorkflowInstance:
        return IsolatedWorkflowInstance(self.pool.host_for(details.info.workflow_id), details, self.validator)


class IsolatedWorkflowInstance(WorkflowInstance):
    def activate(self, activation: WorkflowActivation) -> WorkflowActivationCompletion:
        completion = self.host.activate(activation, timeout=timedelta(seconds=2))   # gVisor, no network, no credentials
        return self.validator.check(activation, completion)
        # allowed: ScheduleActivity of our effect types on forced task queues; timers; child AgentRun on "runs";
        # SignalExternalWorkflowExecution only to peers the swarm policy allows; ContinueAsNew of the same type;
        # payload caps. Anything else → failed workflow task (or fail the run as SECURITY_VIOLATION).
```

For large tenants, add a namespace per tenant per cell as defense in depth, which gives API-level isolation at the
cost of more namespaces and worker pools.

### 4.1 Synchronous RL

**Mechanics.** The rollout controller keeps admission (bounded buffer, ADR-0015). It starts one `AgentRun` per
`count`, sets search attributes `JobId` and `Labels`, and uses the fairness key `job_id`. Weights are fixed for the
batch. The controller waits for every run's outcome, then publishes.

- Stragglers are cancelled with `handle.cancel()`, and `teardown` still runs.
- `run.crashed` does not exist, because a worker crash only delays the run. Resampling applies only to
  infrastructure failures that surface as failed activities.
- Each run ends with one `run.outcome` activity. It publishes `{run_id, rewards keyed by reply effect_id, ending,
  labels}` to the trajectory pipeline and replaces B13's "run events once terminal".

```python
async def run_rows(client: Client, job: RolloutJob, rows: list[TaskRow]) -> list[WorkflowHandle]:
    handles: list[WorkflowHandle] = []
    for row in rows:
        specification = job.specification_for(row)
        handles.append(await client.start_workflow(
            AgentRunWorkflow.run, specification,
            id=new_run_id(job.cell),                      # embeds home cell (identifiers contract)
            task_queue="runs",
            search_attributes=job.search_attributes(row),
            priority=Priority(fairness_key=job.job_id),
        ))
    return handles
```

**Per-turn cost** (one model call + one environment call): 12 history events, 2 actions, about 8 persistence
transactions [analysis, section 5]. Fine for batches of thousands of episodes of tens of turns.

### 4.2 Asynchronous RL

- **In-flight weight updates.** Abort-and-resubmit happens inside the recorder while the `model.request` activity is
  waiting (ADR-0008). Temporal sees only a longer activity. `start_to_close_timeout` must cover the worst case, and
  retries are safe because the recorder deduplicates on the effect identity. Nothing in Temporal's model conflicts
  with R5.
- **Continuous admission.** This works as today. Priority separates evaluation from training on shared task queues.
- **High-rate short episodes are the problem.** A 3-turn math episode in regular mode costs about 1 start + 3 model
  activities + 1 outcome activity = **5 actions and about 40 history events** [analysis]. We propose **compact mode**
  for `Durability.BEST_EFFORT` runs: the whole episode runs in few workflow tasks, and effects are *local
  activities* whose receivers deduplicate on the effect identity. Crash semantics: the workflow task retries
  and the local activities re-run with the **same** activity IDs. The recorder and envd return cached results, so an
  episode is usually recovered rather than lost. P5 is waived exactly as `best_effort` already waives it.

```python
class CompactEffects(Effects):          # selected when RunBinding.durability == BEST_EFFORT
    async def model_request(self, request: ModelRequest) -> ModelReply:
        return await workflow.execute_local_activity(
            "model.request", request,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(maximum_attempts=3),
            result_type=ModelReply)
# started with task_timeout=timedelta(seconds=60) so a 30 s episode needs ≈ 1 workflow-task heartbeat
```

In compact mode a 3-turn episode costs about 2–4 actions and about 10 events. The exact count depends on workflow
task heartbeats, which are counted as actions ([actions](https://docs.temporal.io/cloud/actions)) [unverified until
spike T7]. Constraint: while local activities run, signals wait, so compact mode is for non-interactive episodes only.

### 4.3 Swarms (1,000 agents)

- **Spawn.** A `Swarm` coordinator starts 10 `SwarmGroup` children, and each group starts 100 `AgentRun` children.
  The tree keeps every parent under the 2,000-pending-children limit and keeps each history small. Group workflows
  start children in batches of about 25 per workflow task, because the history transaction limit is 4 MB and child
  inputs count against it. Cost: 2 actions per child start in Cloud.
- **Messaging.** `run.send(run_id, message)` becomes `get_external_workflow_handle(run_id).signal("deliver", …)`.
  That adds 2 events on the sender and 1 on the receiver (plus a workflow task), and costs 1 action.
  - A swarm of 1,000 agents each sending one message every 10 s is **100 signals/s, about 0.1/s per receiver**: well
    inside the ≤ 5/s per-workflow guidance.
  - Each receiver hits the 10,000-signal cap after about 28 h, so continue-as-new triggers on
    `is_continue_as_new_suggested()` or on a signal count threshold.
  - Broadcasts go through one `swarm.broadcast` activity that signals targets through the client. That costs N signal
    actions, but keeps the sender's history at 3 events instead of 2N.
- **Fan-in.** Children report through awaited results at the group level, and groups report to the coordinator.
  A single hub absorbing 1,000 completions is lock-bound per workflow (theoretical ≈ 20 events/s at 50 ms database
  latency) [fact on the bound; latency impact unverified, spike T5].
- **Shared environments.** These follow the existing attachment design; environment ownership stays with the
  coordinator (P11).
- **Cross-cell.** Signals cannot cross namespaces. Nexus operations (including the experimental system Nexus
  signal-with-start in Python SDK 1.29) could relay across cells, but the rule "a swarm lives in one cell" stands.

```python
@workflow.defn(name="SwarmGroup")
class SwarmGroupWorkflow:
    @workflow.run
    async def run(self, plan: SwarmGroupPlan) -> SwarmGroupResult:
        handles: list[ChildWorkflowHandle] = []
        for batch in chunked(plan.members, 25):             # stay under the 4 MB transaction limit
            for member in batch:
                handles.append(await workflow.start_child_workflow(
                    AgentRunWorkflow.run, member.specification,
                    id=f"{plan.swarm_id}/agent/{member.index}",
                    task_queue="runs",
                    parent_close_policy=ParentClosePolicy.REQUEST_CANCEL))
            await workflow.sleep(timedelta(0))              # yield: next workflow task
        results = await asyncio.gather(*handles, return_exceptions=True)
        return SwarmGroupResult.merge(plan, results)
```

### 4.4 Durable agent identities

**Definition (proposed).** A *durable agent identity* is a long-lived, addressable entity that outlives any run. It
consists of:

| Part | Content | Where it lives on Temporal |
|---|---|---|
| Address | `agent://{tenant}/{name}` → workflow ID `identity:{tenant}:{name}` in the home cell's namespace; a global directory maps address → cell | Workflow ID |
| State | Profile, role, default `AgentReference` / deployment, counters, lifecycle state | Workflow state, carried across continue-as-new |
| Memory | `run.memory`: namespaced, versioned key-value records; a small working set in state | Our store (per-cell Postgres table or object store) via `effects.platform` activities |
| Mailbox | Inbound messages and assignments | Signals into an in-workflow queue; overflow above N spills to an external mailbox table |
| Credentials and permissions | Grant *references* (broker `secret_ref`), import bindings, allowed peers, scopes | State; secrets never enter Temporal (P8) |
| Owned resources | Workspaces, environments and snapshots, owned by principal `identity:…` | Environment Manager registry (P11: the identity destroys them on retirement) |
| Policy binding | Model channel(s), sampling, `RunBinding` defaults, training lineage | State; its runs carry label `identity_id` so samples can be grouped and trained |

**Lifecycle.** `created` (update-with-start `register`, idempotent by workflow ID) → `active` (idle in
`wait_condition` most of the time) → `suspended` (no timers, mailbox closed) → `retired` (drain, destroy owned
resources, complete).

**Runs versus identities.** Each unit of work (a conversation, an assignment, a swarm role) is an `AgentRun` child
started with `ParentClosePolicy.ABANDON`. Children do not survive a parent's continue-as-new otherwise
([child workflows](https://docs.temporal.io/child-workflows)). The run signals `run_finished` back. High-rate
conversation traffic goes *directly* to the run via its run key (signal-with-start to `conversation:{identity}:{thread}`),
so the identity handles only lifecycle events and stays far below 5 signals/s.

**Continue-as-new cadence.** After each finished run, or at `is_continue_as_new_suggested()`, or every 1,000
signals. Identities use `AUTO_UPGRADE` + `workflow.patched`, because their code is small and platform-owned.
Long-idle `AgentRun`s waiting for a user use `PINNED`. When they wake with
`is_target_worker_deployment_version_changed()`, they continue-as-new with `AUTO_UPGRADE`, so old worker versions
only need to process one more task per run. Server v1.32 "version reactivation signals" reactivate a drained version
when a pinned workflow needs it ([v1.32.0](https://github.com/temporalio/temporal/releases/tag/v1.32.0)).

```python
@workflow.defn(name="AgentIdentity", versioning_behavior=VersioningBehavior.AUTO_UPGRADE)
class AgentIdentityWorkflow:
    def __init__(self) -> None:
        self.state: IdentityState | None = None
        self.mailbox: collections.deque[MailboxMessage] = collections.deque()
        self.signals_in_segment = 0

    @workflow.signal(name="deliver")
    def deliver(self, message: MailboxMessage) -> None:
        self.signals_in_segment += 1
        self.mailbox.append(message)

    @workflow.signal(name="run_finished")
    def run_finished(self, report: RunReport) -> None:
        self.signals_in_segment += 1
        self.state.active_runs.pop(report.run_id, None)

    @workflow.update(name="register")
    def register(self, request: RegistrationRequest) -> IdentityAddress:
        return self.state.address

    @workflow.run
    async def run(self, state: IdentityState) -> None:
        self.state = state
        self.mailbox.extend(state.carried_mailbox)
        while self.state.lifecycle is not Lifecycle.RETIRED:
            await workflow.wait_condition(
                lambda: bool(self.mailbox) or self.state.lifecycle is Lifecycle.RETIRED,
                timeout=timedelta(hours=24))                      # daily housekeeping: 1 timer action/day
            while self.mailbox and self.state.can_start_run():
                message = self.mailbox.popleft()
                run_id = f"{workflow.info().workflow_id}/run/{self.state.next_run_number()}"
                await workflow.start_child_workflow(
                    AgentRunWorkflow.run, self.state.run_specification_for(message),
                    id=run_id, task_queue="runs", parent_close_policy=ParentClosePolicy.ABANDON)
                self.state.active_runs[run_id] = message.summary()
            if workflow.info().is_continue_as_new_suggested() or self.signals_in_segment > 1_000:
                await workflow.wait_condition(workflow.all_handlers_finished)
                workflow.continue_as_new(self.state.carry(list(self.mailbox)))
        await workflow.execute_activity("identity.retire", self.state.owned_resources(),
                                        task_queue="effects.platform",
                                        start_to_close_timeout=timedelta(minutes=10))
```

**Why this fits well [analysis].** A mostly idle identity costs no worker memory, because it is evicted from the
cache. On the server it costs one mutable-state row plus a short history, and one timer action per day. A wake-up
replays a short history, since the entity keeps it short by continuing as new. This is the virtual-actor behavior
that DBOS needed a new layer for (charter item 5); on Temporal it is the documented entity-workflow pattern.

## 5 Scale and performance

### 5.1 Evidence [fact]

- Temporal Cloud: spikes of "150,000+ actions per second" across the service
  ([Series D](https://temporal.io/blog/temporal-raises-usd300m-series-d-at-a-usd5b-valuation)); "automatic scaling to
  300k Actions/second or more" ([Cloud page](https://temporal.io/cloud)). Neither figure is per namespace.
  The self-service per-namespace ceiling is 6,000 APS ([capacity modes](https://docs.temporal.io/cloud/capacity-modes)).
- Self-hosted benchmark (Temporal, May 2023): MySQL with 4 vCPUs, going from 4 to 512 shards and scaling history CPU,
  took throughput from 150 to 1,350 state transitions/s
  ([scaling basics](https://temporal.io/blog/scaling-temporal-the-basics)). Third-party write-ups report about
  4,500 state transitions/s with 2,048 shards on Postgres. These sources are of low or uncertain quality [unverified].
- Latency: self-hosted MySQL p50 `RespondWorkflowTaskCompleted` 23.9 ms and p90 51.5 ms; Cloud 17.8 ms and 24.7 ms
  ([benchmark](https://temporal.io/blog/benchmarking-latency-temporal-cloud-vs-self-hosted-temporal)).
- No published benchmark shows a *single self-hosted cluster* at tens of thousands of state transitions per second,
  and none characterizes Python worker density. These are our biggest evidence gaps.

### 5.2 Per-turn accounting [analysis]

Base case (B): 1 model call + 1 environment call per turn. High case (H): 1 model call + 2 concurrent environment
calls.

| Per turn | B | H | Derivation |
|---|---|---|---|
| Activities | 2 | 3 | effects |
| History events | 12 | ~15–18 | per activity: Scheduled + Started + Completed (Started is written at close ([forum](https://community.temporal.io/t/when-does-temporal-write-the-activitytaskstarted-event-into-workflow-history/6162))); per workflow task: Scheduled + Started + Completed |
| Workflow tasks | 2 | 2–3 | one per completion batch |
| Actions (Cloud) | 2 | 3 | activity starts; heartbeats rarely reach the server with throttling; retries add |
| Persistence transactions | ~8 | ~11 | ≈ 2 per activity (record started, completed) + 2 per workflow task [unverified] |
| History bytes | 2–5 KB (claim check) / 10–100 KB (inline) | same | context *delta* + reply + tool output; never the full context |

Per-run overhead: start, outcome publication, continue-as-new every ~200–340 turns, and environment lifecycle. That
is about 4–6 actions per run, or ≈ 0.2 per turn for 25-turn RL episodes.

### 5.3 At the design point [analysis]

| Quantity | Fleet (30k turns/s) | Per cell (~2.5k turns/s) | Our run-store design for comparison |
|---|---|---|---|
| Actions/s | 66k (B) – 96k (H) | 5.5k – 8k | n/a |
| History events/s | 360k – 540k | 30k – 45k | ~90k events/s fleet |
| Workflow tasks/s (worker activations) | 60k – 90k | 5k – 7.5k | 60k `Step`s/s |
| Persistence transactions/s | ~240k – 330k | ~20k – 28k | ~60k transactions/s fleet, ~5k per cell |
| History shards | — | 4,096 (fixed at creation; lock-bound per shard) | 4,096 partitions |
| Open workflows | 300k | 25k | same |

- **Persistence per cell.** About 20–28k transactions/s, each writing several rows (history node, mutable-state row,
  task records). That is 4–5× what ADR-0002 sized one Postgres primary for. **One managed Postgres per cell is
  unlikely to hold** [unverified, spike T1]. Realistic options: Cassandra per cell (not a managed multi-cloud
  primitive, which strains N2), cells of about 8–10k runs, or Temporal Cloud.
- **Workers.** To avoid replay, 25k active runs per cell must stay cached. At 2,000 cached runs per process that means
  at least 13 Python processes per cell, plus headroom. Memory per cached run, with conversation history in workflow
  memory, is estimated at 0.3–2 MB [unverified]. Python activation throughput per core and replay speed are not
  published [unverified, spike T2].
- **Continue-as-new cadence.** At 12 events per turn, the 4,096-event suggestion arrives at about 340 turns. Our
  snapshot cadence (every hook, every 10 turns) is irrelevant on Temporal. Only continue-as-new points matter, at
  about every 200 turns.

### 5.4 Cost

**Temporal Cloud** (list prices [fact]): $50 per million actions for the first 5M, falling in tiers to $25 per
million for 100M–200M. Above 200M/month is "Contact Sales". Active storage costs $0.042 per GB-hour and retained
storage $0.00105 per GB-hour. Replicated namespaces cost 2× ([pricing](https://docs.temporal.io/cloud/pricing),
[pricing.md](https://temporal.io/pricing.md)).

| Scenario (30-day month) | Actions/month | At $25/M (lowest published tier) | At $10/M | At $2/M |
|---|---|---|---|---|
| B: 66k APS | ~171 billion | ~$4.3M | ~$1.7M | ~$0.34M |
| H: 96k APS | ~249 billion | ~$6.2M | ~$2.5M | ~$0.50M |
| Compact mode for 70% of turns (B) | ~70–90 billion | ~$1.8–2.3M | ~$0.7–0.9M | ~$0.14–0.18M |

Storage, with payloads offloaded by claim check [analysis]:

- *Active storage.* 300k open histories × ~1 MB ≈ 300 GB, about $9k/month. With inline payloads (~10 MB each) it
  is about $94k/month.
- *Retained storage.* About 10 TB of closed history per day. With 1-day retention that is about $8k/month; with
  30-day retention, about $234k/month.

The negotiated price for more than 170B actions/month is unknown [unverified, spike T8]. Scale check: this volume
would be about 2–3% of Temporal Cloud's reported *lifetime* action count **per month**. It would need a sales
engagement and capacity above the 6,000 APS self-service ceiling in every cell namespace.

**Self-hosted** [analysis, unverified]:

- Per cell: a Cassandra cluster of roughly 9–15 nodes, 100–200 vCPU of Temporal services, and a visibility store,
  for about $25–50k/month. Across 12 cells that is about **$0.3–0.6M/month** of infrastructure, plus a platform team
  of about 3–6 engineers.
- Third-party estimates put the break-even with Cloud at only 30–50M actions/month
  ([Automation Atlas](https://automationatlas.io/guides/temporal-cloud-vs-self-hosted-2026/), low confidence).
  We are roughly 3,000× past that.
- For scale: 15M generated tokens/s implies thousands of GPUs, likely several million dollars per month. Cloud at
  list price would therefore rival the inference bill, while self-hosting would be a modest fraction of it.

### 5.5 Where it breaks

1. **Per-workflow rate:** ≤ 5 signals/s sustained guidance per workflow. Hubs (coordinators, popular identities)
   need trees and direct run-key routing.
2. **History size:** runs past about 340 turns *must* continue-as-new. Payloads over 2 MB or transactions over 4 MB
   (big fan-outs) fail outright.
3. **Actions cost** at Cloud list price, and **persistence write amplification** self-hosted.
4. **Per-namespace APS:** 6,000 self-service, below one cell's 5.5–8k.
5. **Short episodes:** fixed per-run and per-effect overhead. Compact mode is the mitigation.
6. **Visibility queries** in Cloud (30/s): run listing and search for our UI needs our own index.

## 6 Failure semantics

| Failure | What happens on Temporal | Effect on our guarantees |
|---|---|---|
| Workflow worker dies mid-task | Workflow-task timeout or sticky timeout (default 10 s) → the task is redelivered to another worker, which replays | Nothing lost. Latency is up to the timeout, above N5's 50 ms: set the sticky timeout to about 2–5 s |
| Stale worker completes a task | Rejected (task token no longer matches the started event) | Same as our `FENCED` |
| Activity worker dies mid-effect | `start_to_close` or heartbeat timeout → retry with the **same** activity ID | Effectively once where the receiver deduplicates (recorder, envd, Environment Manager); `outcome_unknown` otherwise |
| Late completion after timeout | Rejected by attempt-scoped task token | Same as "first terminal wins" |
| History host dies | Shard re-acquired by another host (`RangeID` fencing) | Stall of seconds for that shard's runs |
| Persistence failover | All shards stall | Cell-scoped, like our Postgres failover |
| Non-determinism on replay | Workflow task fails and retries (the run is stuck, not failed) unless `NondeterminismError` is listed in the workflow's failure exception types | Equivalent to quarantine; can be fixed by deploying corrected code |
| Cancellation | `CancelledError` → `finally` teardown; activities cancelled per `ActivityCancellationType` (needs heartbeats to reach running activities) | Matches the loop. Known SDK bug: cancellation can be lost when another event arrives in the same task (fix behind a flag in 1.31) |
| Termination, run timeout, reset | No workflow code runs | Owned environments leak → an **ownership reaper** in the Environment Manager checks for terminal owners (`DescribeWorkflowExecution`) and TTLs |
| Local activity crash (compact mode) | Workflow task retries; local activities re-run with the same activity IDs | At-least-once; receivers deduplicate |
| Temporal Cloud regional outage | Standard namespace: unavailable; HA namespace: failover with RPO under 1 minute | Duplicates of activities after failover are absorbed by deduplication |

## 7 Security and multi-tenancy

- **Credentials.** A worker holds namespace-scoped credentials that can start, signal and terminate any workflow in
  the namespace. There are no per-workflow ACLs. Untrusted code must therefore live behind the isolated runner
  (4.0), which is our effect validation re-homed.
- **Sandbox.** The Temporal Python sandbox is "not secure" ([README](https://github.com/temporalio/sdk-python)).
  gVisor (trust-boundaries rule 9) remains the boundary in both Q10 answers.
- **Effect authorization.** Workflow ID and run ID in `activity.info()` come from the server. Effect workers check
  ownership against them, which is stronger than trusting arguments.
- **Encryption and deletion.** A payload codec with per-tenant data keys encrypts payloads in workers, so the server
  and Temporal Cloud see ciphertext. Crypto-shredding a tenant key makes its histories unreadable, which satisfies
  the deletion requirement for payloads. Search attributes, workflow IDs, activity types and (unless encoded) failure
  messages stay plaintext, so no PII goes in them. Whether memo passes through the codec in Python SDK 1.33 is
  [unverified].
- **Tenant fairness.** Fairness keys per tenant on shared task queues (GA in the open-source server, paid in Cloud),
  plus our per-tenant quotas at the Control API. Namespaces per tenant are the stronger option.
- **Attribution.** Principal attribution (v1.31) records the authenticated caller on history events, which is useful
  for audit.

## 8 Operations

- **Self-hosted per cell:** the Temporal services and their persistence cluster, a visibility store, dynamic config,
  and a shard count that can never change.
  - Upgrades go one minor version at a time with schema migrations (for example, v1.31 required Postgres schema
    v1.19 and visibility v1.14).
  - Deprecations move fast: the legacy versioning APIs were deprecated in v1.28 and removed in v1.33.
- **Workers:** the Temporal Worker Controller runs rainbow deployments per build ID. We must keep old versions
  alive while `PINNED` runs remain, or rely on upgrade-on-continue-as-new.
- **Cloud:** capacity management (TRUs, support tickets above 6,000 APS), private connectivity, a 30-day default
  retention (1–90 days), and a proprietary persistence layer that we cannot inspect.
- **Observability:** the Web UI and history make debugging individual runs better than anything we would build. SDK
  metrics (schedule-to-start latency, sticky cache hit rate) and server metrics are mature. OpenTelemetry through
  interceptors, with the replay-safe providers added in Python SDK 1.32.

## 9 What we still build

| Item | Rough size [analysis] |
|---|---|
| `RunContext` / `Program` facade over `temporalio.workflow`: handles → activities, deterministic effect identity, `run.patched`, gather, signals and approvals | 3–5k lines of Python |
| Continuation: pickle/restore task and agent state at turn boundaries, history blobs, loop changes for continue-as-new | 1–2k lines of Python |
| Content-addressed code package loader inside the sandbox (passthrough of immutable packages) | ~1k lines |
| Effect workers: model (recorder or direct adapter), environment (ownership, attachment tokens), tool router, platform (blobs, memory, outcomes, `run.emit`) | 3–5k lines of Go or Python |
| **Isolated runner + gVisor task host + command validator** (only if Q10 = tenants) | 2–4k lines + packaging; tracks internal SDK APIs |
| Control API adapter: runs, signals and run keys → Temporal client; run event stream | 2–3k lines of Go |
| Rollout controller integration (admission unchanged) and outcome → trajectory pipeline | ~1k lines |
| Ownership reaper for terminated runs; token-streaming side channel; per-tenant codec and codec server | 2–3k lines |
| Identity service: entity workflow, directory, memory store, mailbox overflow | 3–5k lines |
| Self-hosted cell automation: Temporal + Cassandra + visibility per cell, dashboards, runbooks | significant, continuous |

**No longer built:** the run store schema and fenced appends, partition leases, mailboxes, the timer relay, the replay
driver and divergence detection, version routing, and signal delivery. That is most of ADR-0002/0003/0004 and the
S5/S6 spikes. We estimate it at 15–25k lines of the most failure-sensitive code [analysis].

## 10 Risks, open questions, spike tests

**Risks.**

1. Cost or write amplification at the design point (sections 5.3–5.4).
2. Python worker density and replay cost are unknown.
3. The isolated runner depends on internal SDK interfaces.
4. Continue-as-new with pickled state is required and becomes a correctness hot spot.
5. Cloud excludes Azure and puts the failure domain outside our control.
6. Cancellation-loss bug until the SDK flag is enabled by default.
7. Operating Cassandra per cell.

**Open questions.**

- The negotiated Cloud price per action above 200M/month.
- Whether a Postgres cell can reach 25k transactions/s.
- Whether compact mode's crash-recovery rate is acceptable for RL sampling bias.
- Whether the directory of identities should itself be a Temporal-free table (likely yes).

**Spike tests (each with a pass criterion).**

| # | Test | Measure | Pass |
|---|---|---|---|
| T1 | One cell's cluster (4,096 shards) on (a) the largest managed Postgres 16 and (b) 6-node Cassandra 5; synthetic `AgentRun` with 25k open runs, 2 activities per turn, one turn per 10 s | Sustained state transitions/s, database CPU, p99 `RespondWorkflowTaskCompleted`, schedule-to-start | 2× design load (5k turns/s) with p99 control overhead ≤ 50 ms per half-turn |
| T2 | Python worker density with the real loop, a pydantic-heavy task package, and the sandbox | Activations/s per core, RSS per cached run, replay time for a 4,000-event history | ≥ 500 activations/s per core; ≤ 1 MB per cached run; replay ≤ 200 ms |
| T3 | Isolated runner: gVisor host over a socket, command validator, red team (signal, terminate, schedule foreign activity types) | Added latency per activation; rejected attacks | ≤ 2 ms p99 added; 100% of the attack list rejected |
| T4 | 5,000-turn run with 64 KiB tool outputs, continue-as-new every 200 turns, deploy twice mid-run | History errors, continue-as-new latency, correct upgrade | No limit errors; continue-as-new ≤ 1 s; runs upgrade at the next boundary |
| T5 | Swarm: 1,000 agents at 1 message per 10 s each, plus a hub that absorbs a burst of 1,000 completions | Signal p99, task failures from signals racing continue-as-new, hub drain time | Signal p99 ≤ 200 ms; hub burst drained in ≤ 60 s |
| T6 | Cancel 10k runs at once; terminate 1k | Teardown completion, leaked environments, lost cancellations with the SDK flag on and off | 0 leaks after the reaper; 0 lost cancellations with the flag on |
| T7 | Compact mode: 3-turn episodes at 10k/s per cell, killing workers randomly | Actions and transactions per episode, fraction recovered from recorder cache | ≤ 4 actions per episode; ≥ 95% recovered |
| T8 | Cloud commercial test: quote for 170–250B actions/month and 8k APS per namespace; one-week load test on one namespace | $/M actions; sustained APS | Total cost ≤ 25% of the inference budget |
| T9 | 1M idle identities with a daily timer; wake 10k/s | Storage per identity, wake-to-first-activity latency, upgrade across 3 deploys | ≤ 100 KB per identity; wake p99 ≤ 500 ms |

## 11 Sources

- Temporal Cloud: [limits](https://docs.temporal.io/cloud/limits), [pricing](https://docs.temporal.io/cloud/pricing),
  [pricing.md](https://temporal.io/pricing.md), [actions](https://docs.temporal.io/cloud/actions),
  [capacity modes](https://docs.temporal.io/cloud/capacity-modes),
  [operating envelope](https://docs.temporal.io/cloud/operating-envelope),
  [high availability](https://docs.temporal.io/cloud/high-availability), [regions](https://docs.temporal.io/cloud/regions),
  [Cloud overview page](https://temporal.io/cloud)
- Self-hosted: [defaults](https://docs.temporal.io/self-hosted-guide/defaults),
  [multi-cluster replication](https://docs.temporal.io/self-hosted-guide/multi-cluster-replication),
  [persistence](https://docs.temporal.io/temporal-service/persistence),
  [server architecture](https://docs.temporal.io/temporal-service/temporal-server),
  [history-service design](https://github.com/temporalio/temporal/blob/main/docs/architecture/history-service.md),
  [dynamic config constants](https://github.com/temporalio/temporal/blob/main/common/dynamicconfig/constants.go)
- Releases: [server v1.32.0](https://github.com/temporalio/temporal/releases/tag/v1.32.0),
  [v1.31.0](https://github.com/temporalio/temporal/releases/tag/v1.31.0),
  [v1.29.0](https://github.com/temporalio/temporal/releases/tag/v1.29.0),
  [Python SDK 1.33.0](https://github.com/temporalio/sdk-python/releases/tag/1.33.0),
  [1.31.0](https://github.com/temporalio/sdk-python/releases/tag/1.31.0),
  [1.29.0](https://github.com/temporalio/sdk-python/releases/tag/1.29.0),
  [1.28.0](https://github.com/temporalio/sdk-python/releases/tag/1.28.0)
- Python SDK: [README (sandbox, cancellation)](https://github.com/temporalio/sdk-python),
  [worker options](https://github.com/temporalio/sdk-python/blob/main/temporalio/worker/_worker.py),
  [workflow runner interface](https://github.com/temporalio/sdk-python/blob/main/temporalio/worker/_workflow_instance.py),
  [sandbox docs](https://docs.temporal.io/develop/python/python-sdk-sandbox),
  [worker versioning](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning),
  [local activities](https://docs.temporal.io/local-activity), [message passing](https://docs.temporal.io/sending-messages),
  [Workflow Streams](https://docs.temporal.io/develop/python/workflows/workflow-streams),
  [external storage](https://docs.temporal.io/develop/python/data-handling/external-storage),
  [priority and fairness](https://docs.temporal.io/develop/task-queue-priority-fairness),
  [child workflows](https://docs.temporal.io/child-workflows)
- Agent integrations: [OpenAI Agents contrib](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/openai_agents),
  [OpenAI Agents SDK announcement](https://temporal.io/blog/announcing-openai-agents-sdk-integration),
  [Pydantic AI + Temporal](https://pydantic.dev/docs/ai/integrations/durable_execution/temporal/),
  [docs.temporal.io/ai](https://docs.temporal.io/ai)
- Scale and cost evidence: [Scaling Temporal: the basics](https://temporal.io/blog/scaling-temporal-the-basics),
  [Cloud vs self-hosted latency benchmark](https://temporal.io/blog/benchmarking-latency-temporal-cloud-vs-self-hosted-temporal),
  [Cloud persistence layer](https://temporal.io/blog/higher-throughput-and-lower-latency-temporal-clouds-custom-persistence-layer),
  [Series D announcement](https://temporal.io/blog/temporal-raises-usd300m-series-d-at-a-usd5b-valuation),
  [Quo: Postgres to Cassandra](https://www.quo.com/blog/postgres-to-cassandra/),
  [forum: PostgreSQL in production](https://community.temporal.io/t/postgresql-good-option-for-persistence-in-production/6153),
  [signal throughput guidance (docs issue #3665)](https://github.com/temporalio/documentation/issues/3665),
  [ActivityTaskStarted timing (forum)](https://community.temporal.io/t/when-does-temporal-write-the-activitytaskstarted-event-into-workflow-history/6162),
  [Automation Atlas cost guide (third party, low confidence)](https://automationatlas.io/guides/temporal-cloud-vs-self-hosted-2026/),
  [Temporal Worker Controller](https://github.com/temporalio/temporal-worker-controller)
