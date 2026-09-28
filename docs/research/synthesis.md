# Synthesis: substrate, interfaces, and how the system operates

Status: **Draft** · 2026-09-27 · Inputs: [substrate-dbos](substrate-dbos.md), [substrate-restate](substrate-restate.md),
[substrate-temporal](substrate-temporal.md), [substrate-custom-and-others](substrate-custom-and-others.md),
[interfaces-agent-frameworks](interfaces-agent-frameworks.md), [interfaces-rl-environments](interfaces-rl-environments.md)

This document consolidates the six reports into recommendations and a fleshed-out design at each level. Numbers are
the reports' (sources are there); anything marked *estimate* is analysis, not measurement. Nothing here is
normative until it lands in `architecture/`, `contracts/`, `components/` or an ADR (section 10 lists the proposed
changes).

---

## 1. Recommendations

1. **Buy the engine core; build what no substrate has.** Our own runtime would cost about 39–60 engineer-months to
   a first version plus 2–4 engineers permanently (*estimate*), mostly in versioning, visibility, retention, reset,
   operator tooling and correctness testing — the parts engines spent years on.
2. **Primary substrate: DBOS (MIT library, 3.1.0) in "pump mode", on the per-cell Postgres we already chose.**
   **Designated alternative: Restate.** **Excluded unless spikes overturn it: Temporal** (4–5× our write load per
   turn; Temporal Cloud ≈ $4–6M/month at list price for our action rate). **Fallback: our own runtime**, only if the
   phase-1 spikes fail on both DBOS and Restate. No other contender is viable as the primary substrate (no Python,
   no multi-cloud self-hosting, or published throughput far below a cell).
3. **Make the substrate swappable behind one narrow waist.** A trusted *pump* (the substrate's workflow) drives a
   sandboxed *task host* over `HarnessHost`; every effect the task host requests becomes a substrate step. All four
   substrate reports independently arrived at this shape. It also resolves the untrusted-code problem on every
   substrate, so Q10 no longer blocks the substrate choice.
4. **Durable agent identities are resources, not endless runs.** An identity has an address, a deployment, versioned
   state, scoped memory, single-consumer mailbox lanes, grants as a principal, owned resources and a policy binding.
   Work happens in bounded *activation runs* (one per conversation episode); `WaitFor` ends an activation at a turn
   boundary, and the next message starts a new one.
5. **Replace the generator-style replay driver with a deterministic asyncio event loop** (Temporal's design), so
   third-party agent frameworks run unchanged inside the task host.
6. **Keep the RL design; extend its data and controls.** The field has converged on what we built (renderers,
   masked observations, per-token versions, in-flight weight updates, stale-KV correction). Add turn spans, MoE
   routing replay, finer outcomes, gang admission, run cancellation, multiple trainable channels, staged weight
   publish, rewards to descendants, and a prefill-only `ScoreTokens` operation.
7. **Fix five correctness gaps the reviews found** (section 7), notably effect-identifier reuse after a lossy
   database failover and cleanup that no substrate runs after hard cancellation.

---

## 2. What every report agrees on (engine-independent architecture)

These hold whichever substrate is chosen. They are the core of the proposed design.

### 2.1 The pump and the task host

```
                 ┌──────────────── trusted ────────────────┐     ┌──────── untrusted (gVisor, no network) ────────┐
 substrate ────▶ │ pump workflow (generic, rarely changes) │ ◀──▶│ task host: deterministic asyncio loop,          │
 (DBOS steps)    │  · authorizes each effect request       │ B4  │ Program / Task / Agent code, snapshots          │
                 │  · runs it as a substrate step          │     └─────────────────────────────────────────────────┘
                 │  · feeds completions back in log order  │
                 └───────┬───────────────────────┬─────────┘
                         ▼                       ▼
            recorder · envlet · Environment Manager · tool router · blob store · identity service
```

- **The pump is a pure forwarder.** It starts the task host, sends it inputs, validates the effect requests it
  returns (schemas, sizes, environment ownership, permitted imports and peers, budgets), and runs each as a
  substrate step. On recovery the substrate re-runs the pump; recorded steps return their stored completions, which
  the pump re-feeds to a fresh task host. Because the task host is a pure function of its inputs, it regenerates the
  same requests: that *is* our replay, with no second log.
- **The pump starts steps in the task host's deterministic order**, so substrate step identifiers are deterministic
  even for concurrent tool calls (DBOS numbers steps in call order; running task code directly as a DBOS workflow
  breaks on concurrent multi-effect tools).
- **Only JSON crosses the boundary.** Snapshots are pickled and unpickled inside the sandbox only.
- **Versioning splits in two.** The pump's substrate version changes rarely (DBOS patching). Task and agent code is
  pinned by `code_reference` and routed to sandbox pools, exactly as today.
- **Q10 now only decides how strict the sandbox is** (per-tenant pools, cells for high-trust tenants), not which
  substrate we can use.

### 2.2 Effects are at-least-once; receivers make them effectively-once

Every substrate records a step's result *after* running it (DBOS steps, Restate `ctx.run`) or can redeliver it
(Temporal activities on timeout). So:

- **Effect identity** comes from the substrate's deterministic step identity:
  `effect_id = {run_id}:{generation}:{step_ordinal}`, identical on every re-execution.
- **Receivers deduplicate** with three states (absent → execute; in progress → join; done → return cached) — now
  *mandatory*, not a rare-race optimization — **and check an argument digest** sent with the `effect_id`; a
  mismatch is a conflict error, never a cached answer to a different request (section 7, gap 1).
- **Tools that cannot deduplicate** claim an attempt marker in their own committed transaction before dispatch. A
  retry that finds the marker reports `tool.outcome_unknown` instead of calling again. One extra write per such call.
- P5 ("persist before effect") is restated accordingly (section 10).

### 2.3 Bounded runs

- **Generations (continue-as-new).** When a run exceeds its replay budget (turns, recorded bytes, wall time) or a
  deploy asks it to drain, it hands over at the next resumable point: the task host exports a snapshot, the pump
  starts generation *n+1* with it, and generation *n* ends. This bounds replay cost and log size on every substrate
  (Temporal requires it past ~340 turns; Restate replays the whole journal on every resume).
- **Identities never run forever** (section 5): continuity between activations lives in versioned state records,
  memory and an identity-owned history table — not in a replayed log, and not in pickle, which does not survive code
  upgrades.

### 2.4 Other shared conclusions

- **Short synchronous RL episodes need no per-step durability.** A crashed episode is cheaper to resample than to
  resume. They run as a single substrate step (about 6 rows per episode on DBOS) or bypass the substrate entirely.
- **Cancellation must be cooperative, plus a reaper.** No substrate runs cleanup after termination, timeout or (on
  DBOS) cancellation. Cancel is a control message the pump delivers at turn boundaries so `teardown` runs normally;
  a per-cell reaper destroys whatever terminal runs and archived identities still own (P11).
- **Token streaming is out of band** everywhere (DBOS streams and Temporal Workflow Streams write a row or signal
  per flush; Restate has no streams). Durable outputs go through the log.
- **Admission, quotas and fairness stay ours.** Substrate flow control either co-locates a scope on one partition
  (Restate) or has no bounded start queue (Temporal).
- **Swarms stay in one cell**; cross-cell messages go through the target cell's Control API.

---

## 3. Substrate comparison

| | DBOS (pump mode) | Restate | Temporal | Own runtime |
|---|---|---|---|---|
| Model | Durable workflow library in our process; steps memoized in Postgres | Server with replicated log + RocksDB; services, virtual objects, workflows; push invocations | Server (history, matching); workflows + activities; pull workers | Our Postgres log + Go workers + task hosts |
| License | MIT (Conductor proprietary, not needed) | Server BSL 1.1 (Apache-2.0 after 4 years per version); SDK MIT | MIT; Cloud commercial | Ours |
| Storage per cell | The cell Postgres we already run | A 5–6 node Restate cluster on NVMe + object-store snapshots | Cassandra-class cluster (Postgres unlikely to hold) + visibility store | The cell Postgres |
| Actors / identities | Ours: entity table + `deliver` SQL function + activations on a partitioned queue (`partition_concurrency = 1`), ~2–3k lines | Native virtual objects (serial per key, K/V state, free suspension) | Entity workflows + continue-as-new; ≤ 5 signals/s per workflow | Ours (not yet designed) |
| Write cost per turn (*estimate*) | ~2.5–3 checkpoint rows, ~14k transactions/s per cell (half read-only) | ~4–6 log records | ~12 history events, 8–11 transactions (4–5× ours) | ~3 events, ~2 transactions |
| Evidence at our scale | Vendor benchmark 43k no-step workflows/s, 30.6k queued/s on one large Postgres; our cell needs ~15–25% | 3-node cluster ~85–94k actions/s; Replit runs tens of cells at 25k actions/s each | Cloud reports 150k+ actions/s service-wide; no self-hosted cluster evidence at our rate | None |
| Recovery / fencing | `owner_xid` fencing (2026, young); **no failure detector** → our controller (~0.5–0.8k lines) | Epochs; built in | Shard `RangeID` fencing; built in | Leases + epochs (ours) |
| Main risks | Delete-based retention churn; young fencing code; `NOTIFY` per `send`; recovery reads one transaction per recorded step | Young Python SDK (1.0 in June 2026); key features experimental in 1.7; partition count fixed per cluster; BSL; new cluster type to operate | Cost; write amplification; history limits; Python sandbox is not a security boundary; no Azure in Cloud | Engineering time; correctness of a new engine |
| Glue we still build (*estimate*) | ~11–15k lines | ~15–25k lines | ~20–30k lines + persistence operations | Full engine (39–60 engineer-months) |

**Why DBOS first.** Once identities are resources with bounded activations (section 5), the actor features we need
beyond serial execution per key — lanes, busy policies, grants, memory scopes, owned resources, schedules — are ours
on every substrate, so Restate's native actors save less than they first appear to (~2–3k lines). DBOS adds no new
stateful system to operate: it lives in the managed Postgres per cell that ADR-0002 already chose and that every
cloud offers (N2). It is MIT, so forking is a real option if retention or fencing need changes. Restate's
advantages — actor-native semantics, free suspension, a purpose-built log — become decisive only if DBOS fails the
phase-1 spikes (retention at our write rate, recovery storms, fencing under partitions).

**Switching cost is bounded.** The substrate-specific code is the pump, the actor scheduling, the recovery
controller and the Control API adapter — roughly 5–8k lines. Task code, agents, the recorder, environments, identities
and the RL plane do not change.

---

## 4. The design on DBOS, level by level

### 4.1 Cell layout

| Schema (cell Postgres) | Owner | Holds |
|---|---|---|
| `dbos` | DBOS | workflows, steps (`operation_outputs`), queues, messages, streams, versions |
| `actors` | us | `identity`, `mailbox`, `identity_history` (time-partitioned), `identity_state_version`, `memory` |
| `effects` | us | attempt markers (24 h retention) |
| `platform` | us | run keys, deployments, environment registry, executor heartbeats |

Processes per cell: pump executors (a StatefulSet; stable `executor_id` = pod name; ~20–50 processes), task-host
sandbox pools per `code_reference`, the recovery controller, the reaper, retention/archival, the Control API.

### 4.2 Run kinds

| Kind | DBOS shape | Durability | Used for |
|---|---|---|---|
| **Durable run** | `run_pump` workflow; one step per effect; generations | Resumes after any crash | Agentic RL episodes, coding agents, swarm members, workflows (`Program`) |
| **Activation run** | `activate_identity` workflow on the `identities` queue, partition key = identity lane | Resumes; ends at `WaitFor` | Conversations and other identity work |
| **Best-effort run** | `best_effort_run` workflow with a single `episode_step` | Crash → resample (~6 rows per episode) | Short synchronous RL episodes |

A middle tier (a `chunk_step` covering *K* turns) is possible later if resampling long episodes proves too costly.

### 4.3 Effects and recovery

- `effect_id = f"{run_id}:{generation}:{DBOS.step_id}"`; argument digest sent with every effect.
- Completion order is recorded with `DBOS.asyncio_wait`, so the task host sees completions in the same order on
  replay (one row per wait).
- **Recovery controller** (replaces DBOS Conductor): executors heartbeat every 5 s; an executor is dead when its
  heartbeat is older than 15 s *and* Kubernetes reports the pod gone; the controller re-enqueues its `PENDING`
  workflows (DBOS's own `reenqueue_for_recovery` statement). Takeover ≈ 20 s plus a queue poll. A zombie executor's
  next checkpoint fails the `owner_xid` check, so it can at most re-dispatch steps it had already started; receivers
  absorb them.
- **Poison runs**: the pump counts recoveries per run; after 3 crash-looping recoveries the run is quarantined
  (`run.failed{POISONED}`) before it can take down another task host.

### 4.4 Retention and history

DBOS step rows are deleted by retention (24–72 h). Long-lived records we need later — run events projected for B13,
identity history, recorder sessions — live in our tables and object storage. Inline step payloads are capped at
~4–8 KiB (larger values go to blob storage) to limit write-ahead-log volume; the current 64 KiB inline limit is too
high for this substrate.

---

## 5. Durable agent identities

### 5.1 Definition (proposed normative wording)

A **durable agent identity** is a named, long-lived, owned resource:

1. **Address**: an immutable `agent_id` (`ag_{cell_id}_{ulid}`, embedding the home cell) and a mutable unique name
   `{tenant}/{namespace}/{name}` (e.g. `agent://acme/support/alice-bot`).
2. **Deployment and version policy**: a named deployment (Program, or Task + Agent code references, plus a
   `RunBinding` template). `FOLLOW` (default) takes the deployment's current version at the next activation;
   `PINNED` fixes code references.
3. **State**: small, typed, versioned records (`state_schema` + registered upcasters), read and written only through
   effects. Never a pickled coroutine.
4. **Memory**: namespaced key-value records with versions, compare-and-set and time-to-live, plus pinned context
   blocks; scoped to the identity, shareable only through explicit grants.
5. **Mailbox**: durable, ordered envelopes, deduplicated by `message_id`, with **lanes**; at most one activation
   consumes a lane. Default one lane per identity; optionally one per conversation.
6. **Grants**: the identity is a principal. Imported tools and brokered credentials are authorized as that principal
   (possibly on behalf of a user through a delegated grant). An access control list says who may send to it and
   read its streams.
7. **Owned resources** (P11): named workspaces (`get_or_create`), snapshots, schedules, child identities.
8. **Policy binding**: model slots bound to channels; training advances the channel without code changes (R5).
9. **Lifecycle**: `ACTIVE` (idle or running) ⇄ `PAUSED` (accepts messages, consumes none) → `ARCHIVED` (read-only)
   → `DELETED` (crypto-shredded with its key; owned resources destroyed by the reaper).

### 5.2 Activations

```
 message ──▶ deliver(identity, lane, envelope)          one transaction: insert into mailbox (idempotent);
                                                          if the lane is idle, mark it SCHEDULED and
                                                          dbos.enqueue_workflow('activate_identity', partition = lane)
                 │
                 ▼
 activate_identity(lane)   ── load state (upcast) ── claim messages ── run the Program/Task loop (pump mode)
                 │                                          │
                 │            Task.respond → WaitFor ───────┘ commit state + history (compare-and-set), then
                 ▼                                              park: lock row, re-check mailbox, mark IDLE
            nothing resident while idle; every activation runs the deployment's current version
```

- **Lost wake-ups are impossible**: `park` locks the lane row and re-checks the mailbox in a new statement, so any
  message whose `deliver` committed first is seen.
- **Busy policy** per identity: `ENQUEUE` (default), `INTERRUPT` (cancel the in-flight model effect and feed the new
  message as the next observation), `REJECT`.
- **Idle activation timeout** (default 30 minutes with no messages) ends the activation; `End(continue_as=…)` ends
  the episode and starts a fresh activation carrying explicit state, which gives RL a natural episode boundary.
- **Schedules**: one alarm per identity. Schedules live in identity state; only the next fire time is a delayed
  enqueue. This avoids timer storms (every identity's 00:00 job).
- **Short waits** (approvals inside a tool) stay in the activation as `DBOS.recv` with a timeout.

### 5.3 Interface

```python
# Control plane
client.identities.create(AgentIdentitySpecification(
    name="support/alice-bot", deployment="support-agent", version_policy=VersionPolicy.FOLLOW,
    grants=[GrantReference("zendesk")], mailbox=MailboxPolicy(lanes=LaneMode.PER_CONVERSATION)),
    request_id="…")
client.identities.send(Address.identity("support/alice-bot", conversation="slack:C1/171.2"),
                       Envelope(kind="message", content=[Text("Hi")]),
                       idempotency_key="slack-event-Ev123")          # signal-with-start

# Task code inside an activation
class SupportConversation(Task):
    imports = ["zendesk"]

    async def setup(self, run: RunContext) -> None:
        self.notes = await run.identity.resources.workspace("notes", NOTES_ENVIRONMENT)

    async def start(self, run: RunContext) -> Observation:
        envelope = await run.mailbox.next()
        profile = await run.identity.memory.get(("users", envelope.sender.subject))
        return Observation(render_first_turn(envelope, profile))

    async def respond(self, run: RunContext, reply: Message) -> Observation | WaitFor:
        if reply.tool_calls:
            return await self.run_tools(run, reply)
        await run.emit("reply", reply.content, to=run.origin)       # delivered by the Slack connector
        return WaitFor("message", timeout=timedelta(hours=24), on_timeout=End(truncated=True))

    async def resume(self, run: RunContext, envelope: Envelope) -> Observation:
        return Observation(envelope.content)
```

### 5.4 Training identities

- Each activation is an episode; its trajectory is the activation's recorder sessions.
- **Delayed rewards** (a thumbs-up two days later): `client.runs.reward(run_id, reply_effect_id, value, key=…)`
  appends `reward.assigned` to a terminal run and re-emits the affected samples as a new revision.
- **Samples from production traffic** need an *observe* job mode that collects samples from runs outside rollout
  jobs, labelled by identity, subject to retention and consent rules (open question 5).

---

## 6. How the system operates, per use case

### 6.1 Synchronous RL (on-policy GRPO)

1. The Ray driver starts a job: `rollouts.start(task, agent, binding, buffer_samples)`.
2. It submits rows with `job.run(row, labels={"group": g}, count=8, admit_together=True)`. Gang admission keeps
   the 8 runs of a group together under buffer pressure.
3. Each run is a **best-effort run**: one DBOS step drives the task host through the whole episode against the
   template-built environment; samples come from the recorder; the outcome (rewards bound to reply `effect_id`s,
   ending, observation digests) is the workflow output.
4. The assembler emits one `Sample` per run and trainable slot; the driver groups by label, drops zero-variance
   groups, computes group-relative advantages, and trains.
5. `job.publish(channel, weights, phase=STAGE_AND_COMMIT)` between steps; no episode spans a weight change.

Cost: about 6 rows per episode; a crash resamples; the crash rate by episode length is monitored for bias.

### 6.2 Asynchronous RL with in-flight weight updates

1. Long agentic episodes run as **durable runs** (pump mode) so a worker loss does not discard hours of work.
2. The trainer publishes every step: `STAGE` transfers weights while the old version serves; `COMMIT` runs the
   weight update controller's pause → abort → swap → resume. In-flight generations are split by
   abort-and-resubmit inside one model step; the step just takes longer, and a retried step hits the recorder's
   cached result.
3. **Staleness is the caller's admission rule** (a ten-line AReaL-style rule in the trainer adapter), backed by the
   buffer bound. The caller uses `job.cancel(label_selector=…)` to stop surplus or over-stale work.
4. Samples carry per-token `weights_version`, turn spans and (for MoE) routed experts; the trainer computes
   importance weights against recorded behavior logprobs.

### 6.3 Multi-agent training

- **Self-play**: two slots of one task on the same trainable channel; each run yields two samples; the caller uses
  role-conditioned baselines.
- **One trainable role in a swarm of frozen agents**: the root spawns children; children inherit the job; the root's
  `score` rewards descendants (`run.reward(value, slot=…, run_id=child)`); samples are assembled when the root is
  terminal.
- **Several trainable identities**: one job with `trainable_channels={"planner": …, "coder": …}`; the caller routes
  samples by `policy_id` to separate trainers, each publishing its own channel, and acknowledges the minimum
  consumed cursor.

### 6.4 Swarms

- **Spawn and fan-out**: `run.map(specifications, concurrency=16, quorum=None)` admits children with backpressure;
  each child is a durable run in the same cell; results are recorded in the parent.
- **Messaging**: `run.send / request / reply` with `Address` and `Envelope`. To identities: `deliver` into their
  mailbox (one transaction, no `NOTIFY`). To resident runs waiting in-activation: `DBOS.send_bulk`.
- **Topics**: a per-topic append log that subscribers read by cursor at turn boundaries, instead of one inbox write
  per subscriber (avoids 10⁴-way write amplification).
- **Wide fan-in**: a barrier record counts completions and wakes the parent once.
- **Handoffs** stay in the Agent layer (`HandoffAgent` + a `HandoffTools` mixin that answers transfer calls).
- **Scale** (*estimate*): 1,000 agents at one turn per 10 s with one message each ≈ 100 messages/s ≈ a few hundred
  rows per second — trivial for a cell. Chatty swarms must avoid per-row `NOTIFY` (spike S-D5).

### 6.5 Durable agent identities in production

A Slack message becomes `identities.send(..., idempotency_key=event_id)`; the connector posts `run.emit("reply")`
outputs back; approvals arrive through the Control API `DecideApproval`; the identity's workspace hibernates between
conversations; memory persists across conversations; the activation runs the latest deployment. 300k mostly idle
identities cost 300k rows, not 300k coroutines.

### 6.6 Generic durable workflows

A `Program` with `async def main(self, run)` runs as a durable run: effects are steps, `run.sleep` is a delayed
continuation, triggers (cron via DBOS schedules, webhooks via the Control API) start it, `run.emit` delivers output.

---

## 7. Correctness gaps found

| # | Gap | Where | Fix |
|---|---|---|---|
| 1 | **Effect identifier reuse after a lossy failover.** If the database loses an acknowledged commit, replay can request a *different* effect under an identifier a receiver already executed, and the receiver returns the old cached result | Every design keyed by log position | Synchronous replication with zero data loss (Aurora, RDS Multi-AZ, Cloud SQL regional HA, Azure zone-redundant HA); receivers check an argument digest with the `effect_id` and reject mismatches |
| 2 | **Cleanup does not run** after termination, timeouts or (DBOS) cancellation | All substrates | Cooperative cancellation at turn boundaries; run deadlines computed from recorded time; per-cell reaper for owned resources |
| 3 | **Unbounded logs and pickle-based state** for long-lived entities | Identities, long runs | Bounded activations and generations; versioned state records with upcasters; history in our tables |
| 4 | **Poison runs** crash every task host they are replayed onto | Task hosts | Recovery counting and quarantine before reload |
| 5 | **Silent divergence** when replayed code changes observations or rewards but not effect requests | Replay | Divergence detection also compares digests of `observation.recorded` and `reward.assigned` |
| 6 | Non-determinism that gVisor does not stop (`time.time`, `os.urandom`, `uuid4`, native thread ordering) | Task host | The deterministic asyncio loop patches these (R1) |
| 7 | Timer storms and bloat at millions of timers | Schedules | One alarm per identity; delete-on-fire |

---

## 8. Interface change set

Consolidated from both interface reports, in priority order. "Where" is the document the change lands in.

| # | Change | Where |
|---|---|---|
| I1 | Deterministic asyncio event loop replaces the generator driver; forbids real I/O; maps time and timers to the log | harness/durability.md |
| I2 | Pump / task-host boundary as the substrate waist; effect identity from substrate step identity; argument digests | harness/README.md, contracts/effects.md, delivery-semantics.md |
| I3 | `WaitFor` (from `start` and `respond`), `Task.resume`, `End(continue_as=…)`, generations | harness/task.md, durability.md |
| I4 | Durable agent identities: specification, `run.identity` (state, memory, resources), mailbox lanes, busy policy, schedules, `ag_…` identifiers, deployments | new components/identities/ |
| I5 | Messaging: `Address`, `Envelope`, `send / request / reply / publish`, `run.mailbox.next`, `run.map(concurrency, quorum)`, topics as cursor-read logs | new components/messaging/ (or identities) |
| I6 | Approvals: `@tool(approval=…)`, `RunBinding.approvals`, `Approve / Reject / Respond` with `remember`, `approval.requested / decided`, Control API `DecideApproval` | harness/task.md, tool-router, control-api |
| I7 | Streaming in two tiers: the log as a resumable stream by cursor; ephemeral token and stdout deltas by `effect_id`; `run.emit(kind, payload, to=Address)` → `output.emitted` | contracts/run-events.md, control-api |
| I8 | Memory: `run.memory` with scopes, versions, compare-and-set, time-to-live, pinned blocks; `ContextHints.pinned` | harness/task.md, agent.md |
| I9 | Environments: `get_or_create(name)`, `start_process` / `Process`, `expose(port)`, layered `Template` with `start` and `ready`, `Environment.request`, optional `Template(dockerfile=…)` | environments/*, harness/task.md |
| I10 | `Sample`: turn spans, `routed_experts`, `outcome`, `truncation_reason`, `mask_reasons` (`run.exclude_from_training`), media, optional transcript, `root_run_id`, enrichments | trajectories |
| I11 | `RolloutJobs`: `Cancel(run_ids | label_selector)`, label-filtered `Samples` and per-ticket awaitables, `admit_together`, `trainable_channels`, `Publish(channel, WeightsSource, phase)` | rollouts |
| I12 | Swarm rewards: children inherit the job; rewards to descendants; assembly at root terminal | rollouts, trajectories |
| I13 | `ScoreTokens` on the engine adapter; optional per-job enrichment (teacher / reference logprobs) | recorder/engine-adapter, trajectories |
| I14 | Budgets: `Budget(turns, input_tokens, output_tokens, cost, wall_time, tool_calls)` per binding and identity; exhaustion → `TRUNCATED` | harness, control-api |
| I15 | Tools: toolsets, `ToolCallContext`, binding kinds `external` and `a2a`, MCP Tasks support | tool-router, harness/task.md |
| I16 | Interoperability: tier-1 foreign harness in an environment with a `compatibility_endpoint()`; tier-2 in-process adapters (OpenAI Agents SDK, Pydantic AI); task sessions exported over MCP; environments as a sandbox provider; A2A for identities; OpenTelemetry projection | new components/interoperability/ |
| I17 | Adapters in: Harbor, verifiers, OpenEnv, Inspect, Gymnasium/TextArena. Out: miles/slime, SkyRL, Tinker exporter, AReaL, verl | new components/adapters/ |

---

## 9. Spike plan

Ordered so each phase gates the next. Pass criteria are in the source reports.

| Phase | Spike | Decides |
|---|---|---|
| **1 — substrate go / no-go** | X1 pump + gVisor task host on DBOS: added latency per step (≤ 5 ms p99), replay after kill | The waist is affordable |
| | S-D1 cell throughput (25k runs; 7.5k then 20k checkpoints/s; 2/8/32 KB payloads) | DBOS holds a cell |
| | S-D2 48-hour retention soak | Delete-based retention is sustainable, or we fork the schema |
| | S-D3 recovery storm (kill an executor holding 2k runs × 300 steps) | Recovery time ≤ 2 min |
| | S-D7 zombie fencing under network partition | No duplicate checkpoints; duplicates absorbed |
| | X6 forced managed-Postgres failover under load | No lost acknowledged commit; digest check catches reuse |
| | S-D6 mass cancellation | Zero leaked environments after the reaper |
| **2 — identities and swarms** | S-D4 activations (25k lanes, 2.5k deliveries/s, chaos): zero double activations, zero lost wake-ups | The actor layer |
| | S-D5 `NOTIFY` pressure from swarm messaging | Messaging path choice |
| | X4 identity activation, DBOS vs Restate head-to-head (optional; run if phase 1 is marginal) | Whether Restate's actors justify a second cluster type |
| | S6 gang admission; S7 swarm rewards at root | Rollout additions |
| **3 — RL** | S1 miles adapter (synchronous): recomputed vs recorded logprobs; learning curve parity | Trainer integration |
| | S2 asynchronous with publish every step: cache hit after swap, lag distribution, effective sample size vs `max_kv_age` | ADR-0009 in practice |
| | S3 staged publish (checkpoint, delta, NCCL, checkpoint-engine) | `WeightsSource` choices |
| | S4 routed-expert capture through the recorder; S9 renderer coverage; X3 best-effort runner vs durable | MoE and renderer readiness |
| **4 — interoperability** | Deterministic asyncio loop with the OpenAI Agents SDK in-process; LangGraph compatibility; S5 Harbor adapter with Claude Code / Codex in a microVM | Tier-2 and tier-1 adapters |

---

## 10. Proposed changes to the normative docs

Not applied yet. Each needs agreement.

**New decision records**
- **ADR-0016 — Substrate: DBOS in pump mode behind `HarnessHost`**; Restate as the designated alternative; own
  runtime as fallback. Supersedes the runtime half of ADR-0004 and the Run Store design of ADR-0002/0003 (Postgres per
  cell stays).
- **ADR-0017 — Effects are at-least-once; receivers deduplicate with argument digests; attempt markers** (restates P5
  and ADR-0003).
- **ADR-0018 — Durable agent identities as resources with bounded activations and mailbox lanes.**
- **ADR-0019 — Deterministic asyncio event loop in the task host** (amends ADR-0013).
- **ADR-0020 — Short synchronous RL episodes run best-effort in a single step.**
- **ADR-0021 — Messaging, approvals and two-tier streaming.**
- **ADR-0022 — RL interface extensions** (Sample fields, rollout controls, staged publish, swarm rewards, `ScoreTokens`).

**Principles**: restate P5 (ADR-0017); add "long-lived entities are bounded runs over versioned state"; add
"substrate behind one waist".

**Components**: new `identities/`, `messaging/` (or folded into identities), `interoperability/`, `adapters/`;
`run-store/` and `runtime/` rewritten as DBOS configuration, the pump, the recovery controller and the reaper;
`harness/durability.md` rewritten around the pump and the asyncio loop; `trajectories/` and `rollouts/` extended
(I10–I13); `environments/` extended (I9); `delivery-semantics.md` rewritten for at-least-once steps.

## 11. Decisions needed

1. **Substrate**: DBOS primary with Restate as the alternative (section 3) — agree?
2. **Q10** (who writes task code): no longer blocks the substrate, but decides sandbox pool isolation and whether
   high-trust tenants get their own cells.
3. **Mailbox lanes and busy policy defaults**: one lane per identity or per conversation; `ENQUEUE` or `INTERRUPT`.
4. **First trainer**: miles/slime (recommended; mandatory routing replay for MoE), then SkyRL.
5. **Trainer-participating weight transfer** (`DistributedTransfer`): acceptable coupling of trainer and inference
   networks, or object storage plus deltas only?
6. **Samples from production traffic** (identity *observe* jobs): retention, consent and delayed-reward rules.
