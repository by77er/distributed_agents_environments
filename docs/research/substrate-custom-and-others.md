# Substrate: our own runtime vs other contenders

Status: **Draft** · 2026-09-27 · Question owner: [research charter](README.md)

**Question.** (A) Evaluate the runtime specified in our docs — Postgres run log per cell, partition leases with
epoch fencing, commit-then-dispatch effects, generator-driven replay in Python task hosts, a Go runtime — honestly as
one option against buying a substrate. (B) Survey the contenders other than DBOS, Restate and Temporal (covered in
[substrate-dbos](substrate-dbos.md), [substrate-restate](substrate-restate.md), [substrate-temporal](substrate-temporal.md)).

**Method.** Our design documents (listed in the [charter](README.md)); vendor documentation, changelogs, license
files and GitHub metadata checked on 2026-09-27. Facts carry links; analysis and estimates are labelled as such;
vendor performance claims without published methodology are marked.

## 1. Verdict

1. **No contender beyond DBOS, Restate and Temporal is a viable primary substrate**: each fails Python, self-hostable
   multi-cloud production, or credible throughput at ~25k runs per cell. Hatchet (MIT, Postgres, Python) fits best but
   is queue-shaped; keep it only as a candidate for non-run queues.
2. **Cloudflare Durable Objects / Agents SDK is the best model for durable agent identities** (named single
   activation, per-object storage, free hibernation, one alarm per actor) and cannot be self-hosted: borrow, don't run.
3. **Our own runtime costs ~39–60 engineer-months to v1 plus 2–4 engineers permanently** (estimate), mostly in
   versioning, visibility, retention, reset, operator tooling and correctness testing — not in the Postgres core.
4. **Its unique properties are layerable** (trusted adapter + sandboxed task host over `HarnessHost`; short RL rollouts
   bypass the substrate) except one: a single log that is both durability record and RL record.
5. **Recommendation: build what no substrate has, buy the engine core**; fall back to our own engine only if spikes
   X1/X2 (§6) fail. Ray is our callers' compute plane: integrate with it, do not adopt it.
## 2. Our own runtime

This section evaluates the runtime specified in [run-store](../components/run-store/README.md),
[runtime](../components/runtime/README.md), [delivery-semantics](../architecture/delivery-semantics.md),
[durability](../components/harness/durability.md) and ADR-0002/0003/0004/0013, as if it were one more product on
the shelf. It is the only option whose behaviour we can specify exactly, and the only one whose bugs we would own.

### 2.1 What the specification already is

Read as a product, the specified runtime is a durable execution engine with this shape (all facts from our docs):

| Concern | Specified mechanism |
|---|---|
| Durable state | One Postgres primary per cell; append-only `events` partitioned by time; `runs`, `leases`, `inbox`, `timers`, `snapshots`, `archive` tables |
| Ownership | 4,096 partitions per cell; leases `(partition, owner, epoch, expires_at)`, TTL 15 s, renewed every 5 s in one batch; epoch checked inside every append transaction |
| Execution | Go workers; per-run serial mailbox; Python task hosts (one per core, gVisor, no network) drive `async` code as generators |
| Effects | `*.requested` committed before dispatch (the log is the outbox); `effect_id = {run_id}:{seq}`; receivers dedupe with absent / in progress / done; `retry_class` decides re-dispatch versus `outcome_unknown` |
| Recovery | Replay of the log into the task host; *k*-th requested effect matched to *k*-th committed request; divergence fails the run with `NON_DETERMINISM` |
| Replay bound | Pickled task and agent state at resumable points (every hook, every ~10 turns), capped at 16 MiB, disposable |
| Messaging | Intra-cell `send` / `spawn` / timers written in the same transaction as the step; cross-cell via a relayed outbox |
| Durability tiers | `durable` (append before every dispatch) and `best_effort` (buffered; crash → `run.crashed`, resampled) |
| Versioning | Runs pinned to `code_reference`; `run.patched(change_id)`; `code.upgraded` at resumable points |

That is roughly the core of Temporal's history service, matching service and SDK replayer, compressed into one
Postgres schema and a fixed workflow shape. The honest comparison is therefore not "a thin layer on Postgres" but
"a new durable execution engine with a narrower surface".

### 2.2 Build inventory and effort

Estimates are ours (analysis, not measured), in engineer-months for experienced distributed-systems
engineers, to reach a production-grade first version at the design point. They assume the environment plane,
recorder and tool router are built anyway and are not counted.

| # | Component | Scope beyond the docs | Effort (engineer-months) |
|---|---|---|---|
| 1 | Run Store schema + Go client | Fenced append with side-writes, gapless `seq`, inbox, keyed inserts, `SKIP LOCKED` scheduling views, snapshot index, partition maintenance (`pg_partman` or own), archive reader that makes cold segments transparent | 3–4 |
| 2 | Runtime worker | Lease manager and partition balancing without a coordinator; per-run mailboxes; bounded dispatch; pending-effect timer wheel; takeover sweep; retry classes; validation; environment client; recorder session open; tool resolution; completion backstop; graceful drain | 5–7 |
| 3 | Python task host | Generator driver; deterministic `gather` scheduler; replay matcher; divergence detection; `@checkpoint`; snapshot / restricted unpickler; per-`code_reference` package loading; memory accounting for thousands of runs per process; gVisor packaging | 5–7 |
| 4 | Timers and relays | Timer relay (due timers → inbox), cross-cell outbox relay, nudges, cron triggers and webhooks (new generic additions) | 2–3 |
| 5 | Control API | Create (idempotent), signal, signal-with-start on run keys, cancel, query, `Watch` streams, bulk create for rollouts | 2–3 |
| 6 | Visibility and search | Listing and filtering 300k live runs and millions of terminal runs across 12+ cells by status, labels, tenant, deployment, time; fan-out or a secondary index (e.g. ClickHouse fed from archive segments) | 2–4 |
| 7 | Versioning and deployments | Deployment objects (name → code references), host pools per version, draining old versions, `patched`, replay-compatibility test tooling that replays production histories against new code in CI | 3–4 |
| 8 | Retention, archive, deletion | Archive to columnar segments, drop time partitions, per-tenant encryption keys and crypto-shredding, legal holds | 2–3 |
| 9 | Operator tooling and UI | Run inspector (event timeline, pending effects, host placement), quarantine queue for `NON_DETERMINISM`, reset / rewind a run to an event, bulk cancel / signal / reset, local replay debugger | 4–6 |
| 10 | SDK ergonomics | Typed handles, test harness with time skipping and mocked effects, local single-process dev server (Postgres or SQLite), error messages for determinism violations, documentation | 3–5 |
| 11 | Actor layer for durable agent identities | Entity rows, mailbox semantics, activation per message, `WaitFor`, compaction of long-lived logs | 2–4 |
| 12 | Generic platform additions | `run.emit` connectors, token-streaming side channel, budgets and quotas, `run.memory`, OpenTelemetry | 3–5 |
| 13 | Correctness testing | Fault-injection and deterministic simulation of worker, host and Postgres failures; linearizability checks of the fenced append; soak tests at 20k events/s per cell (spike S5) | 3–5 |
| | **Total** | | **≈ 39–60** |

Read: a team of 5–6 engineers for 9–12 months to a credible v1, then 2–4 engineers permanently for maintenance,
on-call, performance and feature pressure. Durable execution vendors describe the same arc: Temporal's lineage
(Cadence at Uber, then Temporal from 2019) has spent years on the edge cases listed next, and only declared its
current versioning model generally available in March 2026 (facts below).

### 2.3 Hard edge cases engines took years to get right

For each: what went wrong elsewhere (fact, cited), and what our specification says or omits (analysis).

1. **Versioning long-lived code.** Temporal went from `GetVersion` / `patched` markers, through Build-ID based
   versioning, to Worker Deployments with *Pinned* and *Auto-Upgrade* behaviours, which reached GA only in March 2026,
   with "upgrade on Continue-as-New" still in public preview
   ([Temporal changelog](https://temporal.io/changelog/worker-versioning-continue-as-new-worker-controller),
   [Worker Versioning docs](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning)).
   Our docs specify pinning plus `patched` — Temporal's first two generations. Missing: who keeps old host pools
   alive while pinned runs exist (a durable agent identity idle for months pins a version indefinitely), how a
   deployment is declared drained, and replay-compatibility testing against real histories before a deploy. For
   durable agent identities, pinning is the wrong default; they need an upgrade point at each activation (§2.6).
2. **Python non-determinism.** Temporal's Python SDK ships a workflow sandbox that re-imports modules per workflow
   and restricts non-deterministic calls, and documents its performance cost and pass-through escape hatches
   ([Temporal Python sandbox](https://docs.temporal.io/develop/python/python-sdk-sandbox)). Our design relies on gVisor
   with no network plus a fixed `PYTHONHASHSEED`. gVisor does not stop `time.time()`, `os.urandom`, `uuid.uuid4()`,
   thread-pool ordering in native libraries, or a library upgrade changing iteration order; these must be caught by
   patching the standard library in the host (as Temporal does) and by divergence detection. Divergence detection
   compares kind + argument hash, so a change that alters only *observations or rewards* (not effect requests)
   replays silently into a different state — acceptable for recovery, dangerous for RL data if a replayed run
   re-emits different rewards (the log keeps the first, but task state diverges from it).
3. **Timers at scale.** Timers must survive storms (every identity's daily cron at 00:00), cancellation races, and
   table bloat. Our `timers` table updates `fired` in place and polls `fire_at` — at millions of timers this becomes
   an index-bloat and vacuum problem; delete-on-fire plus a time-bucketed in-memory wheel per
   partition is the likely fix (analysis; our understanding is that Temporal and Restate both keep timers per shard /
   partition and load them ahead of time — **[unverified here]**, see the sibling reports).
4. **Cancellation.** Cooperative cancellation must propagate to pending effects, child runs (including cross-cell
   children through the relay), and `teardown`, which itself performs effects *while* cancelled; a second cancel,
   a cancel racing completion, and a cancel arriving during replay all need defined outcomes. Our docs define the
   happy path (`run.cancel_requested` → wind-down → `run.cancelled`) but not child propagation or a teardown deadline.
5. **History growth.** Temporal caps a history at 51,200 events or 50 MB and forces Continue-as-New
   ([limits](https://docs.temporal.io/workflow-execution/limits)). We have no cap, and snapshots bound *replay* but not
   *log size*. A durable agent identity that lives for a year accumulates an unbounded log; it needs log
   compaction: a snapshot that becomes the new base, with older segments archived and no longer needed for recovery.
   Snapshot validity is tied to `code_reference`, so the compaction base must survive code upgrades — which pickle
   does not guarantee. Durable identities therefore need versioned, schema-evolvable state records, not pickle.
6. **Visibility and search.** Temporal originally required Elasticsearch for filtered listing; SQL-based advanced
   visibility arrived in v1.20 with small per-namespace caps (3 custom attributes per type, 10 keywords)
   ([Temporal visibility](https://docs.temporal.io/visibility)). Our `labels jsonb` on a per-cell `runs` table works
   for one cell; a tenant-wide "all runs with label X, failed, last 7 days" across cells needs a global index.
7. **Retention and deletion.** Dropping time partitions works only if nothing live references the partition;
   long-lived runs span partitions, so partition drop must wait for archive of every run touching it, or archive
   must copy live runs' old events forward. Per-tenant crypto-shredding must cover events, snapshots, blobs,
   recorder session trees and archive segments.
8. **Reset, rewind, backfill.** Operators eventually need "re-run this run from event N with fixed code", bulk
   reset after a bad deploy, and bulk backfill of schedules. Temporal and DBOS both expose reset / fork operations
   (DBOS forks a workflow from a chosen step into a new workflow
   ([DBOS workflow management](https://docs.dbos.dev/python/tutorials/workflow-management))); we have none specified.
9. **Poison runs and blast radius.** A run whose step crashes the Python process (native segfault, out-of-memory,
   pathological pickle) takes down every run co-hosted in that process, and replays onto the next host, crashing it
   too. Needs crash-loop detection per run and quarantine before reload. Not specified.
10. **Fairness and priority.** One tenant's 50k-run rollout job must not starve interactive coding runs on the same
    worker. Temporal now ships task-queue priority levels and fairness keys, fairness being a paid Temporal Cloud feature
    ([Temporal priority and fairness](https://docs.temporal.io/develop/task-queue-priority-fairness)). Our runtime
    lists priority as an open question (FIFO initially).
11. **Failover durability.** `effect_id = {run_id}:{seq}`. If a Postgres failover ever loses an acknowledged commit
    (asynchronous replica promoted), the next owner replays, re-requests an effect at the same `seq` — possibly with
    different arguments — and a receiver that already executed the lost request returns the *old* cached result
    for a *different* request. The design therefore requires synchronous replication with zero data loss on
    failover (Aurora, RDS Multi-AZ, Cloud SQL regional HA and Azure zone-redundant HA provide this; asynchronous
    cross-region replicas do not). Receivers should also verify an argument hash with `effect_id`. Neither is in
    the docs today.
12. **Lease clocks.** Fencing stops a deposed worker from *committing*, not from *dispatching* an effect it
    committed just before expiry (ADR-0003 accepts this). Worker pauses (GC, CPU throttling in Kubernetes) longer than
    the TTL cause duplicate dispatch bursts; the three-state receivers absorb them, but `outcome_unknown` tools will
    surface them to the model. Measure in spike S5.
13. **Upgrading the engine itself.** Rolling a new runtime or task host version moves partitions (drain → release →
    acquire → lazy reload). At 2,500 runs per worker and replay of each, a rolling deploy of 50 workers is a
    cell-wide replay wave; snapshots bound it, but the wave's load on Postgres must be tested.

### 2.4 What it uniquely enables

Analysis, with the substrate comparison in brackets.

1. **Untrusted task host split.** Task code holds no database connection, no substrate credentials and no network;
   it can only return events that the trusted runtime validates (schemas, sizes, environment ownership). [DBOS runs
   task code in the process holding the system-database connection; a Temporal worker holds namespace credentials and
   can call any API in the namespace; a Restate service endpoint holds no store credentials but performs its own I/O.]
2. **The log schema is the product.** `observation.recorded`, `reward.assigned`, `model.completed` with
   `reply_effect_id` are typed, versioned, closed-catalog events that the trajectory assembler reads directly (B13).
   [On any substrate, the substrate's history is an opaque, engine-versioned format; our schema becomes a second
   stream written by steps.]
3. **Two durability tiers in one runtime.** `best_effort` runs skip per-step persistence entirely. [No bought
   substrate has an equivalent; each persists every step.]
4. **Replay bounded by snapshots.** Pickled state at resumable points means a 500-turn episode replays from the last
   snapshot, not from `run.created`. [Temporal replays full history and relies on Continue-as-New; DBOS re-executes
   from the start while skipping recorded step bodies.]
5. **Same-transaction intra-cell `send` / `spawn`.** Swarm fan-out and messaging cost one transaction and no
   delivery machinery. [DBOS gets the same because it is also Postgres; Restate has exactly-once messaging between
   its own services; Temporal child starts are atomic with the workflow task.]
6. **Per-cell control.** One Postgres per cell, cell-scoped failure, cell-local data residency, no vendor control
   plane. [Achievable with any self-hostable substrate by deploying one instance per cell.]
7. **Worker never blocks on effects; effect execution is ours.** Model requests go to the recorder, environment
   calls to envlets, with our validation, deadlines, retry classes and `outcome_unknown` semantics. [Temporal
   activities and Restate service calls can express this; DBOS steps run inside the workflow's process.]

### 2.5 Operating cost

Rough, unverified estimates for the design point (12 cells, 300k runs):

| Item | Own runtime | Note |
|---|---|---|
| Postgres | 12 multi-AZ primaries sized for ~20k events/s peak each, plus a read replica each | Needed by DBOS too, and by self-hosted Temporal (as its persistence) — not a differentiator |
| Runtime workers | 10–50 per cell, small Go processes | Temporal's frontend / history / matching services or Restate's server nodes would replace these |
| Task hosts | One Python process per core; the dominant compute | Required under every option (someone must run Python task code) |
| People | 2–4 engineers on the runtime permanently after v1, including on-call | The real cost; comparable to the platform team a self-hosted Temporal or Restate deployment of this size needs, *plus* the engine's own development |

Temporal Cloud list pricing is $50 per million Actions, falling to $25 per million at volume
([Temporal pricing](https://docs.temporal.io/cloud/pricing)). At ~30k turns/s and ~2 billable Actions per turn
(analysis), that is ~155 billion Actions a month, ~$3.9M a month at the lowest list tier before commitments —
a useful ceiling showing that at this scale "buy" means *self-host*, and the comparison is engineering time against
engineering time.

### 2.6 What can be layered onto a bought substrate

| Unique property | Layerable? | How |
|---|---|---|
| Untrusted task host split | **Yes** | The substrate's worker becomes a trusted *adapter* that implements the substrate SDK and speaks `HarnessHost` (B4) over a local socket to a sandboxed task host. The adapter's workflow is one generic, trivially deterministic loop ("send input to host; for each returned effect, perform it as a step / activity / call"). Costs one local hop per step. This fixes DBOS known issue 1. |
| Log schema for RL | **Yes, with dual write** | Our events become step outputs; a projector (DBOS `write_stream`, a Temporal activity, a Restate call to a sink) writes them, keyed by `effect_id`, to a store the trajectory assembler reads. Loses single-log atomicity; needs dedupe at the sink. |
| Snapshots | **Partly** | Natural on actor-state substrates (Restate virtual-object state, the entity row in the DBOS actor proposal). On Temporal, only via Continue-as-New with the snapshot as input. |
| `best_effort` tier | **Yes, by bypass** | Run short RL rollouts outside the substrate: an in-memory runner that uses the same task host and flushes our events to object storage at completion. This is arguably better than a tier inside the engine. |
| Commit-then-dispatch | **Depends** | Temporal (activity scheduled is committed before the task is dispatched) and Restate calls (journaled before send) preserve it; DBOS steps and Restate `ctx.run` record after execution → at-least-once + receiver dedupe, as the working hypothesis already accepts. |
| Per-cell control | **Yes** | One substrate deployment per cell, if the license and operations allow it (Restate BSL permits internal platforms; Temporal MIT; DBOS MIT library, Conductor proprietary). |
| Same-transaction swarm messaging | **Substrate-dependent** | Native in DBOS (Postgres) and Restate; activity-mediated in Temporal. |
| Retry classes and `outcome_unknown` | **Yes** | Implemented in the adapter and effect executors, not the engine. |

Conclusion of the section (analysis): every unique property except "single log that is both the durability record
and the RL record" can be layered. The adapter pattern (trusted substrate worker + sandboxed task host over B4) is the
key move; it keeps `HarnessHost` as the narrow waist (P12) and lets the substrate be swapped.

The adapter, sketched substrate-neutrally (the `substrate` calls map to `@DBOS.step`, a Temporal activity, or a
Restate `ctx.run` / service call; names are illustrative, not an existing API):

```python
async def run_adapter(substrate: SubstrateContext, run_id: str, specification: RunSpecification) -> None:
    """Trusted. Deterministic by construction: it only forwards between the task host and the substrate."""
    host = await TaskHostClient.attach(run_id, specification.task.code_reference)  # sandboxed, no network
    events = await host.step(RunEvent.run_created(specification))                  # pure computation, not a step
    while not any(event.is_terminal for event in events):
        validate(events, specification)                                            # schemas, sizes, ownership
        await substrate.step("emit", lambda: sink.append(run_id, events))          # our typed log, keyed by effect_id
        completions = await gather_in_log_order(
            substrate.step(effect.effect_id, lambda effect=effect: execute_effect(effect))
            for effect in requested_effects(events)
        )
        events = await host.step_many(completions)
```

On recovery the substrate re-runs `run_adapter`; recorded steps return their stored results, and the task host is
driven through the same inputs, which is exactly our replay. Waiting for a signal becomes the substrate's receive
primitive (`DBOS.recv`, a Temporal signal, a Restate awakeable or virtual-object handler).

### 2.7 Fit per use case

| Use case | Own runtime (as specified) | Comment |
|---|---|---|
| Synchronous reinforcement learning (short episodes, trainer waits for a batch) | Good, via `best_effort` | The durability machinery is mostly idle: a crashed episode is cheaper to resample than to resume. Any substrate is overkill here; an in-process runner is the right tool. |
| Asynchronous reinforcement learning (long coding episodes, policy changes mid-rollout) | Good | Resuming a 300-turn episode after a worker loss is worth it. Policy changes are handled by the recorder, not the substrate. |
| Swarms | Good within a cell | Same-transaction `send` / `spawn`; cross-cell is relayed. |
| Durable agent identities | Incomplete | No entity or activation model, no log compaction, pickle-based state tied to code versions (§2.3 items 1 and 5). Needs the actor layer (row 11), whichever substrate is chosen. |
## 3. Contenders

Each subsection separates **facts** (cited; checked on the web on 2026-09-27) from **analysis**. "Fit" is scored
against our four use cases: synchronous reinforcement learning (sync RL), asynchronous reinforcement learning
(async RL), swarms, and durable agent identities. Items marked **[unverified]** come from vendor marketing, search
snippets or secondary sources.

### 3.1 Cloudflare Durable Objects, Workflows and the Agents SDK

**Facts.**

- *Model*: a Durable Object is an addressable actor with one active instance at a time; all requests to it reach the
  same instance ([in-memory state](https://developers.cloudflare.com/durable-objects/reference/in-memory-state/)).
  *Input gates* stop other events from being delivered while a storage operation runs; *output gates* hold outgoing
  messages until pending writes are confirmed, so code need not await `put()`
  ([Cloudflare blog](https://blog.cloudflare.com/durable-objects-easy-fast-correct-choose-three/)).
- *Storage*: SQLite per object (GA 2025-04-07, 10 GB per object), a synchronous SQL API plus a key-value API, and
  30-day point-in-time recovery ([changelog](https://developers.cloudflare.com/changelog/post/2025-04-07-sqlite-in-durable-objects-ga/),
  [storage API](https://developers.cloudflare.com/durable-objects/api/storage-api/)). Each commit's write-ahead log is
  sent to 5 followers and confirmed on 3 acknowledgements; batches go to object storage every 10 s or 16 MB
  ([SQLite in Durable Objects](https://blog.cloudflare.com/sqlite-in-durable-objects/)).
- *Limits*: soft limit ~1,000 requests/s per object; 30 s CPU per request (configurable to 5 min); 128 MB memory
  per isolate, shared by co-located objects; unlimited objects per namespace
  ([limits](https://developers.cloudflare.com/durable-objects/platform/limits/)).
- *Hibernation*: an idle object hibernates after ~10 s if nothing (timers, in-flight fetches, outbound sockets)
  pins it; the WebSocket Hibernation API keeps client sockets open while the object is out of memory; on wake the
  constructor re-runs and in-memory state is gone; no shutdown hook
  ([lifecycle](https://developers.cloudflare.com/durable-objects/concepts/durable-object-lifecycle/),
  [WebSockets](https://developers.cloudflare.com/durable-objects/best-practices/websockets/)).
- *Alarms*: one alarm per object, at-least-once, up to 6 retries with backoff
  ([alarms](https://developers.cloudflare.com/durable-objects/api/alarms/)). The Agents SDK multiplexes many
  schedules (delays, dates, cron) onto that one alarm via a SQLite table
  ([schedule tasks](https://developers.cloudflare.com/agents/api-reference/schedule-tasks/)).
- *Placement*: created near the first request; location hints are best-effort; `eu` and `fedramp` jurisdictions
  enforced; objects never move after creation ([data location](https://developers.cloudflare.com/durable-objects/reference/data-location/)).
- *Workflows*: GA 2025-04-07; each instance is itself a SQLite-backed Durable Object; `step.do` memoized by step
  name, at-least-once; `step.sleep`, `step.waitForEvent` (up to 365 days); 10,000 steps by default (up to 25,000);
  50,000 concurrently *running* instances per account (waiting instances excluded); 300 creations/s per account
  ([limits](https://developers.cloudflare.com/workflows/reference/limits/), [Workflows V2](https://blog.cloudflare.com/workflows-v2/)).
  Cloudflare's own Workflows V1 control plane bottlenecked on a single account-level Durable Object and was
  re-sharded in V2 (same source).
- *Agents SDK* (MIT): one agent = one Durable Object, addressed by name ("same name… always the same instance"),
  with SQL, synced state, schedules, queues, email, human-in-the-loop via Workflows, MCP client
  ([Agents API](https://developers.cloudflare.com/agents/api-reference/agents-api/), [repository](https://github.com/cloudflare/agents)).
  *Fibers* (`runFiber`, `ctx.stash()`) register work before it runs and call `onFiberRecovered(name, snapshot)` after a
  restart — recovery hooks, **not** automatic re-execution
  ([durable execution](https://developers.cloudflare.com/agents/api-reference/durable-execution/)).
  *Project Think* (preview, 2026-04) adds sub-agents as co-located child objects ("facets") with their own SQLite,
  tree-structured sessions, and an execution ladder up to a full sandbox ([blog](https://blog.cloudflare.com/project-think/)).
- *Python*: Python Workers GA on 2026-09-21 (Pyodide in V8 isolates; native extensions "early stages"; `threading`
  not functional) ([GA post](https://blog.cloudflare.com/python-workers-ga/),
  [stdlib](https://developers.cloudflare.com/workers/languages/python/stdlib/)); Durable Objects in Python since
  2025-05 ([changelog](https://developers.cloudflare.com/changelog/post/2025-05-14-python-worker-durable-object/));
  Python Workflows beta since 2025-08 (GA **[unverified]**); the Agents SDK is TypeScript-only.
- *Self-hosting*: `workerd` is Apache-2.0, but its Durable Object storage is `none`, `inMemory` or an experimental
  `localDisk`, and "objects are always local to one instance of the runtime" — no replication, no distribution
  ([workerd.capnp](https://raw.githubusercontent.com/cloudflare/workerd/main/src/workerd/server/workerd.capnp)). A
  clustered self-hosting attempt over NFS was closed in 2026-09 as too slow
  ([workerd PR 6780](https://github.com/cloudflare/workerd/pull/6780)). Deno's `celld` (Apache-2.0) offers self-hosted
  distributed Durable Objects with ownership via conditional writes to object storage, JavaScript only, and is
  described by its author as not ready for important production systems
  ([celld](https://github.com/denoland/celld), [The Register](https://www.theregister.com/devops/2026/08/12/nodejs-creator-liberates-durable-objects-from-cloudflare-with-celld/5286954)).
- *Pricing*: requests $0.15/M, duration $12.50 per million GB-s (billed at 128 MB), SQLite rows written $1.00/M
  ([pricing](https://developers.cloudflare.com/durable-objects/platform/pricing/)).

**Analysis.** The actor model is the closest match anywhere to *durable agent identities*: a stable name, a
private database, alarms, free hibernation, sockets that outlive the process. But the managed service runs only on
Cloudflare's network (violates N2 as a primary), the only production-grade implementation is not self-hostable,
Python is Pyodide inside a 128 MB isolate (no gVisor-style sandbox of our own, no native wheels for many ML
libraries), and Workflows' per-account concurrency ceiling (50k running) is below our design point. The Agents SDK's
own durability is *checkpoint-and-recover-hook*, weaker than our replay. Environments (microVMs) would still live
elsewhere, so every tool call would cross clouds.

| Sync RL | Async RL | Swarms | Durable identities |
|---|---|---|---|
| Poor (no GPU-side locality, Python constraints) | Poor | Fair (object-to-object RPC, facets) | **Excellent model**, unacceptable hosting constraints |

**Verdict: reject as substrate; borrow the model** (§5). Revisit only if `celld` or an equivalent matures *and* adds
Python.

### 3.2 Dapr (Workflows, Actors, Dapr Agents)

**Facts.**

- CNCF Graduated (2024-11), Apache-2.0, latest runtime 1.18.4 (2026-09-09)
  ([CNCF](https://www.cncf.io/announcements/2024/11/12/cloud-native-computing-foundation-announces-dapr-graduation/),
  [releases](https://github.com/dapr/dapr/releases)).
- *Workflow engine*: embedded in the `daprd` sidecar, built on Dapr's fork of `durabletask-go`; each workflow instance
  is an internal actor with `inbox-*` and append-only `history-*` state keys; timers and retries are actor reminders
  held by the Scheduler service (embedded etcd by default); full-history replay; determinism rules like Temporal's
  ([architecture](https://docs.dapr.io/developing-applications/building-blocks/workflow/workflow-architecture/),
  [Scheduler](https://docs.dapr.io/concepts/dapr-services/scheduler/)).
- *State*: any transactional store flagged `actorStateStore`, including PostgreSQL v2
  ([actors overview](https://docs.dapr.io/developing-applications/building-blocks/actors/actors-overview/)).
- *Versioning* (patch-based `IsPatched` and name-based) arrived in 1.17 (2026-02); 1.18 (2026-06) added optional
  history signing, a `WorkflowAccessPolicy`, and Scheduler-enforced concurrency limits
  ([1.17](https://blog.dapr.io/posts/2026/02/27/dapr-v1.17-is-now-available/),
  [1.18](https://blog.dapr.io/posts/2026/06/10/dapr-v1.18-is-now-available/)).
- *Limits*: payloads bounded by `--max-body-size` (default 4 MiB); the docs call workflows unsuited to
  "latency-sensitive workloads" ([features](https://docs.dapr.io/developing-applications/building-blocks/workflow/workflow-features-concepts/)).
- *Published performance*: 10,000 workflows × 15 no-op activities on 3 replicas took 11 min 31 s in the 1.16 test
  (the same test crashed on 1.15) ([1.16 post](https://blog.dapr.io/posts/2025/09/16/dapr-v1.16-is-now-available/));
  vendor claims of "millions of concurrent agent workflows" are unverifiable per
  [InfoQ](https://www.infoq.com/news/2026/08/diagrid-catalyst-ai-agents/).
- *Actors*: virtual actors, turn-based, Placement service with consistent hashing, reminders persistent, timers not
  ([actor concepts](https://docs.dapr.io/developing-applications/building-blocks/actors/actors-features-concepts/)).
- *Python*: `dapr` 1.18.3, production-stable ([PyPI](https://pypi.org/project/dapr/)). *Dapr Agents* 1.0 GA
  (2026-03, Python only; each `DurableAgent` run is a workflow)
  ([CNCF](https://www.cncf.io/announcements/2026/03/23/general-availability-of-dapr-agents-delivers-production-reliability-for-enterprise-ai/));
  an open issue notes the whole conversation is stored in one entry so read/write volume "grows quadratically"
  ([dapr-agents #791](https://github.com/dapr/dapr-agents/issues/791)).

**Analysis.** Dapr satisfies the portability constraints (Kubernetes, Apache-2.0, Postgres, Python) better than any
contender here. But the published throughput (~14 workflows/s, ~220 activities/s on a 3-node test — our arithmetic)
is two to three orders of magnitude below a cell's ~7.5k events/s; the control plane adds a stateful etcd-based
Scheduler and a Placement service to operate; the sidecar is one more process per pod; and the SDK runs task code
in the app process with no sandbox. Its workflow-as-actor internals are nonetheless a close cousin of our design
(inbox + append-only history per instance).

| Sync RL | Async RL | Swarms | Durable identities |
|---|---|---|---|
| Poor (overhead per step) | Fair, unproven at scale | Fair (pub/sub, child workflows) | Fair (actors + reminders), state model immature |

**Verdict: reject as primary substrate** unless a spike at 25k runs per cell overturns the published numbers.

### 3.3 Microsoft Orleans (and the Durable Task family)

**Facts.**

- Orleans 10 (2026-01), latest 10.3.1, MIT, **.NET only**; no official Python
  ([releases](https://github.com/dotnet/orleans/releases), [overview](https://learn.microsoft.com/en-us/dotnet/orleans/overview)).
- *Virtual actors (grains)*: activated on demand, collected when idle (default collection age 15 min; only incoming
  calls, reminders and stream events count as activity)
  ([activation collection](https://learn.microsoft.com/en-us/dotnet/orleans/host/configuration-guide/activation-collection)).
- *Placement*: resource-optimized by default since 9.2 (weighted CPU / memory / activation count), plus random,
  prefer-local, hash, activation-count, silo-role, custom directors and placement filters; experimental *activation
  repartitioning* moves grains towards the grains they talk to; experimental *rebalancing*
  ([placement](https://learn.microsoft.com/en-us/dotnet/orleans/grains/grain-placement)).
- *Directory*: strongly consistent since 9.0 (a distributed hash table with a view-change protocol), removing the
  earlier duplicate-activation caveat ([grain directory](https://learn.microsoft.com/en-us/dotnet/orleans/implementation/grain-directory)).
- *Live grain migration* (Orleans 8): dehydrate / rehydrate in-memory state and queued messages
  ([what's new in Orleans 8](https://devblogs.microsoft.com/dotnet/whats-new-in-orleans-8/)).
- *Versioning*: `[Version(N)]` on grain interfaces, backward-compatible / strict / all-compatible strategies, and
  version selectors; an incompatible activation is deactivated and re-placed on a compatible silo
  ([grain versioning](https://learn.microsoft.com/en-us/dotnet/orleans/grains/grain-versioning/grain-versioning)).
- *Durability*: pluggable grain storage (ADO.NET incl. PostgreSQL), reminders (definition persisted, missed ticks
  skipped), `JournaledGrain` event sourcing; `Microsoft.Orleans.Journaling` (write-ahead log + checkpoint) and
  `DurableJobs` are alpha; a durable-tasks prototype is unmerged ([Orleans PR 10325](https://github.com/dotnet/orleans/pull/10325)).
- *Scale*: used in Azure, Xbox, Skype, Halo, PlayFab (vendor statement; no fresh numbers).
- *Durable Task Scheduler* (Azure-managed; Dedicated GA 2025-11, Consumption GA 2026-03) backs Microsoft Agent
  Framework "durable agents", where each agent session is a durable entity limited to **1 MB** of state including the
  conversation ([durable agents](https://learn.microsoft.com/en-us/azure/durable-task/sdks/durable-agents-microsoft-agent-framework),
  [scheduler](https://learn.microsoft.com/en-us/azure/durable-task/scheduler/durable-task-scheduler)).

**Analysis.** Orleans is the reference design for activation, placement and interface versioning of virtual actors,
and its 2026 direction (journaled durable state, durable inbox/outbox per grain) is exactly our "durable agent
identity" shape. It is disqualified by language: task and agent code are Python, and bridging every grain call to a
Python host would give us Orleans' hardest-to-operate part (the cluster) without its productivity. The Durable Task
Scheduler is Azure-only and its 1 MB entity limit rules out long conversations.

| Sync RL | Async RL | Swarms | Durable identities |
|---|---|---|---|
| n/a (.NET) | n/a | Strong model | Strong model |

**Verdict: reject; borrow placement, idle collection and versioning ideas.**

### 3.4 Akka (Akka SDK, Akka Agentic Platform)

**Facts.**

- *License*: BSL 1.1, each release converting to Apache 2.0 after three years (Akka core 2.10.22: change date
  2029-09-09); production requires a license key; free keys were announced for companies under $25M revenue
  ([license keys](https://akka.io/blog/akka-license-keys-and-no-spam-promise),
  [BSL FAQ](https://akka.io/bsl-license-faq)) — the current FAQ no longer restates that exemption **[unverified as current]**.
- *Model*: actors + Cluster Sharding + Persistence (event-sourced entities with snapshots, durable state); the Akka
  SDK adds Event Sourced Entities, Key Value Entities, Views, Workflows and Agents. Workflows persist state after each
  step transition (not replay); steps are retried and must be idempotent
  ([workflows](https://doc.akka.io/sdk/workflows.html), [agents](https://doc.akka.io/sdk/agents.html)).
- *Passivation*: idle passivation (2 min default) and active-entity limits with LRU / LFU / Window-TinyLFU strategies
  ([cluster sharding](https://doc.akka.io/libraries/akka-core/current/typed/cluster-sharding.html)).
- *Storage*: self-managed SDK services support only Postgres via R2DBC ([configuring](https://doc.akka.io/operations/configuring.html)).
- *Python*: none (Java / Scala).
- *Scale*: "benchmarked to 10 million TPS" and similar — vendor claims without methodology
  ([Akka blog](https://akka.io/blog/akka-launches-new-deployment-options-for-agentic-ai-at-scale)).

**Analysis.** Mature entity sharding and passivation, but JVM-only, commercially licensed for production, and its
workflow durability is at-least-once steps with persisted state rather than replay. Same disqualifier as Orleans.

**Verdict: reject; borrow passivation strategies (active-entity limits with frequency-aware eviction).**

### 3.5 Inngest

**Facts.**

- *Model*: event-triggered durable step functions; the handler re-runs from the top with completed `step.run`
  results injected ([execution model](https://www.inngest.com/docs/learn/how-functions-are-executed)); `sleep`,
  `wait_for_event` with a match expression, `invoke`, parallel steps; flow control with keyed concurrency,
  throttling, rate limits, debounce, priority, singleton functions
  ([flow control](https://www.inngest.com/docs/guides/flow-control)).
- *Python*: `inngest` 0.5.19 (still 0.x); client-side checkpointing (lower inter-step latency) is not available for
  Python ([inngest-py](https://github.com/inngest/inngest-py), [checkpointing](https://www.inngest.com/docs/setup/checkpointing)).
  AgentKit is TypeScript-only.
- *License / self-hosting*: server under SSPL v1 with conversion to Apache 2.0 after three years; production
  self-hosting needs Redis *and* Postgres; Helm chart exists; no automatic cleanup of Postgres runs
  ([LICENSE](https://github.com/inngest/inngest), [self-hosting](https://www.inngest.com/docs/self-hosting)).
- *Limits (cloud)*: 1,000 steps per function, 4 MiB per step, 32 MiB run state
  ([limits](https://www.inngest.com/docs/usage-limits/inngest)).
- *Scale*: "100K+ executions per second" claimed, no methodology ([home](https://www.inngest.com/)).

**Analysis.** Excellent multi-tenant flow control, the best in class for "fair queueing per tenant key". But 1,000
steps per function (a 300-turn coding episode with 2–3 steps per turn exceeds it), SSPL for the server, Redis in the
critical path, HTTP-invoked steps (latency per step) and a 0.x Python SDK make it a poor substrate for our runs.

**Verdict: reject; borrow flow-control vocabulary** (keyed concurrency, throttle, debounce, singleton).

### 3.6 Hatchet

**Facts.**

- *Model*: durable task queue + DAGs + "durable tasks"; regular tasks at-least-once; durable tasks keep an event log
  checkpointed at waits (sleep, event, child completion) and spawns, can be **evicted** while waiting, and resume by
  replaying the log; code between checkpoints must be deterministic
  ([durable tasks](https://docs.hatchet.run/v1/durable-tasks), [guarantees](https://docs.hatchet.run/v1/architecture-and-guarantees)).
- *Features*: concurrency keys as CEL expressions with round-robin / cancel strategies, dynamic rate limits,
  priority, worker affinity, multi-tenancy ([concurrency](https://docs.hatchet.run/home/concurrency)).
- *License*: MIT (whole repository) ([repository](https://github.com/hatchet-dev/hatchet)). *Python*: first-class
  SDK, 1.41.1 (2026-09-24). *Kubernetes*: official Helm charts including an HA chart
  ([high availability](https://docs.hatchet.run/self-hosting/high-availability)).
- *Storage*: Postgres as the source of truth (RabbitMQ optional); partitioned tables
  ([Postgres partitioning](https://hatchet.run/blog/postgres-partitioning)).
- *Scale*: "hundreds of tasks/sec per engine instance" on Postgres only; "high tens of thousands" with sharding by
  arrangement; lists "10,000+ tasks/sec without tuning" as a poor fit
  ([guarantees](https://docs.hatchet.run/v1/architecture-and-guarantees)); cloud bursts >5k tasks/s cited at launch
  ([Hacker News](https://news.ycombinator.com/item?id=43572733)) **[vendor claim]**. An open bug concerns durable-task
  eviction after worker loss ([issue 4969](https://github.com/hatchet-dev/hatchet/issues/4969)).

**Analysis.** The best license and deployment fit of all contenders (MIT, Postgres, Python, Helm). Its durable
tasks are the right shape for agents — eviction while waiting is our "suspended runs cost nothing". But checkpoints
exist only at waits and spawns (model and tool calls would be child tasks, each a queue round trip), the durable
feature is young, and Hatchet's own guidance puts a single Postgres-backed engine at hundreds of tasks/s against a
cell's thousands of effects/s. It is a task *queue* first.

| Sync RL | Async RL | Swarms | Durable identities |
|---|---|---|---|
| Fair (as a queue for batched rollouts) | Fair, throughput-limited | Good (children, DAGs) | Weak (no entity model) |

**Verdict: reject as the run substrate; a credible candidate for the non-hot-path queues** (rollout admission,
triggers, connectors) **if** we want a bought queue there. Keep on the spike list.

### 3.7 Golem

**Facts.**

- *Model*: durable, addressable **agents** (WebAssembly components), identified by constructor parameters; typed
  agent-to-agent RPC ([Golem 1.3](https://golem.cloud/blog/new-golem-1-3-release/),
  [Golem 1.5](https://golem.cloud/blog/golem-1-5-the-agent-runtime/)).
- *Durability*: transparent — host calls recorded in an append-only oplog and replayed; user-defined snapshots since
  1.5; exactly-once claimed for local code and agent-to-agent calls, at-least-once for remote calls in flight at a
  crash ([reliability](https://learn.golem.cloud/concepts/reliability)).
- *Languages*: Rust, TypeScript, Scala, MoonBit in 1.5; **Python dropped** after 1.2.
- *License*: Apache-2.0 → custom (2025-05) → **BSL 1.1** (2026-02), prohibiting offering Golem as a hosted
  developer platform to third parties ([LICENSE](https://github.com/golemcloud/golem/blob/main/LICENSE)).
- *Self-hosting*: Postgres, Redis (oplog primary tier) and S3; no official Helm chart
  ([Kubernetes](https://learn.golem.cloud/v1.5/deploy/kubernetes), [persistence](https://learn.golem.cloud/v1.5/operate/persistence)).
- *Scale*: no published benchmarks; cloud tier "coming soon" with 5,000 concurrent agents ([Golem Cloud](https://golem.cloud/cloud/)).

**Analysis.** Intellectually the closest to "untrusted durable agent code": WebAssembly sandboxes give isolation and
determinism together, as our gVisor task host does. But no Python, three license changes in a year, no scale
evidence, and Redis in the durability path. If tenants ever write agent code in a WebAssembly-friendly language the
idea is worth revisiting; the product is not.

**Verdict: reject; borrow "oplog of host calls + snapshot" as validation of our design.**

### 3.8 Ray

**Facts.**

- Apache-2.0, PyTorch Foundation since 2025-10, latest 2.58.0
  ([PyTorch Foundation](https://pytorch.org/blog/pytorch-foundation-welcomes-ray-to-deliver-a-unified-open-source-ai-compute-stack/)).
- *Actors are not durable*: on restart "its state will be recreated by rerunning its constructor"; "the application
  is responsible for recovering the state"; detached actors die with the cluster
  ([actor fault tolerance](https://docs.ray.io/en/latest/ray-core/fault_tolerance/actors.html)). Ray Workflows is
  deprecated ([Ray forum](https://discuss.ray.io/t/ray-workflows-deprecated/22132)).
- *Scale*: tens of thousands of actors on 2,000–10,000-node clusters
  ([Anyscale](https://www.anyscale.com/blog/how-we-scaled-ray-from-batch-inference-to-10000-node-training-clusters)).
- *RL*: eight of sixteen surveyed RL libraries use Ray as their orchestration backbone
  ([Hugging Face survey](https://huggingface.co/blog/async-rl-training-landscape)).
- *Security*: token auth only since 2.52 and off by default; actively exploited unauthenticated job submission
  ([Ray security](https://docs.ray.io/en/latest/ray-security/index.html)).

**Analysis.** Ray is where our *callers* live (trainers, rollout drivers), not a durability substrate. It is the right
tool for the synchronous-RL compute plane inside one trusted job, and wrong for durable runs, identities, or
untrusted code.

**Verdict: integrate, do not adopt.** The rollout client should be a good citizen inside Ray (our
`RolloutJobs` Python client is already that shape).

### 3.9 Rivet Actors

**Facts.** Apache-2.0 actors "built for AI agents, collaborative apps, and durable execution"; each actor gets
SQLite and a filesystem tiered to S3; workflows and queues added 2026-01; actors in Node.js / Bun, Rust, Effect.ts —
Python only as an experimental client ([repository](https://github.com/rivet-dev/rivet), [actors](https://rivet.dev/actors/)).
Self-hosted on Postgres + NATS is "production-ready for light-to-moderate workloads, up to roughly 1,000 concurrent
actors"; beyond that, FoundationDB "requires an enterprise license"
([storage](https://rivet.dev/docs/deploy/self-host/control-plane/storage/)).

**Verdict: reject** (no Python actors; open-source scale ceiling ~1,000 concurrent actors vs our ~25,000 per cell).
Borrow per-actor SQLite tiered to object storage as an option for identity memory.

### 3.10 AWS Lambda durable functions

**Facts.** Announced 2025-12 with Python and Node; checkpoint-and-replay model; 3,000 durable operations and 100 MB
checkpointed data per execution (hard); executions up to 1 year; 5M running executions per region (soft)
([announcement](https://aws.amazon.com/about-aws/whats-new/2025/12/lambda-durable-multi-step-applications-ai-workflows/),
[concepts](https://docs.aws.amazon.com/lambda/latest/dg/durable-basic-concepts.html),
[quotas](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html)). SDK Apache-2.0, service proprietary.

**Verdict: reject (AWS-only, violates N2).** Borrow its explicit hard per-execution budgets.

### 3.11 Vercel Workflow

**Facts.** GA 2026-04; Apache-2.0; `"use workflow"` / `"use step"` directives compiled into an event log; pluggable
"Worlds" (storage + queue + stream adapters) including a Postgres World that is "a reference implementation … not
optimized for scale"; Python SDK in beta ([Vercel blog](https://vercel.com/blog/a-new-programming-model-for-durable-execution),
[Postgres World](https://workflow-sdk.dev/worlds/postgres), [Python](https://workflow-sdk.dev/docs/getting-started/python)).

**Verdict: reject.** Borrow the "World" service-provider interface: a narrow storage/queue/stream contract under the
durable API — the same idea as our `RunStore` interface.

### 3.12 Agent-specific servers: LangGraph Agent Server, Letta

**Facts.** LangGraph checkpoints per super-step to Postgres (modes `exit` / `async` / `sync`), serializes runs per
thread, and its Agent Server needs an Enterprise license key for self-hosting
([checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers),
[Agent Server](https://docs.langchain.com/langsmith/agent-server), [self-hosted](https://docs.langchain.com/langsmith/self-hosted)).
Letta retired its Python server; `letta-code` (TypeScript) keeps identity, memory and conversations in Letta Cloud
while agents run on registered "computers", with memory versioned in git
([letta](https://github.com/letta-ai/letta), [letta-code](https://github.com/letta-ai/letta-code)).

**Verdict: host them, do not run on them.** LangGraph agents can be *hosted* on our platform (checkpointer pointed
at the cell's store, or as unmanaged harnesses via B18). Letta is a design reference for identities: identity and
memory separate from the compute that runs them.

### 3.13 Small Postgres-native engines: Absurd, Resonate

**Facts.** *Absurd* (Apache-2.0): durable execution as one SQL file of stored procedures, pull-based leased workers,
checkpointed steps with auto-numbered repeated names for agent loops; in production at its
author's company for ~5 months as of 2026-04; admits no partitioning
([announcement](https://lucumr.pocoo.org/2025/11/3/absurd-workflows/),
[in production](https://lucumr.pocoo.org/2026/4/4/absurd-in-production/)). *Resonate* (Apache-2.0): durable
promises ("distributed async await"), Python among five SDKs, a Postgres-only server variant, a formal
specification in TLA+ / Lean; pre-1.0, no scale evidence ([repository](https://github.com/resonatehq/resonate)).

**Analysis.** Neither is a substrate at our scale, but both show that the Postgres core of a durable engine is small
(its author puts Absurd's Python SDK at ~1,900 lines against ~170,000 for Temporal's Python SDK —
[in production](https://lucumr.pocoo.org/2026/4/4/absurd-in-production/)). That supports the claim that our Run
Store is not the expensive part of §2.2 — the expensive parts are the operator surface, versioning and correctness
testing.

**Verdict: reject as substrates; use Absurd's SQL as a reference implementation during spike S5.**

### 3.14 Others checked and dismissed

| System | Why dismissed | Source |
|---|---|---|
| Trigger.dev v4 | Tasks TypeScript-only (Python as scripts); CRIU checkpoints not in self-hosted v4 | [docs](https://trigger.dev/docs/how-it-works), [self-hosting](https://trigger.dev/docs/self-hosting/kubernetes) |
| LittleHorse | Needs Kafka; AGPL-3.0 since 2025-04 | [InfoQ](https://www.infoq.com/news/2025/04/littlehorse/) |
| Obelisk | WebAssembly workflows; AGPL-3.0; pre-release | [repository](https://github.com/obeli-sk/obelisk) |
| Flyte 2 | Kubernetes-native, Python, GA 2026-08; container-per-task granularity fits pipelines, not 300k fine-grained runs | [Union](https://www.union.ai/blog-post/flyte-2-is-generally-available-the-durable-open-source-ai-runtime) |
| Kitaru, Mastra, Convex components, Jido | Early, framework-bound, or not Python | [Kitaru](https://www.zenml.io/blog/kitaru-launch), [Jido](https://github.com/agentjido/jido) |
| Monarch (Meta) | PyTorch-native actor meshes for training, not durability | [repository](https://github.com/meta-pytorch/monarch) |
| Claude Managed Agents | Hosted only; architecture (append-only session log, stateless harness, disposable sandbox) mirrors ours | [engineering post](https://anthropic.com/engineering/managed-agents) |

**Market signal (fact):** agent frameworks now ship *adapters* to durable engines rather than engines: Pydantic AI
supports Temporal, DBOS, Prefect, Restate and AWS Lambda
([Pydantic AI](https://pydantic.dev/docs/ai/integrations/durable_execution/overview/)); the OpenAI Agents SDK
integration with Temporal reached GA in 2026-05 ([Temporal](https://temporal.io/blog/replay-2026-product-announcements)).
**Analysis:** our `HarnessHost` boundary should make our task host one more such adapter target, not a competitor to them.

## 4. Comparison matrix

Summary level. DBOS, Restate and Temporal rows use public facts only; their full evaluations are in the sibling
reports. Public facts used for those three rows: DBOS Transact Python 3.1.0 (2026-09-24) is MIT
([repository](https://github.com/dbos-inc/dbos-transact-py)) and DBOS Conductor is under a proprietary license
([Conductor license](https://www.dbos.dev/conductor-license)); the Restate server 1.7.12 (2026-09-22) is BSL 1.1 whose
additional use grant permits production use for one's own services, internal platforms, and platforms exposing their
own abstraction rather than Restate's APIs ([LICENSE](https://github.com/restatedev/restate/blob/main/LICENSE)), with
MIT SDKs including Python 1.0.5 ([sdk-python](https://github.com/restatedev/sdk-python)); the Temporal server 1.32.0
(2026-09-11) is MIT ([repository](https://github.com/temporalio/temporal)). "Split" = can untrusted task code run without holding substrate credentials or a database connection,
without an adapter.

| Option | Model | Durability | Python | License / self-host | Multi-cloud Kubernetes | Split | Scale evidence | Sync RL | Async RL | Swarms | Identities | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Own runtime** | Actor host + replayed workflows | Commit-then-dispatch log, replay, snapshots | Native (designed for it) | Ours | Yes (Postgres per cell) | **Yes** | None (spikes S5, S6 pending) | Good (`best_effort`) | Good | Good in cell | Incomplete | Build only the parts no substrate has (§1) |
| DBOS | Workflow library in app process | Step outputs recorded after step; resume from last step | Yes (Transact 3.1.0) | MIT library; Conductor proprietary | Yes (Postgres) | No (same process) | Vendor claims | Fair | Good | Good (Postgres send) | Needs actor layer | Working hypothesis; see sibling report |
| Restate | Services, virtual objects, workflows | Journal per invocation; own replicated log | Yes (SDK 1.0.x) | BSL 1.1 server, internal platforms permitted | Yes (own cluster) | Partial (no store credentials in handler) | Vendor claims | Fair | Good | Good | **Virtual objects** | See sibling report |
| Temporal | Workflows (+ entity workflows) | Full history replay | Yes (sandboxed SDK) | MIT server; Cloud $25–50 per million Actions | Yes (own cluster, Cassandra / SQL) | No (worker holds namespace credentials) | Large production base | Fair | Good | Fair | Entity workflows + Continue-as-New | See sibling report |
| Cloudflare Durable Objects / Agents | Actor per object | Quorum write-ahead log per object; fibers | Pyodide; Agents SDK TypeScript | Managed only; `workerd` not distributed | **No** | Isolates (Cloudflare-run) | "Millions" of objects (vendor) | Poor | Poor | Fair | **Best model** | Reject; borrow |
| Dapr | Workflow-as-actor sidecar | Full history replay | Yes | Apache-2.0, CNCF | Yes | No | ~14 workflows/s published test | Poor | Fair | Fair | Fair | Reject unless spike overturns |
| Orleans | Virtual actors | Pluggable storage; journaling alpha | No (.NET) | MIT | Yes | n/a | Halo, Xbox (vendor) | n/a | n/a | Strong model | Strong model | Reject; borrow |
| Akka | Sharded entities, workflows | Event sourcing; step state persisted | No (JVM) | BSL + license key | Yes | n/a | Vendor claims | n/a | n/a | Strong model | Strong model | Reject |
| Inngest | Event-driven step functions | Memoized steps, re-run from top | 0.x | SSPL server | Yes (Redis + Postgres) | No | Vendor claims | Poor | Poor (1,000 steps) | Fair | Weak | Reject; borrow flow control |
| Hatchet | Task queue + durable tasks | Checkpoints at waits/spawns, eviction | Yes | MIT | Yes (Postgres) | No | Hundreds/s per engine (vendor) | Fair | Fair | Good | Weak | Reject for runs; queue candidate |
| Golem | Wasm durable agents | Oplog of host calls + snapshots | **No** | BSL 1.1 | DIY manifests | Yes (Wasm) | None | Poor | Poor | Fair | Good model | Reject |
| Ray | Actors (non-durable) | None for actor state | Yes | Apache-2.0 | Yes (KubeRay) | No | 10k-node clusters | **Compute plane** | Caller side | n/a | No | Integrate as caller |
| Rivet | Actors with SQLite | Per-actor SQLite tiered to S3 | Client only | Apache-2.0; FoundationDB enterprise | Yes | No | ~1,000 actors on open-source Postgres | Poor | Poor | Fair | Good model | Reject |
| AWS Lambda durable functions | Checkpoint + replay | Managed checkpoint log | Yes | Proprietary service | **No** | Lambda isolation | 5M executions/region quota | Poor | Fair | Fair | Weak (3,000 operations) | Reject |
| Vercel Workflow | Directive-compiled workflows | Event log via "Worlds" | Beta | Apache-2.0 | Postgres World not production-grade | No | Vendor claims | Poor | Poor | Fair | Weak | Reject |

## 5. Ideas to borrow

Each is analysis; the source of the idea is cited in §3.

1. **Name → identity → single live activation** (Cloudflare Agents, Orleans, Golem). Durable agent identities should
   be addressed by a stable name (`agents/{tenant}/{name}`), with the platform guaranteeing one live activation. In
   our terms: the identity's partition lease already gives single ownership; add a name → `run_id` (or entity) map
   with signal-with-start, which the "run keys" proposal already sketches.
2. **Hibernation as a first-class, free state** (Cloudflare). Suspended identities cost only rows. Keep external
   connections (Slack sockets, WebSockets to a UI) *outside* the activation, in a connector tier that survives
   eviction and carries a small per-connection attachment — Cloudflare's 16 KiB `serializeAttachment` is the model.
3. **Output gates** (Cloudflare). "Hold every outbound message until the write that caused it is durable" is our P5
   stated as a runtime mechanism. Applying it to the *ephemeral token-streaming side channel* and to `run.emit`
   connectors gives a clean rule: tokens may stream early (marked provisional), but emitted outputs are released only
   after commit.
4. **One alarm per actor, schedules multiplexed on top** (Cloudflare Agents). Store per-identity schedules in the
   identity's own state and keep only the *next* fire time in the cell's `timers` table. That bounds the timer table
   to one row per identity and removes the timer-storm bloat risk of §2.3 item 3.
5. **Idle collection policy by type and by *incoming* activity** (Orleans, Akka). Evict suspended runs by policy
   (collection age per program type; memory-pressure shedding with a frequency-aware policy such as Window-TinyLFU),
   and count only incoming messages and timers as activity. Our runtime evicts on `run.suspended`; identities that
   chat every few seconds would thrash without an idle age.
6. **Version-compatible placement** (Orleans). Route an activation to a task host that advertises a compatible
   interface version, and re-place an activation that lands on an incompatible host. For identities this replaces
   "pin forever" with "upgrade at the next activation", which fixes §2.3 item 1.
7. **Placement filters and locality repartitioning** (Orleans). Prefer placing a run's task host near its environment
   host and recorder instance; move chatty swarm members together. Useful only after the basics work.
8. **Durable inbox/outbox per actor with journaled state** (Orleans 2026 direction). Identity state as a journal of
   typed records plus periodic checkpoints — not a single growing blob. Dapr Agents' "quadratic" single-entry
   conversation state and Microsoft's 1 MB session limit are the cautionary tales.
9. **Hard per-execution budgets** (AWS Lambda durable functions: 3,000 operations, 100 MB; Temporal: 51,200 events /
   50 MB). Make our limits explicit and enforced (events per run, bytes per run, snapshot size) with compaction for
   identities, so one run cannot degrade a cell's Postgres.
10. **Keyed flow control vocabulary** (Inngest, Hatchet). Concurrency keys (per tenant, per repository), throttles,
    debounce and singleton semantics are what "per-tenant quotas" and "rollout admission" need; adopt the vocabulary
    for the Control API and rollout controller.
11. **Pluggable backend interface** (Vercel "Worlds", our `RunStore`). Keep the storage/queue/stream interface narrow
    enough that the backend can be swapped for a bought substrate — the adapter of §2.6 is the same idea one level up.
12. **Identity separate from compute** (Letta, Claude Managed Agents). An identity's memory, credentials references
    and mailbox are durable records; the task host and the environment are disposable and re-attached on activation.
13. **Infrastructure-aware retry** (Flyte 2). When an environment effect fails with out-of-memory or preemption, retry
    with a changed resource request rather than identically — a natural extension of our retry classes.

## 6. Open questions and spike tests

| # | Question / spike | Decides |
|---|---|---|
| X1 | **Adapter spike**: run the §2.6 adapter on the working-hypothesis substrate (DBOS) with a gVisor task host; measure added latency per step (target ≤ 5 ms p99) and replay cost after a kill | Whether the untrusted split is layerable at acceptable cost |
| X2 | **Cell load test** on the chosen substrate at 25k concurrent runs, ~7.5k events/s sustained, 20k/s peak, on one managed Postgres per cell (extends S5) | Whether any bought substrate fits a cell; the same harness then measures our own Run Store |
| X3 | **`best_effort` runner outside the substrate**: in-process runner with buffered log flush for synchronous RL; compare throughput and cost with durable runs | Whether sync RL should bypass the substrate entirely (recommended) |
| X4 | **Identity activation model**: prototype an identity as entity row + per-message activation (DBOS actor proposal) vs Restate virtual object; measure activation latency, idle cost, log growth after 10k messages | Actor layer design; whether Restate's model justifies its BSL server |
| X5 | **Versioning for identities**: can activation-time upgrade (idea 6) be expressed on the chosen substrate without pinning old workers forever? | Versioning policy for long-lived runs |
| X6 | **Failover durability**: force a managed-Postgres failover under load; verify no acknowledged append is lost and no `effect_id` is reused with different arguments (§2.3 item 11) | Required database configuration; receiver argument-hash check |
| X7 | **Poison run containment**: inject a task that segfaults its host on replay; verify quarantine before it crashes a second host | Crash-loop detection requirements |
| X8 | **Hatchet as queue**: evaluate Hatchet only for rollout admission, triggers and connectors | Whether to buy the non-run queue |
| Q-a | Does Q10 (external tenants write task code) resolve to "yes"? If yes, the split (§2.4 item 1) is mandatory and substrates without an adapter path are excluded | Isolation model |
| Q-b | Restate BSL: does a platform where external tenants register task code (Q10 = yes) stay within "internal platform" / "abstraction layer" grants? Our reading: yes, because tenants use our API, not Restate's — needs legal review | Restate eligibility |
| Q-c | What engineering headcount is available for the substrate for the next 18 months? Below ~4 engineers, building §2.2 rows 1, 2, 4, 6, 8, 9 and 13 is not credible | Build vs buy |

## 7. Sources

All web sources accessed 2026-09-27, grouped by the section that first cites them. Internal design documents are linked inline.

**§2.3 Hard edge cases engines took years to get right**

- [Temporal changelog](https://temporal.io/changelog/worker-versioning-continue-as-new-worker-controller)
- [Worker Versioning docs](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning)
- [Temporal Python sandbox](https://docs.temporal.io/develop/python/python-sdk-sandbox)
- [limits](https://docs.temporal.io/workflow-execution/limits)
- [Temporal visibility](https://docs.temporal.io/visibility)
- [DBOS workflow management](https://docs.dbos.dev/python/tutorials/workflow-management)
- [Temporal priority and fairness](https://docs.temporal.io/develop/task-queue-priority-fairness)

**§2.5 Operating cost**

- [Temporal pricing](https://docs.temporal.io/cloud/pricing)

**§3.1 Cloudflare Durable Objects, Workflows and the Agents SDK**

- [in-memory state](https://developers.cloudflare.com/durable-objects/reference/in-memory-state/)
- [Cloudflare blog](https://blog.cloudflare.com/durable-objects-easy-fast-correct-choose-three/)
- [changelog](https://developers.cloudflare.com/changelog/post/2025-04-07-sqlite-in-durable-objects-ga/)
- [storage API](https://developers.cloudflare.com/durable-objects/api/storage-api/)
- [SQLite in Durable Objects](https://blog.cloudflare.com/sqlite-in-durable-objects/)
- [limits](https://developers.cloudflare.com/durable-objects/platform/limits/)
- [lifecycle](https://developers.cloudflare.com/durable-objects/concepts/durable-object-lifecycle/)
- [WebSockets](https://developers.cloudflare.com/durable-objects/best-practices/websockets/)
- [alarms](https://developers.cloudflare.com/durable-objects/api/alarms/)
- [schedule tasks](https://developers.cloudflare.com/agents/api-reference/schedule-tasks/)
- [data location](https://developers.cloudflare.com/durable-objects/reference/data-location/)
- [limits](https://developers.cloudflare.com/workflows/reference/limits/)
- [Workflows V2](https://blog.cloudflare.com/workflows-v2/)
- [Agents API](https://developers.cloudflare.com/agents/api-reference/agents-api/)
- [repository](https://github.com/cloudflare/agents)
- [durable execution](https://developers.cloudflare.com/agents/api-reference/durable-execution/)
- [blog](https://blog.cloudflare.com/project-think/)
- [GA post](https://blog.cloudflare.com/python-workers-ga/)
- [stdlib](https://developers.cloudflare.com/workers/languages/python/stdlib/)
- [changelog](https://developers.cloudflare.com/changelog/post/2025-05-14-python-worker-durable-object/)
- [workerd.capnp](https://raw.githubusercontent.com/cloudflare/workerd/main/src/workerd/server/workerd.capnp)
- [workerd PR 6780](https://github.com/cloudflare/workerd/pull/6780)
- [celld](https://github.com/denoland/celld)
- [The Register](https://www.theregister.com/devops/2026/08/12/nodejs-creator-liberates-durable-objects-from-cloudflare-with-celld/5286954)
- [pricing](https://developers.cloudflare.com/durable-objects/platform/pricing/)

**§3.2 Dapr (Workflows, Actors, Dapr Agents)**

- [CNCF](https://www.cncf.io/announcements/2024/11/12/cloud-native-computing-foundation-announces-dapr-graduation/)
- [releases](https://github.com/dapr/dapr/releases)
- [architecture](https://docs.dapr.io/developing-applications/building-blocks/workflow/workflow-architecture/)
- [Scheduler](https://docs.dapr.io/concepts/dapr-services/scheduler/)
- [actors overview](https://docs.dapr.io/developing-applications/building-blocks/actors/actors-overview/)
- [1.17](https://blog.dapr.io/posts/2026/02/27/dapr-v1.17-is-now-available/)
- [1.18](https://blog.dapr.io/posts/2026/06/10/dapr-v1.18-is-now-available/)
- [features](https://docs.dapr.io/developing-applications/building-blocks/workflow/workflow-features-concepts/)
- [1.16 post](https://blog.dapr.io/posts/2025/09/16/dapr-v1.16-is-now-available/)
- [InfoQ](https://www.infoq.com/news/2026/08/diagrid-catalyst-ai-agents/)
- [actor concepts](https://docs.dapr.io/developing-applications/building-blocks/actors/actors-features-concepts/)
- [PyPI](https://pypi.org/project/dapr/)
- [CNCF](https://www.cncf.io/announcements/2026/03/23/general-availability-of-dapr-agents-delivers-production-reliability-for-enterprise-ai/)
- [dapr-agents #791](https://github.com/dapr/dapr-agents/issues/791)

**§3.3 Microsoft Orleans (and the Durable Task family)**

- [releases](https://github.com/dotnet/orleans/releases)
- [overview](https://learn.microsoft.com/en-us/dotnet/orleans/overview)
- [activation collection](https://learn.microsoft.com/en-us/dotnet/orleans/host/configuration-guide/activation-collection)
- [placement](https://learn.microsoft.com/en-us/dotnet/orleans/grains/grain-placement)
- [grain directory](https://learn.microsoft.com/en-us/dotnet/orleans/implementation/grain-directory)
- [what's new in Orleans 8](https://devblogs.microsoft.com/dotnet/whats-new-in-orleans-8/)
- [grain versioning](https://learn.microsoft.com/en-us/dotnet/orleans/grains/grain-versioning/grain-versioning)
- [Orleans PR 10325](https://github.com/dotnet/orleans/pull/10325)
- [durable agents](https://learn.microsoft.com/en-us/azure/durable-task/sdks/durable-agents-microsoft-agent-framework)
- [scheduler](https://learn.microsoft.com/en-us/azure/durable-task/scheduler/durable-task-scheduler)

**§3.4 Akka (Akka SDK, Akka Agentic Platform)**

- [license keys](https://akka.io/blog/akka-license-keys-and-no-spam-promise)
- [BSL FAQ](https://akka.io/bsl-license-faq)
- [workflows](https://doc.akka.io/sdk/workflows.html)
- [agents](https://doc.akka.io/sdk/agents.html)
- [cluster sharding](https://doc.akka.io/libraries/akka-core/current/typed/cluster-sharding.html)
- [configuring](https://doc.akka.io/operations/configuring.html)
- [Akka blog](https://akka.io/blog/akka-launches-new-deployment-options-for-agentic-ai-at-scale)

**§3.5 Inngest**

- [execution model](https://www.inngest.com/docs/learn/how-functions-are-executed)
- [flow control](https://www.inngest.com/docs/guides/flow-control)
- [inngest-py](https://github.com/inngest/inngest-py)
- [checkpointing](https://www.inngest.com/docs/setup/checkpointing)
- [LICENSE](https://github.com/inngest/inngest)
- [self-hosting](https://www.inngest.com/docs/self-hosting)
- [limits](https://www.inngest.com/docs/usage-limits/inngest)
- [home](https://www.inngest.com/)

**§3.6 Hatchet**

- [durable tasks](https://docs.hatchet.run/v1/durable-tasks)
- [guarantees](https://docs.hatchet.run/v1/architecture-and-guarantees)
- [concurrency](https://docs.hatchet.run/home/concurrency)
- [repository](https://github.com/hatchet-dev/hatchet)
- [high availability](https://docs.hatchet.run/self-hosting/high-availability)
- [Postgres partitioning](https://hatchet.run/blog/postgres-partitioning)
- [Hacker News](https://news.ycombinator.com/item?id=43572733)
- [issue 4969](https://github.com/hatchet-dev/hatchet/issues/4969)

**§3.7 Golem**

- [Golem 1.3](https://golem.cloud/blog/new-golem-1-3-release/)
- [Golem 1.5](https://golem.cloud/blog/golem-1-5-the-agent-runtime/)
- [reliability](https://learn.golem.cloud/concepts/reliability)
- [LICENSE](https://github.com/golemcloud/golem/blob/main/LICENSE)
- [Kubernetes](https://learn.golem.cloud/v1.5/deploy/kubernetes)
- [persistence](https://learn.golem.cloud/v1.5/operate/persistence)
- [Golem Cloud](https://golem.cloud/cloud/)

**§3.8 Ray**

- [PyTorch Foundation](https://pytorch.org/blog/pytorch-foundation-welcomes-ray-to-deliver-a-unified-open-source-ai-compute-stack/)
- [actor fault tolerance](https://docs.ray.io/en/latest/ray-core/fault_tolerance/actors.html)
- [Ray forum](https://discuss.ray.io/t/ray-workflows-deprecated/22132)
- [Anyscale](https://www.anyscale.com/blog/how-we-scaled-ray-from-batch-inference-to-10000-node-training-clusters)
- [Hugging Face survey](https://huggingface.co/blog/async-rl-training-landscape)
- [Ray security](https://docs.ray.io/en/latest/ray-security/index.html)

**§3.9 Rivet Actors**

- [repository](https://github.com/rivet-dev/rivet)
- [actors](https://rivet.dev/actors/)
- [storage](https://rivet.dev/docs/deploy/self-host/control-plane/storage/)

**§3.10 AWS Lambda durable functions**

- [announcement](https://aws.amazon.com/about-aws/whats-new/2025/12/lambda-durable-multi-step-applications-ai-workflows/)
- [concepts](https://docs.aws.amazon.com/lambda/latest/dg/durable-basic-concepts.html)
- [quotas](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html)

**§3.11 Vercel Workflow**

- [Vercel blog](https://vercel.com/blog/a-new-programming-model-for-durable-execution)
- [Postgres World](https://workflow-sdk.dev/worlds/postgres)
- [Python](https://workflow-sdk.dev/docs/getting-started/python)

**§3.12 Agent-specific servers: LangGraph Agent Server, Letta**

- [checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)
- [Agent Server](https://docs.langchain.com/langsmith/agent-server)
- [self-hosted](https://docs.langchain.com/langsmith/self-hosted)
- [letta](https://github.com/letta-ai/letta)
- [letta-code](https://github.com/letta-ai/letta-code)

**§3.13 Small Postgres-native engines: Absurd, Resonate**

- [announcement](https://lucumr.pocoo.org/2025/11/3/absurd-workflows/)
- [in production](https://lucumr.pocoo.org/2026/4/4/absurd-in-production/)
- [repository](https://github.com/resonatehq/resonate)

**§3.14 Others checked and dismissed**

- [docs](https://trigger.dev/docs/how-it-works)
- [self-hosting](https://trigger.dev/docs/self-hosting/kubernetes)
- [InfoQ](https://www.infoq.com/news/2025/04/littlehorse/)
- [repository](https://github.com/obeli-sk/obelisk)
- [Union](https://www.union.ai/blog-post/flyte-2-is-generally-available-the-durable-open-source-ai-runtime)
- [Kitaru](https://www.zenml.io/blog/kitaru-launch)
- [Jido](https://github.com/agentjido/jido)
- [repository](https://github.com/meta-pytorch/monarch)
- [engineering post](https://anthropic.com/engineering/managed-agents)
- [Pydantic AI](https://pydantic.dev/docs/ai/integrations/durable_execution/overview/)
- [Temporal](https://temporal.io/blog/replay-2026-product-announcements)

**§4. Comparison matrix**

- [repository](https://github.com/dbos-inc/dbos-transact-py)
- [Conductor license](https://www.dbos.dev/conductor-license)
- [LICENSE](https://github.com/restatedev/restate/blob/main/LICENSE)
- [sdk-python](https://github.com/restatedev/sdk-python)
- [repository](https://github.com/temporalio/temporal)
