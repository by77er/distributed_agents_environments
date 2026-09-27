# Runtime

Status: **Proposed** · Boundaries B3, B4, B5, B7, B9, B10 · See [ADR-0004](../../decisions/0004-durable-actor-runtime.md), [ADR-0013](../../decisions/0013-replay-durability.md)

## Purpose

Hosts runs as durable actors. A runtime worker leases partitions, keeps each owned run loaded in a task host,
feeds inputs to it, persists the resulting events, and performs the effects they request. It is the only component
that talks to task hosts, and the only one that turns task code's requests into real side effects.

## Owns / does not own

| Owns | Does not own |
|---|---|
| Partition leasing and failover | What the task and agent do (task host) |
| Per-run mailboxes; serial step execution | Imported tool execution (Tool Router) |
| Commit-then-dispatch of effects | Sampling, tokens (model endpoint) |
| Pending-effect tracking, deadlines, re-dispatch | Environment lifecycle implementation (Environment Manager) |
| Validating effects: schemas, sizes, environment ownership | Durable storage (Run Store) |
| Environment client: attachments, tokens, dispatch to envlets | |
| Routing runs to task hosts serving their code version | |
| Opening recorder sessions; resolving tools; backstop cleanup of owned environments | |

## Boundaries

| Direction | Counterpart | Contract |
|---|---|---|
| consumes | Run Store (B3) | [run-store](../run-store/README.md) |
| consumes | Task host (B4) | [harness](../harness/README.md#interface-harnesshost-b4) |
| consumes | Model endpoint (B5) | [model-endpoint](../../contracts/model-endpoint.md) |
| consumes | Tool Router (B7) | [tool-router](../tool-router/README.md#interface) |
| consumes | envlet → envd (B9) | [env-api](../environments/env-api.md) |
| consumes | Environment Manager (B10) | [environments](../environments/README.md#interface) |
| provides | Nudge endpoint (intra-cell) | below |
| provides | Admin endpoint (intra-cell) | below |

## Interface

The runtime's own surface is small; its behavior is defined by the loop.

```proto
service RuntimeWorker {
  rpc Nudge(NudgeRequest) returns (Empty);            // {run_id}: inbox has items; best-effort, idempotent
  rpc Drain(DrainRequest) returns (Empty);            // release all partitions gracefully (deploys)
  rpc Inspect(RunReference) returns (RunDebugView);   // mailbox depth, pending effects, host placement
}

// Partition → worker directory, for routing nudges and completions:
//   leases table in the Run Store (owner column) cached by Control API and relays.
```

### The loop (normative)

```
on input(run_id, event):                            # completion | inbox item | timer | cancel
  enqueue(mailbox[run_id], event)                   # processed serially per run

process(run_id, event):
  host ← placement[run_id] or load(run_id)          # load = pick a host for the run's code_reference;
                                                    #        host.Load(snapshot?, events)
  if event.effect_id ∈ terminal_effects(run_id): drop   # deduplicate (terminal-once)
  events ← host.Step(run_id, event)                 # host advances speculatively
  validate(events)                                  # emittable types, sizes, schemas, environment ownership
  result ← store.Append(run_id, partition, epoch, seq, [event] ++ events, side_writes(events))
  FENCED        → host.Evict(all runs in partition); drop_partition(partition); return
  SEQ_CONFLICT  → host.Evict(run_id); return        # reloaded on next input
  for effect in effects(events): dispatch(effect)   # eager; bounded concurrency
```

Validation failures that are the model's fault are shown to the model, not dispatched: imported-tool arguments
that fail `input_schema` produce `tool.requested` + `tool.failed{INVALID_ARGUMENTS}` in the same append.
Validation failures that are the task's fault (unknown environment, invalid observation) fail the run.

## Semantics & guarantees

- **Serial per run**, concurrent across runs. One worker hosts thousands of runs because task code never blocks
  on I/O.
- **Commit-then-dispatch** for `durable` runs; buffered for `best_effort` (see
  [delivery-semantics](../../architecture/delivery-semantics.md#run-durability-tiers)).
- **Speculative steps**: a task host advances when `Step` returns; if the append fails the runtime evicts the run
  from the host, and the next input replays it from the log.
- **Partition ownership**: workers aim to hold `P / live_workers` partitions each. Acquisition is competitive via
  the Run Store; a worker over target releases partitions at quiescent points. No coordinator.
- **Lease renewal** is batched: one `RenewLeases` call per interval for all owned partitions. Default TTL 15 s,
  renew every 5 s. A worker that fails to renew stops dispatching new effects for those partitions immediately.
- **Takeover**: on acquiring a partition, load `ActiveRuns` and `RunsWithInbox` lazily (on first input or via a
  background sweep), and re-dispatch pending effects per retry class.
- **Suspension**: runs that emitted `run.suspended` with no pending effects are evicted from their task host;
  they are woken by `Nudge`, a timer, or the takeover sweep.
- **Run start** (runtime-internal, invisible to task code): `session.open` per recorded model slot the task
  declares, tool resolution (`@tool` methods reported by the host + imports via the tool router →
  `tools.resolved`). The first `Step` (`run.created`)
  runs after these are committed.
- **Environment client**: dispatches `environment.lifecycle` to the Environment Manager and `environment.call` to
  the environment's envlet (endpoint from `Connect`, cached), with the run's attachment token. Environments a run
  creates are attached to it automatically (`environment.attached`).
- **Run completion**: on a terminal event, the runtime closes recorder sessions and destroys every environment the
  run still owns (`environment.released`) — a backstop for tasks whose `teardown` did not.
- **Code versions**: runs are routed to task hosts serving their `code_reference`; `code.upgraded` moves a run to
  new code at a resumable point.

## State & durability

None durable. Caches: run placement on task hosts, materialized contexts per (run, slot), pending effect table with
deadlines (timer wheel), environment endpoints. All rebuilt from the log.

## Failure modes

| Failure | Effect |
|---|---|
| Worker crash | Its partitions expire (≤ TTL); other workers acquire; runs replay from log; pending effects re-dispatched |
| Worker partitioned from store | Cannot renew → stops dispatching → deposed by fencing; no split-brain effects |
| Task host crash | Worker reloads affected runs on another host (replay) |
| Task code error | `run.failed{TASK_ERROR}` (after `teardown`); other runs unaffected |
| Replay divergence | `run.failed{NON_DETERMINISM}`; run quarantined |
| Slow endpoint | Dispatch queue backs up; effects are already committed; backpressure via per-worker limits |

## Scale envelope

~25k runs per cell over 10–50 workers → ~500–2,500 runs per worker; ~1k steps/s per worker at the design point.
Task hosts are co-located with workers (Python processes, one per core).

## Build vs buy

Build. Temporal / Restate / DBOS solve general workflows; we have one fixed loop and the log schema is our product
(see ADR-0004). Proposed language: Go (runtime) with Python task hosts (ADR-0011).

## Open questions

- Priority between runs on one worker (e.g. interactive coding runs vs reinforcement-learning rollouts) — FIFO
  initially.
- Task host placement: one host pool per code version, or multi-version hosts loading packages on demand?
