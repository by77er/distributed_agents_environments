# Run Store

Status: **Proposed** · Boundaries B2, B3 · See [ADR-0002](../../decisions/0002-postgres-run-store.md), [ADR-0003](../../decisions/0003-log-is-the-outbox.md)

## Purpose

The durable heart of a cell: every run's event log, partition leases, inboxes, timers, and the minimal
materialized run state needed for scheduling. Everything that must survive a worker crash lives here.

## Owns / does not own

| Owns | Does not own |
|---|---|
| Run event logs (append, read, archive) | Task and agent state (in task hosts, derived from the log) |
| Partition leases and fencing epochs | Environment registry (Environment Manager) |
| Inboxes (signals, messages, long completions) | Recorder session trees (object storage via recorder) |
| Timers | Blob content (object storage) |
| Run rows: status, partition, labels, parent | Cross-cell anything |

## Boundaries

| Direction | Counterpart | What |
|---|---|---|
| provides → | Runtime (B3) | append, read, leases, inbox consume, timers, snapshots |
| provides → | Control API (B2) | create run, deliver signal, query runs, stream events |
| provides → | Timer relay | due timers |
| consumes ← | Object storage | archived log segments, state snapshots |

## Interface

Semantic interface. The Postgres reference implementation below is one implementation (P1); a different
backend MUST provide identical semantics.

```proto
service RunStore {
  // Runs
  rpc CreateRun(CreateRunRequest) returns (CreateRunResult);
  //   {run_id, request_id, created_event, labels, parent?} → OK | ALREADY_EXISTS(run_id)
  //   Idempotent on request_id. Writes run row + seq 0 atomically.

  // Log
  rpc Append(AppendRequest) returns (AppendResult);
  rpc Read(ReadRequest) returns (stream RunEvent);          // {run_id, from_seq, to_seq?} — transparently reads archive
  rpc Watch(WatchRequest) returns (stream RunEvent);        // for Control API event streams

  // Leases
  rpc AcquirePartition(AcquireRequest) returns (Lease);     // {partition, worker_id, ttl} → Lease{epoch} | HELD(owner, expires_at)
  rpc RenewLeases(RenewRequest) returns (RenewResult);      // batch: [(partition, epoch)] → per-partition OK | LOST
  rpc ReleasePartition(ReleaseRequest) returns (Empty);

  // Scheduling views (per partition, used on lease acquisition)
  rpc ActiveRuns(PartitionReference) returns (stream RunRow);     // status ∈ {pending, running}
  rpc RunsWithInbox(PartitionReference) returns (stream RunReference);  // suspended runs with undelivered inbox items
  rpc DueTimers(DueTimersRequest) returns (stream Timer);   // {partition?, now, limit} — SKIP LOCKED semantics

  // Inbox (external writers: Control API, relays)
  rpc Deliver(DeliverRequest) returns (DeliverResult);      // {target_run_id, inbox_key, item} → OK | DUPLICATE

  // State snapshots (optional optimization for long runs)
  rpc PutSnapshot(SnapshotRequest) returns (Empty);         // {run_id, at_seq, code_reference, blob_reference}
  rpc GetSnapshot(RunReference) returns (Snapshot);               // latest

  // Lifecycle
  rpc Archive(RunReference) returns (Empty);                      // terminal runs → object storage segment
}

message AppendRequest {
  string   run_id          = 1;
  uint32   partition       = 2;
  uint64   epoch           = 3;   // fencing token
  uint64   expected_seq    = 4;   // seq the first event will receive
  repeated RunEvent events = 5;   // contiguous
  SideWrites side_writes   = 6;   // all applied atomically with the events
}

message SideWrites {
  RunStatus         new_status     = 1;  // materialized status, if changed
  repeated uint64   consume_inbox  = 2;  // inbox items consumed into this append
  repeated InboxItem deliver       = 3;  // intra-cell send / spawn / child.completed to other runs
  repeated CreateRunRequest spawn  = 4;  // intra-cell child runs
  repeated Timer    timers         = 5;  // timer.requested
  repeated uint64   cancel_timers  = 6;
}

message AppendResult {
  oneof r {
    uint64 next_seq       = 1;  // OK
    uint64 fenced_epoch   = 2;  // FENCED: current epoch
    uint64 current_seq    = 3;  // SEQ_CONFLICT
  }
}
```

## Semantics & guarantees

- **Atomicity**: an `Append` and all its `SideWrites` commit together or not at all — including writes to other
  runs' inboxes and child run creation in the same cell. This is what makes intra-cell `send`/`spawn` free of
  delivery machinery.
- **Fencing**: `Append` checks `epoch == lease.epoch` for the run's partition inside the transaction.
- **Linearizable per run**: `seq` is gapless and strictly ordered; `Read` after a successful `Append` returns it.
- **Inbox exactly-once consumption**: an inbox item is consumed by exactly one `Append` (the one that lists it
  in `consume_inbox` and records the corresponding `signal.received`/`child.completed`/… event).
- **Keyed inserts**: `Deliver`, `spawn`, and timers are keyed (`inbox_key = effect_id` or `request_id`) so relay
  retries are no-ops.
- **Archive transparency**: `Read` returns the same events whether hot or archived.

## State & durability

Postgres reference schema (per cell):

```sql
runs        (run_id PK, partition, status, parent_run_id, labels jsonb, head_seq, created_at, updated_at)
            -- index (partition, status) WHERE status IN ('pending','running')
events      (run_id, seq, type, schema_version, writer_epoch, committed_at, payload bytea,
             PRIMARY KEY (run_id, seq))  PARTITION BY RANGE (committed_at)   -- drop partitions, never DELETE
leases      (partition PK, owner, epoch, expires_at)
inbox       (run_id, inbox_id, inbox_key UNIQUE, kind, payload, delivered_at, consumed_seq NULL)
            -- index (run_id) WHERE consumed_seq IS NULL
timers      (timer_id PK, run_id, partition, fire_at, tag, effect_id UNIQUE, fired bool)
            -- index (fire_at) WHERE NOT fired
snapshots   (run_id, at_seq, code_reference, blob_uri, PRIMARY KEY (run_id, at_seq))
archive     (run_id PK, segment_uri, last_seq)
```

- Durability: synchronous commit to a multi-AZ managed Postgres. Replicas serve `Watch` and queries.
- Hot set: active + recently terminated runs. Terminal runs are archived to object storage (columnar segments)
  after a grace period; time partitions are then dropped.

## Failure modes

| Failure | Effect | Mitigation |
|---|---|---|
| Primary failover | Appends stall for the failover window, cell-wide | Workers retry; leases live in the store so ownership is unchanged; runs wait (their turns take seconds anyway) |
| Hot partition (one very chatty run) | Contention on one row range | Per-run in-flight effect limits; event batching |
| Table bloat | Write amplification | Append-only events + time partitioning; inbox/timer rows deleted in bulk |
| Connection exhaustion | Failed appends | PgBouncer; workers per cell are few (tens) |

## Scale envelope

Design point ~7.5k events/s per cell, sized for ~20k/s peak; ~60% of appends carry side-writes. If a cell needs
≫50k appends/s, shrink the cell before changing the store (see ADR-0002).

## Build vs buy

Buy managed Postgres (RDS/Aurora, Cloud SQL/AlloyDB, Azure Flexible Server). Build the schema and a thin client
library. Postgres-compatible distributed SQL (YugabyteDB, CockroachDB, Aurora DSQL) is the escape hatch if a
single primary per cell becomes unacceptable — verify `SKIP LOCKED` and transaction-size support first.

## Open questions

- Archive format: Parquet segments per run vs per time window.
