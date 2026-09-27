# 0002 — Postgres per cell for the run store

Status: **Proposed** · Date: 2026-09-26

## Context

The run store must do: fenced appends (check lease epoch + insert atomically), ordered replay, scheduling queries
(runnable runs, due timers, inbox wakeups), multi-run atomic writes (spawn, send), and ad-hoc operational
queries. It must be available on every major cloud (N2).

## Decision

Managed Postgres, one primary per cell, behind a `RunStore` interface. Append-only events partitioned by time;
terminal runs archived to object storage.

## Consequences

- Fencing, queues (`SKIP LOCKED`), timers and intra-cell `send`/`spawn` are single transactions — no sagas, no
  CDC-derived indexes, no delivery machinery inside a cell.
- Portable: RDS/Aurora, Cloud SQL/AlloyDB, Azure Flexible Server.
- Costs: a primary failover stalls one cell's appends for its duration; per-cell write ceiling (keep cells small
  enough); vacuum/partition hygiene is required.
- Cost is comparable to alternatives at this scale (DynamoDB on-demand is ~$0.625 per million write units,
  doubled for transactional writes) — not the deciding factor.

## Alternatives considered

- **DynamoDB**: AWS-only (violates N2 as a primary); fencing via `TransactWriteItems` at 2× cost; scheduling via
  eventually consistent GSIs with hot-key sharding; TTL unusable as timers.
- **ScyllaDB**: source-available since Dec 2024 (free tier capped at 50 vCPU / 10 TB); LWT is single-partition and
  slow; Raft-backed strongly consistent tables are experimental and single-partition; queue/timer tables are a
  tombstone anti-pattern; no cross-partition transactions.
- **Postgres-compatible distributed SQL** (YugabyteDB, CockroachDB, Aurora DSQL): the escape hatch if the single
  primary becomes unacceptable; keeps the data model; pays consensus latency per write.
