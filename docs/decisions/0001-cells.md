# 0001 — Cells are the unit of scale and failure

Status: **Proposed**, scoped by [0016](0016-layers-and-profiles.md) to the fleet profile · Date: 2026-09-26

## Context

Design point: 300k concurrent runs, ~90k log events/s, up to ~300k environments (N1). Single Kubernetes clusters
are officially limited to ~150k pods / 5k nodes; a single database primary cannot absorb the write rate; and any
single failure must affect a bounded slice of the system (N4).

## Decision

Partition the execution plane into **cells**: each a Kubernetes cluster + Run Store primary + runtime workers +
tool routers + recorders + environment manager + env host pools. ~25k runs per cell. `run_id` embeds the home
cell; runs and their environments never leave it. Inference, object storage and RL-plane services are shared at
region/global scope.

## Consequences

- Linear scaling by adding cells; each cell's components stay at a comfortable size (e.g. ~7.5k appends/s).
- Blast radius of any infrastructure failure is one cell.
- Cross-cell swarms and env attachment are not supported; cross-cell messages go through a relayed outbox.
- Operational cost: many clusters and databases to manage → must be templated and automated from day one.

## Alternatives considered

- **One global store (DynamoDB/Scylla) + one big cluster**: removes the per-cell ceiling but fights Kubernetes
  limits, loses transactional scheduling, and makes every failure global.
- **Shard only the database**: fixes writes but not cluster limits or blast radius.

## Amendment ([0016](0016-layers-and-profiles.md), [0017](0017-dbos-substrate.md))

Cells exist only in the fleet profile. A cell's Postgres is the DBOS system database; the run store and runtime workers
are replaced by DBOS executors ([platform/cells](../platform/cells.md)).
