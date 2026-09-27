# Cells

Status: **Proposed** · See [ADR-0001](../decisions/0001-cells.md).

A **cell** is a self-contained slice of the execution plane: one Kubernetes cluster, one Run Store primary (with
replicas), its runtime workers, tool routers, recorder instances, environment manager and env host node pools, and
a prefix in regional object storage. Scale by adding cells; any single infrastructure failure affects at most one
cell (N4).

## What lives where

| Scope | Components |
|---|---|
| **Per cell** | Control API (cell), Run Store, Runtime, Task hosts, Tool Router, Recorder, Environment Manager, envlets, egress proxies, timer relay |
| **Per region** | Inference fleets (engines + router + WUC), object storage buckets, MCP/HTTP tool services |
| **Global** | Control API edge, Global Router, Policy Registry, Rollout Controller, Trajectory Assembler, sample logs |

Inference is regional, not per cell, because GPU pools are expensive to fragment. Cells share it through the
recorder, which routes by session affinity.

## Run placement

- `run_id` embeds its **home cell** (see [identifiers](../contracts/identifiers.md)). Runs never migrate between
  cells, so routing needs no lookup on the hot path.
- The Global Router picks a cell at creation: explicit request > affinity (parent run's cell for `spawn`, job's
  cell set for rollouts) > capacity (free worker slots, free env capacity, store headroom).
- An environment is always in the same cell as the runs attached to it.

## Cross-cell interaction

Rare, explicit, asynchronous:

| Interaction | Mechanism |
|---|---|
| `send` to a run in another cell | Relayed outbox → Global Router → target cell's inbox |
| `spawn` into another cell | Discouraged; relayed outbox → target cell Control API. Default is same cell as parent |
| Attaching an env in another cell | **Not supported** |
| Reading another cell's run log | Via Control API only (not the Run Store directly) |

**Rule**: a swarm and its environments live in one cell. Size cells so the largest expected swarm fits.

## Sizing (design point)

| Per cell | Value |
|---|---|
| Concurrent runs | ~25k |
| Log events/s | ~7.5k (peak 2–3× headroom → size for ~20k) |
| Partitions | 4,096 |
| Runtime workers | ~10–50 (thousands of runs each) |
| Env host nodes | ~150–200 |
| k8s pods | low thousands (workers, routers, envlets) — well below cluster limits because microVMs are not pods |

## Cell lifecycle

- **Create**: provision cluster + store + node pools from one template; register with Global Router as `accepting`.
- **Drain**: mark `draining` → no new runs; existing runs complete; durable long-lived runs are not migrated
  (there is no cross-cell migration) — they finish or are cancelled with notice.
- **Failure**: cell outage stalls its runs only. Global components treat the cell as unavailable; rollout jobs
  resample on other cells.
