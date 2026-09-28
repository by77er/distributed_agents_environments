# Cells

Status: **Proposed** · Layer: platform (optional) · See [ADR-0001](../decisions/0001-cells.md)

A **cell** is a self-contained slice of the fleet profile: one Kubernetes cluster, one Postgres (the DBOS system
database for the [durable runner](../durability/README.md)), its durable executors and task hosts, recorder and
tool-router instances, and a prefix in regional object storage. Scale by adding cells; any single infrastructure
failure affects at most one cell (N4). Cells exist only in the fleet profile.

## What lives where

| Scope | Components |
|---|---|
| **Per cell** | Control API (cell), Postgres (DBOS), durable executors, task-host pools, recovery controller, reaper, recorder, tool router, connectors |
| **Per region** | inference fleets (engines, router, weight update controller), object storage, MCP / HTTP tool services |
| **Global** | Control API edge, cell router, policy registry, rollout service, trajectory assembler, sample logs |

Inference is regional, not per cell, because GPU pools are expensive to fragment; cells share it through the
recorder, which routes by session affinity. Trainers using distributed weight transfer run in the same region.
Environment infrastructure is placed by the [environment system](../environments/README.md).

## Placement

- `run_id` embeds its **home cell** ([identifiers](../contracts/identifiers.md)). Runs never migrate between cells.
- A conversation's runs live in one cell: the cell router maps the conversation key to a cell on first contact.
- The cell router picks a cell for a new run: explicit request > affinity (a parent's cell for `spawn`, a job's cell
  set for rollouts) > capacity.

## Cross-cell interaction

Rare, explicit, asynchronous: messages and spawns to another cell go through that cell's Control API with an
idempotency key. **A swarm lives in one cell**; size cells so the largest expected swarm fits.

## Sizing (design point)

| Per cell | Value |
|---|---|
| Concurrent runs | ~25k |
| Durable checkpoints | ~7k/s (peak 2–3×) — see the [DBOS research](../research/substrate-dbos.md#53-estimates-at-the-design-point) |
| Durable executors | ~20–50 processes |

## Cell lifecycle

- **Create**: provision cluster, Postgres and pools from one template; register with the cell router as `accepting`.
- **Drain**: mark `draining` → no new runs or conversations; existing runs complete or hand over at their next
  generation; conversations' next runs start in another cell.
- **Failure**: a cell outage stalls its runs only; rollout jobs resample elsewhere.
