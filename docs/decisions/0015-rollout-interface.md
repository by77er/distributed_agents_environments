# 0015 — Rollout interface: rows in, samples out, weights published

Status: **Accepted** · Date: 2026-09-27

## Context

The rollout side had accumulated algorithm concepts: group size, whole-group delivery, staleness bounds and an
adaptive feedback loop, per-trainer sampling constraints, static and pushed datasets, two delivery paths. Each
coupled the rollout system to a particular learning algorithm.

## Decision

- The caller (typically a Ray-based trainer) uses five operations: `start`, `run(row, labels, count)`,
  `samples(cursor)`, `acknowledge(cursor)`, `publish(weights)`.
- **Grouping belongs to the caller.** `count` only batches calls; labels travel with each `Sample`; the caller
  assembles groups.
- **One durable, ordered sample log per job**, read by cursor. Replaying it serves offline training and evaluation.
- **Backpressure is a bounded buffer**: runs are admitted while in-flight runs plus unacknowledged samples stay
  under `buffer_samples`.
- **Staleness is bounded by the buffer and `max_turns`**, not by an adaptive loop. Trainers mask outliers using
  per-token `weights_version`. `max_kv_age` is an operator setting per channel.
- **Trainable channels sample with temperature only.**
- `Sample` carries only what a loss needs; diagnostics stay in session trees and run logs, reachable by identifier.

## Consequences

- The rollout side knows no algorithm; the only per-algorithm code is a thin trainer adapter.
- Adaptive staleness control is gone: when lag spikes, trainers mask more tokens instead of the system throttling
  early. It can return as an optional addition if measurements show the need.
- Incompatibilities between a trainer and the data (e.g. multi-sequence samples) surface in the adapter, not as job
  validation.

## Alternatives considered

- **`TrainingRequirements` declared by trainer adapters and validated per job**: explicit, but keeps nine
  algorithm-shaped parameters in the rollout contract.
- **Group-aware rollout jobs** (`group_size`, group barriers): see Context.
