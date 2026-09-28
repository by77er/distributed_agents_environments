# 0017 — DBOS as the durable execution substrate, in pump mode

Status: **Accepted** · Date: 2026-09-27 · Supersedes the run store of [0002](0002-postgres-run-store.md) and the runtime of [0004](0004-durable-actor-runtime.md)

## Context

The run store, leases, effect journaling, replay, timers, messaging and versioning we had specified are a durable
execution engine. Building our own was estimated at 39–60 engineer-months plus a permanent team
([research](../research/substrate-custom-and-others.md)). DBOS, Restate and Temporal were evaluated in depth
([synthesis](../research/synthesis.md)).

## Decision

- The durability layer's `DurableRunner` is built on **DBOS** (MIT), on Postgres (or SQLite locally).
- **Pump mode**: a small, trusted, generic DBOS workflow drives a sandboxed task host over `HarnessHost` and performs
  each effect the task host requests as a DBOS step, started in the task host's deterministic order. Task code never
  runs as a DBOS workflow directly and never holds database credentials.
- We build: the pump, conversation scheduling on a partitioned queue, a recovery controller (instead of DBOS
  Conductor), a reaper for cleanup, and the run-event projection.
- **Restate** is the designated alternative; our own runtime is the fallback, both only if phase-1 spikes fail.

## Consequences

- One Postgres per cell stays (as the DBOS system database); no new stateful system to operate.
- Effects become at-least-once ([0018](0018-effects-at-least-once.md)).
- DBOS does not run cleanup after hard cancellation or timeout: cancellation is cooperative and a reaper cleans up.
- Risks to validate: delete-based retention at our write rate, recovery storms, `NOTIFY` pressure, young fencing
  code.
- The substrate-specific code is bounded (pump, scheduling, recovery, adapter), so switching remains possible.

## Alternatives considered

- **Temporal**: best semantic fit, but ~4–5× our write load per turn and ~$4–6M/month on Temporal Cloud at list
  price for our rate.
- **Restate**: actor-native and strong for conversations; a new cluster type to operate, a young Python SDK, a BSL
  server.
- **Our own runtime**: full control; the cost of building and operating a new engine.
