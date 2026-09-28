# 0020 — A deterministic asyncio event loop in the task host

Status: **Proposed** · Date: 2026-09-27 · Amends [0013](0013-replay-durability.md)

## Context

ADR-0013 drove coroutines like generators, with no asyncio event loop. Third-party agent frameworks use
`asyncio.gather`, `create_task`, locks and queues, so they could not run under replay. gVisor also does not stop
`time.time`, `uuid4` or `os.urandom`.

## Decision

The durable runner's task host runs code on a custom asyncio event loop (the design Temporal's Python SDK uses):
deterministic ordering of ready callbacks; effects as futures resolved from recorded completions; `loop.time()`
mapped to recorded time; real I/O raises `NonDeterminismError`; clocks and randomness patched to run-scoped sources;
threads rejected. See [task host](../durability/task-host.md).

## Consequences

- Ordinary asyncio code, including third-party frameworks, runs under replay unchanged.
- The loop is a correctness-critical component that needs its own test suite (spike, phase 4).

## Alternatives considered

- **Generator driver** (ADR-0013): simpler, but excludes third-party code.
