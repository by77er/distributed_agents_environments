# 0011 — Go for control and environment planes, Python for recorder and RL plane

Status: **Superseded** by [0023](0023-development-baseline.md) · Date: 2026-09-26

## Context

Language should follow what we adopt. The environment layer starts from E2B's Go codebase; Kubernetes controllers
(controller-runtime) and llm-d are Go. The recorder depends on `renderers` (Python); engines, trainers and RL
tooling are Python.

## Decision

- **Go**: Control API, Runtime, Run Store client, Tool Router, Environment Manager, envlet, envd.
- **Python**: Task hosts and the Task / Agent SDK, Recorder, Rollout Controller, Trajectory Assembler,
  Trainer Adapters.
- Cross-language boundaries use gRPC with the IDL in these docs as the source of truth.

## Consequences

- Two toolchains; contracts must be generated from one IDL source.
- The Runtime ↔ task host boundary (B4) crosses languages → task hosts are co-located Python processes reached
  over a local socket.
- Recorder throughput scales horizontally; if CPU-bound, hot paths can move to Rust without changing its contracts.

## Alternatives considered

- **Rust core** (earlier lean): strong fit for envd and performance, but fights the reuse gradient (E2B, llm-d,
  controller-runtime are Go).
- **All Python**: fastest iteration; heavier envd and weaker fit for the environment plane.

## Amendment ([0016](0016-layers-and-profiles.md), [0017](0017-dbos-substrate.md))

The core, the recorder, the rollout API and the durable runner (DBOS) are Python. The Go runtime is gone. Whether the
pump is written in Python or Go (DBOS Go) is open; platform services and the environment system choose their languages
when they are designed.
