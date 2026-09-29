# 0024 — Validate with a product before RL

Status: **Accepted** · Date: 2026-09-28 · Amends [0023](0023-development-baseline.md)

## Context

The development plan put the recorder and local RL (M1) before the durable runner (M2). Most of the system's risk,
though, sits in parts that have nothing to do with RL: the loop, conversations, delivery modes, tools, the durable
runner and its recovery. An agent product on a third-party model exercises all of them end to end, can be evaluated
objectively, and needs no GPU.

## Decision

**Order of work**: M0 → **P1** → M2 → M1 → M3. Milestone names stay stable; only the order changes.

**P1 — a project assistant.** A long-lived conversational agent about a local code repository:

- Read-only repository tools as `@tool` methods: search, read files, list files, git history.
- Notes as an imported tool (memory is a tool), so writes are effects that deduplicate by `effect_id`.
- Follow-ups the assistant schedules itself, implemented with `WaitFor` timeouts: no new runtime primitive.
- Served by a small HTTP API over `Runner.send` and run event streams; connectors and the Control API come later.
- Evaluated four ways: task success (programmatic checks), judged quality (a second model with a rubric), cost and
  latency (from run events), and, once on the durable runner, durability under faults (processes killed at random
  points; no lost messages, no duplicated model calls or note writes).

**Model endpoint.** A direct adapter for the OpenAI Responses API, authenticated with the developer's local Codex
login (ChatGPT account tokens in `~/.codex/auth.json`, refreshed by the adapter), against the Codex backend. Model
`gpt-6-astra` by default. Checked on 2026-09-28: a request through the backend succeeds with the stored tokens. The
backend is not a stable public API; the adapter sits behind the `ModelEndpoint` protocol, so an API-key adapter can
replace it without touching anything else.

**Durable runner.** DBOS on SQLite first (no container runtime on the development machine); Postgres follows.

## Consequences

- `docs/development/plan.md` gains P1 and an explicit order of work.
- M1's engine measurements (ADR-0023) stay valid; RL work resumes after M2.
- The evaluation harness built for P1 is the first consumer of run events outside tests, and the model for
  `RolloutJobs` in evaluation mode later.
