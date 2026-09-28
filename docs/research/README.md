# Research

Status: **Draft** · Started 2026-09-27

Investigations that inform design decisions. Reports here are evidence and analysis, not normative design; when a
report changes the design, the change lands in `architecture/`, `contracts/`, `components/` or an ADR.

## Charter

Investigate each option for the durable execution substrate, compare the interface we are creating with existing
tools, and work out how the system operates in each use case — **reinforcement learning (synchronous and
asynchronous)**, **swarms**, and **durable agent identities** — in enough detail to make and implement decisions.

## Current design state (read this first)

The normative design is in [`docs/`](../README.md). Read at least: [requirements](../architecture/requirements.md),
[principles](../architecture/principles.md), [overview](../architecture/overview.md),
[harness](../components/harness/README.md) (and its `task.md`, `agent.md`, `durability.md`),
[rollouts](../components/rollouts/README.md), [trajectories](../components/trajectories/README.md),
[environments](../components/environments/README.md), [recorder](../components/recorder/README.md), and the
[decision records](../decisions/README.md).

Design point: 300k concurrent runs, ~30k model turns/s, cells of ~25k runs with one Postgres each, multi-cloud
primitives only, microVM environments on nested-virtualization nodes, untrusted code in environments.

### Decisions and proposals made after the docs were last updated

These are **not yet in the docs**. Treat them as the working hypothesis to evaluate, not as settled.

1. **Program layer.** A generic durable-program layer (`class Program: async def main(self, run)`) under the
   Task / Agent loop, so the platform can host plain durable workflows as well as agents. The agent loop becomes one
   `Program` (`AgentProgram`).
2. **Generic platform additions** (for use beyond RL): `run.emit(kind, payload)` outputs delivered by connectors
   (Slack, email, UI); an ephemeral, non-durable token-streaming side channel; triggers (cron, webhooks); **run keys**
   (external identifiers such as a Slack thread mapped to a run) with signal-with-start; `@tool(requires_approval=True)`
   for human approval; deployments (name → code references + default binding); per-run budgets and per-tenant quotas;
   `run.memory` (namespaced durable key-value store); per-tenant encryption keys for deletion; OpenTelemetry traces.
3. **The core is a durable execution engine.** The run log, leases, effects, replay, timers, signals and versioning
   duplicate what Temporal, Restate, DBOS and others provide. Build-vs-buy for the substrate is reopened (ADR-0004).
4. **Working assumption: DBOS** as the substrate. Mapping: run = DBOS workflow; effects = `@DBOS.step`; signals =
   `DBOS.send` / `DBOS.recv`; timers = `DBOS.sleep`; child runs = child workflows; versioning = application versions +
   `DBOS.patch()`; rollout admission = DBOS queues; outputs and RL events = `DBOS.write_stream`. Known issues:
   - **untrusted task code** would run in the same process as the DBOS system-database connection;
   - **idle workflows** blocked in `recv` stay resident and pinned to one code version;
   - **recovery** of a dead executor's workflows needs DBOS Conductor (commercial tiers) or our own controller;
   - DBOS records step output *after* the step, so our "persist before dispatch" becomes at-least-once steps plus
     receiver deduplication, with attempt markers for tools that cannot deduplicate.
5. **DBOS is not an actor system.** Proposed virtual-actor layer on DBOS: an entity row holds state (versioned
   records, not only pickle); each activation (start, each incoming message) is a workflow on a queue partitioned by
   entity key with `partition_concurrency = 1`; suspension happens only at turn boundaries via a new return value
   from `Task.respond`: `WaitFor("user_message", timeout=...)`; short waits (approvals) stay in-workflow `recv`.
   Alternatives: Restate virtual objects, Temporal entity workflows.
6. **Durable agent identities** (new term, needs definition): long-lived agent entities with a stable address that
   persist across runs and conversations — memory, credentials and permissions, a mailbox, owned resources (e.g.
   workspaces), a policy binding — that can join swarms and be trained.
7. **Open question Q10** (who writes task code: in-house only, or external tenants) is unresolved and decides the
   isolation model. Evaluate both.

## Conventions for reports

- **Never truncate type names** (`Observation`, not `Obs`; `EnvironmentSpecification`, not `EnvSpec`).
- Cite sources as markdown links; mark anything unverified as such. Today is 2026-09-27; verify current versions,
  licenses and limits on the web instead of relying on memory.
- Separate facts (with sources) from analysis and recommendations.
- Code sketches in Python for task / agent / program code; IDL or Go only where the design already uses them.

## Reports

| Report | Question |
|---|---|
| [substrate-dbos.md](substrate-dbos.md) | DBOS as the substrate, including the virtual-actor layer |
| [substrate-restate.md](substrate-restate.md) | Restate (services, virtual objects, workflows) as the substrate |
| [substrate-temporal.md](substrate-temporal.md) | Temporal (workflows, entity workflows) as the substrate |
| [substrate-custom-and-others.md](substrate-custom-and-others.md) | Our own runtime vs other contenders (Cloudflare Durable Objects / Agents, Dapr, Orleans, Akka, Inngest, Hatchet, …) |
| [interfaces-agent-frameworks.md](interfaces-agent-frameworks.md) | Our Task / Agent / tool / environment interface vs agent and durable-agent frameworks |
| [interfaces-rl-environments.md](interfaces-rl-environments.md) | Our interface vs RL environment specifications and RL frameworks' rollout integration |
| [synthesis.md](synthesis.md) | Comparison, recommendation, and how the system operates per use case |
