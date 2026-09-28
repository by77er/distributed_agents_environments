# Open questions

Status: **Draft** · Every unresolved question, with the document that owns the answer. Resolve by editing the
owner document (and adding an ADR if load-bearing), then delete the row.

## Deliberately deferred

Decided *not* to decide yet (P14).

| Topic | Owner | When it matters |
|---|---|---|
| The environment system: drivers, isolation, templates, placement | [environments](../environments/README.md) | when tasks that need computers are built beyond the local profile |
| First trainer to integrate (must support routing replay for mixture-of-experts) | [trajectories](../core/trajectories/README.md#trainer-adapters) | RL spikes |
| Training on production traffic (consent, retention, delayed rewards) | policy, then [rollouts](../core/rollouts/README.md) | before any production data is used for training |
| Which trust tiers are needed, and whether T3 (public) is in scope | [trust tiers](../platform/trust-tiers.md) | before external tenants |
| Human approval of tool calls, run budgets, streaming to clients beyond the Control API | [research](../research/interfaces-agent-frameworks.md#9-recommended-interface-changes-concrete) | when product use cases need them |

## Open interface questions

| # | Question | Owner |
|---|---|---|
| Q1 | Should `score` also run after a hook raised, to score partial work? | [harness](../core/harness/README.md) |
| Q2 | Priority semantics across rows of one rollout job; a cursor per consumer when several trainers share a job | [rollouts](../core/rollouts/README.md) |
| Q3 | Where enrichment (teacher / reference logprobs via `score_tokens`) runs: assembler or trainer | [trajectories](../core/trajectories/README.md) |
| Q4 | Whether the durable pump is written in Python or Go (DBOS Go feature parity) | [durability](../durability/README.md) |
| Q5 | Pool strategy for weight transitions at fleet scale; capacity for pinned historical versions | [inference](../inference/README.md) |
| Q6 | Tenancy model and quota dimensions | [control API](../platform/control-api/README.md) |

## Workload facts needed

| # | Question | Affects |
|---|---|---|
| W1 | Distribution of run lifetimes (seconds-long RL episodes vs days-long conversations) and their mix | durable vs best-effort runs; database sizing |
| W2 | Largest expected swarm | cell sizing (swarms stay in one cell) |
| W3 | Target model families | renderer coverage; routing replay |

## Validation spikes

Each can overturn a decision. Pass criteria are in the linked research.

| Phase | Spikes | Decides |
|---|---|---|
| 1 — durable substrate | pump + task-host latency and replay; DBOS cell throughput, retention soak, recovery storm, fencing under partition, failover durability, mass cancellation ([DBOS research](../research/substrate-dbos.md#10-risks-open-questions-spike-tests)) | ADR-0017 holds, or switch to Restate |
| 2 — conversations and swarms | activation correctness (no double activations, no lost wake-ups), messaging load, gang admission, swarm rewards | conversation scheduling; rollout extensions |
| 3 — RL | trainer adapter parity, asynchronous publish every step, staged weight transfer, routed-expert capture through the recorder, renderer coverage, best-effort runs ([RL research](../research/interfaces-rl-environments.md#82-spike-tests)) | ADR-0009 and ADR-0022 in practice |
| 4 — interoperability | deterministic event loop with an in-process third-party framework; foreign harness via the recorder's compatible endpoints | adapter tiers |
