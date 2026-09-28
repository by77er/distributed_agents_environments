# Architecture decision records

One record per load-bearing decision. Status: `Proposed` (recommended, not yet agreed), `Accepted` (agreed),
`Deferred` (deliberately not decided yet), `Superseded by NNNN`.

| # | Decision | Status |
|---|---|---|
| [0001](0001-cells.md) | Cells are the unit of scale and failure (fleet profile) | Proposed · amended by 0016, 0017 |
| [0002](0002-postgres-run-store.md) | Postgres per cell for the run store | Superseded by 0017 |
| [0003](0003-log-is-the-outbox.md) | The run log is the outbox; eager dispatch | Superseded by 0018 |
| [0004](0004-durable-actor-runtime.md) | Durable actor runtime hosting runs | Superseded by 0017, 0012 |
| [0005](0005-tools-are-spec-plus-binding.md) | Agent = model + tools; tools are specification + implementation; MCP is one binding | Accepted · amended by 0012 |
| [0006](0006-harness-unaware-of-policy.md) | The harness is unaware of the model and policy changes | Accepted |
| [0007](0007-active-recorder.md) | An active recorder owns tokenization and records sessions | Accepted |
| [0008](0008-mid-rollout-policy-change.md) | Policy may change mid-rollout; split generations by abort-and-resubmit | Accepted (requirement) · mechanism Proposed |
| [0009](0009-stale-kv-importance-sampling.md) | Reuse stale KV across weight updates; correct with importance sampling | Accepted · amended by 0015 |
| [0010](0010-envlet-fork-e2b.md) | microVMs via a node daemon (envlet) forked from E2B's orchestrator | Deferred (environment system) |
| [0011](0011-stack.md) | Language stack | Proposed · amended by 0016, 0017 |
| [0012](0012-task-agent-loop.md) | Tasks, agents, and one framework-owned loop; `Observation` | Accepted · amended by 0014 |
| [0013](0013-replay-durability.md) | Durability of task and agent code by deterministic replay | Proposed · amended by 0017, 0020 |
| [0014](0014-no-forks-template-recipes.md) | No forks; setup cost paid by templates | Accepted (templates belong to the deferred environment system) |
| [0015](0015-rollout-interface.md) | Rollout interface: rows in, samples out, weights published | Accepted |
| [0016](0016-layers-and-profiles.md) | A core library with optional layers, and deployment profiles | Accepted |
| [0017](0017-dbos-substrate.md) | DBOS as the durable execution substrate, in pump mode | Accepted |
| [0018](0018-effects-at-least-once.md) | Effects are at-least-once; receivers deduplicate with argument digests | Accepted |
| [0019](0019-conversations-and-priority-delivery.md) | Conversations as keyed runs, with priority-based delivery | Accepted |
| [0020](0020-deterministic-event-loop.md) | A deterministic asyncio event loop in the task host | Proposed |
| [0021](0021-trust-tiers.md) | Isolate task code by trust tier, in one execution architecture | Proposed |
| [0022](0022-rl-interface-extensions.md) | RL interface extensions (routing replay accepted) | Proposed |

## Template

```markdown
# NNNN — Title
Status: Proposed | Accepted | Superseded by NNNN · Date: YYYY-MM-DD

## Context
## Decision
## Consequences
## Alternatives considered
```
