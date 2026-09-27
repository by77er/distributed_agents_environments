# Architecture decision records

One record per load-bearing decision. Status: `Proposed` (recommended, not yet agreed), `Accepted` (agreed),
`Superseded by NNNN`.

| # | Decision | Status |
|---|---|---|
| [0001](0001-cells.md) | Cells are the unit of scale and failure | Proposed |
| [0002](0002-postgres-run-store.md) | Postgres per cell for the run store | Proposed |
| [0003](0003-log-is-the-outbox.md) | The run log is the outbox; eager dispatch | Proposed |
| [0004](0004-durable-actor-runtime.md) | Durable actor runtime hosting runs (original pure-step-function authoring model superseded) | Proposed · amended by 0012, 0013 |
| [0005](0005-tools-are-spec-plus-binding.md) | Agent = model + tools; tools are specification + implementation; MCP is one binding | Accepted · amended by 0012 |
| [0006](0006-harness-unaware-of-policy.md) | The harness is unaware of the model and policy changes | Accepted |
| [0007](0007-active-recorder.md) | An active recorder owns tokenization and records sessions | Accepted |
| [0008](0008-mid-rollout-policy-change.md) | Policy may change mid-rollout; split generations by abort-and-resubmit | Accepted (requirement) · mechanism Proposed |
| [0009](0009-stale-kv-importance-sampling.md) | Reuse stale KV across weight updates; correct with importance sampling | Accepted · amended by 0015 |
| [0010](0010-envlet-fork-e2b.md) | microVMs via a node daemon (envlet) forked from E2B's orchestrator | Proposed |
| [0011](0011-stack.md) | Go for control and environment planes, Python for recorder and RL plane | Proposed |
| [0012](0012-task-agent-loop.md) | Tasks, agents, and one framework-owned loop; `Observation` | Accepted · amended by 0014 |
| [0013](0013-replay-durability.md) | Durability of task and agent code by deterministic replay; disposable snapshots | Proposed · amended by 0014 |
| [0014](0014-no-forks-template-recipes.md) | No forks: setup cost is paid by content-addressed template builds | Accepted |
| [0015](0015-rollout-interface.md) | Rollout interface: rows in, samples out, weights published | Accepted |

## Template

```markdown
# NNNN — Title
Status: Proposed | Accepted | Superseded by NNNN · Date: YYYY-MM-DD

## Context
## Decision
## Consequences
## Alternatives considered
```
