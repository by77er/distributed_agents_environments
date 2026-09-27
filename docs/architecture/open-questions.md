# Open questions

Status: **Draft** · Every unresolved question, with the doc that owns the answer. Resolve by editing the owner
doc (and adding an ADR if load-bearing), then delete the row.

## Blocking interface decisions

Decided since the last revision: trainable channels sample with temperature only (was Q1; see
[ADR-0015](../decisions/0015-rollout-interface.md)).

These change a contract; resolve before implementing the affected boundary.

| # | Question | Owner | Leaning |
|---|---|---|---|
| Q2 | Context shape: agent-owned with task hints (current), or may a task require a shape? | [agent](../components/harness/agent.md) | Agent-owned; tasks hint via `ContextHints` |
| Q3 | Is the inference fleet ours to operate, or an external platform we require the engine contract from? | [inference](../components/inference/README.md) | Undecided — determines who implements the WUC |
| Q4 | Which trainer first? | [trajectories](../components/trajectories/README.md) | Evaluate slime/miles and AReaL for external-trajectory support |
| Q5 | Tool progress streaming: results only, or `tool.progress` events? | [tool-router](../components/tool-router/README.md) | Results only initially |
| Q10 | Who writes task code: in-house researchers only, or external tenants? | [trust-boundaries](trust-boundaries.md), [durability](../components/harness/durability.md) | Sandbox task hosts from day one either way (it also enforces determinism) |
| Q11 | Should `score` run after a hook raised, to score partial work? | [harness](../components/harness/README.md) | No initially |
| Q12 | Template cache eviction and pinning (by age, size, or per job)? | [placement](../components/environments/placement.md) | Age + size, with per-job pins |

## Workload facts needed

| # | Question | Affects |
|---|---|---|
| Q6 | Typical environment footprint (vCPU / memory / disk)? | density, template snapshot size, node shape |
| Q7 | Distribution of run lifetimes (seconds-long RL vs days-long coding) and their mix? | `durable` vs `best_effort` mix, Run Store sizing, snapshot cadence |
| Q8 | Largest expected swarm (runs + envs)? | cell sizing (swarms can't span cells) |
| Q9 | Tenancy model and quota dimensions? | Control API, cells |

## Validation spikes

Each can overturn a Proposed ADR.

| # | Spike | Validates |
|---|---|---|
| S1 | `renderers` fidelity on target model families with tool calls + reasoning, driven by 2–3 real harnesses; measure prefix-mismatch rate | ADR-0007 |
| S2 | SGLang abort-and-resubmit: exact partial tokens/logprobs; continuation correctness across a weight update; stale-KV cache hits | ADR-0008, ADR-0009 |
| S3 | Nested Firecracker on target instance types: density, cold restore from object storage via UFFD, same-host restore rate | ADR-0010, N3 |
| S4 | Extractability of E2B's orchestrator from Nomad/Consul | ADR-0010 |
| S5 | Postgres append throughput per cell with partition fencing and side-writes at ~10–20k events/s | ADR-0002 |
| S6 | Task host prototype: replay driver, divergence detection, sandbox (gVisor, no network), snapshot / restore; measure replay cost of a 500-turn episode and runs hosted per process | ADR-0013 |

## Smaller questions (owned in component docs)

- Run Store: archive format — [run-store](../components/run-store/README.md)
- Runtime: per-worker run priority; task host pools per code version — [runtime](../components/runtime/README.md)
- Environments: remote-backed volumes; cross-zone restore; port forwarding for previews —
  [environments](../components/environments/README.md), [placement](../components/environments/placement.md),
  [env-api](../components/environments/env-api.md)
- Rollouts: priority semantics; per-job concurrency limits across cells — [rollouts](../components/rollouts/README.md)
- Recorder: default flush mode per job type; `renderers` license — [recorder](../components/recorder/README.md)
