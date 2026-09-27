# Requirements

Status: **Accepted** · Every design decision should trace back to an item here.

## Functional

| ID | Requirement |
|---|---|
| R1 | Agents run **headless and durable**: a run survives the loss of any worker process, and can suspend (e.g. waiting on a human) and resume without holding compute. |
| R2 | An agent is **a model plus the tools exposed to it**. What executes a tool is invisible to the agent and swappable per run. |
| R3 | **Pluggable environments**: microVMs, containers, whole VPSs, and other stateful services, chosen per task. The environment backing a tool is invisible to the agent. |
| R4 | First-class use cases: **large swarms**, **RL pipelines** (both sampling and training, on- and off-policy), **remote stateful coding agents**. |
| R5 | The **policy may change mid-rollout** (weights version, sampling parameters, or model) and the harness is not aware of it. |
| R6 | **Any harness is trainable**, including third-party harnesses we do not control, via the model endpoint alone. |
| R7 | Environments run **untrusted code**. Security properties differ per environment and are expressed and enforced **at the environment layer through an interface**. |
| R8 | Training data is **exact**: the tokens trained on are the tokens the policy saw and sampled, with honest behavior-policy logprobs and per-token policy versions. |

## Non-functional

| ID | Requirement |
|---|---|
| N1 | **Scale**: hundreds of thousands of concurrent agents. Design point: **300k concurrent runs**. |
| N2 | **Portable primitives**: managed Kubernetes, managed Postgres, object storage with conditional writes, VM node pools with nested virtualization, L4 load balancers, local NVMe. No single-cloud service on the critical path. |
| N3 | **microVMs require nested-virtualization node pools** (AWS C8i/M8i/R8i, GCP, Azure). This is an accepted infrastructure requirement. |
| N4 | **Failure isolation**: any single infrastructure failure affects at most one cell. |
| N5 | **Model latency dominates**: control-plane latency per step (persist + dispatch) SHOULD stay ≤ 50 ms p99, well under model turn latency (seconds). |

## Design-point arithmetic

Rough numbers used to size components. Revisit when real workload data exists.

| Quantity | Assumption | Result |
|---|---|---|
| Model turns | 300k runs × 1 turn / 10 s | ~30k turns/s |
| Log appends | ~3 events per turn, ~2 transactions per turn | ~90k events/s, ~60k txn/s |
| Per cell (12+ cells) | ~25k runs per cell | ~7.5k events/s per cell |
| Environments | worst case one per run | ~300k microVMs |
| Env hosts | ~150 microVMs per 64-vCPU node | ~2,000 nodes |
| Generated tokens | 500 output tokens per turn | ~15M tokens/s |
| Re-prefill per weight update (if KV recomputed) | 300k × 30k-token contexts | ~9B tokens — avoided by [ADR-0009](../decisions/0009-stale-kv-importance-sampling.md) |

**Implication**: inference dominates cost. Prefer choices that improve inference efficiency (cache affinity,
avoiding re-prefill) and environment density over control-plane elegance.

## Non-goals (initially)

- Multi-region active-active runs (a run lives in one cell).
- Building an inference engine, a trainer, or a container runtime.
- Preemption / priority scheduling across tenants.
- A UI.
