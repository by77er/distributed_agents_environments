# Requirements

Status: **Accepted** · Every design decision should trace back to an item here.

## Functional

| ID | Requirement |
|---|---|
| R1 | Agents run **headless**, and **durably when deployed with the durability layer**: a run survives the loss of any worker process, and can suspend (e.g. waiting on a human) and resume without holding compute. |
| R2 | An agent is **a model plus the tools exposed to it**. What executes a tool is invisible to the agent and swappable per run. |
| R3 | **Pluggable environments**: microVMs, containers, whole VPSs, and other stateful services, chosen per task — or none: many tasks need no environment. The environment backing a tool is invisible to the agent. (The environment system is designed separately.) |
| R4 | First-class use cases: **large swarms**, **RL pipelines** (both sampling and training, on- and off-policy), **remote stateful coding agents**. |
| R5 | The **policy may change mid-rollout** (weights version, sampling parameters, or model) and the harness is not aware of it. |
| R6 | **Any harness is trainable**, including third-party harnesses we do not control, via the model endpoint alone. |
| R7 | Environments run **untrusted code**. Security properties differ per environment and are expressed and enforced **at the environment layer through an interface**. |
| R8 | Training data is **exact**: the tokens trained on are the tokens the policy saw and sampled, with honest behavior-policy logprobs, per-token policy versions and, for mixture-of-experts policies, the routed experts. |
| R9 | **Runs on one machine.** The core — tasks, agents, the recorder, the rollout API, a trainer and one inference engine — works in one process on a single GPU, with no services. Scale-out is a deployment choice, not a different system. |
| R10 | **Conversations**: agents receive messages over time; messages carry a priority that decides whether they queue, steer the current turn, or interrupt it. |

## Non-functional

| ID | Requirement |
|---|---|
| N1 | **Scale**: hundreds of thousands of concurrent agents. Design point: **300k concurrent runs**. |
| N2 | **Portable primitives**: managed Kubernetes, managed Postgres, object storage with conditional writes, VM node pools with nested virtualization, L4 load balancers, local NVMe. No single-cloud service on the critical path. |
| N3 | **microVMs require nested-virtualization node pools** (AWS C8i/M8i/R8i, GCP, Azure). This is an accepted infrastructure requirement. |
| N4 | **Failure isolation** (fleet profile): any single infrastructure failure affects at most one cell. |
| N5 | **Model latency dominates**: durable control overhead per effect (record + dispatch) SHOULD stay ≤ 50 ms p99, well under model turn latency (seconds). |
| N6 | **Colocated weight transfer**: trainers and inference engines may share a GPU fabric (NCCL / RDMA) for weight transfer; placement keeps them in the same region and zone. |

## Design-point arithmetic

Rough numbers for the **fleet profile**, used to size components. Revisit when real workload data exists.

| Quantity | Assumption | Result |
|---|---|---|
| Model turns | 300k runs × 1 turn / 10 s | ~30k turns/s |
| Durable checkpoints | ~2.5–3 per turn | ~84k/s fleet; ~7k/s per cell (12+ cells of ~25k runs) |
| Environments | at most one per run, when tasks use them | up to ~300k |
| Generated tokens | 500 output tokens per turn | ~15M tokens/s |
| Re-prefill per weight update (if KV recomputed) | 300k × 30k-token contexts | ~9B tokens — avoided by [ADR-0009](../decisions/0009-stale-kv-importance-sampling.md) |

**Implication**: inference dominates cost. Prefer choices that improve inference efficiency (cache affinity,
avoiding re-prefill) over control-plane elegance.

## Non-goals (initially)

- Multi-region active-active runs (a run lives in one cell).
- Building an inference engine, a trainer, or a container runtime.
- Preemption / priority scheduling across tenants.
- A UI.
