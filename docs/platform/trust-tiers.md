# Trust tiers

Status: **Proposed** (to be explored further) · Layer: platform · See [ADR-0021](../decisions/0021-trust-tiers.md)

Program, task and agent code runs in [task hosts](../durability/task-host.md) driven by a trusted pump. Every tier
uses that same structure; tiers differ only in three settings. Environments are isolated separately
([environments](../environments/README.md)).

| Setting | Options |
|---|---|
| **Sandbox** for the task host | separate process with no network → gVisor → microVM |
| **Pooling** | shared per code version → one tenant per process → one run per process |
| **Placement** | shared cell → dedicated cell |

## Tiers

| Tier | Who writes the code | Main threat | Sandbox | Pooling | Placement | Additional controls |
|---|---|---|---|---|---|---|
| **T0 platform** | us (default agents, adapters) | bugs | separate process, no network | shared | shared | — |
| **T1 in-house** | our researchers | bugs, non-determinism, runaway resources | separate process, no network; gVisor optional | shared per `code_reference` | shared | per-process resource limits |
| **T2 partner** | contracted, identified tenants | a compromise or bug crossing tenants; noisy neighbours | gVisor | one tenant per process | shared, or dedicated for compliance | per-tenant queues and payload encryption keys; per-tenant prefix-cache salt in inference |
| **T3 public** | anonymous or self-serve tenants | deliberate attack, abuse, resource theft | gVisor with a stricter system-call filter; microVM if gVisor escapes are in scope | one tenant per process, small processes, strict quotas | a tier of cells for public traffic | budget ceilings, abuse detection, egress only through imported tools |

## Why these settings

- **The pump is ours in every tier.** Substrate and database credentials never reach task code, so tenant isolation
  is only about task hosts, environments and shared caches.
- **Process-per-tenant is the real boundary.** A task host process holds many runs; two tenants in one process could
  read each other's memory regardless of the sandbox. From T2 upward a process never mixes tenants.
- **Shared inference caches leak across tenants.** Prompt prefixes can be detected through cache-hit timing, so from
  T2 upward prefix-cache keys carry a per-tenant salt. Reuse within a tenant is unaffected.
- **Promotion is configuration.** Moving a tenant between tiers changes the three settings, not the code path.

## Costs

- Per-tenant pools pack less densely; idle tenants hold processes unless pools scale to zero. A pre-warmed process
  per code version that forks new hosts keeps start-up fast.
- gVisor slows system-call-heavy code; task code does almost no I/O by design, so the cost is small.
- MicroVM task hosts cost roughly 20–50 MB each (*estimate*); use them only when the T3 threat model requires it.

## Open questions

- Which tiers are needed first, and whether T3 is in scope at all.
- Whether T1 runs gVisor by default for uniformity with T2.
