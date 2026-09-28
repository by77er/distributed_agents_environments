# Placement, pools, and the envlet

Status: **Preliminary** · Layer: environments (separate system, designed later)

> **Preliminary.** The environment system is a separate system with its own trade-offs and will be designed later.
> The core depends only on the minimal `Environments` / `Environment` handle protocols in
> [task.md](../core/harness/task.md#environments-optional); many tasks use no environment at all. The notes below
> record earlier thinking and research; none of it is decided.


## Why not one pod per environment

At the design point (~25k envs per cell, RL churn ~100 creates/s) pod-per-env pushes the Kubernetes API server
and scheduler hard and cannot restore memory snapshots. Kubernetes manages the **fleet** (envlet
DaemonSet on nested-virt node pools); the Environment Manager places **environments** onto envlets. The `pod`
driver remains for low-churn workloads.

## envlet

Per-host daemon (forked from E2B's orchestrator node agent; Nomad/Consul removed).

Responsibilities:
- Firecracker lifecycle via jailer; per-VM tap devices, nftables, cgroups.
- Snapshot create/restore; **lazy memory restore via userfaultfd**; lazy disk loading from object storage;
  copy-on-write root filesystems.
- Same-host restores of one snapshot share its memory file copy-on-write.
- Local NVMe cache of hot snapshots/templates.
- Egress proxy + credential injection for its VMs.
- Environment API proxy (mTLS in, vsock out) with attachment-token checks and limits.

### Host protocol (Manager ↔ envlet)

```proto
service Envlet {
  rpc Register(RegisterRequest) returns (stream Assignment);   // long-lived; manager pushes work
  rpc Report(stream HostReport) returns (Empty);               // capacity, cached snapshots, env states (every 2 s)
  // Driver SPI operations, invoked by the manager:
  rpc Create(EnvironmentSpecification) returns (Handle);  rpc Restore(RestoreRequest) returns (Handle);
  rpc Build(TemplateRecipe) returns (Template);  rpc Snapshot(SnapshotRequest) returns (SnapshotReference);
  rpc Pause(Handle) returns (Empty);  rpc Resume(Handle) returns (Empty);  rpc Destroy(Handle) returns (Empty);
}

message HostReport {
  string host = 1; Resources capacity = 2; Resources allocated = 3;
  repeated string cached_snapshots = 4;           // for restore locality
  repeated EnvironmentStatus envs = 5;
  string vmm_version = 6; string cpu_template = 7; // snapshot compatibility
}
```

## Placement algorithm (initial)

Score feasible hosts (capacity, VMM/CPU-template compatibility with the snapshot, zone constraints) by:

1. **Restore locality** — the host already caches the template snapshot (restores there are fast and share memory
   pages; elsewhere the snapshot is pulled lazily from object storage).
2. **Bin-packing** — best fit on the binding resource (usually memory), bounded by a per-host share of any one
   template to limit blast radius.

Placement is per cell and single-leader (leader-elected Manager replica); decisions are recorded in the env
registry before the envlet is instructed (so a manager failover never double-places).

## Templates and pools

Templates are content-addressed build recipes (see [environments](README.md#semantics--guarantees)). Builds run on
envlets with spare capacity: BuildKit resolves `base` and applies `files`, the result becomes a root filesystem, the
VM boots, `run` commands execute, and the booted state is captured as the golden snapshot and uploaded to regional
object storage. The template registry (hash → snapshot) is per region, so a template built in one cell is reused by
every cell in the region.

```proto
message Pool {
  string pool_id = 1; string template_id = 2;
  uint32 min_ready = 3; uint32 max_ready = 4;
  uint32 max_refill_per_s = 5;       // throttle refills after bursts
}
```

- Pools keep `min_ready` restored-and-paused environments of one template per cell; `Claim` hands one out and
  triggers refill. They suit interactive workloads with a few hot templates; reinforcement-learning jobs with one
  template per row rely on restore locality instead.

## Open questions

- Nested-virtualization overhead on target instance types (measure: boot, restore p50/p99, exec-heavy and
  I/O-heavy workloads, density).
- How entangled E2B's orchestrator is with Nomad/Consul — decides whether to fork its code or reimplement (see ADR-0010).
- Template cache eviction: by age, by size, or pinned per job?
- Cross-zone restore: allowed (slower) or restricted to the snapshot's zone?
