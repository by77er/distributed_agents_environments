# Environments

Status: **Preliminary** · Layer: environments (separate system, designed later)

> **Preliminary.** The environment system is a separate system with its own trade-offs and will be designed later.
> The core depends only on the minimal `Environments` / `Environment` handle protocols in
> [task.md](../core/harness/task.md#environments-optional); many tasks use no environment at all. The notes below
> record earlier thinking and research; none of it is decided.


**Built so far** (`rollout.environments`, both implementing the core's `EnvironmentService`): `NamespaceEnvironments`,
a per-environment copy of a minimal image in unprivileged Linux namespaces ([ADR-0025](../decisions/0025-agent-sessions.md));
and `LocalEnvironments`, a workspace directory on the host with no isolation, whose only image is `host` (asking it
for another image is an error, not a silent downgrade). Both keep environments as directories, so they survive
restarts, and neither keeps processes running between commands. Output longer than 2000 lines or 50 KB keeps its end,
and the full output is saved inside the environment (`full_output_path`). `rollout.environments.tools.ComputerTools`
gives a task's agent the usual coding-agent tools over any backend.

## Purpose

Durable, fast, pluggable computers for agents. An environment is a first-class resource with its own lifecycle,
independent of any run: it can outlive a run (coding workspace) or be shared (swarm). Setup cost is paid once per
distinct environment through content-addressed **templates** ([ADR-0014](../decisions/0014-no-forks-template-recipes.md)).
Task code creates and uses environments through Python handles ([task](../core/harness/task.md#environments-optional)); agents and
models never see them (P4).

## Sub-boundaries

The environment layer has four contracts. Each has its own document.

| Contract | Between | Doc |
|---|---|---|
| **Environment API** | Runner (on behalf of task code) → envlet → envd (in guest) | [env-api.md](env-api.md) |
| **Environment Manager API** | Runner (on behalf of task code), rollout service, Control API → Manager | this document |
| **Driver SPI** | Manager → backend (firecracker/envlet, pod, vps, local) | [driver.md](driver.md) |
| **SecurityProfile** | security class named by task code, concrete profile from the RunBinding; compiled and enforced by drivers | [security-profile.md](security-profile.md) |

Placement, pools and the envlet host protocol: [placement.md](placement.md).

## Owns / does not own

| Owns | Does not own |
|---|---|
| Environment registry and lifecycle state | Which runs use which env (RunBinding; attachments are recorded here) |
| Template builds (content-addressed), pools, snapshot metadata | Tool semantics (task `@tool` methods) |
| Placement onto hosts | VMM internals (Firecracker, Cloud Hypervisor — bought) |
| Security profile compilation and driver selection | Credentials themselves (broker) |
| Attachments and attachment tokens | |

## Interface

```proto
service EnvironmentManager {
  // Lifecycle — all mutating calls idempotent on request_id
  rpc Create(CreateEnvironmentRequest)     returns (Environment);
  rpc Claim(ClaimRequest)          returns (Environment);          // from a warm pool
  rpc Snapshot(SnapshotRequest)    returns (SnapshotReference);
  rpc Hibernate(EnvironmentReference)            returns (Environment);
  rpc Resume(EnvironmentReference)               returns (Environment);
  rpc Destroy(DestroyRequest)      returns (Empty);

  // Attachments (borrowing)
  rpc Attach(AttachRequest)        returns (Attachment);           // {env_id, run_id} → {attachment_id, token}
  rpc Detach(DetachRequest)        returns (Empty);

  // Data-plane handoff
  rpc Connect(ConnectRequest)      returns (EnvironmentEndpoint);          // {env_id, attachment_token} → {envlet_addr, env_id, ttl}

  // Reads
  rpc Get(EnvironmentReference)                  returns (Environment);
  rpc List(ListEnvironmentsRequest)         returns (stream Environment);

  // Templates and pools
  rpc EnsureTemplate(TemplateRecipe) returns (Template);           // content-addressed build; single-flight per hash
  rpc GetTemplate(TemplateReference) returns (Template);
  rpc PutPool(Pool)                returns (Pool);
}

message CreateEnvironmentRequest {
  string          request_id  = 1;
  OwnerReference        owner       = 2;   // run_id | job_id | principal — P12: owner destroys
  oneof source { string template_id = 3; string snapshot_id = 4; ImageReference image = 5; }
  Resources       resources   = 6;   // vcpu, memory_mib, disk_gib, gpu?
  SecurityProfile security    = 7;
  Persistence     persistence = 8;   // EPHEMERAL | VOLUME | FULL_SNAPSHOT
  Duration        ttl         = 9;   // hard upper bound on lifetime
  map<string,string> labels   = 10;
  PlacementHints  hints       = 11;  // affinity to a snapshot's cached hosts, anti-affinity, zone
}

message TemplateRecipe {                   // hashed with JCS → template_id
  string base = 1;                          // OCI image reference, pinned by digest at build time
  map<string, FileSource> files = 2;        // path → inline bytes | BlobReference | package asset (archives extracted)
  repeated string run = 3;                  // build commands, in order
  map<string, string> environment_variables = 4;
}

message Template {
  string template_id = 1;                   // tpl_{sha256 of recipe with the base resolved to a digest}
  TemplateState state = 2;                  // BUILDING | READY | FAILED
  string snapshot_id = 3;                   // golden snapshot, when READY
  string build_log = 4;                     // blob reference
}

message Environment {
  string env_id = 1; EnvironmentState state = 2; string driver = 3; string host = 4;
  SecurityProfile security = 5; Persistence persistence = 6;
  OwnerReference owner = 7; repeated Attachment attachments = 8;
  DriverCapabilities capabilities = 9; Timestamp expires_at = 10;
}

enum EnvironmentState { PENDING = 0; READY = 1; HIBERNATED = 2; DESTROYED = 3; FAILED = 4; LOST = 5; }
```

## Semantics & guarantees

- **Templates**: `EnsureTemplate` hashes the recipe (with `base` resolved to an image digest). If the hash is
  `READY`, it returns immediately; otherwise one build runs (BuildKit → root filesystem → boot → `run` commands →
  golden snapshot) and concurrent callers wait for it. A recipe that is only a `base` is a prebuilt image, so
  complicated environments can be baked by any external pipeline. `Create` with a template restores its golden
  snapshot; restores of the same snapshot on one host share memory pages copy-on-write.

- **Idempotency**: every mutating call is keyed by `request_id`; the runner uses `effect_id` as `request_id`
  for `environment.lifecycle`, `environment.attach` and `environment.release` effects.
- **Driver selection**: the manager picks a driver whose `capabilities` satisfy the request (resources,
  security, persistence). If none can, the call fails with `UNSATISFIABLE` — never a silent downgrade.
- **Ownership (P12)**: only the owner (or TTL expiry) destroys. `Detach` never destroys. An env with no
  attachments is not garbage unless its owner is terminal or its TTL expired.
- **Attachment tokens** scope data-plane access to `(env_id, run_id)`; envlets verify them on every call.
- **Lifecycle**:
  ```
  PENDING → READY ⇄ HIBERNATED → DESTROYED
      ↘ FAILED        READY → LOST (host died; recoverable only from a snapshot)
  ```
- **Persistence tiers**: `EPHEMERAL` (nothing survives host loss), `VOLUME` (disk survives; processes don't),
  `FULL_SNAPSHOT` (memory + disk; can hibernate mid-process).

## State & durability

Environment registry, templates, pools, attachments: a small Postgres schema per cell (can share the cell's
instance, separate schema). Snapshot contents: regional object storage. Host-local caches of hot snapshots on NVMe.

## Failure modes

| Failure | Effect |
|---|---|
| Host lost | Its envs → `LOST`. `VOLUME` envs with remote-backed disks and `FULL_SNAPSHOT` envs with a checkpoint can be restored elsewhere by their owner |
| Manager down | Data plane keeps working (Connect endpoints are cached with TTL); lifecycle ops stall |
| Snapshot storage slow | Restore latency rises for templates not cached on the host |

## Scale envelope

~25k environments per cell, ~150 per host. Lifecycle ops: RL churn up to ~100 creates per second per cell, almost all restores of cached template snapshots.

## Build vs buy

Build the manager, placement, pools and security compilation. Fork E2B's node agent as the envlet (ADR-0010).
Use kubernetes-sigs/agent-sandbox (gVisor/Kata) for the pod driver.

## Open questions

- Typical environment footprint (vCPU / memory / disk) — drives density and snapshot size.
- Remote-backed volumes for `VOLUME` tier (network block storage vs lazy-loaded object storage) — affects
  recoverability after host loss.
