# Driver SPI

Status: **Preliminary** · Layer: environments (separate system, designed later)

> **Preliminary.** The environment system is a separate system with its own trade-offs and will be designed later.
> The core depends only on the minimal `Environments` / `Environment` handle protocols in
> [task.md](../core/harness/task.md#environments-optional); many tasks use no environment at all. The notes below
> record earlier thinking and research; none of it is decided.


A driver implements environment lifecycle for one kind of backend. The Environment Manager selects drivers by
**declared capabilities**; a driver either enforces what it is asked for or rejects the request (P8, R7).

## Interface

```go
type Driver interface {
    Name() string                      // "firecracker", "pod", "vps", "local"
    Capabilities() DriverCapabilities

    Create(ctx context.Context, spec EnvironmentSpecification) (Handle, error)
    Restore(ctx context.Context, snap SnapshotReference, spec EnvironmentSpecification) (Handle, error)
    Snapshot(ctx context.Context, h Handle, kind SnapshotKind, quiesce bool) (SnapshotReference, error)
    Pause(ctx context.Context, h Handle) error        // hibernate (FULL_SNAPSHOT drivers)
    Resume(ctx context.Context, h Handle) error
    Destroy(ctx context.Context, h Handle) error

    Endpoint(h Handle) EnvironmentAPIEndpoint                 // how the runner reaches envd (via envlet)
    Status(ctx context.Context, h Handle) (EnvironmentStatus, error)
}

type EnvironmentSpecification struct {
    EnvironmentID       string
    Source      Source           // template snapshot | workspace snapshot | image
    Resources   Resources
    Security    SecurityProfile  // MUST be enforced exactly or Create fails with ErrUnsatisfiable
    Persistence Persistence
    Labels      map[string]string
}

type DriverCapabilities struct {
    IsolationClasses []IsolationClass  // container | sandboxed_container | microvm | dedicated_vm
    NetworkModes     []NetworkMode     // none | allowlist | proxied | open
    CredentialInject bool              // egress-proxy credential injection available
    Persistence      []Persistence     // ephemeral | volume | full_snapshot
    SnapshotKinds    []SnapshotKind    // disk | memory (full); used for templates and hibernation
    GPU              bool
    MaxResources     Resources
    ColdStartP50     time.Duration
    RestoreP50       time.Duration
}
```

## Drivers

| Driver | Backend | Isolation | Snapshots | Persistence | Notes |
|---|---|---|---|---|---|
| `firecracker` | envlet on nested-virt node pools | microvm | **memory** + disk; same-host restores share pages | all three | Primary for untrusted, high-churn, RL. Implemented by the envlet (forked from E2B) |
| `cloud-hypervisor` | envlet | microvm | memory, disk | all three | For GPU passthrough / virtio-fs needs Firecracker lacks. Later |
| `pod` | kubernetes-sigs/agent-sandbox + gVisor/Kata | sandboxed_container / microvm (Kata) | disk (platform snapshots) | ephemeral, volume | Portable default for long-lived, low-churn workspaces |
| `vps` | cloud provider APIs | dedicated_vm | provider snapshots | volume | Whole machines; minutes to start |
| `local` | processes on a dev box | none | none | ephemeral | Development and tests only; never satisfies untrusted profiles |

## Rules

1. **Declare, enforce, or reject.** A driver MUST NOT accept a `SecurityProfile` it cannot enforce. There is no
   best-effort security.
2. **Capabilities are honest.** Advertised latencies are measured (p50 over the last window), not aspirational.
3. **Endpoints are private.** `Endpoint` never returns an address on a guest-reachable network.
4. **Idempotent destroy.** Destroying a missing environment succeeds.
5. **Snapshots are portable within a driver family** (firecracker snapshots restore on any envlet of the same
   VMM version and CPU template in the region); cross-driver restore is not supported.
