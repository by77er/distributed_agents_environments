# SecurityProfile

Status: **Proposed** · Requirement R7 · Principles P7, P8

Each environment declares its security requirements through this interface. Drivers compile and enforce them
**outside the guest**; the Environment Manager rejects combinations no driver can satisfy.

## Schema

```proto
message SecurityProfile {
  IsolationClass isolation   = 1;   // minimum: CONTAINER < SANDBOXED_CONTAINER < MICROVM < DEDICATED_VM
  NetworkPolicy  network     = 2;
  repeated CredentialGrant credentials = 3;
  Limits         limits      = 4;
  repeated string tags       = 5;   // audit / policy labels, e.g. "untrusted-code", "customer-data"
}

message NetworkPolicy {
  NetworkMode mode = 1;                         // NONE | ALLOWLIST | PROXIED | OPEN
  repeated string allow = 2;                    // ALLOWLIST: host patterns ("*.pypi.org", "github.com")
  uint64 egress_bytes_max = 3;
  bool   allow_inbound_preview = 4;             // expose ports via envlet proxy (remote coding)
}

message CredentialGrant {
  string destination = 1;                       // host pattern; MUST also be allowed by NetworkPolicy
  Scheme scheme      = 2;                       // BEARER | BASIC | HEADER(name) | AWS_SIGV4 | …
  string secret_ref  = 3;                       // broker reference; never a secret value
  repeated string methods = 4;                  // optional HTTP method restriction
}

message Limits {
  uint32 vcpu = 1; uint64 memory_mib = 2; uint64 disk_gib = 3;
  uint32 pids_max = 4; uint64 disk_iops = 5;
  Duration wall_time_max = 6;                   // env TTL upper bound
}
```

## Security classes

Task code names a **class** (`EnvironmentSpecification.security`); the run's `RunBinding.security_classes` maps it to
a concrete profile (credential grants, allowlists) that must satisfy the class minimum, or the operator default
for the class applies. The runtime resolves the profile before calling the Environment Manager.
Classes are operator-defined minimums:

| Class | isolation ≥ | network | credentials | Typical use |
|---|---|---|---|---|
| `trusted-dev` | container | open | allowed | local development only |
| `untrusted-offline` | microvm | none | none | RL rollouts on hermetic tasks |
| `untrusted-allowlist` | microvm | allowlist | via grants | coding agents that install packages |
| `untrusted-proxied` | microvm | proxied (all egress through proxy, logged) | via grants | agents browsing / calling APIs |
| `dedicated` | dedicated_vm | per profile | via grants | whole-machine tasks, customer isolation |

## Enforcement matrix

| Property | firecracker (envlet) | pod (agent-sandbox) | vps |
|---|---|---|---|
| isolation | KVM + jailer + seccomp | gVisor / Kata RuntimeClass | provider hypervisor |
| `network: none` | no tap device | NetworkPolicy deny-all | provider firewall + no public IP |
| allowlist / proxied | per-VM tap + nftables + egress proxy (SNI/CONNECT) | egress gateway + NetworkPolicy | provider firewall + proxy |
| credential injection | host egress proxy | egress gateway proxy | proxy VM |
| limits | VMM config + cgroups on VMM process | pod resources | instance size |

## Rules

1. The profile is fixed at creation. Changing it requires a new environment.
2. The profile is not part of a template's hash: one template can be restored under different profiles. Template
   builds run with the `untrusted-proxied` class so `run` commands can fetch packages.
3. Credentials are resolved by the broker at injection time and never written to the guest, logs, recorder,
   or tool results.
4. Every profile compiles to a deterministic host configuration whose hash is recorded on the environment, for audit.
5. Scratch environments for scoring (`run.environments.scratch`) default to `untrusted-offline` regardless of the
   agent's environment profile.
