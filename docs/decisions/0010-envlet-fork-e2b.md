# 0010 — microVMs via a node daemon (envlet) forked from E2B's orchestrator

Status: **Deferred** — the environment system is designed separately ([environments](../environments/README.md)) · Date: 2026-09-26

## Context

Untrusted code (R7) at ~25k environments per cell with RL churn and memory-snapshot forking. Pod-per-environment
strains Kubernetes at this churn and cannot express memory forks. The hard parts of a microVM node agent —
Firecracker lifecycle, userfaultfd lazy memory restore, lazy disks, CoW rootfs, snapshot templates, in-guest
daemon — exist in E2B's Apache-2.0 runtime, which is orchestrated by Nomad/Consul and is GCP-first.

## Decision

- Kubernetes runs an **envlet** DaemonSet on nested-virtualization node pools; our Environment Manager places
  environments onto envlets.
- Fork E2B's orchestrator node agent and envd as the starting envlet/envd. Remove Nomad/Consul and E2B's API;
  replace with our manager, placement, pools and SecurityProfile enforcement.
- Our Environment API and Driver SPI are the contracts; E2B's protocols are starting implementations.
- Use kubernetes-sigs/agent-sandbox (gVisor/Kata) as the `pod` driver for low-churn workloads.

## Consequences

- Months of low-level VMM work avoided; the parts that conflict with our design are the parts we replace.
- We carry a fork (Go) and its upstream-tracking cost.
- Nested virtualization overhead (L2 guests) must be measured on target instance types.

## Alternatives considered

- **Build the envlet from scratch**: full control, many months of UFFD/lazy-block/jailer work.
- **agent-sandbox only**: Kubernetes-native but pod-per-sandbox; Firecracker only on its roadmap; no memory fork.
- **Kata Containers**: pod-per-VM, no memory fork; kept as an isolation option under agent-sandbox.

## Validation required

Spike: how entangled the orchestrator is with Nomad/Consul. If extraction is impractical, fall back to building
the envlet and reusing E2B's components selectively (UFFD handler, template builder).
