# 0016 — A core library with optional layers, and deployment profiles

Status: **Accepted** · Date: 2026-09-27

## Context

The design had grown from the fleet downward: network services with IDL at every boundary, cells, a run store,
tenancy. Most of what task authors, agent authors and trainers use — the loop, tools, the model endpoint, the
recorder, samples, the rollout API, weight publishing — does not depend on any of that. It should work with one
trainer and one inference engine on a single local GPU.

## Decision

- The system is a **core library** plus optional layers: **inference** (some form required), **environments** (a
  separate system, designed later), **durability**, **platform**.
- Core interfaces are **Python protocols first**; every core protocol has an in-process implementation. Network
  forms are implementations, never definitions.
- Upper layers depend on the core, never the reverse.
- Three **deployment profiles**: local (one process, one GPU), cluster, fleet.
- A task that uses no environment is a complete task.

See [layers and profiles](../architecture/layers-and-profiles.md).

## Consequences

- The local profile is the development, test and research path, and the reference implementation of every protocol.
- Durability and fleet concerns no longer shape core interfaces; they implement them.
- Docs are organized by layer.

## Alternatives considered

- **Service-first design** (the previous revision): every boundary a network API. Rejected: it forced
  infrastructure onto single-machine use and made the core hard to test.
