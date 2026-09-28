# Design docs

This tree defines the system's **boundaries and interfaces**. To build with the code as it exists, start with the
[developer guide](guide/README.md). It is organized by layer:
a **core library** that runs on one machine, plus **optional layers** that deploy the same protocols with durability,
at fleet scale, or with computers for tasks ([layers and profiles](architecture/layers-and-profiles.md)).

## Layout

```
docs/
├── README.md                          ← you are here
├── architecture/                      system-level: what exists and how it fits
│   ├── requirements.md                requirements every decision traces back to
│   ├── principles.md                  invariants every layer is held to
│   ├── layers-and-profiles.md         the layers; local, cluster and fleet profiles
│   ├── overview.md                    components by layer, interfaces between layers, visibility
│   ├── turn-lifecycle.md              one turn under the local and the durable runner
│   ├── delivery-semantics.md          effect identity, deduplication, retry classes, messages
│   ├── trust-boundaries.md            trust zones and rules
│   ├── glossary.md
│   └── open-questions.md              deferred topics, open questions, spike plan
├── contracts/                         types that cross layers (defined once, here)
│   ├── identifiers.md  canonical-content.md  run-events.md  effects.md  model-endpoint.md
├── core/                              the library: Python protocols with in-process implementations
│   ├── harness/                       loop, Program, Task, Agent, conversations, Runner, determinism rules
│   ├── recorder/                      token-exact recording, session trees, engine adapter
│   ├── rollouts/                      rows in, samples out, weights published
│   └── trajectories/                  Sample and its assembly
├── inference/                         engines, policy registry, weight transfer and transitions
├── environments/                      PRELIMINARY: a separate system, designed later
├── durability/                        optional: DurableRunner on DBOS, the pump, the task host
├── platform/                          optional: cells, trust tiers, Control API, tool router
├── development/plan.md                milestones, exit criteria, first tasks
├── guide/                             developer guide to the code as built; generated API reference
├── decisions/                         ADRs: why each load-bearing choice was made
└── research/                          investigations behind the decisions (evidence, not normative)
```

## Reading order

1. [requirements](architecture/requirements.md) → [principles](architecture/principles.md) →
   [layers and profiles](architecture/layers-and-profiles.md) → [overview](architecture/overview.md)
2. The core: [harness](core/harness/README.md) → [task](core/harness/task.md) → [agent](core/harness/agent.md) →
   [conversations](core/harness/conversations.md) → [recorder](core/recorder/README.md) →
   [rollouts](core/rollouts/README.md) → [trajectories](core/trajectories/README.md), with
   [contracts](contracts/README.md) as reference
3. [inference](inference/README.md), then the optional layers: [durability](durability/README.md),
   [platform](platform/README.md)
4. [decisions](decisions/README.md) and [research](research/README.md) for the reasoning behind any of the above

## Conventions

- **Normative language.** MUST / MUST NOT / SHOULD / MAY follow RFC 2119.
- **Python protocols first.** Core interfaces are defined as Python protocols and dataclasses. Network forms (gRPC /
  HTTP, written in a protobuf-flavored IDL where shown) are implementations and are normative only for semantics.
- **Define once.** Every type has exactly one defining document. Types crossing layers live in `contracts/`.
- **Full type names.** Never abbreviate type names (`Observation`, not `Obs`).
- **Status.** `Draft` (being written), `Preliminary` (notes; not decided), `Proposed` (complete, awaiting
  agreement), `Accepted` (agreed; changes need an ADR).
- **Defer what is not needed yet** (P14): mark undecided areas as open rather than filling them in.
