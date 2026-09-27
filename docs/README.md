# Design docs

This tree defines the system's **boundaries and interfaces** before any implementation.
Nothing here is code; everything here is a contract that code will be held to.

## Layout

```
docs/
├── README.md                     ← you are here
├── architecture/                 system-level: what exists and how it fits
│   ├── requirements.md           the requirements every decision traces back to
│   ├── principles.md             invariants that constrain every component
│   ├── overview.md               component map, boundary map (B1–B18), visibility matrix
│   ├── turn-lifecycle.md         one agent turn, end to end, across every boundary
│   ├── delivery-semantics.md     fencing, outbox, idempotency, retry classes
│   ├── trust-boundaries.md       trust zones, credentials, untrusted content
│   ├── cells.md                  unit of scale and failure; what is per-cell vs global
│   ├── glossary.md
│   └── open-questions.md         every unresolved question, linked to its owner doc
├── contracts/                    types that cross two or more boundaries (defined once, here)
│   ├── README.md                 evolution and compatibility rules
│   ├── identifiers.md            run_id, effect_id, session_id, versions, …
│   ├── canonical-content.md      model-agnostic messages, content blocks, ToolSpecification
│   ├── run-events.md             every event type in the run log
│   ├── effects.md                effect kinds, identity, completion routing
│   └── model-endpoint.md         the model contract task and agent code depend on
├── components/                   one directory per component; same template everywhere
│   ├── control-api/              external API: runs, signals, environments, jobs
│   ├── runtime/                  durable actor host: partitions, leases, dispatch, timers
│   ├── harness/                  the rollout loop, RunSpecification, task host protocol
│   │   ├── task.md               Task: hooks, Observation, tools, environment handles, RunContext
│   │   ├── agent.md              Agent: context selection, acting
│   │   └── durability.md         replay, checkpoints, snapshots, determinism, versioning
│   ├── run-store/                the log, leases, inbox, timers (Postgres reference impl)
│   ├── tool-router/              imported external tools: binding SPI, MCP import / facade
│   ├── environments/             environment API (envd), driver SPI, security, placement
│   ├── recorder/                 session API, session trees, engine adapter
│   ├── inference/                policy registry, channels, weight updates, engine requirements
│   ├── rollouts/                 rollout jobs: rows in, samples out, weights published
│   └── trajectories/             assembler, Sample schema, per-job sample log, trainer adapters
└── decisions/                    ADRs: why each load-bearing choice was made
```

## Reading order

1. [requirements](architecture/requirements.md) → [principles](architecture/principles.md) → [overview](architecture/overview.md)
2. [contracts](contracts/README.md), then [turn-lifecycle](architecture/turn-lifecycle.md) and [delivery-semantics](architecture/delivery-semantics.md)
3. The authoring model: [harness](components/harness/README.md) → [task](components/harness/task.md) → [agent](components/harness/agent.md) → [durability](components/harness/durability.md)
4. Components, in dependency order: run-store → runtime → tool-router → environments → recorder → inference → rollouts → trajectories → control-api
5. [decisions](decisions/README.md) for the reasoning behind any of the above

## Conventions

- **Normative language.** MUST / MUST NOT / SHOULD / MAY follow RFC 2119.
- **IDL.** Interface blocks use a protobuf-flavored IDL. They are normative for *semantics* — operations,
  fields, errors, guarantees — not for wire format. Wire format is chosen per boundary and recorded in an ADR.
- **Define once.** Every type has exactly one defining document. Types crossing ≥2 boundaries live in
  `contracts/`; types private to one boundary live in that component's doc. Everything else links.
- **Status.** Every doc carries one of: `Draft` (being written), `Proposed` (complete, awaiting agreement),
  `Accepted` (agreed; changes need an ADR).
- **Component template.** Every component doc has these sections, in this order:
  Purpose · Owns / does not own · Boundaries · Interface · Semantics & guarantees ·
  State & durability · Failure modes · Scale envelope · Build vs buy · Open questions.
