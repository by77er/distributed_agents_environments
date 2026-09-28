# Principles

Status: **Proposed** · Invariants every layer is held to. A design that violates one needs an ADR.

### P1 — Contracts, not implementations
Every boundary is an interface plus interchangeable implementations. A consumer MUST NOT branch on which
implementation is behind a contract. *Runners, model endpoints, tool bindings, environment layers and sample sinks
are all swappable this way.*

### P2 — The core is a library; everything else is a deployment
Core interfaces are Python protocols with in-process implementations, and the whole core runs on one machine
(one GPU is enough). Durability, fleet-scale execution, multi-tenancy and environment infrastructure are optional
layers that implement the same protocols ([layers and profiles](layers-and-profiles.md)). Upper layers depend on the
core, never the reverse.

### P3 — Code acts only through effects
Program, task and agent code is ordinary `async` Python, but every interaction with the outside world — model
samples, imported tools, environment operations, messages, time, randomness — goes through `run` and its handles,
which turn into *effects* a runner performs. Code that also follows the [determinism rules](../core/harness/determinism.md)
can be replayed, which is how a durable runner makes it survive crashes without being written as a state machine.

### P4 — The agent and the model see canonical content only
The agent sees messages, tool specifications and a declared capability contract — no tokens, policy versions,
environment identities or credentials. The model sees only what the agent selects and the tools offered. The task
sees the environments it creates, but never tokens, policy versions or credentials. *Consequence: policy swaps,
environment backends and reinforcement-learning recording are invisible to the code that decides behavior
(R2, R3, R5).*

### P5 — Effects are at-least-once; receivers make them effectively-once
A durable runner may execute an effect more than once (a crash between performing it and recording it). Every
effect carries a stable identity and an argument digest; receivers deduplicate on the identity and reject a known
identity with a different digest. Effects whose receivers cannot deduplicate are guarded by an attempt marker and
report `OUTCOME_UNKNOWN` rather than being silently retried ([delivery semantics](delivery-semantics.md)).

### P6 — Every effect has a stable identity
`effect_id = {run_id}:{generation}:{ordinal}` is deterministic and identical across re-executions. Every receiver
that can treats it as an idempotency key with three states: absent → execute; in progress → join; done → return
the recorded result.

### P7 — Long-lived things are bounded runs over explicit state
Nothing runs forever. Long runs hand over to new generations; conversations are sequences of runs separated by
`WaitFor` or `End(continue_as=…)`. What must persist across them is explicit, versioned state or a tool — never a
suspended coroutine or an ever-growing replay log.

### P8 — Enforcement lives outside the trust boundary
Security properties are enforced by what hosts untrusted code — the sandbox, the VMM, the host network — never by
anything running inside it. Anything returned from untrusted code is untrusted.

### P9 — Secrets never reach untrusted code
Credentials are held outside task hosts and environments, and are injected by trusted components (tool bindings,
egress proxies) where they are used.

### P10 — The recorded behavior logprob is the only authority on how a token was sampled
No component assumes "sampled by version v" implies a particular distribution. Stale KV, in-flight updates and
numeric drift are all corrected with the same mechanism: importance sampling against recorded logprobs.

### P11 — Optional means removable
Removing an optional component (the recorder, the trajectory pipeline, an optional layer) MUST NOT change whether
runs work — only what data they produce or what failures they survive.

### P12 — Owners destroy; attachments borrow
Every resource (environment, session, snapshot) has one owner, designated at creation (by default the creator),
who is responsible for destroying it. Attaching to a resource never transfers ownership. When a run ends, its runner
destroys whatever the run still owns.

### P13 — Own the contracts, buy the implementations
Build the narrow waists (the core protocols, the recorder, the pump / task-host boundary). Adopt existing
implementations beneath them (DBOS, inference engines, trainers). Adopt *components*, never *platforms* whose
abstractions conflict with ours.

### P14 — Defer choices that are not yet needed
Decide an interface when a consumer needs it; keep undecided areas (environments, trainer choice, tenancy policy)
explicitly marked as open rather than filled with provisional detail.
