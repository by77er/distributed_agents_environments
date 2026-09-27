# Principles

Status: **Proposed** · Invariants every component is held to. A design that violates one needs an ADR.

### P1 — Contracts, not implementations
Every boundary is an interface plus interchangeable implementations. A consumer MUST NOT branch on which
implementation is behind a contract. *Tools, environments, models, stores, and sinks are all swappable this way.*

### P2 — A run is its log
All durable run state is an append-only event log. Any worker can reconstruct a run by replaying its log.
Nothing about a run lives only in a process. Snapshots (pickled task state) only shorten replay; they may be
discarded at any time.

### P3 — Task and agent code act only through effects, and are deterministic between them
Task and agent code is ordinary `async` Python, but every interaction with the outside world — model samples,
environment operations, imported tools, time, randomness — goes through `run` and its handles, which turn into
*effects* the runtime performs. Between effects the code is deterministic, so replay reproduces it exactly.
*Consequence: workers never block on a model, one worker hosts thousands of runs, and code survives crashes
without being written as a state machine.*

### P4 — The agent and the model see canonical content only
The agent sees messages, tool specifications, and a declared capability contract — no tokens, policy versions,
environment identities or credentials. The model sees only what the agent selects and the tools offered. The task
sees the environments it owns, but never tokens, policy versions or credentials. *Consequence: policy swaps,
environment backends and reinforcement-learning recording are invisible to the code that decides behavior
(R2, R3, R5).*

### P5 — Persist before effect
An effect is dispatched only after the event requesting it is durably committed under a valid lease.
A worker that cannot commit cannot cause effects.

### P6 — Every effect has a stable identity; receivers dedupe
`effect_id` is deterministic and identical across retries. Every receiver treats it as an idempotency key with
three states: absent → execute; in-progress → join; done → return cached result.

### P7 — Enforcement lives outside the trust boundary
Security properties of an environment are enforced by the host, VMM, and network — never by anything running
inside the guest. Anything returned from a guest is untrusted.

### P8 — Secrets never enter environments
Credentials are held by brokers outside the guest and injected at egress. An environment compromise leaks no
credentials.

### P9 — The recorded behavior logprob is the only authority on how a token was sampled
No component assumes "sampled by version v" implies a particular distribution. Stale KV, in-flight updates,
and numeric drift are all corrected with the same mechanism: importance sampling against recorded logprobs.

### P10 — Optional means removable
Removing an optional component (the recorder, the trajectory pipeline) MUST NOT change whether runs work — only what
data they produce.

### P11 — Owners destroy; attachments borrow
Every resource (environment, session, snapshot) has one owner, designated at creation (by default the creator),
who is responsible for destroying it. Attaching to a resource never transfers ownership. When a run ends, the
runtime destroys whatever the run still owns.

### P12 — Own the contracts, buy the implementations
Build the narrow waists (run log, task and agent interfaces, environment contract, recorder). Adopt existing
implementations beneath them. Adopt *components*, never *platforms* whose abstractions conflict with ours.

### P13 — Cells are the unit of scale and failure
A cell is a self-contained slice (cluster, store, workers, env hosts). Scale by adding cells; contain failures
within one. Cross-cell interaction is rare, explicit, and asynchronous.
