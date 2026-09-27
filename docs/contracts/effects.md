# Effects

Status: **Proposed** · Delivery guarantees are defined in [delivery-semantics](../architecture/delivery-semantics.md).

An **effect** is side-effecting work derived from a committed `*.requested` event. Task and agent code never
perform effects; they `await` handles, the task host turns each await into a requested event, and the runtime
performs the effect only after commit (P5).

## Catalog

| Effect | Requested by | Executor (boundary) | Completion events | Delivery pattern | Default retry class | Receiver deduplicates |
|---|---|---|---|---|---|---|
| `model.request` | task host | Model endpoint (B5) | `model.completed` / `model.failed` | eager | idempotent | recorder: yes · direct adapter: no (re-sampling is safe) |
| `environment.lifecycle` | task host | Environment Manager (B10) | `environment.completed` / `environment.failed` | eager | idempotent | yes (`request_id = effect_id`) |
| `environment.call` | task host | envlet → envd (B9) | `environment.completed` / `environment.failed` | eager | per operation (below) | yes (three-state, envd) |
| `tool.request` | task host | Tool Router (B7), imported tools only | `tool.completed` / `tool.failed` / `tool.outcome_unknown` | eager | per tool | per binding |
| `spawn.request` | task host | Run Store (same transaction) or relay | `child.completed` (via inbox when the child terminates) | same-transaction / relay | n/a | keyed insert |
| `message.send` | task host | Run Store (same transaction) or relay | none | same-transaction / relay | n/a | keyed insert |
| `timer.request` | task host | Run Store `timers` + timer relay | `timer.fired` | same-transaction + relay | n/a | keyed insert |
| `session.open` | runtime | Recorder (B5) | `session.opened` | eager | idempotent | yes |
| `environment.attach` | runtime | Environment Manager (B10) | `environment.attached` | eager | idempotent | yes |
| `environment.release` | runtime | Environment Manager (B10) | `environment.released` | eager | idempotent | yes |

`environment.requested` events map to `environment.lifecycle` (`CREATE`, `SNAPSHOT`, `HIBERNATE`,
`RESUME`, `DESTROY`) or `environment.call` (everything else). Retry classes for calls: `GET`, `GET_REFERENCE`,
`LIST`, `STAT`, `LIST_PROVIDED_TOOLS` are `pure`; `EXECUTE`, `PUT`, `REMOVE`, `MOVE`, `CALL_PROVIDED_TOOL` are
`side_effecting` with receiver deduplication, so they are re-dispatched with the same key.

## Envelopes

What executors receive and return. Transport is per boundary; these fields are mandatory on all of them.

```proto
message Effect {
  string      effect_id = 1;   // idempotency key; identical on every attempt
  string      run_id    = 2;
  string      kind      = 3;
  uint32      attempt   = 4;   // 1-based; informational
  Timestamp   deadline  = 5;   // absolute
  bytes       payload   = 6;   // the *.requested payload
  CallContext context   = 7;
}

message CallContext {             // assembled by the runtime from R-visible events
  string run_id = 1;
  string tenant = 2;
  map<string,string> labels = 3;
  map<string, EnvironmentAttachment> environments = 4;   // environment_id → {attachment_id, attachment_token}
  map<string, string> sessions = 5;                      // model slot → session endpoint
}

message Completion {
  string    effect_id   = 1;
  uint32    attempt     = 2;
  Status    status      = 3;   // OK | FAILED | OUTCOME_UNKNOWN
  bytes     payload     = 4;   // e.g. ExecutionResult, ToolResult, SampleResult
  string    error_class = 5;   // matches the *.failed class enums
  string    detail      = 6;
}
```

Attachment tokens live only in `CallContext`, which the runtime assembles; task code holds environment
identifiers, never tokens.

## Rules

1. **Identity.** `effect_id` MUST be forwarded to every downstream system that can use an idempotency key
   (envd, Environment Manager, recorder, HTTP `Idempotency-Key` header, MCP `_meta.idempotency_key`).
2. **Authorization.** The runtime rejects `environment.requested` for any environment the run neither owns nor is
   attached to (task code is untrusted; identifiers can be forged).
3. **Concurrency.** Effects requested in one `Step` are dispatched concurrently. There is no ordering guarantee
   between them; code that needs ordering awaits one before requesting the next.
4. **Limits.** The runtime enforces per-run in-flight effects (default 16) and per-worker in-flight effects.
   Excess effects wait in the worker's dispatch queue — they are already committed, so a crash loses nothing.
5. **Terminal-once.** The first terminal event for an `effect_id` in the log wins; later completions are dropped.
6. **Cancellation** is best-effort: `Cancel(effect_id)` to the executor. The log records the outcome that
   actually arrives (or `DEADLINE`).
