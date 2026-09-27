# Run events

Status: **Proposed**

Every record in a run's log. The log is the run (P2); this catalog is closed — a new event type requires a change
here.

## Envelope

```proto
message RunEvent {
  string    run_id         = 1;
  uint64    seq            = 2;   // gapless, from 0
  string    type           = 3;   // e.g. "environment.completed"
  uint32    schema_version = 4;
  Timestamp committed_at   = 5;   // store commit time; exposed to code as run.now()
  uint64    writer_epoch   = 6;   // lease epoch of the committing worker
  bytes     payload        = 7;   // type-specific, below
}
```

## Visibility

- **T** — task-visible: replayed into the task host (`Load`) and, when an input, passed to `Step`.
- **R** — runtime-only: never passed to the task host (bindings, attachments, operational records).
- **Emitter**: `task` (returned from `Step`: task, agent, or loop code), `runtime`, or `store` (written by another
  run's transaction or by the Control API via the inbox).

Within the task host, the agent never receives environment identities: environment events resolve handle calls
made by task code, and only observations reach the agent.

## Catalog

### Lifecycle

| Type | Vis | Emitter | Payload | Notes |
|---|---|---|---|---|
| `run.created` | T* | runtime | `specification: RunSpecification`, `labels`, `parent{run_id, effect_id}?` | *The host sees task and agent references and parameters — never `binding`. Always `seq = 0`; first `Step` input |
| `run.suspended` | T | task | `waiting_for: SIGNAL \| EFFECTS`, `signal_kind?`, `until?` | runtime may evict the host state; wakes on inbox or timer |
| `run.completed` | T | task | `outcome: SUCCESS \| FAILURE`, `result: [Block]` | terminal |
| `run.failed` | T | task / runtime | `class: TASK_ERROR \| INVALID_OBSERVATION \| NON_DETERMINISM \| INFRASTRUCTURE`, `detail` | terminal |
| `run.cancel_requested` | T | store | `reason`, `by` | input; raised in task code as cancellation; `teardown` still runs |
| `run.cancelled` | R | runtime | | terminal |
| `run.crashed` | R | runtime | `last_seq` | terminal; `best_effort` runs only |
| `code.upgraded` | R | runtime | `from`, `to` (code references) | at resumable points only; see [versioning](../components/harness/durability.md#versioning) |
| `patch.marked` | T | task | `change_id` | recorded by `run.patched()` on live execution |

### Episode

| Type | Vis | Emitter | Payload |
|---|---|---|---|
| `observation.recorded` | T | task | `reply_effect_id?` (absent for the start observation), `messages`, `reward?`, `end?: TERMINATED \| TRUNCATED`, `info` |
| `reward.assigned` | T | task | `slot`, `value`, `key`, `reply_effect_id?` — out-of-band and episode-level rewards (`run.reward`, `score`) |
| `checkpoint.completed` | T | task | `name`, `ordinal`, `return_value`, `task_state: BlobReference` |

### Model

| Type | Vis | Emitter | Payload |
|---|---|---|---|
| `model.requested` | T | task | `slot`, `context: ContextDelta`, `tool_names` (subset exposed this turn), `max_output_tokens?`, `tool_choice?`, `deadline_ms?` |
| `model.completed` | T | runtime | `effect_id`, `message: Message(ASSISTANT)`, `finish_reason`, `usage{context_used, context_limit, input_tokens?, output_tokens?}` |
| `model.failed` | T | runtime | `effect_id`, `class: CONTEXT_OVERFLOW \| CONTRACT_VIOLATION \| DEADLINE \| UNAVAILABLE \| INTERNAL`, `detail` |

### Environments

| Type | Vis | Emitter | Payload |
|---|---|---|---|
| `environment.requested` | T | task | `operation` (below), `environment_id?`, `arguments`, `deadline_ms?` |
| `environment.completed` | T | runtime | `effect_id`, `result` (operation-specific: `ExecutionResult`, `FileInformation[]`, `Snapshot`, environment handle, …) |
| `environment.failed` | T | runtime | `effect_id`, `class: UNSATISFIABLE \| LOST \| DENIED \| INVALID_ARGUMENTS \| DEADLINE \| INTERNAL`, `detail` |

Operations — lifecycle (served by the Environment Manager): `CREATE`, `SNAPSHOT`, `HIBERNATE`, `RESUME`,
`DESTROY`. Calls (served by envlet → envd): `EXECUTE`, `PUT`, `GET`, `GET_REFERENCE`, `LIST`, `STAT`, `REMOVE`,
`MOVE`, `LIST_PROVIDED_TOOLS`, `CALL_PROVIDED_TOOL`.

### Imported tools

| Type | Vis | Emitter | Payload |
|---|---|---|---|
| `tools.resolved` | T* | runtime | `specifications: [ToolSpecification]` (task `@tool` methods + imports); host sees model-visible fields + `retry_class`; import bindings are R-only |
| `tools.changed` | T | task | `added`, `removed` — explicit only (e.g. environment-provided tools added in `setup`) |
| `tool.requested` | T | task | `call_id`, `tool_name`, `spec_hash`, `arguments`, `deadline_ms?` — **imported tools only**; `@tool` methods run in the host |
| `tool.completed` | T | runtime | `effect_id`, `result: ToolResult` |
| `tool.failed` | T | runtime | `effect_id`, `class: INVALID_ARGUMENTS \| DENIED \| DEADLINE \| BINDING_ERROR`, `detail` |
| `tool.outcome_unknown` | T | runtime | `effect_id` — dispatched, not deduplicable, result lost |

### Multi-run and signals

| Type | Vis | Emitter | Payload |
|---|---|---|---|
| `spawn.requested` | T | task | `specification: RunSpecification`, `labels` — child created in the same transaction |
| `child.completed` | T | store | `effect_id`, `child_run_id`, `outcome`, `result` |
| `message.sent` | T | task | `to_run_id`, `payload: [Block]`, `correlation_id?` — no completion |
| `signal.received` | T | store | `kind`, `from`, `payload`, `inbox_id` |

### Timers

| Type | Vis | Emitter | Payload |
|---|---|---|---|
| `timer.requested` | T | task | `after_ms`, `tag` — `run.sleep`, signal timeouts |
| `timer.fired` | T | store | `effect_id`, `tag` |

### Runtime-internal

| Type | Vis | Emitter | Payload |
|---|---|---|---|
| `session.opened` | R | runtime | `slot`, `session_id`, `endpoint` |
| `environment.attached` | R | runtime | `environment_id`, `attachment_id`, `owned: bool` |
| `environment.released` | R | runtime | `environment_id`, `action: DETACH \| DESTROY` — backstop cleanup at terminal state |
| `effect.redispatched` | R | runtime | `effect_id`, `attempt`, `reason` — observability only |

## Run status machine

```
            ┌──────────────── cancel ───────────────┐
pending ──▶ running ⇄ suspended                      ▼
               │                    completed · failed · cancelled · crashed
               └────────────────────────▲
```

`status` is derived from the log and materialized on the run row by the same transaction that appends the
causing event (see [run-store](../components/run-store/README.md)).
