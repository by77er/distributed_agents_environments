# Run events

Status: **Proposed**

The typed record of what happened in a run. Every runner produces this stream: the `LocalRunner` in memory, a
durable runner persisted as a projection alongside its own journal ([durability](../durability/README.md)).
Consumers — the trajectory assembler, client event streams, observability — read these events, never a
substrate's internal format. The catalog is closed: a new type requires a change here.

## Envelope

```python
@dataclass(frozen=True)
class RunEvent:
    run_id: str
    seq: int                       # position in this run's event stream, gapless from 0
    type: str                      # e.g. "observation.recorded"
    schema_version: int
    recorded_at: datetime          # exposed to code as run.now() for inputs
    payload: JsonValue
```

## Catalog

### Lifecycle

| Type | Payload | Notes |
|---|---|---|
| `run.created` | `specification`, `conversation?`, `labels`, `parent{run_id, effect_id}?` | always `seq = 0` |
| `generation.started` | `generation`, `code_references`, `state_reference?` | a hand-over to a new generation ([determinism](../core/harness/determinism.md#generations)) |
| `run.suspended` | `waiting_for: {kind, timeout?}` | a `WaitFor`; no compute held under a durable runner |
| `run.completed` | `outcome: SUCCESS \| FAILURE`, `result` | terminal |
| `run.failed` | `class: TASK_ERROR \| INVALID_OBSERVATION \| NON_DETERMINISM \| POISONED \| INFRASTRUCTURE`, `detail` | terminal |
| `run.cancel_requested` | `reason`, `by` | delivered to code as cancellation; `teardown` still runs |
| `run.cancelled` | | terminal |
| `patch.marked` | `change_id` | recorded by `run.patched()` on new code |

### Episode

| Type | Payload |
|---|---|
| `observation.recorded` | `reply_effect_id?` (absent for the start observation), `messages`, `reward?`, `end?`, `info`, `digest` |
| `reward.assigned` | `slot`, `value`, `key`, `reply_effect_id?`, `target_run_id?` (a descendant, for swarm rewards) |
| `training.excluded` | `reason` — from `run.exclude_from_training` |
| `output.emitted` | `kind`, `payload`, `to?` |

### Effects

| Type | Payload |
|---|---|
| `effect.requested` | `effect_id`, `kind`, `arguments_digest`, `payload` (large payloads by reference) |
| `effect.completed` | `effect_id`, `status: OK \| FAILED \| OUTCOME_UNKNOWN`, `payload`, `error_class?` |

Effect kinds are listed in [effects](effects.md). Model samples complete with the canonical reply and
`usage{context_used, context_limit, input_tokens?, output_tokens?}`; tokens never appear here (they live in the
recorder).

### Conversations and messages

| Type | Payload |
|---|---|
| `message.received` | `envelope`, `priority`, `mode: QUEUE \| STEER \| INTERRUPT` |
| `turn.interrupted` | `reply_effect_id` — the cancelled model sample |

### Tools

| Type | Payload |
|---|---|
| `tools.resolved` | `specifications: [ToolSpecification]` (`@tool` methods + imports) |
| `tools.changed` | `added`, `removed` |

## Run status

```
created ──▶ running ⇄ suspended ──▶ completed · failed · cancelled
```
