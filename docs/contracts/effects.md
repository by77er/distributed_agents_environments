# Effects

Status: **Proposed** · Delivery guarantees: [delivery-semantics](../architecture/delivery-semantics.md) · See [ADR-0018](../decisions/0018-effects-at-least-once.md)

An **effect** is an operation that reaches outside task, agent or program code: a model sample, an imported tool
call, an environment operation, a message, a child run, a timer. Code requests effects by awaiting `run` methods
and handles; a runner performs them. Under the `LocalRunner` an effect is a direct call. Under a durable runner it is
a recorded step ([durability](../durability/README.md)).

## Catalog

| Effect | Requested through | Executor | Completion | Receiver deduplicates |
|---|---|---|---|---|
| `model.sample` | `run.models[slot].sample` | model endpoint (recorder or direct adapter) | reply message | recorder: yes · direct adapter: no (re-sampling is harmless) |
| `tool.call` | an imported tool in the action space | `ToolBinding` (in process, or the tool router) | `ToolResult` or failure | per binding |
| `environment.call` | `Environment` handle methods | environment layer | operation result or failure | yes (required of environment services) |
| `environment.lifecycle` | `run.environments.create`, `destroy`, … | environment layer | handle or failure | yes |
| `message.send` | `run.send` | runner | none | yes (by `message_id`) |
| `message.wait` | `WaitFor` | runner | the envelope, or timeout | n/a |
| `run.spawn` | `run.spawn` | runner | child outcome | yes (by child `run_id`) |
| `timer.sleep` | `run.sleep` | runner | wake-up | n/a |
| `output.emit` | `run.emit` | runner → connectors | none | yes (by `effect_id`) |

`@tool` methods are task code, not effects; the effects they make (e.g. environment calls) are.

## Envelope

What an executor receives. Transport is per implementation; these fields are mandatory everywhere.

```python
@dataclass(frozen=True)
class EffectRequest:
    effect_id: str                 # {run_id}:{generation}:{ordinal}; identical on every attempt
    arguments_digest: str          # sha256 of the canonical JSON of the arguments
    kind: str
    run_id: str
    attempt: int                   # 1-based; informational
    deadline: datetime             # absolute
    payload: JsonValue             # JSON only
    retry_class: RetryClass        # PURE | IDEMPOTENT | SIDE_EFFECTING | UNKNOWN
    context: CallContext           # assembled by the runner, never by task code (tenant, labels, grants, attachments)

@dataclass(frozen=True)
class EffectCompletion:
    effect_id: str
    status: Status                 # OK | FAILED | OUTCOME_UNKNOWN
    payload: JsonValue
    error_class: str | None = None
```

## Rules

1. **Identity.** `effect_id` and `arguments_digest` are forwarded to every receiver that can use them (the
   recorder, environment services, HTTP `Idempotency-Key`, MCP `_meta.idempotency_key`).
2. **Three-state deduplication** at receivers that support it: absent → execute; in progress → join the running
   execution; done → return the recorded result. A known `effect_id` with a different digest is rejected with
   `CONFLICT` — never answered from the cache.
3. **Non-deduplicating receivers.** For `SIDE_EFFECTING` or `UNKNOWN` tools whose binding cannot deduplicate, a
   durable runner writes an attempt marker before dispatch; a re-execution that finds the marker completes with
   `OUTCOME_UNKNOWN` instead of calling again.
4. **Authorization.** Runners reject effects that address resources the run does not own or is not attached to.
   Task code can forge identifiers; `CallContext` is assembled by the runner.
5. **Concurrency.** Effects awaited concurrently are dispatched concurrently; there is no ordering between them.
6. **Terminal-once.** The first completion recorded for an `effect_id` wins; later ones are dropped.
7. **Cancellation** is best-effort and reported by the completion that actually arrives.
