# Effects

For whoever implements a runner or a receiver of effects: the operations that leave program code, their identity,
completion and deduplication.

**Read first:** [Determinism](../determinism.md). **Next:** [Model endpoint](model-endpoint.md).

Code: `rollout.contracts.effects`, `rollout.harness.model` (`Effects`) · See
[API reference](../../../guide/reference.md#effectkind), [determinism](../determinism.md)

An **effect** is an operation that reaches outside task, agent or program code. Code requests effects by awaiting
`run` methods and the handles `run` gives; the run context performs them. Under the `LocalRunner` an effect is a
direct call, recorded as run events.

## Catalog

| Kind | Requested through | Performed by | Completes with |
|---|---|---|---|
| `model.sample` | `run.models[slot].sample` | the slot's model endpoint | the sample result: message, finish reason, usage |
| `tool.call` | `run.tools.call`, which the default `respond` uses for imported tools; `run.sandbox(name).call` | the import's tool set; the sandbox's [pool](../sandboxes.md#pools) | the tool result |
| `output.emit` | `run.emit` | the run context | nothing. An `output.emitted` event follows it. |

`@tool` methods are task code, not effects. The effects they request, such as imported tool calls, are.

## Identity

Every effect has an `effect_id` and an `arguments_digest`.

- The **`effect_id`** is `{run_id}:{generation}:{ordinal}` ([identifiers](identifiers.md)). The ordinal is the
  effect's place in the order the run requested its effects. Code that runs again requests the same effects in the
  same order, so an effect has the same id on every execution.
- The **`arguments_digest`** is the [digest](canonical-content.md#digests) of the effect's arguments. They are the
  `payload` of its `effect.requested` event.

| Kind | Arguments |
|---|---|
| `model.sample` | the `session_id`, the context digest, the spec hashes of the tools offered, `max_output_tokens`, `tool_choice` |
| `tool.call` | the tool's name and the call's arguments, and for a sandbox's operation the sandbox's name |
| `output.emit` | `kind` and `payload` |

Both identifiers go to whatever performs the effect: a model endpoint receives them in the `SampleRequest`, a tool
set or a pool as arguments of `call` (and in the body of `POST /call` when it is served over HTTP).

## Completion

An effect records two events: `effect.requested` when it starts and `effect.completed` when it finishes
([run events](run-events.md#effects)). The status is one of
[`EffectStatus`](../../../guide/reference.md#effectstatus):

| Status | When | What the code that requested it sees |
|---|---|---|
| `ok` | the effect was performed | its result |
| `failed` | whatever performed it raised, or the run was cancelled while it was in flight | the exception |

## Receivers that deduplicate

A receiver that performs each `effect_id` at most once makes an effect safe to request again. Whether an effect is
safe to request again depends on its receiver:

| Effect | Receiver | Safe to request again |
|---|---|---|
| `model.sample` | the [gateway](../../rollout-train/gateway.md) returns the recorded result for an `effect_id` it has recorded | yes |
| `model.sample` | a direct adapter does not deduplicate | yes: it samples again, and nothing is lost |
| `tool.call` of a tool whose `retry_class` is `PURE` or `IDEMPOTENT` | any tool set | yes |
| `tool.call` of a tool whose `retry_class` is `SIDE_EFFECTING` or `UNKNOWN` | a [`DeduplicatingToolSet`](../../../guide/reference.md#deduplicatingtoolset) whose `deduplicates` is true | yes: the tool set performs it at most once |
| the same | any other tool set | no |
| `output.emit` | the run context | yes: it has nothing to perform |

A tool set says that it deduplicates with a `deduplicates` attribute. One that does not say is treated as one that
does not. A tool set served over HTTP reports the attribute of the tool set behind it, so a `ToolBinding(url=...)`
keeps the guarantee ([tools](../../../guide/tools.md#retry-classes)). A sandbox's operations follow the same rules, with
its pool's `deduplicates` in place of a tool set's ([sandboxes](../sandboxes.md#what-a-program-sees)).

A receiver that deduplicates is sent the `arguments_digest` so that it can tell a repeat from a different call: a tool
set that finds a known `effect_id` with a different digest raises
[`Conflict`](../../../guide/reference.md#rolloutcontractsconflict), because the two requests were not the same effect.
The gateway looks a recorded turn up by its `effect_id` alone.

## Rules

1. **Concurrency.** Effects awaited concurrently are performed concurrently. They are numbered in the order they
   are requested, which is the order their branches start.
2. **Cancellation.** Cancelling a sample cancels the task that awaits it and asks the endpoint to cancel the
   `effect_id`. The endpoint's `cancel` is best-effort.
3. **Retries inside an effect.** `Model.sample` retries an overloaded or failing endpoint within one effect, under
   one `effect_id` ([model endpoint](model-endpoint.md#errors)).
