# Effects

Code: `rollout.contracts.effects`, `rollout.harness.model` (`Effects`) · See
[API reference](../../../guide/reference.md#effectkind), [determinism](../determinism.md)

An **effect** is an operation that reaches outside task, agent or program code. Code requests effects by awaiting
`run` methods and the handles `run` gives; the run context performs them. Under the `LocalRunner` an effect is a
direct call. Under the `DurableRunner` it is a recorded step: once it has completed, a replay returns its recorded
result and performs nothing ([durability](../../../implementations/rollout-durable/README.md#effects)).

## Catalog

| Kind | Requested through | Performed by | Completes with |
|---|---|---|---|
| `model.sample` | `run.models[slot].sample` | the slot's model endpoint | the sample result: message, finish reason, usage |
| `tool.call` | `run.tools.call`, which the default `respond` uses for imported tools | the import's tool set | the tool result |
| `environment.call` | `Environment.execute`, `put`, `get` | the runner's environment service | the execution result; nothing; the size read |
| `environment.lifecycle` | `run.environments.create`, `Environment.destroy` | the runner's environment service | the `environment_id`; nothing |
| `output.emit` | `run.emit` | the run context | nothing. An `output.emitted` event follows it. |

`@tool` methods are task code, not effects. The effects they request, such as environment calls, are.

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
| `tool.call` | the tool's name and the call's arguments |
| `environment.call` | the `environment_id`, the operation, and the command with its time limit and directory, or the path. A `put` gives the SHA-256 and size of the content, not the content. |
| `environment.lifecycle` | the operation, and the environment's specification for a creation or its `environment_id` for a destruction |
| `output.emit` | `kind`, `payload`, and `to` when given |

Both identifiers go to whatever performs the effect: a model endpoint receives them in the `SampleRequest`, a tool
set as arguments of `call` (and in the body of `POST /call` when it is served over HTTP), an environment service
receives the `effect_id` of a command.

## Completion

An effect records two events: `effect.requested` when it starts and `effect.completed` when it finishes
([run events](run-events.md#effects)). The status is one of
[`EffectStatus`](../../../guide/reference.md#effectstatus):

| Status | When | What the code that requested it sees |
|---|---|---|
| `ok` | the effect was performed | its result |
| `failed` | whatever performed it raised, or the run was cancelled or interrupted while it was in flight | the exception |
| `outcome_unknown` | a guarded effect was interrupted by a crash in an earlier attempt, under the durable runner | `OutcomeUnknown`: it may or may not have happened, and it is not performed again |

## Receivers that deduplicate

A receiver that performs each `effect_id` at most once makes an effect safe to request again. What a durable runner
does with an effect that a crash interrupted depends on whether its receiver does:

| Effect | Receiver | After a crash interrupted it |
|---|---|---|
| `model.sample` | the gateway returns the recorded result for an `effect_id` it has recorded | requested again |
| `model.sample` | a direct adapter does not deduplicate | requested again: it samples again, and nothing was lost |
| `environment.lifecycle`, and the `put` and `get` of `environment.call` | an environment service makes these safe to repeat. A creation finds the environment it already made, because the id comes from the `effect_id`. | requested again |
| `Environment.execute` | a command is not safe to repeat | guarded: `OutcomeUnknown` |
| `tool.call` of a tool whose `retry_class` is `PURE` or `IDEMPOTENT` | any tool set | requested again |
| `tool.call` of a tool whose `retry_class` is `SIDE_EFFECTING` or `UNKNOWN` | a [`DeduplicatingToolSet`](../../../guide/reference.md#deduplicatingtoolset) whose `deduplicates` is true | requested again: the tool set performs it at most once |
| the same | any other tool set | guarded: `OutcomeUnknown`. The task loop gives the model an error result that says the call may or may not have taken effect. |
| `output.emit` | the run context | requested again: it has nothing to perform |

A tool set says that it deduplicates with a `deduplicates` attribute. One that does not say is treated as one that
does not. A tool set served over HTTP reports the attribute of the tool set behind it, so a `ToolBinding(url=...)`
keeps the guarantee ([tools](../../../guide/tools.md#after-a-crash)).

A receiver that deduplicates is sent the `arguments_digest` so that it can tell a repeat from a different call: a
tool set that finds a known `effect_id` with a different digest raises
[`Conflict`](../../../guide/reference.md#conflict), because the code that ran again did not request the same effect.
The helper `rollout_durable.database.recorded` does this for a tool set that writes to a database. The gateway
looks a recorded turn up by its `effect_id` alone.

## Rules

1. **Concurrency.** Effects awaited concurrently are performed concurrently. They are numbered in the order they
   are requested, which is the order their branches start.
2. **Cancellation.** Cancelling a sample cancels the task that awaits it and asks the endpoint to cancel the
   `effect_id`. The endpoint's `cancel` is best-effort.
3. **Retries inside an effect.** `Model.sample` retries an overloaded or failing endpoint within one effect, under
   one `effect_id` ([model endpoint](model-endpoint.md#errors)).
