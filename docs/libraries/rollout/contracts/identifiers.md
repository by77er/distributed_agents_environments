# Identifiers

Code: `rollout.contracts.identifiers` · See
[API reference](../../../guide/reference.md#effectidentity)

An identifier is an opaque string unless this page gives it a structure.

| Identifier | Structure | Made by | Meaning |
|---|---|---|---|
| `run_id` | `r_{ulid}` | the runner (`new_run_id`), or the caller of `Runner.start` | One run. |
| `seq` | an integer from 0, without gaps | the runner | An event's position in its run's stream. `run.created` is 0. |
| `effect_id` | `{run_id}:{generation}:{ordinal}` | the run context | One effect of a run ([effects](effects.md#identity)). The generation is 0 for every run. The ordinal counts the run's effects from 0 in the order they are requested. |
| `arguments_digest` | SHA-256 in hexadecimal | the run context | The digest of an effect's arguments ([digests](canonical-content.md#digests)). It goes with the `effect_id` to whatever performs the effect. |
| `session_id` | `{run_id}/{model_slot}` | the run context | One model slot of one run: what a model endpoint is asked to describe and sample for. |
| context digest | SHA-256 in hexadecimal | `Model.sample` | Names the messages of one sample request ([digests](canonical-content.md#digests)). |
| spec hash | SHA-256 in hexadecimal | `spec_hash` | Names what a model sees of a tool. |
| blob `sha256` | SHA-256 in hexadecimal | a blob store | The bytes of a blob. The same bytes always have the same reference. |

A ULID is 48 bits of Unix time in milliseconds and 80 random bits, written as 26 characters of Crockford base 32.
Identifiers made from one sort by the time they were made. Runners and services make them; task code does not: its
randomness is `run.random`, seeded from the run's id ([determinism](../determinism.md)).

Identifiers above the harness are defined where they are used: a training run, a group, an episode and an attempt
in [rollouts](../../rollout-train/rollouts.md), a channel's weights version in
[channels](../../rollout-train/channels.md#publishing-weights).

## Rules

- **Only two structures are parsed.** `EffectIdentity.parse` splits an `effect_id` into run, generation and
  ordinal. `SessionIdentity.parse` splits a `session_id` into run and model slot. No other identifier should be
  taken apart.
- **The separators are reserved.** A `run_id` contains no `:` and no `/`.
- **An `effect_id` is the same every time the run's code runs again.** It is the idempotency key for everything a
  run does outside its own code.
- **Labels are not identifiers.** A run's labels are free-form strings, recorded in its `run.created` event and
  copied to its episode. They are for filtering and grouping; nothing is routed by them.
