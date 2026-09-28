# Identifiers

Status: **Proposed**

All identifiers are opaque strings to consumers unless a structure is listed here as normative. ULIDs give
time-ordering and collision resistance without coordination.

## Core

| Identifier | Normative format | Minted by | Notes |
|---|---|---|---|
| `run_id` | `r_{ulid}`; in the platform layer `r_{cell_id}_{ulid}` | runner / Control API | never reused |
| `generation` | uint32, from 0 | runner | increments at each hand-over ([determinism](../core/harness/determinism.md#generations)) |
| `seq` | uint64, from 0, gapless per run | runner | position in the run's event stream; `seq = 0` is `run.created` |
| `effect_id` | `{run_id}:{generation}:{ordinal}` | derived | `ordinal` = position of the effect in the generation's deterministic request order; identical across re-executions → the universal idempotency key |
| `arguments_digest` | `sha256(JCS(arguments))` | derived | sent with every `effect_id`; receivers reject a known `effect_id` with a different digest |
| `conversation key` | `{deployment}/{key}` | caller | `key` is caller-chosen, e.g. `slack:T1/C2/171.2` |
| `message_id` | the sender's `effect_id`, or the caller's idempotency key | runner | deduplication key for envelopes |
| deployment name | `{namespace}/{name}` | operator / API | e.g. `acme/support-bot` |
| `code_reference` | `{package}@{sha256 of package contents}` | package registry | runs are pinned to it |
| `spec_hash` | `sha256(JCS(model-visible ToolSpecification fields))` | core | see [canonical-content](canonical-content.md#toolspecification) |
| `session_id` | `{run_id}/{model_slot}` (runs) · `u_{ulid}/{model_slot}` (unmanaged) | runner / recorder | one recorder session per model slot per run |
| `context_digest` | hash chain, see [model-endpoint](model-endpoint.md#contextdelta) | core / recorder | |
| `renderer_id` | `{family}@{version}` | recorder | e.g. `qwen3@2` |
| `policy_id` | `[a-z0-9-]+` | policy registry | a weights lineage, e.g. `exp42` |
| `weights_version` | uint64, monotonic per `policy_id` | policy registry | `exp42@1731` is a policy version reference |
| `channel` | `{namespace}/{name}` | policy registry | movable pointer, e.g. `exp42/latest` |
| `job_id` | `j_{ulid}` | rollout API | |
| `sample_id` | `{session_id}#{leaf_node_id}` | trajectory assembler | deterministic → consumers deduplicate on it |
| `cursor` | uint64, gapless per job | sample log | position in a job's sample log |
| `request_id` | client-chosen string ≤ 128 B | client | idempotency key for creates |

## Platform layer

| Identifier | Normative format | Notes |
|---|---|---|
| `cell_id` | `[a-z0-9]{2,12}` | embedded in `run_id`; runs never migrate between cells |
| `executor_id` | `{pod_name}` | stable identity of a durable executor, used for recovery |

Environment identifiers (environments, templates, snapshots) are defined by the
[environment system](../environments/README.md).

## Rules

- Identifiers MUST NOT be reused. Deleting a resource does not free its identifier.
- Components MUST NOT parse identifiers except where a structure is normative above (`run_id` → cell in the
  platform layer, `effect_id` → run, generation and ordinal, `session_id` → run and slot).
- Labels (free-form key/value metadata on runs and sessions) are for filtering and joining; they are never used for
  routing or authorization.
