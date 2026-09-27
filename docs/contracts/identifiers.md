# Identifiers

Status: **Proposed**

All identifiers are opaque strings to consumers unless a structure is listed here as normative. ULIDs give
time-ordering and collision resistance without coordination.

| Identifier | Normative format | Scope | Minted by | Notes |
|---|---|---|---|---|
| `cell_id` | `[a-z0-9]{2,12}` | global | operator | e.g. `use1a` |
| `run_id` | `r_{cell_id}_{ulid}` | global | Control API (cell) | **Embeds the home cell.** Runs never migrate, so routing needs no lookup |
| `partition` | `fnv1a64(run_id) mod P` | cell | derived | `P` is fixed at cell creation (default 4096) and never changes |
| `seq` | uint64, from 0, gapless | run | Run Store (by append) | `seq = 0` is always `run.created` |
| `effect_id` | `{run_id}:{seq}` | global | derived | `seq` of the committed `*.requested` event. Deterministic across retries → the universal idempotency key |
| `lease_epoch` | uint64, monotonic | partition | Run Store | fencing token |
| `worker_id` | `{pod_name}.{boot_nonce}` | cell | runtime | distinguishes restarts of the same pod |
| `session_id` | `{run_id}/{model_slot}` (managed) · `u_{ulid}/{model_slot}` (unmanaged) | global | runtime / Control API | one recorder session per model slot per run |
| `env_id` | `e_{cell_id}_{ulid}` | global | Environment Manager | environments are cell-scoped |
| `attachment_id` | `{env_id}/{run_id}` | cell | Environment Manager | one per (env, run) |
| `snapshot_id` | `s_{region}_{ulid}` | region | Environment Manager | snapshots live in regional object storage and may be restored in any cell of the region |
| `template_id` | `tpl_{sha256}` of the recipe (JCS, `base` resolved to a digest) | region | Environment Manager | content-addressed; equal recipes share one build |
| `pool_id` | `pool/{name}` | cell | operator / API | |
| `policy_id` | `[a-z0-9-]+` | global | Policy Registry | a weights lineage, e.g. `exp42` |
| `weights_version` | uint64, monotonic per `policy_id` | lineage | Policy Registry | `exp42@1731` is a policy version ref |
| `channel` | `{namespace}/{name}` | global | Policy Registry | movable pointer, e.g. `exp42/latest` |
| `renderer_id` | `{family}@{version}` | global | Recorder | e.g. `qwen3@2` |
| `spec_hash` | `sha256(JCS(model-visible ToolSpecification fields))` | global | Tool Router | see [canonical-content](canonical-content.md#toolspecification) |
| `context_digest` | hash chain, see [model-endpoint](model-endpoint.md#contextdelta) | session | runtime / recorder | |
| `job_id` | `j_{ulid}` | global | Rollout Controller | |
| `sample_id` | `{session_id}#{leaf_node_id}` | global | Trajectory Assembler | deterministic → consumers deduplicate on it |
| `cursor` | uint64, gapless per job | job | sample log | position in a job's sample log |
| `request_id` | client-chosen string ≤ 128 B | tenant | client | idempotency key for Control API creates |
| `code_reference` | `{package}@{sha256 of package contents}` | global | package registry | task and agent code; runs are pinned to it |
| checkpoint identity | `{method name}#{ordinal}` | run | task host | ordinal counts calls of that method within the run |

## Rules

- Identifiers MUST NOT be reused. Deleting a resource does not free its id.
- Components MUST NOT parse identifiers except where a structure is normative above (`run_id` → cell,
  `effect_id` → run and seq, `session_id` → run and slot).
- Labels (free-form key/value metadata on runs, sessions, environments) are for filtering and joining; they are
  never used for routing or authorization.
