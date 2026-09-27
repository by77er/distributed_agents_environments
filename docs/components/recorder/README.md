# Recorder

Status: **Proposed** · Boundaries B5, B6, B14, B17, B18 · See [ADR-0007](../../decisions/0007-active-recorder.md), [ADR-0008](../../decisions/0008-mid-rollout-policy-change.md), [ADR-0009](../../decisions/0009-stale-kv-importance-sampling.md)

## Purpose

An optional proxy between model consumers (managed runtime or unmanaged harnesses) and inference. In **active**
mode it owns tokenization: it keeps each session's exact token sequence, extends it incrementally, sends
tokens-in requests to engines, and records every sampled span with honest behavior logprobs and the weights
version that produced it. It is what makes any harness trainable (R6, R8) without the harness knowing (R5).

## Owns / does not own

| Owns | Does not own |
|---|---|
| Rendering canonical content ↔ tokens (renderers) for recorded channels | Which replica serves a request (inference router) |
| Session trees: token spans, logprobs, versions, attribution | Weights and weight loading (WUC) |
| Sample dedupe on `effect_id` | Channel → version resolution policy (Policy Registry) |
| Validating sampling params against channel contracts | Rewards (task code) and trajectories (Assembler) |
| Splitting generation across weight transitions (abort + resubmit) | The run log |

## Boundaries

| Direction | Counterpart | Contract |
|---|---|---|
| provides → | Runtime (B5) | [model-endpoint](../../contracts/model-endpoint.md) via [session-api](session-api.md) |
| provides → | Unmanaged harnesses (B18) | OpenAI/Anthropic-compatible endpoints in [session-api](session-api.md) |
| provides → | Trajectory Assembler (B14) | [session-tree](session-tree.md) export |
| consumes | Inference engines (B6) | [engine-adapter](engine-adapter.md) |
| consumes | Policy Registry / WUC (B17) | [inference](../inference/README.md) |
| consumes | Object storage | tree segments |

## Modes

| Mode | Tokenization | Drift | Requires | Use |
|---|---|---|---|---|
| **active** | recorder (renderers) → tokens-in | none within an epoch | renderer for the model family; tokens-in engine | **trainable channels** (default) |
| passive | engine renders messages | detected, not prevented | engine returns prompt/completion token ids | API models, evals |

## Semantics & guarantees

- **Optional (P10).** Runs work identically without it; the RunBinding simply uses a direct adapter.
- **Exactness.** Within a renderer epoch, the tokens recorded as context for turn *k+1* contain turn *k*'s
  sampled tokens verbatim — never re-tokenized text.
- **Honest logprobs (P9).** Recorded logprobs are of the distribution actually sampled from (processed, after
  temperature). Requests with sampling params that would make this false on a trainable channel are rejected.
- **Single version per span.** Every engine response is produced by one weights version (guaranteed by the WUC's
  abort-before-update); a generation interrupted by a weight transition becomes two spans.
- **Idempotent samples.** A repeated `effect_id` returns the cached result — no second sample, no orphan branch.
- **Invisible.** Nothing in responses to model consumers reveals tokens, versions, or engines.

## State & durability

- Hot: active sessions in memory, sticky to one recorder instance by consistent hashing on `session_id`
  (the same key the inference router uses for cache affinity).
- Durable: tree segments flushed to object storage — per turn (`flush: TURN`) or batched (`flush: BATCH`,
  default every 10 s and at close). Loss of unflushed state marks the session **incomplete**.

## Failure modes

| Failure | Effect |
|---|---|
| Instance crash | Sessions it held lose unflushed spans → marked incomplete; runs continue (runtime retries; new instance re-renders from full context via `NEED_FULL_CONTEXT`) |
| Renderer mismatch (harness normalized a message) | New epoch started; `epoch_reason = PREFIX_MISMATCH`; mismatch rate is a metric |
| Engine abort not returning partials | Falls back to discarding the partial and resampling; recorded as `ABORT_DISCARDED` |

## Scale envelope

~30k samples/s fleet-wide; per-sample work is render-delta + tree append + parse. Storage on the order of
hundreds of MB/s at the design point (prefix-shared); top-k logprobs are opt-in per job.

## Build vs buy

Build, in Python, on the `renderers` library (render, parse, `bridge_to_next_turn`, per-message attribution).
Reference designs: slime's agent adapters, Agent Lightning's LLM proxy.

## Open questions

- Default flush mode per job type, given the crash-bias trade-off.
- License and model-family coverage of `renderers` for target models — verify before committing.
