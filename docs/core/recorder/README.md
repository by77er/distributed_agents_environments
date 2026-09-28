# Recorder

Status: **Proposed** · Layer: core · See [ADR-0007](../../decisions/0007-active-recorder.md), [ADR-0008](../../decisions/0008-mid-rollout-policy-change.md), [ADR-0009](../../decisions/0009-stale-kv-importance-sampling.md)

## Purpose

The recorder sits between model consumers (runs, or unmanaged third-party harnesses) and inference engines. In
**active** mode it owns tokenization: it keeps each session's exact token sequence, extends it incrementally, sends
tokens-in requests to engines, and records every sampled span with honest behavior logprobs, the weights version
that produced it and, for mixture-of-experts models, the routed experts. It is what makes any harness trainable
(R6, R8) without the harness knowing (R5).

It is a **library** (`Recorder`) that implements the [model endpoint](../../contracts/model-endpoint.md) in
process; a service wrapper exposes the same operations over the network for the cluster and fleet profiles.

## Owns / does not own

| Owns | Does not own |
|---|---|
| Rendering canonical content ↔ tokens (renderers) for recorded channels | Which replica serves a request (inference router) |
| Session trees: token spans, logprobs, versions, routed experts, attribution | Weights and weight loading (weight update controller) |
| Sample deduplication on `effect_id` + argument digest | Channel → version resolution (policy registry) |
| Validating sampling parameters against channel contracts | Rewards (task code) and samples (trajectory assembler) |
| Splitting generation across weight transitions (abort + resubmit) | Run events |

## Interfaces

| Direction | Counterpart | Contract |
|---|---|---|
| provides | runs (any runner) | [model endpoint](../../contracts/model-endpoint.md); service form in [session-api](session-api.md) |
| provides | unmanaged harnesses | OpenAI/Anthropic-compatible endpoints ([session-api](session-api.md)) |
| provides | trajectory assembler | [session trees](session-tree.md) |
| consumes | inference engines | [engine adapter](engine-adapter.md) |
| consumes | policy registry, weight update controller | [inference](../../inference/README.md) |

## Modes

| Mode | Tokenization | Drift | Requires | Use |
|---|---|---|---|---|
| **active** | recorder (renderers) → tokens-in | none within an epoch | a renderer for the model family; a tokens-in engine | **trainable channels** (default) |
| passive | engine renders messages | detected, not prevented | engine returns prompt / completion token IDs | API models, evaluations |

## Semantics and guarantees

- **Optional (P11).** Runs work identically without it; the `RunBinding` then uses a direct adapter.
- **Exactness.** Within a renderer epoch, the tokens recorded as context for turn *k+1* contain turn *k*'s sampled
  tokens verbatim — never re-tokenized text.
- **Honest logprobs (P10).** Recorded logprobs are of the distribution actually sampled from (processed, after
  temperature). Trainable channels sample with temperature only; other sampling parameters are rejected on them.
- **Single version per span.** Every engine response is produced by one weights version (guaranteed by
  abort-before-update); a generation interrupted by a weight transition becomes two spans.
- **Routing replay.** For mixture-of-experts policies, routed experts are captured for every position, including
  positions served from the prefix cache, and stored per span.
- **Idempotent samples.** A repeated `effect_id` with the same digest returns the recorded result — no second
  sample, no orphan branch. A different digest is a `CONFLICT`.
- **Interrupted samples** (delivery mode `INTERRUPT`) are recorded as aborted branches and never trained on.
- **Invisible.** Nothing in responses to model consumers reveals tokens, versions or engines.

## State

- **Local profile:** sessions in memory, trees written to a local directory.
- **Service form:** active sessions in memory, sticky to one instance by consistent hashing on `session_id` (the key
  the inference router uses for cache affinity); tree segments flushed to object storage per turn or in batches.
  Loss of unflushed state marks the session **incomplete**.

## Failure modes

| Failure | Effect |
|---|---|
| Recorder instance crash (service form) | Sessions it held lose unflushed spans → marked incomplete; runs continue (the runner retries the sample; a new instance re-renders from the full context via `NEED_FULL_CONTEXT`) |
| Renderer mismatch (a harness normalized a message) | New epoch started (`PREFIX_MISMATCH`); the mismatch rate is a metric |
| Engine abort not returning partials | Falls back to discarding the partial and resampling; recorded as `ABORT_DISCARDED` |

## Scale envelope

At the fleet design point, ~30k samples/s; per-sample work is render-delta + tree append + parse. Storage on the
order of hundreds of MB/s (prefix-shared), more with routed experts; top-k logprobs are opt-in per job.

## Build vs buy

Build, in Python, on Prime Intellect's `renderers` library (Apache-2.0: render, parse, `bridge_to_next_turn`,
per-message attribution). Reference designs: slime's agent adapters, Agent Lightning's LLM proxy.

## Open questions

- Default flush mode per job type, given the crash-bias trade-off.
- `renderers` coverage for our target model families (spike S9 in the [research](../../research/interfaces-rl-environments.md#82-spike-tests)).
