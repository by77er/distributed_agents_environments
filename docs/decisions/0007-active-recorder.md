# 0007 — An active recorder owns tokenization and records sessions

Status: **Accepted** · Date: 2026-09-26

## Context

RL needs the exact tokens the policy saw and sampled, with honest logprobs and versions (R8), for any harness
(R6), without the harness knowing (R5). Re-rendering messages each turn causes tokenization drift at turn
boundaries.

## Decision

An optional proxy — the **recorder** — sits between model consumers and inference. For trainable channels it runs
in **active** mode: it renders canonical content to tokens itself (via `renderers`), extends each session's token
sequence incrementally with the exact sampled tokens, calls engines tokens-in, and stores sessions as prefix trees
of spans (tokens, behavior logprobs, weights version, attribution). Unmanaged harnesses use it through
OpenAI/Anthropic-compatible per-session endpoints. Passive mode (engine renders, drift detected) remains for API
models and evals.

## Consequences

- Drift-free token sequences within a renderer epoch; loss masks fall out of span sources.
- Any harness, including third-party ones, is trainable by pointing its base URL at a session.
- The recorder must maintain renderers per model family and handle harnesses that normalize messages (prefix
  mismatch → new epoch, tracked as a metric).
- Retries, best-of-n and compaction are handled uniformly by the tree.

## Alternatives considered

- **Passive recorder only**: simpler, but drift corrupts gradients for self-hosted training.
- **Tokens in the run log / harness**: rejected by ADR-0006.
