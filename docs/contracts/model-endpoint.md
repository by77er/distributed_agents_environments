# Model endpoint contract

Status: **Proposed** · Boundary B5 (and the native half of B18). See [ADR-0006](../decisions/0006-harness-unaware-of-policy.md).

The only thing the runtime (and therefore task and agent code) knows about models. Implemented by:

- the **Recorder** — for trainable channels and anything we want recorded;
- **direct provider adapters** in the runtime's model client — for API models we don't record.

The contract is identical for both. Which one serves a model slot is decided by the RunBinding; task and agent code
cannot tell.

## Interface

```proto
service ModelEndpoint {
  rpc Describe(DescribeRequest) returns (CapabilityContract);
  rpc Sample(SampleRequest)     returns (SampleResult);      // MAY also offer a streaming variant
  rpc Cancel(CancelRequest)     returns (Empty);             // best-effort, by effect_id
}

message SampleRequest {
  string          effect_id         = 1;  // REQUIRED; idempotency key
  string          session_id        = 2;
  ContextDelta    context           = 3;
  repeated ToolSpecification tools           = 4;  // model-visible fields only; the subset exposed this turn
  uint32          max_output_tokens = 5;  // optional; MUST be ≤ contract.max_output_tokens
  ToolChoice      tool_choice       = 6;  // AUTO | NONE | REQUIRED | {name}
  Timestamp       deadline          = 7;
}

message SampleResult {
  Message      message       = 1;  // role ASSISTANT; canonical blocks
  FinishReason finish_reason = 2;  // STOP | LENGTH | TOOL_USE | CONTENT_FILTER
  Usage        usage         = 3;
}

message Usage {
  uint64 context_used   = 1;  // REQUIRED: drives the agent's compaction decisions
  uint64 context_limit  = 2;  // REQUIRED
  uint64 input_tokens   = 3;  // optional; accounting
  uint64 output_tokens  = 4;  // optional; accounting
}

message CapabilityContract {
  string contract_version      = 1;
  uint64 context_limit         = 2;  // minimum guaranteed
  uint32 max_output_tokens     = 3;
  repeated string modalities_in = 4; // "text", "image", …
  bool   tool_calling          = 5;
  bool   parallel_tool_calls   = 6;
  ReasoningSupport reasoning   = 7;  // NONE | PORTABLE | POLICY_SCOPED
}
```

**Deliberately absent**: sampling parameters (temperature, top-p, …), model names, versions, tokens, logprobs.
Sampling parameters belong to the policy — they are configured on the RunBinding's model binding / channel,
below this contract.

## ContextDelta

Context is described as an edit of the previous request's context rather than resent. Task and agent code pass
complete message lists to `Model.sample`; the SDK computes the delta.

```proto
message ContextDelta {
  bytes    parent_digest = 1;  // digest of the previous model.requested context in this slot; empty = full context
  uint32   keep_prefix   = 2;  // number of parent items retained (== parent length for pure append)
  repeated Message append = 3; // items appended after the retained prefix
  bytes    digest        = 4;  // digest of the resulting context
}
```

**Digest chain**: `d₀ = sha256("")`, `dᵢ = sha256(dᵢ₋₁ ‖ sha256(JCS(itemᵢ)))`. The digest of a context of length
`k` is `d_k`, so any retained prefix is identifiable by its own chain value. Compaction is `keep_prefix < len`
plus appended summary items; pure continuation is `keep_prefix == len`.

**Materialization**: the runtime maintains the materialized context per (run, slot) by applying deltas (rebuilt
on replay from `model.requested` events). Endpoints declare in `Describe` whether they accept deltas:

- **Recorder**: accepts deltas; matches `parent_digest` against its session tree.
- **Direct adapters**: always receive the full context (the runtime sends `parent_digest = empty`).
- If an endpoint cannot resolve `parent_digest` (e.g. recorder state lost), it returns `NEED_FULL_CONTEXT` and
  the runtime resends the full context. Deltas are an optimization with a defined fallback, never a requirement.

## Errors

| Error | Meaning | Runtime handling |
|---|---|---|
| `OVERLOADED(retry_after)` | admission control | retry with backoff until deadline; not shown to task code unless the deadline passes (`model.failed{UNAVAILABLE}`) |
| `NEED_FULL_CONTEXT` | delta parent unknown | resend full context; not shown to task code |
| `CONTEXT_OVERFLOW(context_limit)` | context exceeds contract | `model.failed{CONTEXT_OVERFLOW}` — raised in `Model.sample`; the agent compacts |
| `CONTRACT_VIOLATION` | request exceeds capability contract | `model.failed{CONTRACT_VIOLATION}` |
| `DEADLINE` | deadline passed | `model.failed{DEADLINE}` |
| `INTERNAL` | anything else | retry per policy, then `model.failed{INTERNAL}` |

## Guarantees

- **Idempotency.** The recorder MUST return the identical `SampleResult` for a repeated `effect_id` while its
  dedupe record exists. Direct adapters MAY re-sample.
- **Contract stability.** A model slot's `CapabilityContract` MUST NOT weaken during a run. Policy changes that
  would weaken it require a different channel, which is a RunBinding change, not a silent swap.
- **No leakage.** Nothing in `SampleResult` identifies the policy, weights version, or engine.
