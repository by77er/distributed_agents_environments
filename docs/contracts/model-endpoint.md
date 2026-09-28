# Model endpoint contract

Status: **Proposed** · See [ADR-0006](../decisions/0006-harness-unaware-of-policy.md)

The only thing task, agent and program code knows about models. Implemented by:

- the **recorder** — for trainable channels and anything we want recorded (in process, or as a service);
- **direct adapters** — for API models we do not record.

The contract is identical for both. Which one serves a model slot is decided by the `RunBinding`; code cannot tell.

## Interface

```python
class ModelEndpoint(Protocol):
    def describe(self, session_id: str) -> CapabilityContract: ...
    async def sample(self, request: SampleRequest) -> SampleResult: ...
    async def cancel(self, effect_id: str) -> None: ...                  # best-effort

@dataclass(frozen=True)
class SampleRequest:
    effect_id: str                            # REQUIRED; idempotency key
    arguments_digest: str
    session_id: str                           # {run_id}/{slot}
    context: ContextDelta
    tools: Sequence[ToolSpecification] = ()   # model-visible fields only; the subset exposed this turn
    max_output_tokens: int | None = None      # MUST be ≤ contract.max_output_tokens
    tool_choice: ToolChoice | None = None     # AUTO | NONE | REQUIRED | {name}
    deadline: datetime | None = None

@dataclass(frozen=True)
class SampleResult:
    message: Message                          # role ASSISTANT; canonical blocks
    finish_reason: FinishReason               # STOP | LENGTH | TOOL_USE | CONTENT_FILTER
    usage: Usage

@dataclass(frozen=True)
class Usage:
    context_used: int                         # REQUIRED: drives the agent's compaction decisions
    context_limit: int                        # REQUIRED
    input_tokens: int | None = None
    output_tokens: int | None = None

@dataclass(frozen=True)
class CapabilityContract:
    contract_version: str
    context_limit: int                        # minimum guaranteed
    max_output_tokens: int
    modalities_in: frozenset[str]             # "text", "image", …
    tool_calling: bool
    parallel_tool_calls: bool
    reasoning: ReasoningSupport               # NONE | PORTABLE | POLICY_SCOPED
    accepts_context_delta: bool
```

A service implementation exposes the same operations over HTTP or gRPC (see [recorder session API](../core/recorder/session-api.md)).

**Deliberately absent**: sampling parameters (temperature, top-p, …), model names, versions, tokens, logprobs.
Sampling parameters belong to the policy; they are configured on the `RunBinding`'s model binding or channel.

## ContextDelta

Context is described as an edit of the previous request's context rather than resent. Code passes complete message
lists to `Model.sample`; the core computes the delta.

```python
@dataclass(frozen=True)
class ContextDelta:
    parent_digest: bytes | None     # digest of the previous request's context in this slot; None = full context
    keep_prefix: int                # number of parent items retained (== parent length for pure append)
    append: Sequence[Message]       # items appended after the retained prefix
    digest: bytes                   # digest of the resulting context
```

**Digest chain**: `d₀ = sha256("")`, `dᵢ = sha256(dᵢ₋₁ ‖ sha256(JCS(itemᵢ)))`. The digest of a context of length
`k` is `d_k`, so any retained prefix is identifiable by its own chain value. Compaction is `keep_prefix < len` plus
appended summary items; pure continuation is `keep_prefix == len`.

**Materialization**: the core keeps the materialized context per (run, slot). Endpoints whose contract says
`accepts_context_delta` (the recorder) receive deltas; others receive the full context. An endpoint that cannot
resolve `parent_digest` (e.g. recorder state lost) returns `NEED_FULL_CONTEXT` and the core resends the full
context. Deltas are an optimization with a defined fallback, never a requirement.

## Errors

| Error | Meaning | Handling |
|---|---|---|
| `OVERLOADED(retry_after)` | admission control | retry with backoff until the deadline; then `Model.sample` raises `ModelUnavailable` |
| `NEED_FULL_CONTEXT` | delta parent unknown | resend full context; invisible to code |
| `CONTEXT_OVERFLOW(context_limit)` | context exceeds the contract | `Model.sample` raises `ContextOverflow`; the agent compacts |
| `CONTRACT_VIOLATION` | request exceeds the capability contract | `Model.sample` raises `ContractViolation` |
| `CONFLICT` | known `effect_id` with a different argument digest | raised; indicates non-determinism or a lost commit ([delivery-semantics](../architecture/delivery-semantics.md)) |
| `DEADLINE`, `INTERNAL` | as named | retry per policy, then raise |

## Guarantees

- **Idempotency.** The recorder MUST return the identical `SampleResult` for a repeated `effect_id` with the same
  digest while its deduplication record exists. Direct adapters MAY re-sample.
- **Contract stability.** A model slot's `CapabilityContract` MUST NOT weaken during a run. Policy changes that would
  weaken it require a different channel, which is a `RunBinding` change, not a silent swap.
- **No leakage.** Nothing in `SampleResult` identifies the policy, weights version or engine.
