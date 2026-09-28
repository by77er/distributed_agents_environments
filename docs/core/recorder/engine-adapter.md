# Engine adapter

Status: **Proposed** · Layer: core (the protocol); engines are in the [inference layer](../../inference/README.md)

What the recorder requires from an inference engine, and the protocol by which weight transitions split
generations. Engines are bought (SGLang primary, vLLM alternative); this contract is what an adapter must provide
on top of them.

## Interface

```python
class EngineAdapter(Protocol):
    def capabilities(self) -> EngineCapabilities: ...
    async def generate(self, req: GenerateRequest) -> AsyncIterator[GenerateEvent]: ...
    async def abort(self, request_id: str) -> PartialResult: ...
    async def score_tokens(self, tokens: list[int], positions: range) -> list[float]: ...   # prefill-only logprobs

@dataclass
class GenerateRequest:
    request_id: str                 # effect_id + segment index
    input_tokens: list[int]
    sampling: EngineSampling        # translated from SamplingParameters; logprobs = processed
    max_new_tokens: int
    stop_token_ids: list[int]
    routing_key: str                # session_id → prefix-cache affinity (and, in the platform layer, a per-tenant cache salt)
    pool: str                       # replica pool serving the channel's current version (one local engine in the local profile)
    return_routed_experts: bool     # mixture-of-experts policies

@dataclass
class GenerateEvent:                # streamed
    tokens: list[int]
    logprobs: list[float]           # processed distribution, one per token
    finish: FinishReason | None     # STOP | LENGTH | ABORTED
    weights_version: int            # of the replica that produced these tokens
    routed_experts: BlobReference | None   # [positions, layers, top_k], including cached prefix positions
```

## Engine requirements

| Requirement | Level | SGLang | vLLM |
|---|---|---|---|
| Tokens-in generation | MUST | `/generate` with `input_ids` | prompt token ids |
| Per-token logprobs of the *processed* distribution | MUST | verify | `logprobs_mode="processed_logprobs"` (≥ 0.10.2) |
| Abort returns partial tokens + logprobs | MUST | `abort_request` returns partial output | verify; else pause(`keep`) path |
| Pause admission / resume | MUST | `pause_generation` / `continue_generation` | `pause_generation` / `resume_generation` |
| In-place weight update | MUST | `update_weights_from_distributed` / `_from_tensor` / `_from_disk` | weight-sync APIs |
| Prefix caching + sticky routing | MUST | RadixAttention + sgl-model-gateway routing key | APC + llm-d router |
| Weights version reported per response | SHOULD | verify | verify |
| Cache entries tagged with weights version; age-based eviction | SHOULD (upstream patch) | no | no |
| Routed experts for every position, including prefix-cache hits (mixture-of-experts policies) | MUST for MoE | `return_routed_experts` (verify cached positions) | verify |
| Prefill-only scoring (`score_tokens`) for teacher / reference logprobs | SHOULD | `return_logprob` with `logprob_start_len` | `prompt_logprobs` |
| Per-tenant prefix-cache salt (platform layer, trust tiers T2+) | SHOULD | verify | `cache_salt` |
| Top-k logprobs | MAY | yes | yes |

If an engine does not report the weights version per response, the adapter tags each response with the replica's
version as tracked from weight update controller notifications. That is sound only because of the abort-before-update invariant below.

## Weight transition protocol

**Invariant: every engine response (including a partial) is produced by exactly one weights version.**

```
WUC                        Engine replica R                  Recorder
 │ 1. pause admission ───▶  new requests queue
 │ 2. abort in-flight ───▶  returns partials ─────────────▶  record SAMPLED span @ v_old
 │                                                           mark turn interrupted (WEIGHT_UPDATE)
 │ 3. load v_new ────────▶  (cache kept unless max_kv_age exceeded → flush)
 │ 4. resume ────────────▶
 │ 5. notify ─────────────────────────────────────────────▶  replica R now at v_new
 │                                                           6. resubmit input_tokens ++ partial_tokens
 │                                                              (prefix cache hit on old KV → no re-prefill)
 │                                                           7. append SAMPLED span @ v_new; continue
```

- Task and agent code see one ordinary assistant message: the parser runs over the concatenation of all sampled spans
  in the turn.
- **Stale KV is allowed** (ADR-0009): after step 3 the prefix cache may still hold KV computed at `v_old`.
  Correctness does not depend on it — the recorded logprobs are those actually sampled from.
- **KV age bound**: the WUC flushes a replica's cache when its oldest possible entry exceeds the channel's
  `max_kv_age` versions (whole-cache flush, staggered across replicas, until version-tagged eviction exists).

**Local profile.** With one engine colocated with the trainer on one GPU, the weight update controller (WUC) is a
function in the same process: pause admission and abort in-flight requests, put the engine to sleep (free its
memory) for the train step, update weights from tensors, wake it, resume. The invariant above still holds.

## Sampling validation (trainable channels)

Rejected with `CONTRACT_VIOLATION` unless the channel allows them: top-p < 1, top-k, min-p, repetition/presence
penalties, logit bias, grammar-constrained decoding. Speculative decoding is allowed only if the returned
logprobs are the target model's (verify per engine).
