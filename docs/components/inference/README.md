# Inference

Status: **Proposed** · Boundaries B6 (provider side), B16, B17 · See [ADR-0008](../../decisions/0008-mid-rollout-policy-change.md), [ADR-0009](../../decisions/0009-stale-kv-importance-sampling.md)

## Purpose

Serves policies. Most of this layer is **bought** (engines, cache-aware router). We build the thin control layer
that makes policies versioned, addressable by channel, and changeable mid-rollout: the **Policy Registry** and the
**Weight Update Controller (WUC)**.

## Owns / does not own

| Owns | Does not own |
|---|---|
| Policy lineages, weights versions, channels, channel contracts | Tokenization and recording (Recorder) |
| Sampling configurations allowed per channel | Training (trainer) |
| Weight rollout: pause, abort, load, flush policy, resume, readiness gating | Anything visible to task or agent code (nothing here is) |
| Replica pools per (policy, version) and admission signals | |

## Boundaries

| Direction | Counterpart | What |
|---|---|---|
| provides → | Recorder (B6, B17) | engines (via router); channel resolution; transition notifications |
| provides → | Rollout Controller (B16) | `PublishVersion`, `AdvanceChannel` on behalf of `RolloutJobs.Publish` |
| provides → | Runtime (direct adapters) | API-model policy descriptions (provider, model, contract) |
| consumes | Engines, router | bought: SGLang / vLLM; sgl-model-gateway / llm-d router |

## Interface

```proto
service PolicyRegistry {
  rpc RegisterPolicy(Policy)            returns (Policy);        // {policy_id, renderer_id, architecture, kind: WEIGHTS | API | HUMAN}
  rpc PublishVersion(PublishRequest)    returns (PolicyVersion); // {policy_id, weights_version, weights_ref, parent_version}
  rpc PutChannel(Channel)               returns (Channel);
  rpc AdvanceChannel(AdvanceRequest)    returns (Operation);     // {channel, to_version, rollout} — completes when readiness met
  rpc Resolve(ChannelReference)               returns (Resolution);    // {policy_id, weights_version, renderer_id, pool, contract}
  rpc Watch(ChannelReference)                 returns (stream Resolution);
}

message Channel {
  string channel = 1;                       // {namespace}/{name}
  string policy_id = 2;
  CapabilityContract contract = 3;          // what agents and tasks may rely on; MUST NOT weaken while runs use it
  bool trainable = 4;                       // enables recorder sampling validation + honest-logprob requirements
  SamplingConstraints sampling_allowed = 5; // trainable channels: temperature only (fixed); others: operator-defined
  KVCachePolicy kv = 6;                          // {max_kv_age: J versions; 0 = always flush}; operator setting, default 8
  RolloutPolicy rollout = 7;                // {mode: IN_PLACE | STAGGERED, stagger_fraction, min_ready_fraction}
}

message SamplingParameters {                    // configured on RunBindings / sessions, never by task or agent code
  float temperature = 1; float top_p = 2; int32 top_k = 3; float min_p = 4;
  repeated string stop = 5; uint64 seed = 6;
  Json constrained_schema = 7;              // grammar/JSON-constrained decoding (non-trainable channels only)
}

service WeightUpdateController {
  rpc Transition(TransitionRequest) returns (Operation);   // {pool, to_version, flush: NEVER | BY_AGE | ALWAYS}
  rpc Status(PoolReference)               returns (PoolStatus);  // per-replica version, cache age, readiness
  rpc Subscribe(SubscribeRequest)   returns (stream TransitionEvent);  // recorder: {replica, from, to, phase}
}
```

## Semantics & guarantees

- **Versions are monotonic** per `policy_id`; a channel only moves forward unless an operator explicitly rolls
  it back (recorded).
- **Readiness gating**: `AdvanceChannel` completes only when `min_ready_fraction` of the pool serves the new
  version. Channels never point at versions nobody can serve.
- **Abort-before-update**: the WUC MUST pause admission and abort in-flight requests on a replica before loading
  new weights (see [engine-adapter](../recorder/engine-adapter.md#weight-transition-protocol)).
- **KV age**: the WUC tracks, per replica, the oldest version whose KV may still be cached, and flushes when it
  exceeds `max_kv_age`. Flushes are staggered so the pool is never cold at once.
- **Contract stability**: `PutChannel` rejects a contract change that weakens the current one.
- **API models** are registered as `kind: API` policies with a contract; they are served by direct adapters,
  not the WUC.

## Scale envelope

~30k samples/s, ~15M generated tokens/s at the design point — the dominant cost of the system. Weight updates
can run every trainer step because stale-KV reuse removes the re-prefill cost (~9B tokens per update otherwise).

## Build vs buy

Buy engines (SGLang primary, vLLM swappable) and the cache-aware router (sgl-model-gateway or llm-d router);
weight transfer via the engines' distributed update APIs (and P2P transfer where available). Build the registry
and WUC (thin), plus pool/version-aware routing (router config or plugin).

## Open questions

- Is running the inference fleet in scope for us, or is it an external platform we require this contract from?
- Per-token version reporting and version-tagged cache eviction: upstream patches worth contributing?
- Pool strategy during transitions: in-place (all replicas move, brief pause) vs staggered (mixed versions,
  split capacity).
