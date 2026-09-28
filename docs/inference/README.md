# Inference

Status: **Proposed** · Layer: inference · See [ADR-0008](../decisions/0008-mid-rollout-policy-change.md), [ADR-0009](../decisions/0009-stale-kv-importance-sampling.md), [ADR-0022](../decisions/0022-rl-interface-extensions.md)

## Purpose

Serves policies. Engines and routers are **bought** (SGLang, vLLM; a cache-aware router at fleet scale). We build
the thin control layer that makes policies versioned, addressable by channel and changeable mid-rollout: the
**policy registry** and the **weight update controller**. Both are Python protocols with an in-process
implementation for the local profile.

## Owns / does not own

| Owns | Does not own |
|---|---|
| Policy lineages, weights versions, channels, channel contracts | Tokenization and recording (recorder) |
| Sampling configurations allowed per channel | Training (trainer) |
| Weight transfer and transitions: stage, pause, abort, swap, resume, readiness | Anything visible to task or agent code (nothing here is) |
| Replica pools per (policy, version); admission signals | |

## Interface

```python
class PolicyRegistry(Protocol):
    def register_policy(self, policy: Policy) -> None: ...          # {policy_id, renderer_id, architecture, kind: WEIGHTS | API}
    def put_channel(self, channel: Channel) -> None: ...
    def resolve(self, channel: str) -> Resolution: ...              # {policy_id, weights_version, renderer_id, pool, contract}
    def watch(self, channel: str) -> AsyncIterator[Resolution]: ...

class WeightUpdateController(Protocol):
    async def stage(self, channel: str, weights: WeightsSource) -> StagedVersion: ...   # transfer while the old version serves
    async def commit(self, staged: StagedVersion) -> WeightsVersion: ...                # pause → abort → swap → resume → advance
    def transitions(self) -> AsyncIterator[TransitionEvent]: ...    # for the recorder: {replica, from, to, phase}

@dataclass(frozen=True)
class Channel:
    name: str                               # {namespace}/{name}
    policy_id: str
    contract: CapabilityContract            # MUST NOT weaken while runs use it
    trainable: bool                         # enables sampling validation and honest-logprob requirements
    sampling_allowed: SamplingConstraints   # trainable: temperature only; otherwise operator-defined
    kv_cache: KVCachePolicy                 # {max_kv_age: versions; 0 = always flush}; default 8
    pinned_version: int | None = None       # a frozen channel (e.g. a past self for self-play)

@dataclass(frozen=True)
class SamplingParameters:                   # configured on RunBindings, never by task or agent code
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float = 0.0
    stop: Sequence[str] = ()
    seed: int | None = None
    constrained_schema: JsonValue | None = None   # non-trainable channels only
```

`WeightsSource` and `PublishPhase` are defined with the [rollout API](../core/rollouts/README.md); `RolloutJob.publish`
calls `stage` and `commit`.

## By profile

| | Local | Cluster / fleet |
|---|---|---|
| Engine | one SGLang or vLLM engine in process, sharing the GPU with the trainer | engine pools behind a cache-aware router, sticky by `session_id` |
| Registry | in memory | a small service backed by a database |
| Weight transfer | tensors in process (`update_weights_from_tensor`, CUDA IPC); the engine sleeps during the train step | distributed transfer (the trainer joins an NCCL / RDMA collective with the engines), deltas, LoRA adapters or checkpoints |
| Transition | pause → abort → sleep → update → wake → resume | stage while serving; commit = pause → abort → swap → resume, staggered across replicas |
| Asynchronous RL | time-slicing: generation and training alternate | overlapped: generation continues across versions |

## Semantics and guarantees

- **Versions are monotonic** per `policy_id`; a channel only moves forward unless an operator rolls it back
  (recorded).
- **Readiness gating**: a commit completes only when enough of the pool serves the new version. Channels never point
  at versions nobody can serve.
- **Abort-before-update**: in-flight requests on a replica are aborted before its weights change, so every engine
  response comes from one version ([engine adapter](../core/recorder/engine-adapter.md#weight-transition-protocol)).
- **KV cache age**: the controller tracks, per replica, the oldest version whose KV entries may still be cached, and
  flushes when it exceeds `max_kv_age`, staggered so the pool is never cold at once.
- **Routing replay**: engines serving mixture-of-experts policies must return routed experts for every position,
  including cached prefixes.
- **Contract stability**: `put_channel` rejects a contract change that weakens the current one.
- **Tenant isolation** (platform layer, [trust tiers](../platform/trust-tiers.md) T2 and above): prefix-cache keys
  carry a per-tenant salt so tenants cannot detect each other's prompts through cache timing.
- **API models** are registered as `kind: API` policies with a contract and served by direct adapters.
- **Distributed transfer** couples trainer and inference networks: they share an RDMA/NCCL fabric, so they are
  placed in the same region and zone. The trainer participates in the trusted zone.

## Scale envelope

At the fleet design point: ~30k samples/s, ~15M generated tokens/s — the dominant cost of the system. Weight
updates can run every trainer step because stale-KV reuse removes the re-prefill cost (~9B tokens per update
otherwise).

## Open questions

- Per-token version reporting and version-tagged cache eviction: upstream patches worth contributing?
- Pool strategy during transitions at fleet scale: in place (all replicas move) or staggered (mixed versions).
- Capacity for pinned historical versions (self-play opponents); LoRA makes pools of past versions cheap.
