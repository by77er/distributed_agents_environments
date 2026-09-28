# Trajectories

Status: **Proposed** · Layer: core · See [ADR-0015](../../decisions/0015-rollout-interface.md), [ADR-0022](../../decisions/0022-rl-interface-extensions.md)

## Purpose

Turns recorder sessions plus the rewards and endings in [run events](../../contracts/run-events.md) into `Sample`s —
flat training records holding what a loss needs — and keeps them in one ordered sample log per rollout job that
trainers read with a cursor.

Components: the **trajectory assembler** (join + path selection), the **sample log**, and **trainer adapters**
(thin per-framework libraries over the rollout API).

## Owns / does not own

| Owns | Does not own |
|---|---|
| The `Sample` schema | The loss, advantages, importance-sampling estimator (trainer) |
| Joining session paths with run rewards and endings | Rewards (task code), tokens (recorder) |
| Placing rewards at token positions | Grouping samples (caller) |
| The per-job sample log and its cursors | Weight loading |

## Sample

One sample per trainable model slot per run.

```python
@dataclass(frozen=True)
class Sample:
    sample_id: str                          # {session_id}#{leaf_node_id}; consumers deduplicate on it
    cursor: int                             # position in the job's sample log
    job_id: str
    run_id: str
    root_run_id: str                        # the swarm root; equals run_id otherwise
    slot: str
    labels: Mapping[str, str]               # from RolloutJob.run
    sequences: list[Sequence]               # one per renderer epoch on the selected path; usually one
    ending: Ending | None                   # TERMINATED | TRUNCATED | None (did not end normally)
    truncation_reason: TruncationReason | None   # MAX_TURNS | TIME | BUDGET | CONTEXT_OVERFLOW | OUTPUT_LENGTH | TASK
    outcome: Outcome                        # COMPLETED | TASK_ERROR | CANCELLED | INFRASTRUCTURE_EXHAUSTED
    mask_reasons: list[str]                 # from run.exclude_from_training; trainers skip by default
    complete: bool                          # False if recorder data was lost
    policy_id: str
    sampling: SamplingParameters
    transcript: BlobReference | None = None # optional: canonical messages of the path (group judges)
    enrichments: Mapping[str, Enrichment] = field(default_factory=dict)   # optional, e.g. teacher logprobs

@dataclass(frozen=True)
class Sequence:
    tokens: list[int]
    loss_mask: list[bool]                   # True ⇔ sampled by this slot's policy
    behavior_logprobs: list[float]          # NaN where loss_mask is False
    weights_version: list[int]              # per token; 0 where loss_mask is False
    rewards: list[Reward]                   # {token_index, value, key}
    turns: list[TurnSpan]                   # per sampled turn
    routed_experts: BlobReference | None    # mixture-of-experts routing replay: [positions, layers, top_k]
    media: list[MediaReference]             # images / audio at token offsets
    epoch_reason: EpochReason               # why this sequence starts (START | CONTEXT_EDIT | PREFIX_MISMATCH | …)

@dataclass(frozen=True)
class TurnSpan:
    context_start: int                      # first token of the observation this turn responds to
    sampled_start: int
    sampled_end: int                        # exclusive
    reply_effect_id: str                    # joins to run events
    finish: FinishReason                    # STOP | LENGTH | TOOL_USE | ABORTED_RESUBMITTED
    interruptions: int                      # weight transitions inside this turn
```

Loss-mask conventions: renderer-inserted boundary tokens are context (mask 0); the stop token the model sampled is
mask 1 and has a logprob. Reference-model logprobs, advantages and returns are computed by the trainer (or supplied
as an optional enrichment); they are not part of `Sample`.

## Semantics and guarantees

- **Assembly trigger**: the run is terminal and its sessions are closed. For swarms, the *root* run is terminal
  and every session in the tree is closed.
- **Path selection** per [session trees](../recorder/session-tree.md#path-selection). Interrupted and aborted
  branches are never on the path.
- **Reward placement**:
  - rewards bound to a reply (`observation.recorded.reward`, `reward.assigned` with `reply_effect_id`) → the last
    sampled token of that reply, on the slot that produced it;
  - episode-level rewards (including `Task.score`) → the last sampled token of the slot's final sequence.
- **Sample log**: ordered by `cursor`; delivery is at-least-once and consumers deduplicate on `sample_id`. Local
  profile: a directory of Parquet files. Service form: Parquet segments plus a manifest in object storage. Replaying
  a job's log from cursor 0 gives offline training and evaluation.
- **Honesty**: `behavior_logprobs` pass through exactly as recorded; nothing recomputes or rescales them.

## Swarm rewards

A run may assign rewards to slots of its **descendants**, even after they have ended:
`run.reward(value, slot=…, key=…, run_id=child_run_id)`. The event is recorded in the parent's run events with
`target_run_id`, and the descendant's samples are assembled when the root ends. This is how a team outcome reaches
the trainable members of a swarm.

## Trainer adapters

A trainer adapter is a library: it reads `job.samples(cursor)`, converts `Sample`s to its framework's batch format,
and calls `job.publish`. It may split multi-sequence samples into sequences that share the sample's advantage, or sum
per-token rewards for trainers that accept one scalar per sequence, logging what it changed. Candidates, in the
order the [research](../../research/interfaces-rl-environments.md#62-trainers-out) suggests: miles/slime, SkyRL, a
Tinker exporter, AReaL, verl. The choice of the first trainer is deferred; mixture-of-experts training requires one
that supports routing replay.

## Open questions

- Storage format for very long samples (hundreds of thousands of tokens): chunked sequences?
- Whether enrichment (teacher logprobs via `score_tokens`) runs in the assembler or in the trainer.
