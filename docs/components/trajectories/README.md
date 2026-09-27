# Trajectories

Status: **Proposed** · Boundaries B13, B14 (consumer), B15 · See [ADR-0015](../../decisions/0015-rollout-interface.md)

## Purpose

Turns recorder sessions plus the rewards and endings in run logs into `Sample`s — flat training records holding
only what a loss needs — and keeps them in one durable, ordered log per rollout job that trainers read with a
cursor.

Components: **Trajectory Assembler** (join + path selection), **sample log** (durable, ordered), **trainer
adapters** (thin per-framework libraries over the rollout client).

## Owns / does not own

| Owns | Does not own |
|---|---|
| The `Sample` schema | The loss, advantages, importance-sampling estimator (trainer) |
| Joining session paths with run rewards and endings | Rewards (task code), tokens (Recorder) |
| Placing rewards at token positions | Grouping samples (caller) |
| The per-job sample log and its cursors | Weight loading (WUC) |

## Boundaries

| Direction | Counterpart | What |
|---|---|---|
| consumes | Run events (B13) | `observation.recorded`, `reward.assigned`, terminal events — via the Control API event stream once the run is terminal |
| consumes | Recorder export (B14) | session manifests and segments |
| provides → | Callers via `RolloutJobs.Samples` (B15) | samples in cursor order |

## Interface

### Sample

One sample per trainable model slot per run.

```proto
message Sample {
  string sample_id = 1;                    // {session_id}#{leaf_node_id}; consumers deduplicate on it
  uint64 cursor = 2;                       // position in the job's sample log
  string job_id = 3; string run_id = 4; string slot = 5;
  map<string, string> labels = 6;          // from RolloutJobs.Run, e.g. {"group": "g17"}
  repeated Sequence sequences = 7;         // one per renderer epoch on the selected path; usually exactly one
  Ending ending = 8;                       // TERMINATED | TRUNCATED | NONE (run failed or was cancelled)
  bool complete = 9;                       // false if recorder data was lost; filtered out by default
  string policy_id = 10;
  SamplingParameters sampling = 11;
}

message Sequence {
  repeated int32  tokens = 1;
  repeated bool   loss_mask = 2;           // true ⇔ sampled by this slot's policy
  repeated float  behavior_logprobs = 3;   // NaN where loss_mask is false
  repeated uint64 weights_version = 4;     // per token; 0 where loss_mask is false
  repeated Reward rewards = 5;             // {token_index, value, key}
}
```

Not in `Sample`, but reachable by identifier: KV-cache epochs, off-path branches, and timing (session trees);
reward `info`, observations, and metrics (run log).

### Trainer adapters

A trainer adapter is a library, not a service: it iterates `job.samples(cursor)`, converts `Sample`s to its
framework's batch format, and calls `job.publish` with new weights. Normalizations it may apply, logged when
applied:

- splitting multi-sequence samples into separate sequences that share the sample's advantage;
- summing per-token rewards into one scalar for trainers that only accept one per sequence.

## Semantics & guarantees

- **Assembly trigger**: the run is terminal and its sessions are `CLOSED`.
- **Path selection** per [session-tree](../recorder/session-tree.md#path-selection).
- **Reward placement**:
  - `observation.recorded.reward` and `reward.assigned` with a `reply_effect_id` → the last sampled token of that
    reply (joined to the session tree by `effect_id`), on the slot that produced the reply.
  - `reward.assigned` without a reply (episode-level, including `Task.score`) → the last sampled token of the
    slot's final sequence.
- **Ending** comes from the final `observation.recorded`.
- **Sample log**: Parquet segments plus a manifest in object storage, ordered by `cursor`. `Samples(from_cursor)`
  tails it. Delivery is at-least-once; consumers deduplicate on `sample_id`. Replaying a job's log from cursor 0
  gives offline training and evaluation for free.
- **Honesty**: `behavior_logprobs` are passed through exactly as recorded. The assembler never recomputes or
  rescales them.

## Build vs buy

Build the assembler, sample log and adapters. Buy the trainer; the selection criterion is that it accepts
externally generated samples. Evaluate slime/miles and AReaL first.

## Open questions

- Which trainer first? Determines which adapter is built first.
- Storage format for very long samples (hundreds of thousands of tokens): chunked sequences?
