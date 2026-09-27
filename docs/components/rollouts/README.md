# Rollouts

Status: **Proposed** · Boundaries B12, B15, B16 · See [ADR-0015](../../decisions/0015-rollout-interface.md), [ADR-0014](../../decisions/0014-no-forks-template-recipes.md)

## Purpose

Runs task rows at scale for training or evaluation and delivers one stream of `Sample`s per job. The caller —
usually a Ray-based trainer — decides what to run, how many times, and how to group the results. The rollout side
knows no algorithm concepts: no group sizes, no advantages, no staleness policy beyond a bounded buffer.

## Owns / does not own

| Owns | Does not own |
|---|---|
| Rollout jobs: queued rows, created runs, acknowledged position | What to sample and how to group it (caller) |
| Admission under the buffer bound (backpressure) | Rewards (task code) |
| Creating runs in cells; resampling infrastructure failures | Tokens and versions (Recorder) |
| Publishing weights for the job's channel | Sample assembly (Trajectory Assembler) |
| | Environment preparation (content-addressed templates, [environments](../environments/README.md)) |

## Boundaries

| Direction | Counterpart | What |
|---|---|---|
| provides → | Caller (e.g. Ray driver) | `RolloutJobs` API and Python client |
| consumes | Control API (B12) | bulk `CreateRuns`, run completion watch |
| consumes | Sample log (B15) | unacknowledged sample count, for admission |
| consumes | Policy Registry + WUC (B16) | `Publish` → version, transition, channel advance |

## Interface

```proto
service RolloutJobs {
  rpc StartJob(StartJobRequest)       returns (Job);
  //   {request_id, task: TaskReference (without parameters), agent: AgentReference, binding: RunBinding,
  //    buffer_samples, mode: TRAIN | EVALUATE, cells[]}
  rpc Run(RunRequest)                 returns (RunTicket);     // {job_id, request_id, parameters (one row), labels, priority, count} → run_ids
  rpc Samples(SamplesRequest)         returns (stream Sample); // {job_id, from_cursor, include_incomplete}; resumable
  rpc Acknowledge(AcknowledgeRequest) returns (Empty);         // {job_id, through_cursor}: consumed; frees buffer space
  rpc Publish(PublishRequest)         returns (Operation);     // {job_id, weights_reference} → new version on the job's channel
  rpc GetJob(JobReference)            returns (Job);
  rpc CancelJob(JobReference)         returns (Job);
}
```

Python client:

```python
job = rollouts.start(task=FixFailingTest, agent=DefaultAgent, binding=binding, buffer_samples=512)

job.run(row, labels={"group": "g17"}, count=8)   # 8 independent runs of one row; count only batches the calls
for sample in job.samples(cursor):               # one Sample per finished run and trainable slot
    ...                                          # the caller groups by label
job.acknowledge(cursor)                          # consumed through cursor; the client can do this automatically
job.publish(weights)                             # abort-before-update, advance the channel
```

## Semantics & guarantees

- **Admission**: the controller creates runs while `runs in flight + unacknowledged samples < buffer_samples`.
  Otherwise `Run` requests wait in the job's queue (by priority, then arrival).
- **Runs**: created through the Control API with the job's binding, the row as task parameters, and the labels,
  which propagate to the run, its recorder sessions, and its samples.
- **Identical start states** come from the task's content-addressed environment template, not from the
  controller. `count` creates independent runs; it carries no grouping semantics.
- **Staleness** is bounded by the buffer (how long samples wait) and the task's `max_turns` (how long episodes
  run), with in-flight weight updates reaching running episodes through the channel. The trainer masks outliers
  using per-token `weights_version`. There is no adaptive control loop.
- **Resampling**: `run.crashed` and infrastructure failures are re-created with the same labels (new `run_id`), up
  to a retry limit. Task errors are not retried. The crash rate by episode length is monitored for sampling bias.
- **Publish** requires the job's policy slot to be bound to a trainable recorded channel; it runs the
  [weight transition protocol](../recorder/engine-adapter.md#weight-transition-protocol). `EVALUATE` jobs cannot
  publish.
- **Trainable channels sample with temperature only** (no top-p, top-k, penalties or constrained decoding), so the
  learner can reproduce the behavior distribution.

## State & durability

Job state (queue, run identifiers, acknowledged cursor) in a small global Postgres schema. Samples are in the
durable sample log ([trajectories](../trajectories/README.md)), so a caller that crashes reconnects with its
`job_id` and resumes from its last acknowledged cursor.

## Failure modes

| Failure | Effect |
|---|---|
| Controller restart | Resumes from job state; runs keep going meanwhile |
| Caller crash | Runs continue until the buffer fills, then admission stops; the caller resumes from its acknowledged cursor |
| Cell loss | Affected runs are re-created in other cells |

## Scale envelope

Hundreds of thousands of concurrent runs across cells; admission decisions are per job and cheap.

## Build vs buy

Build the controller and the Python client. Trainers (slime/miles, AReaL, verl, …) plug in through a thin adapter
over the client; the only requirement is that they accept externally generated samples.

## Open questions

- Priority semantics across rows of one job (strict vs weighted).
- Per-job concurrency limits across cells, beyond the buffer bound.
