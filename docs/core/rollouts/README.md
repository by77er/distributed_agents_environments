# Rollouts

Status: **Proposed** · Layer: core · See [ADR-0015](../../decisions/0015-rollout-interface.md), [ADR-0022](../../decisions/0022-rl-interface-extensions.md)

## Purpose

Runs task rows at scale for training or evaluation and delivers one stream of `Sample`s per job. The caller —
typically a trainer — decides what to run, how many times, how to group the results and how stale it tolerates
data. The rollout side knows no algorithm concepts: no group sizes, no advantages, no staleness policy beyond a
bounded buffer.

`RolloutJobs` is a Python protocol. `LocalRolloutJobs` implements it in process on a `LocalRunner` (the local
profile); a rollout service implements it over the network on a `DurableRunner`.

## Owns / does not own

| Owns | Does not own |
|---|---|
| Rollout jobs: queued rows, created runs, acknowledged position | What to sample, how to group it, staleness policy (caller) |
| Admission under the buffer bound (backpressure) | Rewards (task code) |
| Creating and cancelling runs; resampling infrastructure failures | Tokens and versions (recorder) |
| Publishing weights for the job's trainable channels | Sample assembly ([trajectories](../trajectories/README.md)) |

## Interface

```python
class RolloutJobs(Protocol):
    def start(self, *, program: ProgramReference, binding: RunBinding, buffer_samples: int,
              trainable_channels: Mapping[str, str] = {},     # model slot → channel (several policies per job)
              mode: JobMode = JobMode.TRAIN, attach_transcripts: bool = False,
              enrichments: Sequence[EnrichmentSpecification] = ()) -> RolloutJob: ...

class RolloutJob(Protocol):
    job_id: str
    def run(self, parameters: Any, *, labels: Mapping[str, str] = {}, count: int = 1,
            priority: int = 0, admit_together: bool = False) -> Ticket: ...
    def cancel(self, *, run_ids: Sequence[str] = (), label_selector: Mapping[str, str] | None = None) -> None: ...
    def samples(self, cursor: int = 0, *, label_selector: Mapping[str, str] | None = None,
                include_masked: bool = False) -> AsyncIterator[Sample]: ...
    def acknowledge(self, cursor: int) -> None: ...            # consumed through cursor; frees buffer space
    async def publish(self, channel: str, weights: WeightsSource, *,
                      phase: PublishPhase = PublishPhase.STAGE_AND_COMMIT) -> WeightsVersion: ...

class Ticket(Protocol):
    run_ids: list[str]
    def samples(self) -> AsyncIterator[Sample]: ...            # only this ticket's samples

@dataclass(frozen=True)
class WeightsSource:                                           # exactly one of:
    tensors: Mapping[str, Tensor] | None = None               # in process (local profile, CUDA IPC)
    checkpoint_uri: str | None = None                          # full checkpoint in object storage
    delta: Delta | None = None                                 # {parent_version, uri}
    lora: LoraAdapter | None = None                            # {base_version, uri}
    distributed: DistributedTransfer | None = None             # {backend: NCCL | NIXL | …, rendezvous}; trainer participates

class PublishPhase(Enum):
    STAGE = "stage"                     # move bytes to engines while the old version serves
    COMMIT = "commit"                   # pause → abort in flight → swap → resume
    STAGE_AND_COMMIT = "stage_and_commit"
```

Example (local profile, GRPO-style groups formed by the caller):

```python
job = jobs.start(program=AgentProgramReference(Wordle, DefaultAgent), binding=binding, buffer_samples=512,
                 trainable_channels={"policy": "exp/latest"})
for row in rows:
    job.run(row, labels={"group": row["id"]}, count=8, admit_together=True)
async for group in complete_groups(job.samples(), by="group", size=8):
    trainer.step(group)
    await job.publish("exp/latest", WeightsSource(tensors=trainer.state_dict()))
```

## Semantics and guarantees

- **Admission**: runs are created while `runs in flight + unacknowledged samples < buffer_samples`. Otherwise
  `run` requests wait in the job's queue (strict priority, then arrival). `admit_together` admits all `count` runs of a
  ticket at once or none; it is an admission rule, not grouping semantics.
- **Runs** receive the row as program parameters and the labels, which propagate to the run, its recorder sessions
  and its samples. `count` creates independent runs.
- **Cancellation** stops admission or cancels running runs by identifier or label (oversampling, dynamic sampling,
  over-stale work). Cancelled runs still yield a `Sample` with `outcome = CANCELLED`, so counts stay exact.
- **Identical start states** for runs of the same row come from the task and, when it uses one, its environment
  template — not from the rollout side.
- **Staleness** is bounded by the buffer and the task's `max_turns`; in-flight weight updates reach running episodes
  through the channel. Adaptive staleness control is the caller's admission rule (e.g. AReaL-style), not the
  system's.
- **Child runs** spawned inside a job's runs inherit the job and its labels (plus `root_run`), so swarm members'
  samples appear in the same log ([trajectories](../trajectories/README.md#swarm-rewards)).
- **Resampling**: infrastructure failures are re-run with the same labels (new `run_id`), up to a retry limit; task
  errors are not. The crash rate by episode length is monitored for sampling bias.
- **Publish** requires the slot's binding to be a trainable recorded channel and runs the
  [weight transition protocol](../recorder/engine-adapter.md#weight-transition-protocol). `EVALUATE` jobs cannot
  publish. `distributed` transfer means the trainer joins a collective with the engines, so both need a shared
  RDMA/NCCL fabric (same region and zone).
- **Trainable channels sample with temperature only**, so the learner can reproduce the behavior distribution.

## State

Local profile: in memory, with the sample log in a local directory. Service form: job state in a small database;
samples in the durable sample log, so a caller that crashes resumes from its last acknowledged cursor.

## Open questions

- A cursor per consumer when several trainers share one job (one cursor per job until then).
- Samples from runs outside jobs (e.g. production conversations): a policy question first.
