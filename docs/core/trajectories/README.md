# Episodes

Status: **Working** (2026-10-02) · Code: `rollout_train.rollouts.episodes`, `rollout_train.recorder` · See [rollouts](../rollouts/README.md), [recorder](../recorder/README.md)

An episode is one finished run as whoever trains on it sees it. Nothing in it says where the run executed.

```python
@dataclass(frozen=True)
class Episode:
    cursor: int                      # its place in the job's log
    job: str
    ticket: str
    run_id: str
    labels: Mapping[str, str]        # from Job.run, plus job, ticket and episode
    parameters: JsonValue            # the row the run was given
    outcome: Outcome                 # COMPLETED | FAILED | CANCELLED
    detail: str | None               # what a failed run raised
    info: Mapping[str, JsonValue]    # the program's result: run.emit("result", {...})
    excluded: str | None             # why the program asked to be left out of training
    traces: Mapping[str, Trace]      # by model slot

@dataclass(frozen=True)
class Trace:
    epochs: list[Epoch]
    rewards: Mapping[str, float]     # by key; `reward` is the key "default"

@dataclass(frozen=True)
class Epoch:                         # one token sequence, as the policy saw and continued it
    tokens: list[int]
    spans: list[Span]                # the tokens the policy sampled: start, end, weights version
    logprobs: list[float]            # behavior logprobs of the tokens inside the spans, in order
```

An episode is assembled when its run ends, from the run's events (labels, rewards, result, ending) and what the
recorder kept of each model slot. Rewards assigned to a slot (`run.reward`) and rewards of observations (the task
loop's, for the `policy` slot) are summed by key.

## Result conventions

A program reports how its episode went with `run.emit("result", {...})`. Three keys are read by training:

| Key | Meaning |
|---|---|
| `solved` | The team did what the row is about (decides what a curriculum unlocks) |
| `saturated` | Nothing was left to earn: the episode ended because the goal was reached in full |
| `duration` | How long it took, in the world's own units (breaks ties among saturated episodes) |

## Swarm rewards

A program with several model slots has a trace per slot. A team that is rewarded together assigns the same reward to
each; `Episode.reward` is the mean over slots.
