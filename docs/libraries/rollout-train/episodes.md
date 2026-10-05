# Episodes

For people who design training: what a finished episode carries for training, and how it is assembled from the recorded
turns.

**Read first:** [Episode runners](rollouts.md). **Next:** [The training loop](training.md).

Code: `rollout_train.rollouts.episodes` · See [`Episode`](../../guide/reference.md#episode),
[`Trajectory`](../../guide/reference.md#trajectory), [`Segment`](../../guide/reference.md#segment),
[rollouts](rollouts.md)

An episode is one finished run as whoever trains on it sees it. Its labels say which group and row it came from. Its
outcome and result say how it went. Its trajectories, one per model slot (each slot's rollout), hold the segments the
policy saw and continued, with the logprobs it sampled them at
([what a session exports](recorder.md#what-a-session-exports)).
Nothing else about the run is needed to compute a loss, and nothing in an episode says where the run executed.

## How an episode is assembled

A [runner](rollouts.md#a-runner) assembles an episode when its run ends, from the run's [events](../rollout/contracts/run-events.md) and from
what the [gateway](gateway.md) recorded of each of its model slots.

| Part of the episode | Comes from |
|---|---|
| `run`, `group`, `number` | the episode the runner claimed: its run, its group and its number in the group |
| `labels` | the run's `run.created` event: `run`, `group` and `episode` (its start is there too, and in its group's record) |
| `outcome`, `detail` | the terminal event: `completed`; `failed` with what the program raised; `cancelled`, also for a run whose events have no ending |
| `info` | the payload of the run's `output.emitted` event of kind `result`: what the program reported with `run.emit("result", {...})` |
| `excluded` | the reason of a `training.excluded` event |
| `trajectories` | one per model slot that sampled or was rewarded: the slot's segments from the gateway's turn store, and its rewards |

A run that could not start is a failed episode whose `detail` says why, with no trajectories.

The runner stores the episode as it is assembled: a small record in the [ledger](checkpoints.md#the-ledger), and its
trajectories and its run's events as blobs ([the record](rollouts.md#the-record)).

## Rewards

Rewards are summed by model slot and by key.

- A reward assigned with `run.reward(value, slot=..., key=...)` goes to that slot and key.
- A reward on an observation goes to the slot `policy`, under the key `default`: it is the reward of the agent the
  task loop drives.
- `Trajectory.reward` is the slot's reward under the key `default`. `Episode.reward` is the mean of that over the
  episode's trained slots, so a team that is rewarded together, each slot the same, has that reward. A slot that is
  not trained (a judge, a fixed opponent: `Trajectory.trained` is false) keeps its segments, marked untrained, and
  counts in what the episode sampled, but not in its reward.

The [monitor](monitor.md) sums rewards with the same function (`rewards`), so the page and the trainer agree.

## What is fit to train on

`Episode.trainable` is true for an episode that completed and was not excluded. A program excludes its run with
`run.exclude_from_training(reason)`, for example after a fault that is not the policy's.

## Result conventions

A program reports how its episode went with `run.emit("result", {...})`. Training reads three entries, through
typed accessors:

| Accessor | Entry of `info` | Meaning |
|---|---|---|
| `Episode.solved` | `solved` is `true` | The players did what the row is about. It decides what a curriculum unlocks. |
| `Episode.saturated` | `saturated` is `true` | Nothing was left to earn: the episode ended because the goal was reached in full. |
| `Episode.duration` | `duration` is a number | How long it took, in the world's own units. It breaks ties among saturated episodes. An episode that does not say is `None`, and is not compared for speed. |

Everything else in `info` is the environment's own and is carried through unread.
