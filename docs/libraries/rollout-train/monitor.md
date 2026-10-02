# Monitor

Code: `rollout_train.monitor` · See [`RunFeed`](../../guide/reference.md#runfeed),
[hooks](../rollout/hooks.md), [rollouts](rollouts.md#watching)

The monitor is a web page for watching training as it happens, at both levels: what the job is doing (groups
submitted, episodes ended, updates, weights published, how the engines are doing), and for each run, every model
slot's turns: what the model was sent, what it thought, what it did and what came back.

```python
from rollout_train.monitor import RunFeed

feed = RunFeed(directory)
runner = LocalRunner(hooks=[feed])                    # every run's events and samples
jobs = RolloutJobs(runner, recorder, hooks=[feed])    # and the job's
```

```bash
rollout monitor DIRECTORY                             # read it: http://localhost:8765
```

An open [profile](../../guide/deploying.md) attaches a feed itself, in the run's directory under `feed`.

## The feed

`RunFeed` is a `RunHooks` and a `JobHooks`. It writes what the page shows to a directory, as it happens.

- **One file per run**, `<run_id>.jsonl`, with one JSON line per run event and per model sample. A sample's line
  holds every message sent (text, reasoning, tool calls, tool results), the names of the tools offered, and the
  reply.
- **One file for jobs**, `_job.jsonl`, with one line per [job event](rollouts.md#watching): tickets, admissions,
  episodes, published weights, the training loop's `iteration` notes, and the engines' throughput (`inference`).
- **The directory is bounded.** `keep` is the number of runs kept; the oldest are deleted.
- **One writer at a time.** Runs that an earlier writer left without an end, because its process was stopped, are
  marked cancelled when the next writer starts, so that the page does not show them running for ever.

## The page

`create_app(directory)` is a Starlette application over a feed directory. It only reads; the process that runs the
runs writes. The routes are listed in `rollout_train.monitor.app`.

| Part | Shows |
|---|---|
| Across the top | the job: episodes ended, groups waiting, the weights being served, the engines' throughput, and the latest groups with their rewards and what their updates did |
| On the left | runs, grouped by the label `group` and titled by `title` or `task`. `episode` names a run within its group. |
| For the chosen run | one column per slot, a slider over turns, and for the turn shown: **Sees** (the last message sent, or the whole context), **Thinks** (reasoning), **Does** (tool calls), **Result** (what the call returned, from the next turn) |
| Below | other effects: the tool calls the program itself made |

A grid of single characters in what a model sees (a map) is drawn with its symbols colored. A run's reward on the
page is summed as an episode's is ([rewards](episodes.md#rewards)).

The feed is a copy for people to read. It is not the run's record: the run's events are, and under a durable runner
they are in its store. Code that watches a job reads its episodes and events from the job itself
([rollouts](rollouts.md)).
