# Monitor

Code: `rollout_train.monitor` · See [`RunFeed`](../../guide/reference.md#runfeed),
[hooks](../rollout/hooks.md), [rollouts](rollouts.md#watching)

The monitor is a web page over a run's directory. It shows two things: where the system stands (the groups in
flight and the stage each is at, the policies' versions, the jobs, the engines, the machine), and for each run every
model slot's turns: what the model was sent, what it thought, what it did and what came back.

```bash
rollout monitor RUN                                   # http://localhost:8765
```

It only reads the directory, as an open [profile](../../guide/deploying.md) lays it out (`rollout_train.layout`).
The process that trains writes it, and need not be running: the page shows a stopped run as it was left.

| Under the run's directory | Holds | The page reads it for |
|---|---|---|
| `ledger` | the run's tables, the policies' versions, the fences ([policies](policies.md)) | groups and their stages, outcomes, versions |
| `jobs` | each job's tickets, episodes and acknowledged cursor ([rollouts](rollouts.md)) | which episodes of a group have ended |
| `feed` | what is happening now, written by `RunFeed` | episodes still running, transcripts, throughput |

## The system view

`System(directory, feed).snapshot()` assembles it, and `/api/system` serves it.

| Panel | Shows |
|---|---|
| In flight | every group that is decided and not done with: its stage, its episodes (each opens its transcript), and its step if one is decided |
| Groups | every group that is done with: its episodes' rewards, whether it was trained on, skipped or failed, and the newest in a table |
| Policy | each policy's head, the size of its versions, how far each step moved it, and the newest versions with their statistics |
| Channel | the version each channel serves, and the engines' throughput minute by minute |
| Tasks played | each row of the catalog that has been played: how often, how often trained on, its last rewards |
| Job | tickets, episodes in the log by outcome, how far its caller has acknowledged, tokens sampled |
| Machine | memory, accelerators and disk of the machine the monitor is on, and the run's processes |
| Ledger | the fence of every scope, the records in every table, and what is kept in the blob store |

A group's stage is read from the records alone, so it is what a [loop](training.md) that started now would find:

| Stage | The records say |
|---|---|
| `decided` | the group is in the run's `groups` table and the job has no ticket for it |
| `waiting` | the job has its ticket and none of its episodes has started |
| `playing` | some of its episodes are in the feed or the job's log, and fewer have ended than were asked for |
| `ended` | as many episodes have ended as were asked for (interrupted ones are run again and do not count) |
| `stepping` | the `steps` table has its decision and the version it names is not there |
| `made` | that version is there, and the `iterations` table has no line for the group |

## The feed

```python
from rollout_train.monitor import RunFeed

feed = RunFeed(directory / "feed")
runner = LocalRunner(hooks=[feed])                    # every run's events and samples
jobs = RolloutJobs(runner, recorder, hooks=[feed])    # and the job's
```

An open profile attaches a feed itself. `RunFeed` is a `RunHooks` and a `JobHooks`. It writes what the page shows
to a directory, as it happens.

- **One file per run**, `<run_id>.jsonl`, with one JSON line per run event and per model sample. A sample's line
  holds every message sent (text, reasoning, tool calls, tool results), the names of the tools offered, and the
  reply.
- **One file for jobs**, `_job.jsonl`, with one line per [job event](rollouts.md#watching): tickets, admissions,
  episodes, published weights, the training loop's `iteration` notes, and the engines' throughput (`inference`).
- **The directory is bounded.** `keep` is the number of runs kept; the oldest are deleted.
- **One writer at a time.** Runs that an earlier writer left without an end, because its process was stopped, are
  marked cancelled when the next writer starts, so that the page does not show them running for ever.

`FeedReader` keeps a summary of each run and reads a run's lines from its file when the page asks for them.

## The runs view

| Part | Shows |
|---|---|
| On the left | **System**, then the job in brief, then runs grouped by the label `ticket` (or `group`) and titled by `title` or `task`. `episode` names a run within its group. |
| For the chosen run | one column per slot, a slider over turns, and for the turn shown: **Sees** (the last message sent, or the whole context), **Thinks** (reasoning), **Does** (tool calls), **Result** (what the call returned, from the next turn) |
| Below | other effects: the tool calls the program itself made |

A grid of single characters in what a model sees (a map) is drawn with its symbols colored. A run's reward on the
page is summed as an episode's is ([rewards](episodes.md#rewards)). The page's address names the view (`#system`,
or `#` and a run's id), so a reload stays where it was.

`create_app(directory)` is the Starlette application; its routes are listed in `rollout_train.monitor.app`.

The feed is a copy for people to read. It is not the run's record: the run's events are, and under a durable runner
they are in its store. Code that watches a job reads its episodes and events from the job itself
([rollouts](rollouts.md)).
