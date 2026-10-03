# Monitor

Code: `rollout_train.monitor` · See [`RunFeed`](../../guide/reference.md#runfeed),
[hooks](../rollout/hooks.md), [rollouts](rollouts.md#watching)

The monitor is a web page over a run's directory, laid out as the run is:

- the **training run**, which plays groups and trains its policy on them;
- a **group**: a row of the catalog, one start, a number of **rollouts**, and the step taken on them;
- a **rollout**: one **episode** of the program, which produces a group of **trajectories**, one per agent (model
  slot): what each was sent, what it thought, what it did and what came back, turn by turn;
- beside them, the **policy** (its versions) and the **machine**, the engines and the ledger.

```bash
rollout monitor RUN                                   # http://localhost:8765
```

It only reads the directory, as an open [profile](../../guide/deploying.md) lays it out (`rollout_train.layout`).
The process that trains writes it, and need not be running: the page shows a stopped run as it was left.

| Under the run's directory | Holds | The page reads it for |
|---|---|---|
| `ledger` | the run's tables, the policies' versions, the fences ([policies](policies.md)) | groups and their stages, outcomes, versions |
| `jobs` | each job's tickets, ended rollouts (episodes) and acknowledged cursor ([rollouts](rollouts.md)) | which rollouts of a group have ended, and what each episode reported |
| `feed` | what is happening now, written by `RunFeed` | rollouts still running, trajectories, throughput |
| `blobs` | each ended rollout's events | trajectories of rollouts the feed has let go |

## The page

The hierarchy is on the left: the run, its groups in flight and then every group it is done with (newest first, a
square for each rollout), the policies, and the machine. Runs and groups fold open and closed, and an open group lists
its rollouts; what is folded is remembered in the browser. The address names what is shown, so a reload stays there.

| View | Address | Shows |
|---|---|---|
| Run | `#/run/RUN` | groups done, trained on and in flight, rows unlocked, the policy's head and what is served; each group in flight with its stage and rollouts; every group's rewards (a column opens its group); the latest groups; the tasks played |
| Group | `#/run/RUN/group/N` | the group's stage, its rollouts (each with its reward and what its episode reported), what was done with it (the step's statistics and the version it made, or why it was skipped), and its start |
| Rollout | `#/rollout/RUN_ID` | what the episode reported, and its trajectories: all agents side by side or one, a slider over turns, and for the turn shown **Sees**, **Thinks**, **Does** and **Result**; the program's own tool calls below |
| Policy | `#/policy/NAME` | how far each step moved the policy, and every version with the group that made it |
| Machine, engines and ledger | `#/system` | memory, accelerators and disk, each channel's throughput, the jobs, the fences and tables |
| Rollouts outside a run | `#/rollouts` | runs in the feed that no training run asked for: evaluations, tests, programs run by hand |

A rollout whose feed file has been pruned is read back from the events its job kept in the blob store: its replies
and tool calls are there, and what each model was sent is not (it is kept as tokens in the traces). A grid of single
characters in what a model sees (a map) is drawn with its symbols colored. A rollout's reward is summed as an
episode's is ([rewards](episodes.md#rewards)).

`System(directory, feed)` reads the directory: `snapshot()` (where the run stands), `group(run, number)` and
`rollout(run_id)`. `create_app(directory)` serves them and the page; its routes are listed in
`rollout_train.monitor.app`.

A group's stage is read from the records alone, so it is what a [loop](training.md) that started now would find:

| Stage | The records say |
|---|---|
| `decided` | the group is in the run's `groups` table and the job has no ticket for it |
| `waiting` | the job has its ticket and none of its rollouts has started |
| `playing` | some of its rollouts are in the feed or the job's log, and fewer have ended than were asked for |
| `ended` | as many rollouts have ended as were asked for (interrupted ones are run again and do not count) |
| `stepping` | the `steps` table has its decision and the version it names is not there |
| `made` | that version is there, and the `iterations` table has no line for the group |
| `done` | the `iterations` table has the group's outcome |

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

The feed is a copy for people to read. It is not the run's record: the run's events are, and under a durable runner
they are in its store. Code that watches a job reads its episodes and events from the job itself
([rollouts](rollouts.md)).
