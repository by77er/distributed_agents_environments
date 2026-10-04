# Monitor

Code: `rollout_train.monitor` · See [`RunFeed`](../../guide/reference.md#runfeed),
[hooks](../rollout/hooks.md), [rollouts](rollouts.md#watching)

The monitor is a web page over a ledger and every run in it: one monitor shows all the runs that share a ledger,
on whichever machines they run. It has three pages, switched along the top:

- **Runs**: every run, running ones first, and each run laid out as it runs: the **run**, which plays groups and
  takes steps on them; a **step**, one update of the policy over the groups queued when it was taken, and the version
  it made; a **group**, one start of one row of the catalog, played as a number of **episodes**; an **episode**, one
  run of the program, with what it reported when it ended; a **rollout**, one agent's (model slot's) part of an
  episode, turn by turn: what it was sent, what it thought, what it did and what came back. Each rollout becomes a
  **trajectory**, the tokens the trainer learns from;
- **Policies**: every policy in the ledger, each alone and all of them as a graph;
- **Statistics**: figures across the runs, a series for each, and the machine the monitor is on, the engines and
  the ledger.

```bash
rollout monitor RUN                                   # http://localhost:8765: RUN's ledger, and every run in it
rollout monitor sqlite:///~/.cache/rollout/ledger.db  # a database's runs (or postgresql://…, or a ledger's directory)
```

`WHERE` is a run's directory (its ledger, as `ledger.json` there says, or files under `ledger`), a ledger's
directory of files, or a database's URL. The monitor only reads; the runs' processes write, and need not be running:
the page shows a stopped run as it was left.

## Where it reads what

The ledger holds what is durable of every run ([the record](training.md#the-record)). Each run keeps the rest in its
own directory, as an open [profile](../../guide/deploying.md) lays it out (`rollout_train.layout`):

| | Holds | The page reads it for |
|---|---|---|
| the ledger | each run's `starts`, `groups`, `results`, `steps` and `failures`; the policies' versions; the fences ([policies](policies.md)) | every run, its steps, groups and their stages, outcomes, versions, the statistics |
| a run's `jobs` | each job's tickets, ended episodes and acknowledged cursor ([rollouts](rollouts.md)) | which episodes of a group have ended, and what each reported |
| a run's `feed` | what is happening now, written by `RunFeed` | episodes still running, their rollouts turn by turn, the engines' throughput |
| a run's `blobs` | each ended episode's events | the rollouts of episodes the feed has let go |

Where a run's episodes are read is decided in one place (`System._source`), from the run's newest `starts` record:

1. its **directory**, if that is on this machine (for a run that wrote no start, the directory the monitor was
   opened on, if the run is its: its job's log is there, or it is named after it);
2. else the **monitor on its machine**, at the `address` its start names (`rollout train --monitor URL`): that
   monitor is asked for the run's groups in flight, its groups, its episodes and its engines, with the
   `x-rollout-monitor-relayed` header, and answers from its own machine only. What it says is kept for a few
   seconds; one that does not answer is left alone for half a minute;
3. else **nowhere**: the page shows what the ledger has (groups, results, steps, versions) and says so.

A run is **running** while it writes (something this reads was written within 20 minutes: a record in the ledger,
its start, its jobs' logs or its feed), **idle** until three hours have passed without a write, and **ended** after.
No process is asked, so a run on any machine is told apart the same way.

## The pages

Each page has a sidebar of its own. On **Runs**, every run (its state, and its host when it is not this one)
folding open to the groups toward its next step (in flight, or recorded and waiting for a step), then its steps,
newest first, each with the groups that went into it (a square for each episode); then the episodes outside a run.
A step's groups need not be consecutive. A group that gave nothing to train on is listed with the step decided after
it, marked skipped. Runs, steps, groups and episodes fold open and closed: an open step lists its groups, an open
group its episodes, and an open episode its rollouts. On **Policies**, the graph (with or without the sample
fixture) and every policy. On **Statistics**, its sections, and the runs drawn: a click leaves a run out or takes it
back. What is folded and what is left out are remembered in the browser. The address names what is shown, so a reload
stays there.

| Page | View | Address | Shows |
|---|---|---|---|
| Runs | Every run | `#/runs` (and `#/`) | each run, running ones first: its state and host, groups in flight and done, steps, the share solved over its last groups, each group's mean reward in order, and where its episodes are read |
| Runs | Run | `#/run/RUN` | its state, steps taken, groups done and in flight, rows unlocked, the policy's head and what is served; the step being taken, with its groups; the groups recorded and waiting for a step; each group in flight with its stage (decided, asked, played, recorded) and episodes; every group's rewards, in the order of the steps they went into (a column opens its group); the latest steps; the tasks played |
| Runs | Step | `#/run/RUN/step/N` | the version the step made and its parent, the groups that went into it (and those decided before it that gave nothing to train on), and the update's statistics |
| Runs | Group | `#/run/RUN/group/N` | the group's stage, its episodes (each with its reward and what it reported), the step it went into, what was done with it (the step's statistics and the version it made, or why it was skipped), and its start |
| Runs | Episode | `#/episode/RUN_ID`, `#/episode/RUN_ID/SLOT` | what the episode reported, and its rollouts, every agent's side by side or one: **turn by turn** (a slider over turns, and for the turn shown **Sees**, **Thinks**, **Does** and **Result**), or the **whole trajectory** (every turn a row: what each agent did and what came back, with what it saw and thought a click away); the program's own tool calls below |
| Runs | Episodes outside a run | `#/episodes` | episodes in the feeds that no training run asked for: evaluations, tests, programs run by hand |
| Policies | Every policy, as a graph | `#/policies`, `#/policies/sample` | every policy as a lane of versions, with forks and distillations between lanes; the trainers and their queues over time; each version's way to the engines and the workers that serve it; the runs and distillations; evaluation suites, a column per version or model ([the policies view](#the-policies-view)) |
| Policies | Policy | `#/policy/NAME` | how far each step moved the policy, and every version with the step that made it |
| Statistics | Across every run | `#/statistics`, `#/statistics/SECTION` | the sections below, each run in its own color |

The statistics, read from the ledger (`rollout_train.monitor.statistics`) and, for the engines, from each run's
feed; every chart is drawn to scale and says each series' value under the pointer:

| Section | Shows |
|---|---|
| `outcomes` | the solve rate and the mean reward over groups, in the order their results were written: a line for each run, the mean of the last 4, 8, 16 or 32 groups, and a dot for each group |
| `rows` | every row played: groups, episodes, the share solved, mean and best reward, the newest group's rewards; and, where groups' starts list `names`, the same by how many names a start gives |
| `steps` | the trainer's statistics over the steps, a chart each: KL moved, KL floor, clip fraction, mean mismatch, mean weight, truncated fraction, loss, the step's time and its start time (as far as each version says them) |
| `pace` | episodes and groups an hour (counted when each group's result was written); what was done with each group (trained on, in a step being taken, waiting for a step, nothing to train on, no episode or a failed step, in flight); why groups gave nothing to train on |
| `queue` | how many groups were in flight (decided, their result not written), and how many waited for a step (recorded with something to train on, no step begun over them), over time |
| `inference` | each run's engines: tokens a second and requests at once, a measurement a minute while they are busy |
| `machine` | the machine the monitor is on (memory, accelerators and disk, now and over the last hour, measured every 15 seconds while the monitor runs), each channel's throughput, the jobs, the ledger's fences and tables (`#/system` opens it) |

An episode whose feed file has been pruned is read back from the events its job kept in the blob store: its replies
and tool calls are there, and what each model was sent is not (it is kept as tokens in the trajectories). A grid of
single characters in what a model sees (a map) is drawn with its symbols colored. An episode's reward is summed as the
trainer sums it ([rewards](episodes.md#rewards)).

`System(directory)` reads a run's directory and its ledger, and `System(ledger=…)` a ledger alone: `snapshot()`
(where every run stands), `group(run, number)`, `episode(run_id)`, `feeds()` (the episodes in the feeds),
`lineage(sample)` (the policies view) and `statistics()`. `create_app(where)` serves them and the page; its routes
are listed in `rollout_train.monitor.app`.

A group's stage is read from the records alone, so it is what a [loop](training.md) that started now would find:

| Stage | The records say |
|---|---|
| `decided` | the group is in the run's `groups` table and the job has no ticket for it |
| `waiting` | the job has its ticket and none of its episodes has started |
| `playing` | some of its episodes are in the feed or the job's log, and fewer have ended than were asked for |
| `ended` | as many episodes have ended as were asked for (interrupted ones are run again and do not count), and no result is written yet |
| `done` | the `results` table has its result |

A step's state is read likewise: `stepping` while it has neither made the version it names nor failed, then
`committed` or `failed`. A group that is done carries the number and state of the step that covers it, if one does;
a recorded group that no step covers waits toward the next one.

## The policies view

A lane for each policy, its versions from the left. A lane is folded to the versions something points at (its first
and newest, the ends of each run's stretch, a fork's parent, a teacher, an evaluated version, one on its way to the
engines), with a gap for the rest; a click on its label opens it (what is open is remembered in the browser). A
version opens the step that made it. Between lanes:

- a **fork**: a policy whose first version's parent is another policy's version;
- a **distillation**: a diamond in its student's lane, before the versions it made, with an edge from each teacher
  and, dashed, from the version the student starts from. It is off policy when it trains on others' samples, on
  policy when on the student's own.

A version that something here starts from and that this ledger does not have (a policy in another ledger) stands in
a lane of its own at the top.

`rollout_train.monitor.lineage` reads it from the ledger's tables and the runs' feeds' job lines, at `/api/policies`. What a
ledger has today is read as it is: the policies' versions and the runs' steps. A run's steps stand for its trainer's
queue (a run takes one step at a time), and the `published` notes in the feed for what its engines serve. The tables
for distillation, trainers, inference workers and evaluations are proposed in
[the policy graph](../../research/policy-dag.md), and nothing writes them yet. `#/policies/sample` (`?sample=1`) shows
the view with a fixture of them (`rollout_train/monitor/sample-lineage.json`) beside the ledger, everything from it
marked sample.

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
  episodes, published weights, the training loop's `result` and `step` notes, and the engines' throughput
  (`inference`).
- **The directory is bounded.** `keep` is the number of runs kept; the oldest are deleted.
- **One writer at a time.** Runs that an earlier writer left without an end, because its process was stopped, are
  marked cancelled when the next writer starts, so that the page does not show them running for ever.

`FeedReader` keeps a summary of each run and reads a run's lines from its file when the page asks for them.

The feed is a copy for people to read. It is not the run's record: the run's events are, and under a durable runner
they are in its store. Code that watches a job reads its episodes and events from the job itself
([rollouts](rollouts.md)).
