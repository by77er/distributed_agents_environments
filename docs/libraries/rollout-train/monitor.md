# Monitor

Code: `rollout_train.monitor` (the service), `libraries/rollout-train/web` (the page) · See
[`RunFeed`](../../guide/reference.md#runfeed), [hooks](../rollout/hooks.md), [rollouts](rollouts.md#watching)

The monitor is a web page over a ledger and every run in it: one monitor shows all the runs that share a ledger,
on whichever machines they run. It has three pages, switched along the top:

- **Runs**: every run, running ones first, and each run laid out as it runs: the **run**, which plays groups and
  takes steps on them; a **step**, one update over the groups queued when it was taken, and the version it made; a **group**, one start of one row of the catalog, played as a number of **episodes**; an **episode**, one
  run of the program, with what it reported when it ended; a **rollout**, one agent's (model slot's) part of an
  episode, turn by turn: what it was sent, what it thought, what it did and what came back. Each rollout becomes a
  **trajectory**, the tokens the trainer learns from;
- **Versions**: every version, each alone and all of them as a graph growing from their base models;
- **Statistics**: figures across the runs, a series for each, and the machine the monitor is on, the engines, the
  runners and the ledger.

```bash
rollout monitor RUN                                   # http://localhost:8765: RUN's ledger, and every run in it
rollout monitor sqlite:///~/.cache/rollout/ledger.db  # a database's runs (or postgresql://…, or a ledger's directory)
```

`WHERE` is a run's directory (its ledger, as `ledger.json` there says, or files under `ledger`), a ledger's
directory of files, or a database's URL. The monitor reads, and writes names in the registry
([the registry](versions.md#the-registry)): a run's, when it is renamed on its page, and bookmarks, made, moved and
taken away on a version's page. The runs' processes write the rest, and need not be running: the page shows a stopped
run as it was left.

A run is shown by its name; its id (what everything kept of it is under) is on its page and under the pointer. A
version is shown by where it came from (the run that made it and its step: `diamonds · S3`) and the shortest start of
its id that no other has (`zwzk`), with the bookmarks that name it; its whole id, its depth and its parents are under
the pointer, and its page says them all.

## How the page is kept current

The page is a single-page application (React and TypeScript, in `libraries/rollout-train/web`). It is built into the
package (`rollout_train/monitor/static`), and the monitor serves it as files: `rollout monitor` needs no Node. The
page draws each view from what it has read, and reads again only what the monitor says changed.

- **Topics.** Each thing a view shows is a topic (`rollout_train.monitor.stream`): `system` (where every run stands),
  `feeds`, `machine`, `statistics`, `versions` (or `versions/sample`), `group/RUN/N`, `episode/RUN_ID`. The
  monitor reads a topic at most once a beat (1.5 seconds), whoever asks, and gives each reading a **version**, a hash
  of what it says (leaving out when it was read).
- **The stream.** `/api/stream?topic=…` is a stream of server-sent events: the version of each topic asked for at
  once, then a `version` event each time one changes. The page watches the topics of the place shown (always
  `system` and `feeds`; a group, an episode, the statistics or the graph as they are shown), and opens a new stream
  when it moves elsewhere. While nobody watches, the monitor reads nothing.
- **ETags.** Each JSON answer carries its topic's version as its ETag. The page sends the version it has
  (`If-None-Match`), and the monitor answers 304, with nothing, when it is the same. An episode's lines are asked for
  after those the page has (`?after=N`).
- **Drawing only what changed.** An answer that changed in part keeps the parts that did not as they were (TanStack
  Query's structural sharing), and each run, group, episode, tile and chart is a component that draws again only when
  what it shows changed. The frame, the sidebar, what is folded, the scroll, an open episode's turn and what one
  opened stay as they are; "how long ago" counts up on its own between readings.
- **When the stream is cut** (a proxy that does not pass it, a monitor restarted), the page says so along the top,
  the browser connects again, and every view is read again every half minute meanwhile.

Answers are compressed (gzip). Building the page (Node 20 or later):

```bash
cd libraries/rollout-train/web
npm ci
npm run build    # type-checks, then writes rollout_train/monitor/static
npm run check    # type-check and lint
npm test         # the page's tests
npm run dev      # the page with hot reloading, asking a monitor on :8765 (MONITOR=http://… names another)
```

The built files are part of the repository, so a change to the page is a change to `web/src` and a build.

## Where it reads what

The ledger holds what is durable of every run ([the record](training.md#the-record)), its episodes as runners record
them ([rollouts](rollouts.md)), and the versions. Each run keeps the rest in its own directory, as an open
[profile](../../guide/deploying.md) lays it out (`rollout_train.layout`):

| | Holds | The page reads it for |
|---|---|---|
| the ledger | each run's `starts`, `groups`, `results`, `steps` and `failures`; its `episodes`, `claims` and `interrupted`; the `versions`; the fences ([versions](versions.md)) | every run, its steps, groups and their stages, the episodes that ended and what each reported, which runner plays what, outcomes, versions, the statistics |
| the registry | each run's id and name, and the bookmarks ([the registry](versions.md#the-registry)) | what each run is called, and the bookmarks that name each version |
| a run's `feed` | what is happening now, written by `RunFeed` | episodes still running, their rollouts turn by turn, the engines' throughput |
| a run's `blobs` | each ended episode's events | the rollouts of episodes the feed has let go |

Where a run's episodes are read is decided in one place (`System._source`), from the run's newest `starts` record:

1. its **directory**, if that is on this machine (for a run that wrote no start, the directory the monitor was
   opened on, if the run is its: as its `run.json` says, or by its name);
2. else the **monitor on its machine**, at the `address` its start names (`rollout train --monitor URL`): that
   monitor is asked for the run's groups in flight, its groups, its episodes and its engines, with the
   `x-rollout-monitor-relayed` header, and answers from its own machine only. What it says is kept for a few
   seconds; one that does not answer is left alone for half a minute;
3. else **nowhere**: the page shows what the ledger has (groups, results, steps, versions, ended episodes) and says
   so.

A run is **running** while it writes (something this reads was written within 20 minutes: a record in the ledger,
its start, or its feed), **idle** until three hours have passed without a write, and **ended** after. No process is
asked, so a run on any machine is told apart the same way.

## The pages

Each page has a sidebar of its own. On **Runs**, every run (its state, and its host when it is not this one)
folding open to the groups toward its next step (in flight, or recorded and waiting for a step), then its steps,
newest first, each with the groups that went into it (a square for each episode); then the episodes outside a run.
A step's groups need not be consecutive. A group that gave nothing to train on is listed with the step decided after
it, marked skipped. Runs, steps, groups and episodes fold open and closed: an open step lists its groups, an open
group its episodes, and an open episode its rollouts. On **Versions**, the graph (with or without the sample
fixture), the versions bookmarks name, and each run's newest. On **Statistics**, its sections, and the runs drawn: a click leaves a run out or takes it
back. What is folded and what is left out are remembered in the browser. The address (after `#`) names what is shown,
so a reload stays there.

| Page | View | Address | Shows |
|---|---|---|---|
| Runs | Every run | `#/runs` (and `#/`) | each run, running ones first: its state and host, groups in flight and done, steps, the share solved over its last groups, each group's mean reward in order, and where its episodes are read |
| Runs | Run | `#/run/RUN` | its name (with a control to rename it), id, state, base model, the version it started from and the one it is at, what is served; figures: where it started and is now, steps, groups done and solved (all, some, none), episodes, the share solved early and late, mean reward, rows unlocked, inference; the step being taken, with its groups; the groups recorded and waiting for a step; each group in flight with its stage (asked, claimed, played, recorded) and episodes; every group's rewards, in the order of the steps they went into (a column opens its group); the latest steps; the tasks played |
| Runs | Step | `#/run/RUN/step/N` | the version the step made and its parent, the groups that went into it (and those decided before it that gave nothing to train on), and the update's statistics |
| Runs | Group | `#/run/RUN/group/N` | the group's stage, its episodes (each with its reward and what it reported; one asked for and not started holds a place; one cut short is marked interrupted), which runners play it, the step it went into, what was done with it (the step's statistics and the version it made, or why it was skipped), and its start |
| Runs | Episode | `#/episode/RUN_ID`, `#/episode/RUN_ID/SLOT` | what the episode reported, and its rollouts, every agent's side by side or one: **turn by turn** (a slider over turns, following the newest unless one moves it, and for the turn shown **Sees**, **Thinks**, **Does** and **Result**), or the **whole trajectory** (every turn a row: what each agent did and what came back, with what it saw and thought a click away); the program's own tool calls below |
| Runs | Episodes outside a run | `#/episodes` | episodes in the feeds that no training run asked for: evaluations, tests, programs run by hand |
| Versions | Every version, as a graph | `#/versions`, `#/versions/sample` | each base model a root, and under it a lane for each run with its versions, a run that starts from another's version hanging under it; what each distillation does, in words; the trainers and their queues over time; each version's way to the engines and the workers that serve it; the runs and distillations; evaluation suites, a column per version or model ([the versions view](#the-versions-view)) |
| Versions | Version | `#/version/ID` (or the start of one) | where it came from (its parents, base model, run and step), its bookmarks (with controls to make one, move one here, or take one away), how far it moved and what is kept of it, its line back to the base model, and what grew from it |
| Statistics | Across every run | `#/statistics`, `#/statistics/SECTION` | the sections below, each run in its own color |

Renaming on a run's page asks the monitor (`POST /api/rename`, `{"id", "name"}`), which renames it in the registry
(`System.rename`; a run from before the registry is registered under its key first). A version's page makes a bookmark
name it, or moves one there (`POST /api/bookmarks`, `{"name", "version"}`, where the version may be any
[reference](versions.md#references)), and takes one away (`DELETE /api/bookmarks/NAME`). A name another run has, or
one that says `/`, `@` or `:`, is refused (409) and the page says why. Every page open on the monitor hears of the
change through its stream.

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
| `machine` | the machine the monitor is on (memory, accelerators and disk, now and over the last hour, measured every 15 seconds while the monitor runs), each channel's throughput, each runner in the ledger (what it plays now, the claims it has made, its fence), the ledger's fences and tables (`#/system` opens it) |

An episode whose feed file has been pruned is read back from the events its runner kept in the blob store: its
replies and tool calls are there, and what each model was sent is not (it is kept as tokens in the trajectories). A
grid of single characters in what a model sees (a map) is drawn with its symbols colored. An episode's reward is
summed as the trainer sums it ([rewards](episodes.md#rewards)).

`System(directory)` reads a run's directory and its ledger, and `System(ledger=…)` a ledger alone: `snapshot()`
(where every run stands), `group(run, number)`, `episode(run_id)`, `feeds()` (the episodes in the feeds),
`lineage(sample)` (the versions view), `statistics()` (with each run's name), `rename(who, name)`,
`bookmark(name, version)` and `unbookmark(name)`. `create_app(where)` serves them,
the stream and the page; its routes are listed in `rollout_train.monitor.app`.

A group's stage is read from the records alone, so it is what a [loop](training.md) that started now would find:

| Stage | The records say |
|---|---|
| `waiting` | the group is in the run's `groups` table, asking for its `episodes`, and no runner has claimed any of them |
| `playing` | a runner has claimed one of its episodes, at least, and fewer have ended than were asked for |
| `ended` | as many episodes are in the run's `episodes` table as were asked for, and no result is written yet |
| `done` | the `results` table has its result |

A claim holds while its runner keeps the fence it was made under and the attempt was not cut short; the page lists
the claims that hold as what each runner plays. A step's state is read likewise: `stepping` while it has neither made
the version it names nor failed, then `committed` or `failed`. A group that is done carries the number and state of
the step that covers it, if one does; a recorded group that no step covers waits toward the next one.

## The versions view

Every version grows from a base model along its first parents ([versions](versions.md#versions)). Each base model is a
root, with a lane of its own; under it a lane for each run whose first version was trained from it, the run's versions
from the left; and under a run's lane, each run that starts from one of its versions: a fork, or a distillation's
start. A lane is folded to the versions something points at (its first and newest, a version another run starts from,
a teacher, a bookmarked or evaluated version, one on its way to the engines), with a gap for the rest; a click on its
label opens it (what is open is remembered in the browser). A version opens its page. Between lanes:

- from a **base model** to the first version of each line trained from it;
- a **fork**: a version trained from another run's version;
- **learned from**: a version's other parents (dashed);
- a **distillation**: a diamond in its student's lane, before the versions it made, with an edge from each teacher
  and, dashed, from the version the student starts from. Its mode is said in words on its lane and in the
  distillations below the graph: off-policy, the student is trained on its teachers' samples; on-policy, the student
  samples and its teachers score every token it sampled; mixed, both.

A version that something here starts from and that this ledger does not have stands in a lane of its own at the top.

`rollout_train.monitor.lineage` reads it from the ledger's tables and the runs' feeds' notes, at `/api/versions`. What
a ledger has today is read as it is: the versions, the runs' steps, bookmarks. A run's steps stand for its trainer's
queue (a run takes one step at a time), and the `published` notes in the feed for what its engines serve. The tables
for distillation, trainers, inference workers and evaluations are proposed in
[the version graph](../../research/policy-dag.md), and nothing writes them yet. `#/versions/sample` (`?sample=1`)
shows the view with a fixture of them (`rollout_train/monitor/sample-lineage.json`) beside the ledger, everything from
it marked sample.

## The feed

```python
from rollout_train.monitor import RunFeed

feed = RunFeed(directory / "feed")
runner = LocalRunner(hooks=[feed])                                   # every run's events and samples
episodes = EpisodeRunner(name, ledger, runner, recorder, blobs, places, hooks=[feed])   # and the runner's notes
```

An open profile attaches a feed itself. `RunFeed` is a `RunHooks` and a [`Hooks`](rollouts.md#watching). It writes
what the page shows to a directory, as it happens.

- **One file per run**, `<run_id>.jsonl`, with one JSON line per run event and per model sample. A sample's line
  holds every message sent (text, reasoning, tool calls, tool results), the names of the tools offered, and the
  reply.
- **One file of notes**, `_notes.jsonl`, with one line per [note](rollouts.md#watching): episodes as runners start
  and end them, published weights, the training loop's `result` and `step` notes, and the engines' throughput
  (`inference`).
- **The directory is bounded.** `keep` is the number of runs kept; the oldest are deleted.
- **One writer at a time.** Runs that an earlier writer left without an end, because its process was stopped, are
  marked cancelled when the next writer starts, so that the page does not show them running for ever.

`FeedReader` keeps a summary of each run (a line that is neither a run's event nor a sample is passed over) and reads
a run's lines from its file when the page asks for them.

The feed is a copy for people to read. It is not the run's record: the run's events are, and under a durable runner
they are in its store. Code that watches a run reads its episodes from the ledger ([rollouts](rollouts.md)).
