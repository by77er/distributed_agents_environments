# Monitor

Code: `rollout_train.monitor` (the service), `libraries/rollout-train/web` (the page) · See
[`RunFeed`](../../guide/reference.md#runfeed), [hooks](../rollout/hooks.md), [rollouts](rollouts.md#watching)

The monitor is a web page over a ledger and every run in it: one monitor shows all the runs that share a ledger,
on whichever machines they run. It has three pages, switched along the top:

- **Runs**: every run, running ones first, and each run laid out as it runs: the **run**, which plays groups and
  takes steps on them; a **step**, one update over the groups queued when it was taken, and the checkpoint it made; a **group**, one start of one row of the catalog, played as a number of **episodes**; an **episode**, one
  run of the program, with what it reported when it ended; a **rollout**, one agent's (model slot's) part of an
  episode, turn by turn: what it was sent, what it thought, what it did and what came back. Each rollout becomes a
  **trajectory**, the tokens the trainer learns from;
- **Checkpoints**: every checkpoint, each alone and all of them as a graph growing from their base models;
- **Statistics**: figures across the runs, a series for each, and the machines that run things (as their runners' and
  launchers' heartbeats say), the engines, the runners and the ledger.

```bash
rollout monitor RUN                                   # http://localhost:8765: RUN's ledger, and every run in it
rollout monitor sqlite:///~/.cache/rollout/ledger.db  # a database's runs (or postgresql://…, or a ledger's directory)
```

`WHERE` is a run's directory (its ledger, as `ledger.json` there says, or files under `ledger`), a ledger's
directory of files, or a database's URL. The monitor reads, and writes three things: names in the registry
([the registry](checkpoints.md#the-registry)) (a run's, when it is renamed on its page, and bookmarks, made, moved and
deleted on a checkpoint's page), and launches: runs asked for from the page ([launching a run](#launching-a-run)),
and asked to stop. The runs' processes write the rest, and need not be running: the page shows a stopped run as it
was left.

A run is shown by its name; its id (what everything kept of it is under) is on its page and under the pointer. A
checkpoint is shown by where it came from (the run that made it and its step: `diamonds · S3`) and the shortest start of
its id that no other has (`zwzk`), with the bookmarks that name it; its whole id, its depth and its parents are under
the pointer, and its page says them all.

## How the page is kept current

The page is a single-page application (React and TypeScript, in `libraries/rollout-train/web`). It is built into the
package (`rollout_train/monitor/static`), and the monitor serves it as files: `rollout monitor` needs no Node. The
page draws each view from what it has read, and reads again only what the monitor says changed.

- **Topics.** Each thing a view shows is a topic (`rollout_train.monitor.stream`): `system` (where every run stands),
  `feeds`, `machines`, `launches`, `statistics`, `checkpoints` (or `checkpoints/sample`), `group/RUN/N`,
  `episode/RUN_ID`. The
  monitor reads a topic at most once a beat (1.5 seconds), whoever asks, and gives each reading a **checkpoint**, a hash
  of what it says (leaving out when it was read).
- **The stream.** `/api/stream?topic=…` is a stream of server-sent events: the checkpoint of each topic asked for at
  once, then a `checkpoint` event each time one changes. The page watches the topics of the place shown (always
  `system` and `feeds`; the launches on Runs and New run; a group, an episode, the statistics and the machines, or
  the graph, as they are shown), and opens a new stream
  when it moves elsewhere. While nobody watches, the monitor reads nothing.
- **ETags.** Each JSON answer carries its topic's checkpoint as its ETag. The page sends the checkpoint it has
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
them ([rollouts](rollouts.md)), and the checkpoints. Each run keeps the rest in its own directory, as an open
[profile](../../guide/deploying.md) lays it out (`rollout_train.layout`):

| | Holds | The page reads it for |
|---|---|---|
| the ledger | each run's `starts`, `groups`, `results`, `steps` and `failures`; its `episodes`, `claims` and `interrupted`; the `checkpoints`; the fences ([checkpoints](checkpoints.md)) | every run, its steps, groups and their stages, the episodes that ended and what each reported, which runner plays what, outcomes, checkpoints, the statistics |
| the registry | each run's id and name, and the bookmarks ([the registry](checkpoints.md#the-registry)) | what each run is called, and the bookmarks that name each checkpoint |
| the heartbeats | each runner's and launcher's newest beat, with its recent beats' measurements (`rollout_train.presence`) | the machines (memory, accelerators, disk, the engines' processes), each channel's checkpoint and throughput, whether a run is running, which launchers are alive and what they offer |
| the launches | the runs and evals asked for and how each goes (`rollout_train.launches`) | the launches on Runs and Evals, the New run form, and a suite's Run this suite form |
| `evaluations/` | each suite's starts, and each eval's subject and results ([evals](evals.md)) | the Evals page, a suite's page, and the suites a checkpoint played |
| a run's `feed` | what is happening now, written by `RunFeed` | episodes still running, their rollouts turn by turn |
| the run's blob store | each ended episode's events, where the run's newest start says its blobs are (`blobs`: files, or S3) | the rollouts of episodes no longer in the feed |

Where a run's episodes are read is decided in one place (`System._source`), from the run's newest `starts` record:

1. its **directory**, if that is on this machine (for a run that wrote no start, the directory the monitor was
   opened on, if the run is its: as its `run.json` says, or by its name);
2. else the **monitor on its machine**, at the `address` its start names (`rollout train --monitor URL`): that
   monitor is asked for the run's groups in flight, its groups, its episodes and its engines, with the
   `x-rollout-monitor-relayed` header, and answers from its own machine only. What it says is kept for a few
   seconds; one that does not answer is left alone for half a minute;
3. else **nowhere**: the page shows what the ledger has (groups, results, steps, checkpoints, ended episodes) and says
   so.

Everything but the live feed is in the database (with a database ledger) or the blob store, so a monitor anywhere that
reaches them shows every run, its machines and its finished episodes; only episodes still playing need the run's
directory or the monitor on its machine.

A run is **running** while it writes (something this reads was written within 20 minutes: a record in the ledger,
its start, a runner's beat, or its feed), **idle** until three hours have passed without a write, and **ended** after. No process is
asked, so a run on any machine is told apart the same way.

## The pages

Each page has a sidebar of its own. On **Runs**, every run (its state, and its host when it is not this one)
folding open to the groups toward its next step (in flight, or recorded and waiting for a step), then its steps,
newest first, each with the groups that went into it (a square for each episode); then the episodes outside a run.
A step's groups need not be consecutive. A group that gave nothing to train on is listed with the step decided after
it, marked skipped. Runs, steps, groups and episodes fold open and closed: an open step lists its groups, an open
group its episodes, and an open episode its rollouts. On **Checkpoints**, the graph (with or without the sample
fixture), the checkpoints bookmarks name, and each run's newest. On **Evals**, the suites, then the evals playing
now. An eval's run is on Evals, not among the runs (its page opens from there). On **Statistics**, its sections, and the runs drawn: a click leaves a run out or takes it
back. What is folded and what is left out are remembered in the browser. The address (after `#`) names what is shown,
so a reload stays there.

| Page | View | Address | Shows |
|---|---|---|---|
| Runs | Every run | `#/runs` (and `#/`) | the launches (each asked-for run: its state, launcher, directory, settings changed, why it failed, a Stop button); each run, running ones first: its state and host, groups in flight and done, steps, the share solved over its last groups, each group's mean reward in order, and where its episodes are read |
| Runs | New run | `#/runs/new` | a form asking for a run ([launching a run](#launching-a-run)) |
| Runs | Run | `#/run/RUN` | its name (with a control to rename it), id, state, base model (or the full checkpoint its adapters build on), the checkpoint it started from and the one it is at, what each of its channels serves; figures: where it started and is now, steps, groups done and solved (all, some, none), episodes, the share solved early and late, mean reward, rows unlocked, inference; the step being taken, with its groups; the groups recorded and waiting for a step; each group in flight with its stage (asked, claimed, played, recorded) and episodes; every group's rewards, in the order of the steps they went into (a column opens its group); the latest steps; the tasks played |
| Runs | Step | `#/run/RUN/step/N` | the checkpoint the step made and its parent, the groups that went into it (and those decided before it that gave nothing to train on), and the update's statistics |
| Runs | Group | `#/run/RUN/group/N` | the group's stage, its episodes (each with its reward and what it reported; one playing with its reward so far, and each slot's where they differ; one asked for and not started holds a place; one cut short is marked interrupted), which runners play it, the step it went into, what was done with it (the step's statistics and the checkpoint it made, or why it was skipped), and its start |
| Runs | Episode | `#/episode/RUN_ID`, `#/episode/RUN_ID/SLOT` | what the episode reported, and its rollouts, every agent's side by side or one: **turn by turn** (a slider over turns, following the newest unless one moves it, and for the turn shown **Sees**, **Thinks**, **Does** and **Result**), or the **whole trajectory** (every turn a row: what each agent did and what came back, with what it saw and thought a click away); the program's own tool calls below ([an episode's rollouts](#an-episodes-rollouts)) |
| Runs | Episodes outside a run | `#/episodes` | episodes in the feeds that no run asked for: tests, programs run by hand |
| Checkpoints | Every checkpoint, as a graph | `#/checkpoints`, `#/checkpoints/sample` | each base model a root, and under it a lane for each run with its checkpoints, a run that starts from another's checkpoint hanging under it; what each distillation does, in words; the trainers and their queues over time; each checkpoint's way to the engines and the workers that serve it; the runs and distillations; evaluation suites, a column per checkpoint or model ([the checkpoints view](#the-checkpoints-view)) |
| Checkpoints | Checkpoint | `#/checkpoint/ID` (or the start of one) | where it came from (its parents, run and step), what its weights are (a LoRA adapter, or full) and what they build on (a base model, or the full checkpoint an adapter is over), its bookmarks (with controls to make one, move one here, or take one away), how far it moved and what is kept of it, its line back to the base model, what grew from it, and the suites it played (each opening the suite) |
| Evals | Every suite and eval | `#/evals` | the evals asked for from the page; each suite (its catalog, starts, subjects, and the subject that did best); every eval, newest first (its suite, who played, episodes played of those asked for, the share solved, whether it is done), each opening its run |
| Evals | Suite | `#/evals/SUITE` | its catalog, rows and seeds; the subject that did best; a **Run this suite** form ([evals](evals.md#asked-for-from-the-page)); its launches and the evals playing it; every subject's episodes at every start, with totals; two subjects compared at the starts both played |
| Statistics | Across every run | `#/statistics`, `#/statistics/SECTION` | the sections below, each run in its own color |

Renaming on a run's page asks the monitor (`POST /api/rename`, `{"id", "name"}`), which renames it in the registry
(`System.rename`; a run from before the registry is registered under its key first). A checkpoint's page makes a bookmark
name it, or moves one there (`POST /api/bookmarks`, `{"name", "checkpoint"}`, where the checkpoint may be any
[reference](checkpoints.md#references)), and takes one away (`DELETE /api/bookmarks/NAME`). A name another run has, or
one that says `/`, `@` or `:`, is refused (409) and the page says why. Every page open on the monitor hears of the
change through its stream.

## Launching a run

A **launcher** on a training machine (`rollout launcher --ledger URL --profiles DIR --catalog module:name --runs DIR
[--at-once N]`, `rollout_train.launcher`) beats every 15 seconds, saying what it offers: each profile under
`--profiles` that names a trainer, with what its trainer makes (`weights`: `lora` or `full`, where the trainer's class
says, as `rollout_lora`'s do) and the settings a launch may change and their values in the profile (the trainer's
settings, `trainer.start`, `trainer.bookmark`, `episodes_at_once`, each channel's `thinking_tokens` and
`answer_tokens`, and the `evals.` settings), the catalogs, and how many runs it plays of how many it may.

**New run** (`#/runs/new`, from the Runs page) offers what the launchers alive offer: a profile and a catalog, the
run's name, the checkpoint it starts from (the base model, a bookmark, or any checkpoint whose weights are kept, by where it
came from and what its weights are; a profile whose trainer trains every weight starts only from full weights, so an
adapter is merged first), a bookmark for it to carry, its groups, groups a step and seed, and every setting of the profile as a field
holding the profile's value, with rows for any other `trainer.KEY`. Values are read as numbers, true or false, or JSON
where they look like them, and as text otherwise. Launching asks the monitor (`POST /api/launches`, with the settings
changed only), which checks the ask (`System.launch`: a launcher alive offers the profile and the catalog, the name is
no other run's, every setting is the profile's or a trainer's, the checkpoint is one) and appends it to the launches; a
refusal (409 or 404) is said under the button. With no launcher alive, the form says so and gives the command that
starts one.

The launcher claims it, makes the run's directory under `--runs` (`NAME-ID`), and starts `rollout train` there with
`--name` and each setting as `--set KEY=VALUE`; its output goes to `train.log` in that directory. The launch's state
follows: `asked`, `claimed`, `running` (with the process), and `ended`, `failed` (with the end of its output) or
`stopped`. **Stop** (`POST /api/launches/ID/stop`) cancels a launch not yet claimed at once; a running one is sent an
interrupt, and the run stops as on Ctrl-C, at a group boundary. Once the run writes its start, its tile links to it.

The statistics, read from the ledger (`rollout_train.monitor.statistics`) and, for the engines, from each run's
runners' heartbeats; every chart is drawn to scale and says each series' value under the pointer:

| Section | Shows |
|---|---|
| `outcomes` | the solve rate (where a task says whether it solved) and the mean reward over groups, in the order their results were written: a line for each run, the mean of the last 4, 8, 16 or 32 groups, and a dot for each group |
| `rows` | every row played: groups, episodes, the share solved, mean and best reward, the newest group's rewards; and, where groups' starts list `names`, the same by how many names a start gives |
| `steps` | the trainer's statistics over the steps, a chart each: KL moved, KL floor, clip fraction, mean mismatch, mean weight, truncated fraction, loss, the step's time and its start time (as far as each checkpoint says them) |
| `pace` | episodes and groups an hour (counted when each group's result was written); what was done with each group (trained on, in a step being taken, waiting for a step, nothing to train on, no episode or a failed step, in flight); why groups gave nothing to train on |
| `queue` | how many groups were in flight (decided, their result not written), and how many waited for a step (recorded with something to train on, no step begun over them), over time |
| `inference` | each run's engines: tokens a second and requests at once, a measurement each beat while they are busy |
| `machines` | a card for each runner's and launcher's machine, as its heartbeats say: its host, alive or gone (no beat for 90 seconds), memory, accelerators and disk now and over its recent beats, its engines' processes, what each channel serves and how fast, and a launcher's profiles; then each channel's throughput, each runner in the ledger (what it plays now, the claims it has made, its fence), the ledger's fences and tables (`#/system` opens it) |

An episode's reward is summed as the trainer sums it ([rewards](episodes.md#rewards)): the mean of its slots'. One
still playing is shown with its reward so far, read from its feed the same way, and each slot's where they differ.

Whether an episode solved its task is what its program reported (`info["solved"]`), and a task need not say. The
monitor serves `solved` as null for an episode that did not say, and, in a group's result (whose `solved` the loop
writes as true or false for each episode), for each episode of a group none of whose episodes said; an eval's results
and counts likewise. Where nothing says, the page shows no solve figures (no share solved, no solve rate, no solved and
not-solved colors) and the mean reward in their place.

### An episode's rollouts

Each agent's rollout is its samples in order, its slots in their numbers' order (`agent-2` before `agent-10`), each in
a color its whole name gives it. An agent offered tools on its turns that samples with none offered (summarising its
memory, say) takes no turn then: those samples are folded, closed, under its turn before. For the turn shown:

- **Sees**: what came back since its newest reply (the messages after it, with any tool results), or with **whole
  context**, every message it was sent. A map in it (the rows under a line opening `Map of what you have seen`, as
  Minecraft's observations write one) is drawn with its symbols colored;
- **Thinks**, **Does**: its reasoning, and its reply and calls;
- **Result**: what came back from each call (the result the next turn's context carries for it), or, for a reply
  that called nothing, the words the next turn's context added after this one's.

An episode whose feed file has been pruned is read back from the events its runner kept in the blob store: its
replies and tool calls are there, and what each model was sent is not (it is kept as tokens in the trajectories).

`System(directory)` reads a run's directory and its ledger, and `System(ledger=…)` a ledger alone: `snapshot()`
(where every run stands), `group(run, number)`, `episode(run_id)`, `feeds()` (the episodes in the feeds),
`lineage(sample)` (the checkpoints view), `evals()` (every suite and eval), `statistics()` (with each run's name), `machines()` (the heartbeats),
`launches()`, `launch(asked)`, `stop(id)`, `rename(who, name)`, `bookmark(name, checkpoint)` and `unbookmark(name)`. `create_app(where)` serves them,
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
the checkpoint it names nor failed, then `committed` or `failed`. A group that is done carries the number and state of
the step that covers it, if one does; a recorded group that no step covers waits toward the next one.

## The checkpoints view

Every checkpoint grows from a base model along its first parents ([checkpoints](checkpoints.md#checkpoints)). Each base model is a
root, with a lane of its own; under it a lane for each run whose first checkpoint was trained from it, the run's checkpoints
from the left; and under a run's lane, each run that starts from one of its checkpoints: a fork, or a distillation's
start. A lane is folded to the checkpoints something points at (its first and newest, a checkpoint another run starts from,
a teacher, a bookmarked or evaluated checkpoint, one on its way to the engines), with a gap for the rest; a click on its
label opens it (what is open is remembered in the browser). A checkpoint opens its page. Between lanes:

- from a **base model** to the first checkpoint of each line trained from it;
- a **fork**: a checkpoint trained from another run's checkpoint;
- **learned from**: a checkpoint's other parents (dashed);
- a **distillation**: a diamond in its student's lane, before the checkpoints it made, with an edge from each teacher
  and, dashed, from the checkpoint the student starts from. Its mode is said in words on its lane and in the
  distillations below the graph: off-policy, the student is trained on its teachers' samples; on-policy, the student
  samples and its teachers score every token it sampled; mixed, both.

A checkpoint that something here starts from and that this ledger does not have stands in a lane of its own at the top.

`rollout_train.monitor.lineage` reads it from the ledger's tables and the runners' heartbeats, at `/api/checkpoints`. What
a ledger has today is read as it is: the checkpoints, the runs' steps, bookmarks. Each checkpoint says what its weights
are (a LoRA adapter, or full weights: those are resharded for the engines' layout on their way there). A run's steps
stand for its trainer's queue (a run takes one step at a time), whose weights are what the run's checkpoints are, and
the channels its runners' beats name for what its engines serve. The tables
for distillation, trainers and inference workers are proposed in [the checkpoint graph](../../research/policy-dag.md),
and nothing writes them yet; evaluations are written by [evals](evals.md), and a suite's name opens its page. `#/checkpoints/sample` (`?sample=1`)
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
  and end them, published weights, and the training loop's `result` and `step` notes.
- **The directory is bounded.** `keep` is the number of runs kept; the oldest are deleted.
- **One writer at a time.** Runs that an earlier writer left without an end, because its process was stopped, are
  marked cancelled when the next writer starts, so that the page does not show them running for ever.

`FeedReader` keeps a summary of each run (a line that is neither a run's event nor a sample is passed over) and reads
a run's lines from its file when the page asks for them.

The feed is a copy for people to read. It is not the run's record: the run's events are, and under a durable runner
they are in its store. Code that watches a run reads its episodes from the ledger ([rollouts](rollouts.md)).
