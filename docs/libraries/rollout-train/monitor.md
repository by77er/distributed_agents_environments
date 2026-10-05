# Monitor

Code: `rollout_train.monitor` (the service), `libraries/rollout-train/web` (the page) · See
[`RunFeed`](../../guide/reference.md#runfeed), [hooks](../rollout/hooks.md), [rollouts](rollouts.md#watching)

The monitor is a web page over a ledger and every run in it: one monitor shows all the runs that share a ledger,
on whichever machines they run. Its pages are switched along the top:

- **Runs**: every run, running ones first, and each run laid out as it runs: the **run**, which plays groups and
  takes steps on them; a **step**, one update over the groups queued when it was taken, and the checkpoint it made; a **group**, one start of one row of the environment, played as a number of **episodes**; an **episode**, one
  run of the program, with what it reported when it ended; a **rollout**, one agent's (model slot's) part of an
  episode, turn by turn: what it was sent, what it thought, what it did and what came back. Each rollout becomes a
  **trajectory**, the tokens the trainer learns from;
- **Checkpoints**: every checkpoint, each alone and all of them as a graph growing from their base models;
- **Evals**: the suites, every eval, and each checkpoint's or base model's evals over time;
- **Statistics**: figures across the training runs, a series for each, what each channel serves, and the ledger;
- **Machines**: every machine that beats and the roles on it (runners, sandbox pools, engine hosts, launchers and
  gateways), with what each holds and how full it is ([the machines](#the-machines)).

```bash
rollout monitor RUN                                   # http://localhost:8765: RUN's ledger, and every run in it
rollout monitor sqlite:///~/.cache/rollout/ledger.db  # a database's runs (or postgresql://…, or a ledger's directory)
```

`WHERE` is a run's directory (its ledger, as `ledger.json` there says, or files under `ledger`), a ledger's
directory of files, or a database's URL. The monitor reads, and writes four things: names in the registry
([the registry](checkpoints.md#the-registry)) (a run's, when it is renamed on its page, and bookmarks, made, moved and
deleted on a checkpoint's page); suites and their versions, made and edited on the Evals page
([evals](evals.md#made-edited-and-asked-for-from-the-page)); launches: runs and evals asked for from the page
([launching a run](#launching-a-run)), and asked to stop or to resume a run; and what is wanted of a run's settings
([a run's settings](#a-runs-settings)), whether it is paused among them
([pausing, resuming and stopping](#pausing-resuming-and-stopping)). The runs' processes write the rest, and need not be
running: the page shows a stopped run as it was left.

A run is shown by its name; its id (what everything kept of it is under) is on its page and under the pointer. A
checkpoint is shown by where it came from (the run that made it and its step: `diamonds · S3`) and the shortest start of
its id that no other has (`zwzk`), with the bookmarks that name it; its whole id, its depth and its parents are under
the pointer, and its page says them all.

## How the page is kept current

The page is a single-page application (React and TypeScript, in `libraries/rollout-train/web`). It is built into the
package (`rollout_train/monitor/static`), and the monitor serves it as files: `rollout monitor` needs no Node. The
page draws each view from what it has read, and reads again only what the monitor says changed.

- **Topics.** Each thing a view shows is a topic (`rollout_train.monitor.stream`): `system` (where every run stands),
  `feeds`, `machines`, `launches`, `statistics`, `checkpoints`, `checkpoint-evals/ID`, `path/ID`, `environments`,
  `environment/MODULE:NAME` (a published one's `environment/NAME@VERSION`), `environment-versions`,
  `environment-version/VERSION`, `imports`, `evals`, `eval-subjects`, `history/checkpoint/ID`, `history/model/NAME`,
  `settings/RUN`, `group/RUN/N`, `episode/RUN_ID`. The monitor reads a topic at most once a beat (1.5 seconds), whoever
  asks, and gives
  each reading a **version**, a hash of what it says (leaving out when it was read). Each beat it reads the ledger once
  for every topic watched (`System.one_reading`; a database ledger in one query, `read_all`).
- **The stream.** `/api/stream?topic=…` is a stream of server-sent events: the version of each topic asked for at
  once, then a `version` event each time one changes. The page watches the topics of the place shown (always
  `system` and `feeds`; the launches on Runs and New run; a group, an episode, the statistics, the machines, the
  environments (with the imports on their list) and an environment, or the graph, as they are shown), and opens a new
  stream
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
them ([rollouts](rollouts.md)), and the checkpoints. Each run keeps the rest in its own directory, as an open
[profile](../../guide/deploying.md) lays it out (`rollout_train.layout`):

| | Holds | The page reads it for |
|---|---|---|
| the ledger | each run's `starts`, `groups`, `results`, `steps` and `failures`; its `episodes`, `claims` and `interrupted`; the `checkpoints`; the fences ([checkpoints](checkpoints.md)) | every run, its steps, groups and their stages, the episodes that ended and what each reported, which runner plays what, outcomes, checkpoints, the statistics |
| the registry | each run's id and name, and the bookmarks ([the registry](checkpoints.md#the-registry)) | what each run is called, and the bookmarks that name each checkpoint |
| the heartbeats | each runner's, pool's, engine host's, launcher's and gateway's newest beat, with its recent beats' measurements (`rollout_train.presence`) | the machines and the roles on them (memory, accelerators, disk, places, pools, what each channel serves and how fast), whether a run is running or paused, which launchers are alive and what they offer |
| the launches | the runs and evals asked for and how each goes (`rollout_train.launches`) | the launches on Runs and Evals, the New run form, a suite's Run this suite form, and each launcher's launches going |
| the published environments | each version of an environment imported from git: its source, commit, entry point, blob, runtime environment, what it says of itself and its check (`rollout_train.published`) | the published environments on Environments, their pages, and the eval data a launch on one may name |
| the sandboxes' leases | each pool's leases, by key (`rollout_train.sandboxes`) | each pool's leases on the Machines page: the episode each was acquired for, how long it has been held, its time limit |
| `evaluations/` | each suite's versions, and each eval's subject (with the version it played) and results ([evals](evals.md)) | the Evals page, a suite's page, and the suites a checkpoint played |
| the registry's suite names | the version each suite's name points to ([versions](evals.md#versions)) | which version a suite's page, its tile and the forms take as its newest |
| a run's `feed` | what is happening now, written by `RunFeed` | episodes still running, their rollouts turn by turn |
| the run's blob store | each ended episode's events, where the run's newest start says its blobs are (`blobs`: files, or S3) | the rollouts of episodes no longer in the feed |
| the ledger's turns | each turn the gateway recorded (`runs/RUN/turns/RUN_ID`), its reply in the blob store | the rollouts of episodes whose feed and kept events hold no sample: a harness's, recorded by a gateway elsewhere |

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

A run's state:

| State | When |
|---|---|
| **finished**, **stopped**, **failed** | its newest start said how it ended: it played what it was asked, it was interrupted or stopped, or it raised (the page shows the error). Every command that starts a run (`rollout train`, `eval`, `imitate`, `env check`) says so as it exits, in the run's `ends` table under the fence its start was written with (`rollout_train.record.ending`), so a process another has replaced says nothing of the newer start |
| **running** | one of its runners beat ([heartbeats](rollouts.md#heartbeats)) within the last 90 seconds (by the clock of the store that keeps the beats) and something this reads was written within 20 minutes |
| **idle** | one of its runners beat within 90 seconds, but nothing was written for 20 minutes: its process is there and waiting |
| **paused** | running or idle, and a runner alive says in its beat that the run is paused ([pausing and resuming](training.md#pausing-and-resuming)); the page says **pausing** while the run is wanted paused (`pause`) and no runner says so yet |
| **lost** | its runners beat once and no longer do, and it never said how it ended: it crashed or was killed |
| **ended** | a run with no beat and no word of how it ended; one started within the last 90 seconds is **running**, its first beat yet to come |

Only while it beats does its page say what its channels serve; a runner beats again as soon as a channel serves a
new checkpoint, so that says the checkpoint served now. An eval that played every start has finished; one a training
run's schedule asked for, and not done, is as that run is (that run's runner plays it), or paused on its own. No process is asked, so a run
on any machine is told apart the same way.

## The pages

Each page has a sidebar of its own. On **Runs**, every run (its state, and its host when it is not this one)
folding open to the groups toward its next step (in flight, or recorded and waiting for a step), then its steps,
newest first, each with the groups that went into it (a square for each episode); then the episodes outside a run.
A step's groups need not be consecutive. A group that gave nothing to train on is listed with the step decided after
it, marked skipped. Runs, steps, groups and episodes fold open and closed: an open step lists its groups, an open
group its episodes, and an open episode its rollouts. On **Checkpoints**, the graph, the base models checkpoints were
trained from, the checkpoints bookmarks name, and each run's newest. On **Evals**, the suites, each folding open to the
evals that played it (each marked with the version it played, where a suite's evals played more than one), and the
eval shown; then the checkpoints (and base models) that have had an eval, the one evaluated last first, each opening
its history and folding open to its evals. An eval's run is on Evals, not among the runs, nor in the statistics: its
page is `#/eval/RUN`. On **Environments**, every
environment (a square, lit where a launcher alive offers it), each folding open to its training runs and the suites
with a version that plays it. On **Statistics**, its sections, and the training runs drawn: a click leaves a run out or takes it
back. What is folded and what is left out are remembered in the browser. The address (after `#`) names what is shown,
so a reload stays there.

| Page | View | Address | Shows |
|---|---|---|---|
| Runs | Every run | `#/runs` (and `#/`) | the launches (each asked-for run: its state, launcher, directory, settings changed, why it failed, a Stop button); each run, running ones first: its state and host, groups in flight and done, steps, the share solved over its last groups, each group's mean reward in order, and where its episodes are read |
| Runs | New run | `#/runs/new` | a form asking for a run ([launching a run](#launching-a-run)) |
| Runs | Run | `#/run/RUN` | its name (with a control to rename it) and the buttons its state allows ([pausing, resuming and stopping](#pausing-resuming-and-stopping)), id, state, base model (or the full checkpoint its adapters build on), the checkpoint it started from and the one it is at, what each of its channels serves; figures: where it started and is now, steps, groups done and solved (all, some, none), episodes, the share solved early and late, mean reward, rows unlocked, inference; the step being taken, with its groups; the groups recorded and waiting for a step; each group in flight with its stage (asked, claimed, played, recorded) and episodes; every group's rewards, in the order of the steps they went into (a column opens its group); the latest steps; the tasks played; each suite's score along the line to its newest checkpoint ([scores along a line](#scores-along-a-line)); its settings ([a run's settings](#a-runs-settings)). An eval's run shows the eval's page instead |
| Runs | Step | `#/run/RUN/step/N` | the checkpoint the step made and its parent, the groups that went into it (and those decided before it that gave nothing to train on), and the update's statistics |
| Runs | Group | `#/run/RUN/group/N` | the group's stage, its episodes (each with its reward and what it reported; one playing with its reward so far, and each slot's where they differ; one asked for and not started holds a place; one cut short is marked interrupted), which runners play it, the step it went into, what was done with it (the step's statistics and the checkpoint it made, or why it was skipped), and its start |
| Runs | Episode | `#/episode/RUN_ID`, `#/episode/RUN_ID/SLOT` | what the episode reported, and its rollouts, every agent's side by side or one: **turn by turn** (a slider over turns, following the newest unless one moves it, and for the turn shown **Sees**, **Thinks**, **Does** and **Result**), or the **whole trajectory** (every turn a row: what each agent did and what came back, with what it saw and thought a click away); the program's own tool calls below ([an episode's rollouts](#an-episodes-rollouts)) |
| Runs | Episodes outside a run | `#/episodes` | episodes in the feeds that no run asked for: tests, programs run by hand |
| Checkpoints | Every checkpoint, as a graph | `#/checkpoints` | each base model a root (one checkpoints were trained from, or one that has had an eval), and under it a lane for each run with its checkpoints, a run that starts from another's checkpoint hanging under it; each run's trainer and its queue over time; each checkpoint's way to the engines and the run engines that serve it; the runs ([the checkpoints view](#the-checkpoints-view)) |
| Checkpoints | Checkpoint | `#/checkpoint/ID` (or the start of one) | where it came from (its parents, run and step), what its weights are (a LoRA adapter, or full) and what they build on (a base model, or the full checkpoint an adapter is over), its bookmarks (with controls to make one, move one here, or take one away), how far it moved and what is kept of it, its line back to the base model, what grew from it, each version of a suite's score along its line, each environment's apart for a suite of several ([scores along a line](#scores-along-a-line)), every eval it had (by hand or by its run's schedule: the suite and, for a suite of more than one version, the version played, the share solved where its episodes say, the mean reward, each environment's apart for a suite of several, episodes, who asked for it, when; each opening the eval; and **history**, opening its history on Evals), a **Run an eval** form (a suite, its version, newest by default, episodes per start, a profile and a name: it posts the same launch as a suite's **Run this suite** form), and the evals of it asked for so, with how each goes |
| Checkpoints | Base model | `#/base/NAME` | its name; the runs trained from it and those forked from theirs (each run with what it started from, its checkpoints and its newest, opening the run); every eval it has had, a card for each version of a suite it played ([a subject's history](#a-subjects-history)); a **Run an eval** form that posts the launch with the base model as `model`, offering the profiles whose launchers offer it ([evals](evals.md#made-edited-and-asked-for-from-the-page)), and the evals of it asked for so, with how each goes |
| Evals | Every suite and eval | `#/evals` | a **New suite** form ([made, edited and asked for from the page](evals.md#made-edited-and-asked-for-from-the-page)); the evals asked for from the page; each suite (the version its name points to, where it has more than one, its environments, starts, subjects, and, for a suite of one environment, the subject that did best at that version); every checkpoint and base model that has had an eval, the one evaluated last first (where it came from, its evals counted and those playing, the suites they played, when the newest began), each opening its history; every eval, newest first (its suite and, for a suite of more than one version, the version it played, who played, episodes played of those asked for, the share solved, each environment's for a suite of several, whether it is done), each opening its page |
| Evals | Eval | `#/eval/RUN` | the buttons its state allows, while it is not done ([pausing, resuming and stopping](#pausing-resuming-and-stopping)); who played (a checkpoint, or the base model), the suite and the version it played, who asked for it (by hand, or a run's schedule at a step), when it started; its share solved and mean reward (each environment's, for a suite of several), episodes played of those asked for, how long it took; and at each start of that version (by environment, for a suite of several), its episodes, how many solved and their mean reward. A run that plays one environment of an eval (`part_of`) shows its eval |
| Evals | A checkpoint's or base model's evals | `#/evals/checkpoint/ID` (or the start of one), `#/evals/model/NAME` | what it is (a checkpoint's shortest id and bookmarks, its id opening its page, the run and step that made it; or the base model's name), its evals and suites counted; for a checkpoint, each version of a suite's score along its line ([scores along a line](#scores-along-a-line)); then every eval it has had ([a subject's history](#a-subjects-history)), a card for each version of a suite it played (suites by name, a suite's versions newest first, a suite's name opening its page): each environment's share solved where its episodes say (else its mean reward) over time, and its evals, newest first, each with its share solved and mean reward at each environment, episodes, who asked for it (by hand, or a run's schedule at a step), when it started and whether it is done, each opening the eval |
| Evals | Suite | `#/evals/SUITE` | the version its name points to (another, once picked): its version, environment and its version, how its starts were chosen, rows, seeds, episodes per start, limits (for a suite of several environments, a table of them, one each), whether it is held out of training; an **Edit** form that saves its next version, adding and removing environments; for a suite of one environment, the subject that did best at that version; a **Run this suite** form ([evals](evals.md#made-edited-and-asked-for-from-the-page)); its launches and the evals playing it; every subject's episodes at every start, with totals: a column group for each version played, newest first and marked off from the next (a version picker shows one), within a version of several environments a column group for each environment with each subject's total there, a row for each start of the versions shown (struck through under a version that does not have it), and within a version a column per subject: the base model first, then each run's checkpoints under the run's name (those made outside a run together), runs in the order their checkpoints grew and within a run by depth, each column saying the checkpoint, what its weights are (`full`, or `over full` for an adapter over full weights), its step, its episodes per start (opening the eval) and whether a schedule asked for it; two subjects of one version compared at the starts both played |
| Environments | Every environment | `#/environments` | an **Import from git** form ([importing from git](#importing-from-git)); every environment the system knows of ([environments](#environments)), by name: its `module:name` (a published one's entry point), whether a launcher alive offers it, the versions seen, a published one's source and commit and what its check found, its training runs and suites counted, and when a run last started on it; each opening its page |
| Environments | Environment | `#/environment/MODULE:NAME`, `#/environment/NAME@VERSION` | its name and `module:name` (a published one's source, commit and ref, entry point, version and when it was imported, and its import's check, each finding with what it said), whether a launcher alive offers it, its version as it loads on the monitor's machine (or that it does not load there) and every version seen, what its results say (the reward range, whether they say solved, what a duration counts) and its curriculum (the generic one, or its own by class); **New run** (the new run's form with it chosen) and **New suite** (the new suite's form with it as the first entry); its rows, easiest first as it orders them, each with the groups and episodes its training runs played of it, the share solved, the mean reward and its eval starts (a row only its eval data has is marked eval only), and every row together; each list of its eval data and how many starts; the suites with a version that plays it (each version, the one its name points to marked); its training runs (state, version, groups, episodes, share solved, when started); every eval of such a suite, by who played it, with its score at this environment's entry; its newest check, each group with its rewards and whether every episode scored the same |
| Statistics | Across every training run | `#/statistics`, `#/statistics/SECTION` | the sections below, each run in its own color; evals and their parts are on Evals |

Renaming on a run's page asks the monitor (`POST /api/rename`, `{"id", "name"}`), which renames it in the registry
(`System.rename`). A checkpoint's page makes a bookmark
name it, or moves one there (`POST /api/bookmarks`, `{"name", "checkpoint"}`, where the checkpoint may be any
[reference](checkpoints.md#references)), and takes one away (`DELETE /api/bookmarks/NAME`). A name another run has, or
one that says `/`, `@` or `:`, is refused (409) and the page says why. The **New suite** and **Edit** forms make a
suite or its next version (`POST /api/suites/NAME`, `System.save_suite`), which moves the suite's name in the registry
([made, edited and asked for from the page](evals.md#made-edited-and-asked-for-from-the-page)). Every page open on the
monitor hears of the change through its stream.

## Environments

The **Environments** page lists every environment the system knows of, by `module:name`, and each published one by
`NAME@VERSION` ([importing from git](#importing-from-git)) (`rollout_train.monitor.environments`). It knows them from
several sources, each a function from what the monitor read (the ledger's tables, the launchers alive, the registry's
names, the published versions) to what it saw of environments: the launchers alive that offer one, the runs started on
one (training runs, evals and checks), the suites whose versions play one, and the published versions the ledger
keeps. The list folds every source's word of an environment into one line: the versions seen (in training runs'
starts and suites' entries; an eval's version is its suite's), whether a launcher alive offers it, its training runs,
the suites with a version that plays it, when a run last started on it, and for a published one where its source came
from (URL, ref, commit, subdirectory, entry point) and what its check found (`published`). A source is one function
added to `SOURCES`. `/api/environments` serves the list (topic `environments`); the pickers on the forms read it too.

An environment's page (`/api/environments/MODULE:NAME`, topic `environment/MODULE:NAME`; 404 for one neither known nor
loading) imports the environment in the monitor's process, once, for what it says of itself: its version, description,
rows, eval data and curriculum. A published one's page (`/api/environments/NAME@VERSION`) takes what it says of itself
from its version's record, as its check in its own runtime environment found it, and adds its source and check
(`published`). Where an environment does not load, the page says why, its rows are those its runs played
(in the order first played), and its description is its newest run's start's. An environment of a project with an
environment of its own, such as GSM8K (`rollout_verifiers.environments:gsm8k`, [verifiers
environments](../../implementations/rollout-verifiers.md#gsm8k)), is listed from a launcher started in that project
that offers it, from the runs and evals played there, and from the suites that name it; its page shows those, without
its own rows and eval data. Beside that, from the ledger:

- each row's groups and episodes in the training runs on it (their `results` tables, matched to the row by the key in
  each group's record), and how many solved, of the episodes fit to train on in runs whose description says their
  results report solving; failed episodes count as played;
- the training runs on it (a start that names it and is no eval and no check), newest first;
- the suites with a version that plays it, and every eval of such a version with its score at this environment's entry
  ([scores along a line](#scores-along-a-line));
- its newest check (`rollout env check` with a profile, whose start says `kind: check`): each group, its rewards, its
  failed episodes, and whether every episode scored the same, which teaches a group-relative update nothing.

The forms the page opens are the new run's (`#/runs/new?environment=MODULE:NAME` chooses it) and the new suite's,
with it as the first entry. The suites' and the new run's forms read the same answer for an environment's rows and
eval data, and say "does not load here" where it does not.

### Importing from git

**Import from git**, on the Environments page, imports an environment from a git repository
([writing an environment others can import](../../guide/publishing.md)): its URL, a ref (a branch, a tag or a commit;
empty for the default branch), the project's subdirectory (empty for the root) and its entry point (empty for the one
its `pyproject.toml` declares). It posts them to `POST /api/environments/import` (`System.import_environment`,
`rollout_train.publishing.publish`), which fetches the source at that ref, packs and stores it, checks it in a Ray job
in its own runtime environment and records the version; it answers with the version (`version`, and `existing` where
the same source was imported before: that version, not checked again), or 422 with why the import was refused (a source
that does not clone, no `pyproject.toml`, no entry point, an entry point that does not load in its runtime environment,
checks that fail), which the form says under its button. While the import goes, the button says its stage (fetching,
reading, packing, storing, checking on Ray, recording), from `/api/environments/imports` (topic `imports`: the imports
this monitor made since it started, newest first, each with its stage, when it began and ended, and the version it made
or why it was refused). Once done, the page opens the version's page.

The published versions are read at `/api/environments/versions` (topic `environment-versions`; with `importing`,
whether this monitor imports) and one at `/api/environments/versions/VERSION`, by its id or `NAME@VERSION` (topic
`environment-version/VERSION`; 404 for one the ledger does not keep).

A monitor imports with the blob store, the Ray cluster's job server and the scratch directory of a cluster config
(`rollout monitor WHERE --cluster [PATH or NAME]`, [the cluster config](../../guide/cluster.md);
`rollout_train.publishing.Importer`), and records the versions beside the ledger it watches; one started without a
cluster config imports nothing, says so (409), and its button is greyed. It reaches the job server as the launchers do
(with the cluster's token where the cluster asks for one), and clones into `[scratch]/imports`, deleting each clone once
it is read.

## Scores along a line

A checkpoint's line is its first parents back to the checkpoint trained from the base model, and the base model before
it (`/api/checkpoints/ID/path`, `rollout_train.monitor.scores.path_of`). It crosses runs (a run that starts from
another's checkpoint), merges (a full checkpoint made from an adapter, outside a run) and forks. The chart draws a line
for each version of a suite any point was evaluated on (named by the suite, and the version where the line has more
than one of it), and for a version of several environments a line for each environment (named by it too): its share
solved at each point where its episodes say whether they solved their start, else its mean
reward, by depth (0 is the base model), every eval of a point on that version pooled. Its
scale runs from the data's own lowest value. A rule marks where the line enters a run (with the run's name, or
"outside a run" for a merge) and where its weights change between full and LoRA; each point is named by a bookmark
that names it, else its short id. A checkpoint's page draws the line ending at it, a run's page the line to the run's
newest checkpoint.

A checkpoint's evals (`/api/checkpoints/ID/evals`, `rollout_train.monitor.scores.evals_of`) are read from each
suite's subjects (`evaluations/SUITE/EVAL/subject`, whose `checkpoint` it is) and their results, the eval's start (when,
and `by` and `step` for one a schedule asked for), and the training runs' `evals` tables.

## A subject's history

A subject is what an eval played: a checkpoint, or a base model by name. Every subject that has had an eval
(`/api/evals/subjects`, `rollout_train.monitor.scores.subjects_in`) is listed the one evaluated last first:

```json
{"subjects": [{"kind": "checkpoint", "id": "oznzomovunlxwyow", "short": "oznz", "run": "lora-run",
               "name": "lora-run", "step": 2, "base": "tiny", "bookmarks": ["best"],
               "evals": ["eval-second-v2", "eval-second"], "suites": ["words-v1"], "started": 1790904111.6,
               "playing": 0}]}
```

`evals` are its evals' runs, newest first; `started` is when the newest began, and `playing` how many have not played
everything yet. A base model's `kind` is `model`, its `id` and `short` its name, and it has no run, step, base or
bookmarks.

A subject's history (`/api/evals/checkpoint/ID`, where the id may be the start of one, and `/api/evals/model/NAME`,
`rollout_train.monitor.scores.history_of`) is `{"subject": ..., "evals": [...]}`: the subject as listed, and every
eval it has had, newest first, each as a checkpoint's evals are (its suite and version, who asked for it and when,
episodes played of those expected, the share solved and the mean reward, and the same at each entry of its version,
by environment). A checkpoint or base model with no eval has a history with none; one the ledger neither has, nor
builds a checkpoint on, nor has evaluated answers 404. Both are scored as a checkpoint's evals are, by one reading of
the evals' subjects and results.

## A run's settings

A training run's page ends with its settings (`/api/runs/RUN/settings`, `System.settings`), as its newest start
records them ([run settings](training.md#changing-a-running-runs-settings)). **Changeable** is a form of the settings it
takes between steps: `groups_per_step`, `evals.suite` (a suite, or none), `evals.every`, `evals.episodes` (empty for
the suite's own), and its
trainer's (`trainer.learning_rate`, say), each holding what is wanted of it, else what the newest step used; one wanted
and not yet used says what the run uses now. Under it, each step that used other settings than the step before, with
what changed. **Save** asks the monitor (`POST /api/runs/RUN/settings`, `{"settings": {KEY: VALUE}}`, only those
changed), which checks each (`System.want`: a setting the run can change, a whole number of 1 at least where one is
needed, no version a suite does not have; a suite may play other environments than the run's, each on the run's
channel; a name the ledger has no suite of is the environment's eval data of that name, frozen when the run first plays
it) and keeps it beside the
ledger; a fixed setting, or one the run does not have, is
refused (409) and the page says why. The run takes them when it next decides a step (or, stopped, when it is started
again). **Fixed** lists the rest as they are: the model, what the trainer is and makes, the adapter's rank, the
channels and their engines (each channel's `thinking_tokens` and `answer_tokens`, "none" for no budget), how many
episodes it plays at once, the groups and seed it was started with; a value that is unset reads "none". A run whose
start records no settings shows none.

## Pausing, resuming and stopping

A run's page and an eval's page show the buttons its state allows ([pausing and resuming](training.md#pausing-and-resuming)):

| Button | Shown | Asks |
|---|---|---|
| **Pause** | running or idle, and not wanted paused | `POST /api/runs/RUN/pause` (`System.pause`): its desired settings say paused |
| **Resume** | wanted paused while its process is there | `POST /api/runs/RUN/resume` (`System.resume`): it goes on in place |
| **Resume** | stopped, failed, lost or ended, with no launch of it going (not an eval a run's schedule asked for) | the same: a launch that starts it again in its directory, which then shows on Runs (marked `resumes`) |
| **Stop** | a launch going started it (in its directory) or resumes it | `POST /api/launches/ID/stop` |

A refusal (409: running and not paused, finished, being launched already; 404: no such run, or no launcher alive offers
its profile and environments) is said beside the buttons. The Runs page counts paused runs, and a paused run's tile
and its dot in the sidebar are drawn in violet. Every page open on the monitor hears of the change through its stream.

## Launching a run

A **launcher** on a training machine (`rollout launcher --ledger URL --profiles DIR --environment module:name --runs DIR
[--at-once N] [--ray URL]`, `rollout_train.launcher`, [launchers](../../guide/deploying.md#launchers)) beats every 15 seconds, saying what it offers: each profile under
`--profiles` that names a trainer, with what its trainer makes (`weights`: `lora` or `full`, where the trainer's class
says, as `rollout_lora`'s do), the base models an eval may play with it (`models`: its channel's model, and with
`--cluster` the models the cluster's inference providers of its engine's kind serve) and the settings a launch may
change and their values in the profile (the trainer's
settings, `trainer.start`, `trainer.bookmark`, `episodes_at_once`, each channel's `thinking_tokens` and
`answer_tokens`, and the `evals.` settings), the environments, and how many runs it plays of how many it may. A
launcher with `--ray` offers the published environments too, each by `NAME@VERSION` beside those it names (and in
`published`, with its source and commit), so the form lists them.

**New run** (`#/runs/new`, from the Runs page) offers what the launchers alive offer: a profile and an environment
(picked from those the monitor knows, by name with `module:name` under it; one no launcher alive offers cannot be
picked), the
run's name, the checkpoint it starts from (the base model, a bookmark, or any checkpoint whose weights are kept, by where it
came from and what its weights are; a profile whose trainer trains every weight starts only from full weights, so an
adapter is merged first), a bookmark for it to carry, its groups, groups a step and seed, the evals it makes of its
checkpoints (a suite whose every environment a launcher of the profile offers, by its name, which follows its newest
version, or none; every how many steps;
and episodes per start, empty for the suite's own: the launch's `evals.suite`, `evals.every` and `evals.episodes`),
and every other setting of the profile as a field holding the profile's value (a channel's `thinking_tokens` or
`answer_tokens` the profile leaves unset reads "none", and one emptied is sent as `none`: no budget),
with rows for any other `trainer.KEY`. Values are read as numbers, true or false, or JSON
where they look like them, and as text otherwise. The suite is the profile's `[evals]` suite, else the environment's
own eval data (the first it has), else chosen: the run is not launched until it says a suite or none. Launching asks
the monitor (`POST /api/launches`, with the settings changed and the evals), which checks the ask (`System.launch`: a
launcher alive offers the profile and every environment the run plays (its own and its evals' suite's, which the
launch names in `environments`), the name is no other run's, every setting is the profile's, a trainer's or one every
training run can change, the checkpoint is one, and the run says the evals it makes: a suite in `evals.suite`, null for
none, or its profile's `[evals]`; a suite in the ledger, a version the suite has, or, where the environment loads on the
monitor's machine, its eval data of that name; for a published environment, the eval data its version recorded) and
appends it to the launches; a
refusal (409 or 404) is said under the button. With no launcher alive, the form says so and gives the command that
starts one.

A launcher claims only a launch whose every environment it offers (one that names no environments offers any), and,
for an eval that names a base model (`model`), whose profile it offers that model with. It
makes the run's directory under `--runs` (`NAME-ID`), and starts `rollout train` there with
`--name` and each setting as `--set KEY=VALUE` (no evals as `--set evals.suite=""`, so that the profile's `[evals]` is
not used), as a process of its own whose output goes to `train.log` in that directory, or, with `--ray`, as a Ray job;
a run on a published environment is a Ray job in that version's runtime environment, whose start records the version
(`published`: its name, id, source, ref, commit, subdirectory and entry point). The launch's state
follows: `asked`, `claimed`, `running` (with the process), and `ended`, `failed` (with the end of its output) or
`stopped`. **Stop** (`POST /api/launches/ID/stop`) cancels a launch not yet claimed at once; a running one is sent an
interrupt, and the run stops as on Ctrl-C, at a group boundary. Once the run writes its start, its tile links to it.

## Statistics

The statistics, read from the ledger (`rollout_train.monitor.statistics`) and, for the engines, from each run's
runners' heartbeats; every chart is drawn to scale and says each series' value under the pointer:

| Section | Shows |
|---|---|
| `outcomes` | the solve rate (where a task says whether it solved) and the mean reward over groups, in the order their results were written: a line for each run, the mean of the last 4, 8, 16 or 32 groups, and a dot for each group |
| `rows` | every row played: groups, episodes, the share solved, mean and best reward, the newest group's rewards; and, where groups' starts list `names`, the same by how many names a start gives |
| `steps` | the trainer's statistics over the steps, a chart each: KL moved, KL floor, clip fraction, mean mismatch, mean weight, truncated fraction, loss, the step's time and its start time (as far as each checkpoint says them) |
| `pace` | episodes and groups an hour (counted when each group's result was written); what was done with each group (trained on, in a step being taken, waiting for a step, nothing to train on, no episode or a failed step, in flight); why groups gave nothing to train on |
| `queue` | how many groups were in flight (decided, their result not written), and how many waited for a step (recorded with something to train on, no step begun over them), over time |
| `inference` | each run's engines: tokens a second and requests at once, a measurement each beat while they are busy; then each channel, what it serves and how fast now |
| `ledger` | each runner that took its fence in the ledger (its fence, the claims it has made, how many hold, when it last claimed), the ledger's fences (but [episodes'](rollouts.md#each-episodes-fence), one per episode claimed) and tables |

An episode's reward is summed as the trainer sums it ([rewards](episodes.md#rewards)): the mean of its slots'. One
still playing is shown with its reward so far, read from its feed the same way, and each slot's where they differ.

Whether an episode solved its task is what its program reported (`info["solved"]`), and a task need not say. The
monitor serves `solved` as null for an episode that did not say, and, in a group's result (whose `solved` the loop
writes as true or false for each episode), for each episode of a group none of whose episodes said; an eval's results
and counts likewise. Where nothing says, the page shows no solve figures (no share solved, no solve rate, no solved and
not-solved colors) and the mean reward in their place.

## The machines

The Machines page (`#/machines`) shows every process that beats beside the ledger
([heartbeats](rollouts.md#heartbeats)) by what its beat says it is, beside what the ledger says of it
(`rollout_train.monitor.machines`). A process is alive while its newest beat is younger than 90 seconds by the clock of
the store that keeps the beats; when it last beat is shown by the monitor's clock. The page has a section for each
kind present, alive ones as cards and the gone ones in a table under them:

| Section | Each card shows |
|---|---|
| Runners | its places, how many it plays and how many are free; its machine's memory, accelerators and disk; the episodes its claims hold (each run, group, episode and attempt, linked, and since when); what its channels serve and how fast |
| Sandbox pools | the kind of sandbox, the runner it is in (a runner's pool is `KIND@RUNNER`) or its own name, how many sandboxes are leased of how many and how many are free; each lease in the `sandboxes` table: the run's episode it was acquired for (linked), the sandbox's name, how long it has been held, its time limit and what is left of it, and whether its claim holds, has lapsed, or its sandbox is lost |
| Engine hosts | the run it follows, its machine; for each channel, what it serves, what the run wants it to (the checkpoint of the greatest depth in the run's `serving` table), tokens a second and requests at once, and each engine's address, the checkpoint it serves and how many checkpoints behind the wanted one it is |
| Launchers | how many launches it plays of how many at once, the profiles and environments it offers, and its launches going (claimed, running or stopping) |
| Gateways | where it listens, and what each channel it samples serves and how fast |

The sidebar lists the machines (by the host each beat names), alive ones first, each opening to its roles; a role
opens its card on its machine's page (`#/machines/HOST/NAME`). A machine's page has everything on that host: its
memory, accelerators and disk now, and memory and accelerator memory in use, how busy each accelerator was and disk in
use over its recent beats (the last hour, at one beat every 15 seconds), then its roles.

`/api/machines` answers `{"now", "hosts", "runners", "pools", "engines", "launchers", "gateways"}`. Each role has its
`name` among the beats, its `host`, `alive` and `at` (when it last beat), and what its kind adds: a runner its `run`,
`places`, `playing`, `free`, `claims`, `pools` and `channels`; a pool its `kind`, `runner`, `size`, `leased`, `free` and
`leases`; an engine host what it `follows` and its `channels`, each with `wanted`, `behind` and its `engines`; a
launcher its `profiles`, `environments`, `at_once`, `playing` and `launches`; a gateway where it `listen`s and its
`channels`. Each host has `alive`, `at`, its newest measurements (`machine`), their `history`, and its `roles`
(`{"kind", "name", "alive"}`).

## An episode's rollouts

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
Where neither the feed nor the kept events hold a sample of an episode (a harness's samples, which a gateway elsewhere
recorded and the runner's hooks never saw), its rollouts are the replies of the turns the gateway recorded, read from
the ledger, live and once it ended.

## Reading it from Python

`System(directory)` reads a run's directory and its ledger, and `System(ledger=…)` a ledger alone: `snapshot()`
(where every run stands), `group(run, number)`, `episode(run_id)`, `feeds()` (the episodes in the feeds), `lineage()`
(the checkpoints view), `evals()` (every suite, its versions, and every eval), `checkpoint_evals(id)`, `path(id)`,
`eval_subjects()`, `history(kind, reference)`, `statistics()` (with each run's name), `machines()` (every machine that
beats and the roles on it), `environments()` (every environment the system knows), `environment(name)` (an
environment's page: [environments](#environments)), `settings(run)` and `want(run, settings)`, `pause(run)`,
`resume(run)`, `launches()`, `launch(asked)`, `stop(id)`, `rename(who, name)`, `bookmark(name, checkpoint)`,
`unbookmark(name)`, `save_suite(name, body)`, `environment_versions()`, `environment_version(reference)`,
`import_environment(body)` and `imports()` (with `System(…, importer=Importer(blobs, jobs, scratch))`).
`create_app(where, importer=…)` serves them, the stream and the page; its routes are listed in
`rollout_train.monitor.app`.

A group's stage is read from the records alone, so it is what a [loop](training.md) that started now would find:

| Stage | The records say |
|---|---|
| `waiting` | the group is in the run's `groups` table, asking for its `episodes`, and no runner has claimed any of them |
| `playing` | a runner has claimed one of its episodes, at least, and fewer have ended than were asked for |
| `ended` | as many episodes are in the run's `episodes` table as were asked for, and no result is written yet |
| `done` | the `results` table has its result |

A claim holds while it is its episode's latest attempt, its runner keeps the fence it was made (or adopted) under,
and the attempt was not cut short ([claims](rollouts.md#what-runners-write)); the page lists the claims that hold as
what each runner plays. A step's state is read likewise: `stepping` while it has neither made
the checkpoint it names nor failed, then `committed` or `failed`. A group that is done carries the number and state of
the step that covers it, if one does; a recorded group that no step covers waits toward the next one.

## The checkpoints view

Every checkpoint grows from a base model along its first parents ([checkpoints](checkpoints.md#checkpoints)). Each base model is a
root, with a lane of its own (a base model that has had an eval is one too, whether or not anything was trained from
it); under it a lane for each run whose first checkpoint was trained from it, the run's checkpoints
from the left; and under a run's lane, each run that starts from one of its checkpoints: a fork. A lane is folded to the
checkpoints something points at (its first and newest, a checkpoint another run starts from, another checkpoint's other
parent, a bookmarked checkpoint, one on its way to the engines), with a gap for the rest; a click on its
label opens it (what is open is remembered in the browser). A base model opens its page (`#/base/NAME`), as a checkpoint
opens its own, by a click, or by Enter or Space once it has the focus (Tab moves it from one to the next). Between
lanes:

- from a **base model** to the first checkpoint of each line trained from it;
- a **fork**: a checkpoint trained from another run's checkpoint;
- **learned from**: a checkpoint's other parents (dashed): a merge's, or the checkpoints that sampled the
  [dataset](datasets.md) a supervised step trained on, which the checkpoint names (`dataset`, served with each
  checkpoint here and at `/api/system`).

A checkpoint that something here starts from and that this ledger does not have stands in a lane of its own at the top.

`rollout_train.monitor.lineage` reads it from the ledger's tables and the runners' heartbeats, at `/api/checkpoints`. What
a ledger has today is read as it is: the checkpoints, the runs' steps, bookmarks. Each checkpoint says what its weights
are (a LoRA adapter, or full weights: a bridge makes those into the engines' files on their way there). A run's steps
stand for its trainer's queue (a run takes one step at a time), whose weights are what the run's checkpoints are, and
the checkpoints its runners' beats say it published for what its engines serve: each until the next. Evals are on the
Evals page ([a subject's history](#a-subjects-history)). Tables for distillation, shared trainers and inference workers
are proposed in [the checkpoint graph](../../research/policy-dag.md).

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

The feed is a copy for people to read, not the run's record. Code that watches a run reads its episodes from the
ledger ([rollouts](rollouts.md)).
