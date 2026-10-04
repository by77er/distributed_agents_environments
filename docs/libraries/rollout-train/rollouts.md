# Rollouts

Code: `rollout_train.rollouts` · See [episodes](episodes.md), [training](training.md),
[API reference](../../guide/reference.md#rollout_trainrollouts)

A run asks for episodes in the [ledger](checkpoints.md#the-ledger), and runners play them. The run decides what to play,
how often and how to group it; a runner knows no algorithm, and the run never learns where its episodes were played.
Several runners, on one machine or many, share the work of every run whose ledger they reach: which machine plays a
group's episodes is only a matter of where runners are.

```python
await plan(ledger, "train", Plan(program, binding), fence)        # how the run's episodes are played
await ledger.append(table("train", GROUPS), "1", {"parameters": row, "episodes": 4}, fence)
async with playing(EpisodeRunner("host/train", ledger, runner, recorder, blobs, places=6)):
    episodes = await episodes_of(ledger, blobs, "train", 1, 4)     # the group's four, once all have ended
```

The [training loop](training.md) writes the plan and the groups; a [profile](../../guide/deploying.md) opens a runner
beside it. Everything goes through the ledger, so the run and its runners may be in one process or on different
machines.

## What a run writes

| Table | Keyed by | Holds |
|---|---|---|
| `runs/RUN/plans` | the fence the run's loop took | how its episodes are played from then on: a [`Plan`](../../guide/reference.md#plan), the program (each group's start is its row) and the [`RunBinding`](../rollout/README.md#run-specifications) |
| `runs/RUN/groups` | group | its row, the start each of its episodes is given (`parameters`), and how many episodes it asks for (`episodes`) |

An episode is `GROUP/EPISODE`, numbered from 1 within its group. A group with a result (`runs/RUN/results`) asks for
nothing more.

## What runners write

| Table | Keyed by | Holds |
|---|---|---|
| `runs/RUN/claims` | `GROUP/EPISODE/ATTEMPT` | the runner that plays that attempt, the number of its fence, when |
| `runs/RUN/episodes` | `GROUP/EPISODE` | the episode's [`Record`](../../guide/reference.md#record), once it has ended |
| `runs/RUN/interrupted` | `GROUP/EPISODE/ATTEMPT` | an attempt its runner cut short by closing, and why |

- **First append wins.** A claim is an append to a key no one has written, so two runners never play one attempt:
  the one whose append is refused looks for other work.
- **A claim holds while its runner keeps its fence and beats.** A runner takes the fence of `runners/NAME` when it
  starts. Its claims hold until it is started again (under the same name, its fence moves on), it notes an attempt
  as interrupted, or, where runners beat ([heartbeats](#heartbeats)), its newest beat is older than 90 seconds (its
  machine died, say). An episode with no record and no claim that holds is open: the next claim is its next attempt.
  A runner does not wait on its own beat: its own claims hold for it while its fence is its own.
- **Every attempt that ends is an episode**, whatever its outcome: completed, failed (the program raised, or the run
  could not start), cancelled. The first record of an episode is its record.
- **An episode's trajectories and its run's events go to the blob store**; the record names both
  ([the record](#the-record)). A run started by `rollout train` says in its `starts` record where that store is
  (`rollout_train.stores`: its kind and settings, never a credential), so that any machine can read a finished
  episode back.

## A runner

[`EpisodeRunner(name, ledger, runner, recorder, blobs, places, ...)`](../../guide/reference.md#episoderunner) claims
open episodes and plays them on a [`Runner`](../rollout/README.md#runner).

- **`places`**: how many episodes it plays at once. When one ends it looks again at once; otherwise every `every`
  seconds.
- **Oldest group first.** Open episodes are claimed in the order their groups were decided, across every run it
  serves.
- **What it serves.** A run whose plan's recorded models are all on channels its recorder serves, whose local
  imports are all among `imports` (the tool sets it has), and whose local pools are all among `pools` (the sandbox
  pools it has). With `runs`, those runs only.
- **Room in the pools.** An episode whose program declares sandboxes is claimed only while their pools have room
  for them: each pool's `capacity()` is asked once a look, and what the runner claims is counted against it as it
  goes ([sandboxes](../rollout/sandboxes.md#in-training-a-lease-ends-with-its-claim)).
- **`guard`** is called before claiming, and raises to wait: a machine short of memory claims nothing until it has
  room.
- **An episode is played** as the run's program with the group's `parameters` as its row, labelled `run`, `group`
  and `episode`, with the claim's key (`RUN/GROUP/EPISODE/ATTEMPT`) as the run's lease: its sandboxes are leased
  under it, and their leases end with the claim. When it ends, the runner takes the run's segments from the recorder (which then forgets the run),
  assembles the [episode](episodes.md), stores it and appends its record.
- **Closing** cancels what it plays, in the runner too, and notes each attempt whose run had started in
  `interrupted`: the episode is open again, for any runner with room. An attempt cancelled before its run started
  is noted nowhere; its claim lapses once its runner's fence moves on or its beats stop.

`serve()` runs until cancelled; `async with playing(runner):` serves while a block runs.
[`episodes_of(ledger, blobs, run, group, count)`](../../guide/reference.md#episodes_of) waits until all `count`
episodes of a group have records and returns them, trajectories and all.

### Heartbeats

With `presence` (a `Presence`, `rollout_train.presence`), a runner beats when it
starts, before it claims anything, and every `beating` seconds (15) after; `beat()` beats at once besides (an open
profile's runner does when a channel serves a new checkpoint). A beat holds what `about()` says of its machine (called
in a thread: it may measure), with its `places`, how many episodes it is `playing`, and, when it has pools, how full
each is (`pools`: `size`, `leased`, `free`). Each runner's newest beat
is kept with the measurements of its recent ones (240: an hour), so its machine can be shown from anywhere.

Beats are kept beside the ledger, as ordinary state changed in place, not appended: `presence.json` beside a ledger
of files, the `presence` table in a database ledger's database (`DatabasePresence`); `presence_of(ledger)` finds
them. A runner whose newest beat is older than `STALE` (90 seconds) is taken to be gone, and its claims lapse.

An open profile's runner says, in each beat: its host, the run it serves, the run's directory, its machine
(`rollout_train.machine`: memory, each GPU's memory and how busy, the disk the directory is on), its engines'
processes and whether each is alive, and what each channel serves (adapter and version) with what passed through it
since the beat before (requests, tokens, tokens a second, requests at once). The [monitor](monitor.md) shows
machines, inference throughput and what is served from these beats.

## The record

| Kept | Where | What |
|---|---|---|
| A [`Record`](../../guide/reference.md#record) per episode | `runs/RUN/episodes`, under `GROUP/EPISODE` | everything about the episode but its trajectories' segments: labels, outcome, result, rewards, tokens sampled by slot. It names two blobs |
| The trajectories | a blob | each slot's segments, spans and logprobs |
| The run's events | a blob | its tool calls and their results, observations and rewards, as the runner recorded them |

- **Nothing is deleted.** Every episode stays in the ledger and can be read and trained on again (`loaded(record,
  blobs)`), for as long as the blob store keeps its blobs.
- **Blobs** go to the [`Blobs`](../../guide/reference.md#blobs) store the runner is given. They are JSON, compressed.
  Runners on several machines need a store they all reach, as they need a ledger they all reach.
- **A span names its sample.** `Span.effect_id` is the effect the run's events know the sample by, so a trajectory
  can be joined to what the action it sampled did. `events_of(record, blobs)` reads the events.

## Environment

What a run trains on and an eval measures is an [`Environment`](../../guide/reference.md#rolloutenvironmentenvironment):

| It says | As | Read by |
|---|---|---|
| what plays an episode | `program` | runners, through each run's plan |
| every situation, easiest first | `rows()`: each a `Row` (key, title, parameters, the rows it `counts_for`) | the curriculum |
| one start of a row | `start(row, rng)`: the parameters every episode of a group is given | the loop, through `train_start` |
| its eval data | `evals()`: named lists of `Start`s (a row's key and title, a seed, the parameters) | `rollout eval`, which freezes each as a suite ([evals](evals.md)) |
| what its results say | `description`: a `Description` (the range of its rewards; whether results say `solved` and `saturated`; what `duration` counts; how its observations are shown) | each run's start record, and so the monitor |
| which it is | `version`, changed whenever rows, starts, eval data or scoring change | each run's start record and each suite's record |
| what to train on next (optional) | `curriculum()`: a `Curriculum` of its own | the loop (`curriculum_of`) |

A row may name other rows its groups count for too (`Row.counts_for`, [the curriculum](training.md#the-curriculum)).
All of it lives in `rollout.environment` and `rollout.curriculum`, in the core library, so an environment needs the
harness and nothing above it. `binding_for(environment, channel, tools)` binds every model slot of the environment's
program to one channel ([three ways in](../../guide/perspectives.md#building-an-environment)).

### Train and eval

Training never draws an eval start, by construction: the loop draws each group's start with `train_start(environment,
row, rng, held_out(environment))`, which draws again while what it drew is one of the environment's eval starts (and
gives up on a row whose every start is one). The check compares starts, not seeds. Keeping train and eval seeds apart
would hold only for an environment whose starts differ whenever their seeds do, which the platform cannot see; comparing
what `start` returned holds for every environment, whatever it does with its random numbers.

A start is told apart by its parameters alone (as canonical JSON, `start_key`), so the guarantee is as fine as they
are: two starts that differ only in a number the episode never reads are the same situation. An environment whose
evals should hold out whole situations lists eval starts of rows it does not offer to train on, or keeps a part of
what its starts are drawn from (worlds, say) for its eval data. `drawn(environment, seeds=…, rows=…)` derives eval
data from rows and seeds when that is enough.

### Checking an environment

```sh
uv run rollout env check minecraft_team.environment:environment --tools minecraft=minecraft_team.worlds:tools
uv run rollout env check tests.rollout_train.rollouts.games:guessing --profile PROFILE --groups 4   # with a model
```

`rollout env check ENVIRONMENT` says, a line for each, whether its rows build (keys and titles unique, `counts_for`
naming rows it has); whether its description and version say something; whether one seed draws one start and its eval
data is the same each time; how many of the starts drawn for training were eval starts, and were drawn again; and how
one episode went on the local runner with a scripted model (`--reply` is what it says each turn; `--row` the row; the
tool sets its program imports from `--tools NAME=module:factory` or a URL, or the profile's), with a reward in the
described range and a result that says what the description says it does.

With `--profile P --groups N` it also plays N groups (of the algorithm's group size, or `--episodes`) of the rows the
environment's curriculum would choose first, on the profile's channel, served by its base model with nothing trained,
as a run of its own (start `kind: check`, in `~/.cache/rollout/checks/NAME` unless `--directory` says). A group whose
episodes all scored the same is flagged: it teaches a group-relative update nothing. When every group is so, the check
fails: a run would take no step at all. It exits 1 when any check fails.

## Watching

[`Hooks.on_note(event)`](../../guide/reference.md#hooks) receives what a runner and a run do, at the level they think
at. Every event has `kind` and `at`; a runner's also have `runner`, a run's `run`.

| `kind` | From | When | Also carries |
|---|---|---|---|
| `started` | a runner | an episode's run started | `run`, `group`, `episode`, `run_id` |
| `ended` | a runner | an episode was recorded | `run`, `group`, `episode`, `run_id`, `labels`, `outcome`, `detail`, `reward`, `info`, and `sampled`: tokens sampled by slot |
| `published` | the run | weights were published | `channel`, `adapter`, `version` |
| `result`, `step` | the run | a group's result, a step's outcome | [the record](training.md#the-record) |

The [monitor](monitor.md)'s feed is one such hook.
