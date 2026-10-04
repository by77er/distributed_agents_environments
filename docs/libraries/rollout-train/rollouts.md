# Rollouts

Code: `rollout_train.rollouts` · See [episodes](episodes.md), [training](training.md),
[API reference](../../guide/reference.md#rollout_trainrollouts)

A run asks for episodes in the [ledger](policies.md#the-ledger), and runners play them. The run decides what to play,
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
- **A claim holds while its runner keeps its fence.** A runner takes the fence of `runners/NAME` when it starts. Its
  claims hold until it is started again (under the same name, its fence moves on) or it notes an attempt as
  interrupted. An episode with no record and no claim that holds is open: the next claim is its next attempt.
- **Every attempt that ends is an episode**, whatever its outcome: completed, failed (the program raised, or the run
  could not start), cancelled. The first record of an episode is its record.
- **An episode's trajectories and its run's events go to the blob store**; the record names both
  ([the record](#the-record)).

## A runner

[`EpisodeRunner(name, ledger, runner, recorder, blobs, places, ...)`](../../guide/reference.md#episoderunner) claims
open episodes and plays them on a [`Runner`](../rollout/README.md#runner).

- **`places`**: how many episodes it plays at once. When one ends it looks again at once; otherwise every `every`
  seconds.
- **Oldest group first.** Open episodes are claimed in the order their groups were decided, across every run it
  serves.
- **What it serves.** A run whose plan's recorded models are all on channels its recorder serves, and whose local
  imports are all among `imports` (the tool sets it has). With `runs`, those runs only.
- **`guard`** is called before claiming, and raises to wait: a machine short of memory claims nothing until it has
  room.
- **An episode is played** as the run's program with the group's `parameters` as its row, labelled `run`, `group`
  and `episode`. When it ends, the runner takes the run's segments from the recorder, assembles the
  [episode](episodes.md), stores it and appends its record. The recorder then forgets the run.
- **Closing** cancels what it plays, in the runner too, and notes each attempt in `interrupted`: the episode is open
  again, for any runner with room.

`serve()` runs until cancelled; `async with playing(runner):` serves while a block runs.
[`episodes_of(ledger, blobs, run, group, count)`](../../guide/reference.md#episodes_of) waits until all `count`
episodes of a group have records and returns them, trajectories and all.

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

## Catalog

What an environment offers to be trained on is a [`Catalog`](../../guide/reference.md#catalog): the program, its
rows (easiest first), and how one start of a row is drawn. A row may name other rows its groups count for too
(`Row.counts_for`, [the curriculum](training.md#the-curriculum)). `Catalog`, `Row` and `binding_for` live in
`rollout.catalog`, in the core library, so an environment needs the harness and nothing above it.
`binding_for(catalog, channel, tools)` binds every model slot of the catalog's program to one channel
([three ways in](../../guide/perspectives.md#building-an-environment)).

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
