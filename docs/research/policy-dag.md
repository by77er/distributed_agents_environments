# The checkpoint graph

**Status: proposed.** The monitor's view of the checkpoints that exist is built. A design note: see [Design
notes](README.md) for the others.

Code: `rollout_train.monitor.lineage` (the view of what exists) · See [checkpoints, runs and the
ledger](../libraries/rollout-train/checkpoints.md),
[training](../libraries/rollout-train/training.md), [the
monitor](../libraries/rollout-train/monitor.md#the-checkpoints-view)

**A proposal.** It describes the graph of every checkpoint and the records behind it: training, distillation,
trainers and their queues, and the way from written weights to served ones. The monitor's Checkpoints view
(`#/checkpoints`) draws the graph from what a ledger has today ([the checkpoints
view](../libraries/rollout-train/monitor.md#the-checkpoints-view)). Every table marked *proposed* below is a proposal:
no code writes or reads it.

## What exists

- A **checkpoint** (the ledger's `checkpoints` table) names its parents: first the checkpoint it was trained from,
  then any others it learned from. It names its base model, its depth, what its weights are (`lora` or `full`), and
  the run and step that made it. A checkpoint trained from another run's checkpoint is a fork. Bookmarks name
  checkpoints in the registry ([checkpoints](../libraries/rollout-train/checkpoints.md#checkpoints)).
- A **run** trains one line. Its `steps` table names, for each step, the groups it covers, the checkpoint it starts
  from and the checkpoint it makes. The checkpoint, appended, is the step's outcome.
- **Imitation** (`rollout imitate`) takes a supervised step on a [dataset](../libraries/rollout-train/datasets.md) of
  episodes. The checkpoint it makes names the dataset, and its parents after the first are the checkpoints that
  sampled the dataset's examples. A **merge** (`rollout merge`) makes a full checkpoint from an adapter.
- **Serving**: the loop writes down what each channel should serve (`runs/RUN/serving`). Engines in its process load
  it; engine hosts elsewhere follow it, keeping the run's `max_lag + 1` newest adapters loaded. A checkpoint whose
  files the engines cannot load as written is bridged first (`checkpoints/resharding`, `checkpoints/resharded`:
  [bridges](../libraries/rollout-train/checkpoints.md#bridges)).
- **Evals**: suites and their versions, played by a checkpoint or a base model, by hand or on a run's schedule, with
  each episode's result recorded ([evals](../libraries/rollout-train/evals.md)). The Evals page shows them, and a
  checkpoint's page its scores along its line.

## Nodes and edges

| | What it is | Read from |
|---|---|---|
| Base model | a root, with a lane of its own | each checkpoint's `base` |
| Checkpoint | a point on its run's lane; it opens its page | `checkpoints` |
| Training | a stretch of a lane: the checkpoints one run's steps made | `runs/RUN/steps` |
| Fork | an edge from a checkpoint to the first checkpoint of a run trained from it | a checkpoint's first parent, made by another run |
| Learned from | a dashed edge from each of a checkpoint's other parents: a merge's, a dataset's samplers | a checkpoint's parents after the first |
| Distillation | a merge node in its student's lane, before the checkpoints it made, with an edge from each teacher and from the checkpoint the student starts from | `runs/RUN/plan` (proposed) |

The graph is a DAG of checkpoints. The view lays out checkpoints, not runs: every item stands to the right of what it
comes from, in its run's lane, and a lane is folded to the checkpoints something points at.

## Distillation

One or more **teachers** (checkpoints, by any reference) are distilled into a **student**: a new run, which starts
from a checkpoint (`from`), or an existing run, which goes on from its newest checkpoint. Each step makes one
checkpoint of the student, as a training step does, whose parents after the first are the step's teachers.

Whether it is on policy or off it is decided by its inputs: whose samples it trains on.

| | Off policy | On policy |
|---|---|---|
| Samples | the teachers' trajectories | the student's own |
| Where from | episodes in other runs (the teachers'), or episodes it plays with the teachers served | episodes it plays with the student served, as a training run does |
| Objective | `likelihood` on the teachers' sampled tokens (forward KL on samples), as imitation does | per-token reverse KL toward the teachers: each sampled token's advantage is `log teacher(x) − log student_old(x)`, under the token ratio and importance weight the policy gradient already uses |
| Needs | the teachers' tokens only | the teachers' logprobs of the student's tokens |

Several teachers:

- Off policy, their samples are pooled; a segment can be weighted by its teacher or by its row.
- On policy, each segment needs one target. Either a teacher is chosen by row (`teacher_by_row`: the teacher that
  was trained on those rows), or the target is the teachers' mixture (`log mean exp` of their logprobs).

The teachers' logprobs come from the trainer, in the step: teachers of the same base are LoRA adapters it can load
beside the student's, without gradient. Teachers of another base are scored by an engine, through a request that
names the exact teacher checkpoint. The `Trainer` protocol would take the teachers' checkpoints with the batch.

### The records (proposed)

A distillation is a run. It uses the run's tables as a training run does (`groups` and `results` when it plays,
`steps`, `failures`), and says what it is in one more table.

| Table | Key | Record |
|---|---|---|
| `runs/RUN/plan` | `plan` | `kind` (`train`, `distill`, `imitate`); `from` (the checkpoint a new student starts from; none: the run goes on from its newest); `teachers` (checkpoints, by id); `data`: `sampled_by` (checkpoints), `runs` (whose episodes), `episodes` (`solved`, `all`), `rows`, `teacher_by_row`; `objective` (`likelihood`, `reverse_kl`, `policy_gradient`); `decided` |
| `runs/RUN/steps` | step number | as today, and `teachers` (the checkpoints this step was taught by), `objective`, `trainer` (whose queue it went to), and `sources` (segments per run they came from) |

The checkpoint's `batch` blob lists its segments as today, each by its source (`RUN/GROUP/EPISODE/SLOT/INDEX`), which
already names the run a segment came from.

What it reuses:

- Appending checkpoints under the run's fence. A teacher must keep its weights while a distillation uses it, so
  `Retention` would also keep every checkpoint a plan or a running step names.
- The loop, for on-policy distillation: it plays groups as training does, with a distillation algorithm in place of
  `Grpo` (every segment kept, its advantage per token).
- Datasets and `imitate`, for off-policy distillation: the same supervised step, over several runs' episodes, with a
  plan and step records.

## Trainers and their queues

A trainer is **registered** as able to train some lines. A LoRA trainer holds a base model and trains any LoRA
checkpoint of that base: steps of different runs take turns on it, each starting from its own parent's adapter. A
full-weight trainer is dedicated to one run.

| Table (proposed) | Key | Record |
|---|---|---|
| `trainers/NAME/registered` | the fence it registered under | `weights` (`lora`, `full`); `base` (the model it holds); `runs` (those it may train); `colocated` (it shares the engines' accelerator); `where`; `at` |
| `trainers/NAME/queue` | `RUN/STEP` | `run`, `step`, `makes`, `at` (when it was queued) |
| `trainers/NAME/taken` | `RUN/STEP` | `began` |

A run finds its trainer by its checkpoints' kind and base: a registered LoRA trainer of its base, or the full-weight
trainer registered for it. With several, the one with the shortest queue.

The way of a step:

1. A run's finished groups collect toward a step (`groups_per_step`; today the loop's in-memory queue).
2. The run decides the step (`runs/RUN/steps`, as today, naming its `trainer`) and appends it to the trainer's queue.
3. The trainer takes steps in order (or in turn by run), appends `taken` when it begins, and appends the checkpoint.
   A failure is written to the run's `failures`, as today.

Today a run takes one step at a time, and fixes the step's parent when it decides it (the run's newest checkpoint
then). With a shared trainer a run may decide its next step while the last is still queued. The step's parent is then
the checkpoint the step before makes, whatever is newest when it is decided; tokens sampled under an older checkpoint
are corrected by the objective's importance weight, as they are today. A run would keep at most a few steps queued.

The queue is traceable from these records alone: a step waits from `queue.at` to `taken.began`, and is taken until its
checkpoint's `made` or its failure's `at`. The view draws each trainer's depth over time (waiting, and being taken),
and for each run the groups waiting toward a step. Without these tables, a run's own steps stand for its trainer's
queue (the view says so: "the run's own").

## From written to served

Appending a checkpoint is its commit: its files are in the blob store before it. The view shows where each checkpoint
is on its way:

| State | The records say |
|---|---|
| `written` | the checkpoint is appended |
| `resharding` | a bridge began on it (`checkpoints/resharding`) |
| `resharded` | the files the engines load are written (`checkpoints/resharded`) |
| `serving` | the run's engines serve it, as its runners' beats say |
| `superseded` | it was served, and nothing serves it now |

An adapter loads as the trainer wrote it, and a checkpoint in a format the engines do not load (Tinker's archive, say)
is bridged into one they do. The view reads what engines serve from the runners' beats.

| Table (proposed) | Key | Record |
|---|---|---|
| `workers/NAME/loaded` | `CHECKPOINT/N` (its *N*th load of that checkpoint) | `at`, `seconds` |
| `workers/NAME/unloaded` | `CHECKPOINT/N` | `at` |

With these, each engine's span of serving a checkpoint is drawn from records rather than from beats, and requests
waiting, by the checkpoint they name, are a measurement beside them. A request names an exact checkpoint, and the
exact checkpoint served is stamped on every sampled token (its depth on each span), so serving several checkpoints at
once never confuses the trainer.

## What the monitor reads

`System.lineage()` reads the ledger's tables and the runners' beats, and `/api/checkpoints` serves it.

| Read from | Drawn as |
|---|---|
| `checkpoints`, `checkpoints/released` | the points, kept or released, and the edges between them |
| `runs/*/steps`, `failures`, `results`, `starts` | each checkpoint's step; the stretches of training; each run's own trainer and its queue; groups waiting toward a step |
| `checkpoints/resharding`, `checkpoints/resharded` | the bridge stage of a checkpoint's way |
| the runners' beats (what each channel serves, as `published` notes) | the run's engines, and the checkpoints they serve |
| the registry | the runs' names and the bookmarks |
| `runs/*/plan`, `trainers/*`, `workers/*` (proposed) | distillations, shared trainers, each engine's loads |

## Open questions

- **On-policy teachers.** A teacher chosen by row, or the teachers' mixture? And may a teacher of another base be
  used on policy (it needs an engine to score the student's tokens, and the two tokenizers to agree)?
- **Mixed samples.** Train on both the student's and the teachers' samples in one step (each with its own
  objective), or keep a distillation one mode?
- **Queue order.** First in, first out on a shared trainer, or in turn by run; and how many steps may one run keep
  queued?
- **Batching LoRA steps.** May a LoRA trainer take steps of several runs in one pass (several adapters, one base
  forward), and record them as one taken?
