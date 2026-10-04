# The policy graph

Code: `rollout_train.monitor.lineage` (the draft view) · See [policies, versions and the ledger](../libraries/rollout-train/policies.md),
[training](../libraries/rollout-train/training.md), [the monitor](../libraries/rollout-train/monitor.md#the-policies-view)

**A proposal.** It describes a view of every policy as a graph, and the records behind it: training, distillation,
trainers and their queues, the way from written weights to served ones, and evaluations. The monitor's policies view
(`#/policies`) draws the graph from what a ledger has today. Every table marked *proposed* below is a proposal: no
code writes it. The view reads them all, and `#/policies/sample` shows it with a fixture of them
(`rollout_train/monitor/sample-lineage.json`). The router that sends requests to inference workers has a design of
its own; this page uses only what the graph needs of it.

## What exists

- A **policy** is a line of versions (`Policies`, the table `policies/NAME/versions`). A `Version` names its
  `parent` by name. The parent may be another policy's version: a fork.
- A **run** trains one policy. Its `steps` table names, for each step, the groups it covers, the version it starts
  from and the version it makes. The version, appended to the policy's table, is the step's outcome.
- **Imitation** (`rollout imitate`) takes a supervised step on the run's own solved episodes, with the trainer's
  `likelihood` objective, and commits the next version. It writes no step record.
- **Serving**: the loop publishes each new version to its channel (`Channel.publish`: every engine loads the LoRA
  adapter, and the one before stays loaded for turns in flight). The feed notes each publish.
- An **evaluation** can be played as a run of the catalog's program on one start of a row
  (`catalog.start(row, random.Random(seed))`), every model slot bound to a version or to another model through
  `DirectModel`, with a `RunFeed` to watch it. Nothing records its score.

## Nodes and edges

| | What it is | Read from |
|---|---|---|
| Policy | a node: a lane of versions, folded until opened | `policies/NAME/versions` |
| Version | a point on its lane; it opens the step that made it | the version, and the step whose `makes` names it |
| Training | a stretch of a lane: the versions one run's steps made | `runs/RUN/steps` |
| Fork | an edge from a version to another policy's first version | `Version.parent` of another policy |
| Distillation | a merge node in its student's lane, before the versions it made, with an edge from each teacher and from the version the student starts from | `runs/RUN/plan` (proposed) |
| Evaluation | a score on a version, and a column of a suite's table | `evaluations/...` (proposed) |

The graph is a DAG of versions. Between policies it may loop: a distillation into an existing student goes from the
student's version *n* to its version *n*+1 through the merge node. The view therefore lays out versions, not
policies: every item stands to the right of what it comes from (the longest path to it), in its own lane.

A folded lane shows the versions something points at: its first and newest, the ends of each run's stretch, a
fork's parent, a teacher, a distillation's start, an evaluated version, and any version on its way to the engines.
The rest are a gap ("+9"). A policy with hundreds of versions stays a short lane until it is opened.

A version this ledger does not have, but that a fork or a distillation starts from (a policy in another run's
directory), stands in a lane of its own, "outside this ledger".

## Distillation

One or more **teachers** (versions, by name) are distilled into a **student**: a new policy, which starts from a
version (`from`), or an existing one, which goes on from its newest version. Each step makes one version of the
student, as a training step does, and commits it with `Policies.add` under the student's writer.

Whether it is on policy or off it is decided by its inputs: whose samples it trains on.

| | Off policy | On policy |
|---|---|---|
| Samples | the teachers' trajectories | the student's own |
| Where from | episodes in other runs' job logs (the teachers' runs), or episodes it plays with the teachers served | episodes it plays with the student served, as a training run does |
| Objective | `likelihood` on the teachers' sampled tokens (forward KL on samples), as imitation does with the run's own | per-token reverse KL toward the teachers: each sampled token's advantage is `log teacher(x) − log student_old(x)`, under the token ratio and importance weight the policy gradient already uses |
| Needs | the teachers' tokens only | the teachers' logprobs of the student's tokens |

`mode(student, sampled_by)` in `rollout_train.monitor.lineage` says which: `on-policy` when every sample is the
student's, `off-policy` when none is, `mixed` when both are.

Several teachers:

- Off policy, their samples are pooled; a segment can be weighted by its teacher or by its row.
- On policy, each segment needs one target. Either a teacher is chosen by row (`teacher_by_row`: the teacher that
  was trained on those rows), or the target is the teachers' mixture (`log mean exp` of their logprobs).

The teachers' logprobs come from the trainer, in the step: teachers of the same base are LoRA adapters it can load
beside the student's, without gradient. Teachers of another base are scored by an engine, through a request that
names the exact teacher version. The `Trainer` protocol would take the teachers' checkpoints with the batch.

### The records (proposed)

A distillation is a run. It uses the run's tables as a training run does (`groups` and `results` when it plays,
`steps`, `failures`), and says what it is in one more table.

| Table | Key | Record |
|---|---|---|
| `runs/RUN/plan` | `plan` | `kind` (`train`, `distill`, `imitate`); `student` (the policy it makes versions of); `from` (the version a new student starts from; none: the student goes on from its newest); `teachers` (versions, by name); `data`: `sampled_by` (policies or versions), `runs` (whose job logs), `episodes` (`solved`, `all`), `versions` (which of the teachers' versions sampled them), `rows`, `teacher_by_row`; `objective` (`likelihood`, `reverse_kl`, `policy_gradient`); `evaluate` (see [scheduling](#scheduling-from-a-run)); `decided` |
| `runs/RUN/steps` | step number | as today, and `teachers` (the versions this step was taught by), `objective`, `trainer` (whose queue it went to), and `sources` (segments per run they came from) |

The step's `batch` blob lists its segments as today (`[source, advantage]`), with each source qualified by its job
(`JOB:cursor/slot/index`), since a distillation reads several runs' logs.

The version itself is unchanged. Its `parent` is the student's previous version, or `from` for a new student's
first. Its teachers are in the step that made it. An optional `Version.teachers` field would let a reader of
`Policies` alone draw the merge edges.

What it reuses:

- `Policies.add`, `writer` and `thin`. A teacher must keep its weights while a distillation uses it, so `Retention`
  would also keep every version a plan or a running step names.
- The loop, for on-policy distillation: it plays groups as training does, with a distillation algorithm in place of
  `Grpo` (every segment kept, its advantage per token).
- `imitation.examples` and `imitate`, for off-policy distillation from logs: the same supervised step, over several
  runs' logs, with a plan and step records.

## Trainers and their queues

A trainer is **registered** as able to train some policies. A LoRA trainer holds a base model and trains any LoRA
policy of that base: steps of different policies take turns on it, each starting from its own parent's adapter. A
full-weight trainer is dedicated to one policy.

| Table (proposed) | Key | Record |
|---|---|---|
| `trainers/NAME/registered` | the fence it registered under | `weights` (`lora`, `full`); `base` (the model it holds); `policies` (those it may train); `colocated` (it shares the engines' accelerator); `where`; `at` |
| `policies/NAME/definition` | `definition` | `weights` (`lora`, `full`); `base`; `rank` or `layout` |
| `trainers/NAME/queue` | `RUN/STEP` | `run`, `step`, `policy`, `makes`, `at` (when it was queued) |
| `trainers/NAME/taken` | `RUN/STEP` | `began` |

A policy finds its trainer by its definition: a registered LoRA trainer of its base, or the full-weight trainer
registered for it. With several, the one with the shortest queue.

The way of a step:

1. A run's finished groups collect toward a step (`groups_per_step`; today the loop's in-memory queue).
2. The run decides the step (`runs/RUN/steps`, as today, naming its `trainer`) and appends it to the trainer's queue.
3. The trainer takes steps in order (or in turn by policy), appends `taken` when it begins, and commits the version.
   A failure is written to the run's `failures`, as today.

Today a run takes one step at a time, and fixes the step's parent when it decides it (the policy's head then). With a
shared trainer a run may decide its next step while the last is still queued. The step's parent is then the version
before the one it makes (`number − 1`), whatever the head is when it is decided; tokens sampled under an older version
are corrected by the objective's importance weight, as they are today. A run would keep at most a few steps queued.

The queue is traceable from these records alone: a step waits from `queue.at` to `taken.began`, and is taken until its
version's `made` or its failure's `at`. The view draws each trainer's depth over time (waiting, and being taken), and
for each run the groups waiting toward a step. Without these tables, a run's own steps stand for its trainer's queue
(the view says so: "the run's own").

## From written to served

`Policies.add` writes a version's files to the blob store and then appends the version: the append is the commit.
What follows is triggered by the commit, not by each file write, since a file may belong to a version that never
commits. With an object store's notifications, the trigger is the version's manifest, written as an object last.

| State | The records say |
|---|---|
| `written` | the version is appended |
| `resharding` | a resharding worker claimed it (`policies/NAME/resharding`) |
| `resharded` | the engines' layout is written (`policies/NAME/resharded`: a `Manifest` whose `layout` is the engines') |
| `rolling out` | it is a run's latest, and some worker still serves an older version of its policy in its place |
| `serving` | every worker that serves its policy serves it; or, not a run's latest, some worker serves it by exact version |
| `superseded` | it was served, and no worker serves it now |

A LoRA adapter loads as the trainer wrote it: it needs no resharding. A full-weight checkpoint, written by the trainer
in its own division (FSDP shards, say), is rewritten in the engines' (tensor-parallel shards, an engine-native
format) by a background resharding worker.

| Table (proposed) | Key | Record |
|---|---|---|
| `policies/NAME/resharding` | version number | `at`, `worker` |
| `policies/NAME/resharded` | version number | `at`, `layout`, `weights` (the manifest the engines load) |
| `runs/RUN/published` | version name | `at`: a request for the run's latest goes to this version from then on |
| `workers/NAME/registered` | the fence it registered under | `machine`, `accelerators`, `holds` (`base`, or a full-weight `policy`), `adapters` (slots), `share` (`training`, `evaluations`) |
| `workers/NAME/loaded` | `VERSION/N` (its *N*th load of that version) | `at`, `seconds` |
| `workers/NAME/unloaded` | `VERSION/N` | `at` |

A request names an exact version (`miner@27`: evaluations, distillation teachers) or a run's latest (resolved per
turn, as `Channel.publish` behaves today). The router keeps, from `loaded` and `unloaded`, which worker serves which
version. It sends a LoRA request to a worker that has the adapter loaded, or has a free slot (loading it while the
request waits), and a full-weight request only to a worker that holds that policy. The exact version served is
stamped on every sampled token (`Span.version`), so a roll-out in progress mixes versions without confusing the
trainer. Requests waiting, by the version they name, are a measurement: the router notes them in the feed
(`routing`: `at`, `waiting` by version), as the engines note their throughput.

A roll-out takes one worker at a time: it drains, loads the new version, and takes requests again. Workers of the
evaluation share load exact versions as suites ask for them.

## Evaluations

An evaluation suite is a fixed set of starts: rows of a catalog, each with a seed, and the parameters
`catalog.start(row, random.Random(seed))` gave. The parameters are kept, so a suite plays the same starts even if the
catalog's `start` changes. A **subject** plays it: a version, or another model (a frontier model through `DirectModel`, say). Each episode is scored as a training group's are: reward and solved.

| Table (proposed) | Key | Record |
|---|---|---|
| `evaluations/SUITE/starts` | start number, from 1 | `task`, `title`, `seed`, `parameters`. Appended once: a changed suite is a new suite |
| `evaluations/SUITE/SUBJECT/subject` | `subject` | `kind` (`version`, `model`); `version`, or `model`, `provider` and `sampling`; `episodes` (per start); `asked_by`; `decided` |
| `evaluations/SUITE/SUBJECT/results` | `START-EPISODE` | `run_id`, `reward`, `solved`, `duration`, `outcome`, `time` |

A subject's name is its version's (`curriculum-9@30`) or a model's (`codex.gpt-6-astra.medium`). Its evaluation
holds the fence of `evaluations/SUITE/SUBJECT`. It asks a job for each start under a key of its own
(`eval-SUITE-SUBJECT-START`), so an evaluation that dies is taken up again as a run is, and appends each result when
its episode ends.

In the graph, a version's score is drawn above it (solved of played, "…" while starts remain). Each suite has a table:
a row for each start, a column for each subject (versions in lane order, then other models).

### Scheduling from a run

A run's plan may ask for evaluations: `evaluate: {suite, every}`. When a step makes a version whose number is a
multiple of `every`, the run appends `runs/RUN/evaluations` (proposed; key: the version, record: `suite`, `decided`)
and starts the evaluation, its requests naming the exact version (on the evaluation share of the workers, or within
the run's `episodes_at_once`). The version must keep its weights until its evaluation ends: `Retention` would keep
every version with an evaluation not yet done.

## What the monitor reads

`System.lineage(sample)` reads every table of the ledger and the feed's job lines, and `/api/policies` serves it.

| Read from | Drawn as |
|---|---|
| `policies/*/versions`, `released` | the lanes, their versions (kept or released) |
| `runs/*/steps`, `failures`, `results` | each version's step; the stretches of training; groups waiting toward a step |
| `runs/*/plan` | distillations: merge nodes, teacher and start edges, their mode |
| `trainers/*` | trainers, their queues and depth over time; else a run's own, from its steps |
| `policies/*/definition`, `resharding`, `resharded` | the reshard stage of a version's way |
| `runs/*/published`, `workers/*` | the roll-out, the workers serving each version; else the feed's `published` notes, as the run's engines |
| feed `routing` notes | requests waiting by version |
| `evaluations/*` | scores on versions, a table for each suite |

## Open questions

- **One ledger or many.** Each run's directory has its own ledger today, so a fork from an earlier run's policy
  points outside it. A graph of every policy needs one ledger shared by the runs (as the durable runner's database
  could be), or a reader over several directories.
- **On-policy teachers.** A teacher chosen by row, or the teachers' mixture? And may a teacher of another base be
  used on policy (it needs an engine to score the student's tokens, and the two tokenizers to agree)?
- **Mixed samples.** Train on both the student's and the teachers' samples in one step (each with its own
  objective), or keep a distillation one mode?
- **Teachers in the version.** Add `Version.teachers`, or keep teachers in the step record only?
- **Queue order.** First in, first out on a shared trainer, or in turn by policy; and how many steps may one run
  keep queued?
- **Batching LoRA steps.** May a LoRA trainer take steps of several policies in one pass (several adapters, one base
  forward), and record them as one taken?
- **The resharding trigger.** The ledger's append, or an object store's notification on the manifest written last?
- **Evaluation episodes.** Within a run's `episodes_at_once`, or on workers of their own; one episode per start or
  several; and are evaluated versions' weights kept for ever?
