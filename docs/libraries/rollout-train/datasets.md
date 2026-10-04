# Datasets and supervised steps

Code: `rollout_train.datasets`, `rollout_train.imitation` · See [training](training.md#imitation),
[checkpoints](checkpoints.md), [`Dataset`](../../guide/reference.md#dataset),
[`make_dataset`](../../guide/reference.md#make_dataset), [the first datasets](../../research/sft-datasets.md)

A **dataset** is a set of examples to imitate, chosen from runs' episodes by a rule: rejection sampling over what the
policy played. It is made once and never changed. A **supervised step** on one (`rollout imitate --dataset`) raises the
likelihood of what its examples sampled, and makes a checkpoint that says which dataset it learned from and which
checkpoints sampled that dataset.

```bash
rollout dataset make best-of-group --run curriculum-9 --turns all --turns minecraft_team.datasets:worked \
    --name diamonds-worked --ledger sqlite:///~/.cache/rollout/ledger.db
rollout dataset list --ledger sqlite:///~/.cache/rollout/ledger.db
rollout imitate profile.toml --dataset diamonds-worked --start curriculum-9 --name diamonds-sft
```

## What a dataset is

- **A record**, in the ledger's `datasets` table, keyed by a random id of sixteen letters, like a checkpoint's. It is
  appended once, under the fence `datasets/ID`. It says how its examples were chosen, what came of that, the
  checkpoints that sampled them, where its manifest is, and who made it and when.
- **A manifest**, a blob (JSON lines, compressed with xz): one line per example. The examples themselves are made from
  the episodes' blobs when a step trains on them, so a dataset holds no copy of its trajectories: curriculum-9's first
  dataset is 1,966 examples in a 17 KB manifest.
- **A name**, if it is given one, in the [registry](checkpoints.md#the-registry) beside run names and bookmarks
  (`dataset_names` in a database, `datasets` in `registry.json`). A name says one dataset for good. Wherever a dataset
  is asked for, it is named by its name, its id, or the start of its id (at least four letters that no other id
  begins with): `resolved_dataset`.

| Record field | Holds |
|---|---|
| `rule`, `per_task` | the episode rule, by name, and (for `capped-per-task`) the most episodes of a task |
| `runs` | the runs its episodes are from, by id |
| `turns` | the turn filters, by name |
| `cut` | the kinds of guidance cut from its examples' prompts (`way` by default) |
| `counts` | `episodes` the rule picked and the `groups` they are of; `turns_seen`, every turn of those episodes; of its examples, `tasks`, `turns`, `sampled_tokens` and `context_tokens` |
| `left_out` | the turns of its episodes that are no examples, by why (`action failed`: 2,969) |
| `checkpoints` | the checkpoints that sampled its examples, by id, by depth |
| `supervision` | `importance` where every example's turns were sampled with their exact tokens and behaviour logprobs; `supervised` where some were not (the trainer computes what it needs of their logprobs, and nothing is importance-corrected). A step on its examples records the same on the checkpoint it makes, and in its start |
| `manifest`, `blobs` | the manifest's blob, and the store it is in, as any process opens it ([`rollout_train.stores`](rollouts.md#what-runners-write)) |
| `made`, `by` | when, and who (`user@host`) |

| Manifest field | Holds |
|---|---|
| `source` | `RUN/GROUP/EPISODE/SLOT/INDEX`: the episode, the model slot and the segment's place in its trajectory |
| `task`, `reward` | the row's key, and the episode's reward |
| `depth`, `checkpoint` | the depth it was sampled at (its spans' newest), and the checkpoint the run served at that depth (none for the base model) |
| `tokens`, `sampled` | its tokens, and the tokens the policy sampled |
| `guidance` | the kinds of guidance to cut from its prompt (those of the dataset's `cut` that its episode carried) |
| anything else | what the turn filters saw of it (`action`, for `minecraft_team.datasets:worked`) |

The checkpoint served at a depth is found along the line of first parents back from each checkpoint the run made and
each it started from (`served_at`). The manifest is kept beside the first run's episodes (the blob store its newest
start names, or the directory of files that holds its episodes), or in the directory `--blobs` names.

## Choosing examples

Episodes first: an **episode rule** picks among the episodes the runs completed (not failed, not excluded).

| Rule | Picks |
|---|---|
| `solved-all` | every solved episode |
| `best-of-group` | of each group's solved episodes, the one with the highest reward, then the shortest (`duration`), then the first |
| `capped-per-task` | solved episodes by that same order, at most `--per-task` (3) of each task |

Then turns: **turn filters** pick among the turns (segments) of those episodes. A turn is an example if every filter
keeps it. `all` keeps every turn that sampled something. An environment supplies others, named `module:name`. A turn
filter is a function

```py
def worked(turns: Sequence[Turn], events: Sequence[RunEvent], info: Mapping[str, JsonValue]) -> Sequence[Mapping[str, JsonValue]]
```

given an episode's turns (each `Turn`: its slot, its place, the samples its spans came from by `effect_id`, its
tokens, its sampled tokens and its depth), its run's events and its result. It says what it saw of each turn, in
order; a turn it gives a `left_out` (why) is no example, and the rest of what it saw goes into the manifest.

The Minecraft team supplies **`minecraft_team.datasets:worked`**: the turns whose action came back ok in the agent's
next observation (`last_action`), each with the action's name. A sample names neither its agent nor its slot, so the
filter follows the run's events: a turn's sample is requested for one agent, and that agent's next `observe`
completes with the outcome. Slots `agent-1` to `agent-4` play under the names in the result's `team`, in slot order.
A turn whose action failed is left out (`action failed`), and so is one whose outcome was never seen (`no action
seen`: a summary of old turns, a sample that called no action, the last turn).

Guidance is cut when the examples are made: each example's prompt loses the guidance its manifest line names, as
[imitation](training.md#imitation) cuts it (`without`). A turn where the cut cannot be made exactly is left out then.

## A step on a dataset

```bash
rollout imitate PROFILE --dataset REF [--start REF] [--name NAME] [--directory RUN] [--limit N] [--seed N] \
    [--learning-rate R] [--warmup N] [--passes N] [--resume-optimizer]
```

The profile's trainer takes the step with `objective = "likelihood"`: the LoRA trainer, or the trainer of every
weight (`rollout_lora:FullTrainer`), each through the same [weighted segments](training.md#the-trainer), each
example weighted 1. The step is a run of its own (the profile's directory, or `--directory`), registered under
`--name`; its start record says `kind: imitation`, the dataset, and the checkpoint it trains from. It trains from
the newest checkpoint the run made, else `--start` (any [reference](checkpoints.md#references)), else the profile's
`[trainer] start`, else the base model.

- **Its parents** are the checkpoint it trained from, then the checkpoints that sampled the examples it trained on,
  by depth. The checkpoint graph shows those as learned-from edges. A step
  from the base model has no parents; its `dataset` still says where its examples came from.
- **Its record names the dataset** (`dataset`, the id), and its metrics add `imitated_episodes`,
  `imitated_segments` and `optimizer_resumed`, beside the trainer's `learning_rate`, `warmup_updates` and `passes`.
  Its batch records each example's source, as a training step's does. The monitor serves the `dataset` of each
  checkpoint with the rest of its record.
- **From where it starts:** a step goes on from its parent's weights when the trainer makes what the parent is (an
  adapter from an adapter, every weight from every weight), with its optimizer started afresh: the parent's trainer
  state holds the moments of another objective (a policy gradient's, say). `--resume-optimizer` goes on from that
  state instead. The metric `optimizer_resumed` (1 or 0) says which. A LoRA step from a full checkpoint begins a
  new adapter over those weights; its trainer and its base are that checkpoint. A step of every weight from an
  adapter is refused: [merge](checkpoints.md#full-weights-and-merges) the adapter first.
- **`--limit N`** trains on N examples drawn at random (by `--seed`); the batch says which.

### Its schedule

A supervised step has a rate, a warmup and passes of its own, whatever the profile's `[trainer]` says for training
runs:

| | Default | Flag |
|---|---|---|
| rate | 1e-6 for every weight, 1e-4 for an adapter (`imitation.RATES`) | `--learning-rate` |
| warmup | a fresh optimizer's rate rises linearly over its first 4 updates (`imitation.WARMUP`) | `--warmup` |
| passes | enough passes for at least 8 optimizer updates of `tokens_per_step` sampled tokens (`passes_for`, `imitation.UPDATES`): one for a large dataset, more for a small one | `--passes` |

Passes rather than a smaller `tokens_per_step`: each update stays the size the trainer's settings make it, over
examples shuffled anew each pass, and a dataset large enough for its updates takes one pass as before. They are the
trainer's settings `passes` and `warmup_updates` ([LoRA trainer](../../implementations/rollout-lora.md)); a
training run's steps use them only if its profile sets them, and a step that goes on from an optimizer's state is
not warmed up.

The defaults come from steps on `slqm` (12 short answers the base model gave in the guessing game, 107 sampled
tokens) from the full checkpoint `qvqy` of Qwen3-0.6B, optimizer fresh. Mean logprob of a sampled token before and
after, on the 12 and on 24 held-out answers of the same game (`qwtv`):

| Step | Updates | Trained on | Held out |
|---|---|---|---|
| every weight, 1e-5, no warmup | 1 | −0.352 → −0.944 | −0.457 → −1.615 |
| every weight, 1e-6, warmup 4 | 8 | −0.352 → −0.291 | −0.457 → −0.374 |
| every weight, 1e-5, warmup 4 | 8 | −0.352 → −0.149 | −0.457 → −0.708 |
| adapter over `qvqy`, 3e-4, warmup 4 | 8 | −0.352 → −0.245 | −0.457 → −0.593 |

A fresh Adam's first update moves every weight by about the full rate, so one update at 1e-5 wrecks the model; eight
warmed-up updates at 1e-6 raise both. At 1e-5 (and an adapter at 3e-4) the step fits the twelve and loses the
held-out answers. The adapter's default, 1e-4, is a third of the rate that overfitted here; it has not been measured
on its own.

In Python:

```py
made = await make_dataset(ledger, "best-of-group", [run], into=blobs, at=where, turns=["all"], cut=["way"])
taught = await examples(ledger, made, renderer)                  # the segments, cut, each weighted 1
checkpoint = await imitate(checkpoints, trainer, taught, fence=fence, run=step_run, start=start, base=model,
                           directory=directory / "checkpoints")  # parents: start, then the samplers by depth
```
