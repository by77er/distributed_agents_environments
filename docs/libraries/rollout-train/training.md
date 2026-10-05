# Training

Code: `rollout_train` · See [rollouts](rollouts.md), [episodes](episodes.md), [datasets](datasets.md),
[API reference](../../guide/reference.md#rollout_train)

The training loop, what it asks of an algorithm and of a trainer, and the curriculum. The loop is written against
the [ledger](checkpoints.md#the-ledger), an [`Environment`](rollouts.md#environment), `Trainer`, `Algorithm` and
[`Checkpoints`](checkpoints.md) only: it asks for each group's episodes in the ledger, and [runners](rollouts.md) play them,
wherever they are. The same loop runs with everything in one process and with the runners, the engines and the
trainer on machines of their own.

```python
await train(environment, trainer, checkpoints, start=None, base="Qwen/Qwen3.5-9B", channel="policy",
            directory=cache, publish=platform.publish, run=run.id, groups=100)
```

`rollout train PROFILE ENVIRONMENT [--groups N] [--groups-per-step N]` runs this loop over what a profile describes:
the profile opens into a trainer, the checkpoints, a way to publish checkpoints and a runner that plays the run's
episodes, says the checkpoint a new run starts from (`[trainer] start`, by default the base model) and the channel that
serves what it trains, and sets `episodes_at_once` ([deploying](../../guide/deploying.md)). The run is the one in its
directory: its id is in the directory's `run.json`, and its name is chosen with `--name` and changed with
`rollout rename` ([runs](checkpoints.md#runs)).

A run's [settings](../../guide/cluster.md#run-settings) are given in layers over what the profile gives, each over
the last, without editing its file: `--preset NAME[@N]` (a [preset](../../guide/cluster.md#presets) kept beside the
profile's ledger), `--settings FILE` (TOML or JSON, dotted keys or tables), `--set KEY=VALUE` (repeatable), then
`--model`, `--renderer` (of `--channel`, by default the trained one), `--groups`, `--groups-per-step` and `--seed`.
A value is read as JSON, then TOML, then as the text it is: `--set trainer.learning_rate=3e-5`,
`--set episodes_at_once=4`, `--set start=curriculum-9:20`, `--set bookmark=diamonds`, `--set max_lag=2`,
`--set objective.preset=dapo`, `--set objective.kl.coefficient=0.01`. The run
settings the profile keeps are applied to it (`start` and `bookmark` as `[trainer] start` and `bookmark`, `max_lag` as
the trained channel's); one it has no place for (`trainer.provider`, a channel's `provider`, `limits.spend`) is
refused, as it needs the cluster config. The objective's (`objective.preset` and its components,
[objectives](#objectives)) reach the trainer as its `objective`. A key that is no run setting is the profile's own
(`memory.runs_gib`), and a
key the profile cannot have is an error, as in the file: when the profile is loaded, for its top level and its
tables; for a `trainer.` key, when the trainer is made with its settings
([`Profile.load(path, settings=...)`](../../guide/reference.md#profile)). The run's start records the profile's
settings (`settings`) and, beside them, its run settings as they ran, with the objective they resolve to and the preset
they came from (`run_settings`: `rollout_train.run_settings.recorded`). A run started again trains with the objective
its newest training start recorded, so what a preset means later does not change it.

## The loop

[`train`](../../guide/reference.md#train) trains a line of [checkpoints](checkpoints.md) on an environment, from `start`
(a checkpoint's id, of this run or another: a fork; the base model, named `base`, if None), and serves each checkpoint
it makes on one channel, which `publish` serves checkpoints on. Unless a `binding` says otherwise, every model slot of
the environment's program is served from that channel. When it starts it writes the run's
[plan](rollouts.md#what-a-run-writes) (the environment's program and that binding) and its start record, which says,
besides where and by what it was started, the environment (as `module:name`), its `version` and its `description`
(what its results say, for the monitor).

- **Groups.** A group is `algorithm.group_size` episodes of one start of one row. The curriculum picks the row and
  the environment draws the start, never one of its eval starts (`train_start`,
  [train and eval](rollouts.md#train-and-eval)); the run's `groups` table keeps both, with how many episodes the group asks for and
  when it was decided. Runners play its episodes from there, and `episodes_of` waits for them.
- **Play and training go their own ways.** Enough groups are kept asked for that `episodes_at_once` episodes (6 by
  default) have work waiting, whatever groups they are of. Runners claim the oldest group's episodes first, as many
  at once as each has places, so a group may begin before the one before it is done, and end first.
- **When a group's last episode ends**, its result is written at once: the curriculum records it, and the algorithm
  says what in it to train on. A group with nothing to train on is done with, with the algorithm's reason
  (`skipped`); the others join a queue.
- **A step is taken over the queue** once at least `groups_per_step` groups are in it (4 by default, so that no
  step leans toward one task), over every group queued by then, while play goes on; at the end of the run, over
  whatever is left. The trainer's segment budget is spread over the groups. The step starts from the newest
  checkpoint the run made (its first, from `start`); the checkpoint it makes is appended under the run's fence and served
  on the channel, and `made` is told of it (a profile's carried bookmark moves there). Tokens sampled under an
  older checkpoint are corrected for by the trainer's objective. One step is taken at a time.
- **Evals between steps.** With `evals` (a [`Schedule`](../../guide/reference.md#schedule)), the checkpoint of every
  `every`th step is evaluated once it is served, and the next step waits until the eval has played every start
  ([evals during training](evals.md#evals-during-training)).
- **A step that fails** (`StepFailed`) is written down with its `error`, its groups are done with, and the weights
  stay as they were. `FAILED_UPDATES` in a row stop the loop.
- **Serving waits for the engines' files.** With `reshard` (a function of a checkpoint and the run's fence, giving a
  manifest), the channel is given the files `reshard` makes for a checkpoint
  ([bridges](checkpoints.md#bridges)): the trainer's files made into what its engines load by a bridge, noted in
  the ledger once per checkpoint and bridge. Without it, the engines load the trainer's files as they are. An open profile's
  `reshard` (when its trained channel names one) runs as a Ray task when the profile names `ray`.

## Dying and starting again

The loop can be killed at any moment and started again. It keeps nothing it cannot read back: what it decides and
what happens are appended to the run's tables in the [ledger](checkpoints.md#the-ledger) (these five, besides its
`plans` and `starts`), and every action is one that can be taken twice.

| Table | Keyed by | Written | Holds |
|---|---|---|---|
| `runs/RUN/groups` | group | when a group is decided: runners play it from there | the row, the start every episode of the group is given, and how many episodes it asks for |
| `runs/RUN/results` | group | when its last episode ends | how it went: a [`Result`](../../guide/reference.md#result) |
| `runs/RUN/steps` | step | before the trainer is called | the groups it covers, the checkpoint it starts from (`parent`), the id of the one it will make (`makes`), the batch (a blob) and how many segments it has, the seed, when it was decided, its changeable settings (`settings`) and the version of the suite its evals play (`suite_version`) |
| `runs/RUN/failures` | step | when a step's trainer fails | its error |
| `runs/RUN/evals` | step | when the eval of the checkpoint the step made has played every start | the suite, the version played, the checkpoint, the eval's run, and how it went ([evals during training](evals.md#evals-during-training)) |

A step's outcome is the checkpoint it makes, in the ledger's `checkpoints` table. A group is done with once its result trains on
nothing, or the step that covers it has made its checkpoint or failed.

| It died | Started again, it |
|---|---|
| after deciding a group, or while it played | waits for its episodes: runners play them, and play again any a runner cut short ([claims](rollouts.md#what-runners-write)) |
| after a group ended | finds no result, and writes it |
| with groups queued | finds results to train on that no step covers, and queues them again |
| during a step | finds the step decided and no checkpoint made, and takes it again over the same groups, from the same parent |
| after the step | finds the checkpoint, serves it, and goes on |
| during an eval of its checkpoint | serves that checkpoint (the newest), and finishes the eval before it decides anything |

- **A step that was decided is finished before another is decided.** A decision names the checkpoint it will make,
  and only one step may make it.
- **Every episode stays in the ledger**, so a loop started again finds every episode a queued group or an
  unfinished step still needs. What a step trains on is the algorithm's batch of those episodes, the same each time
  it is computed.
- **The checkpoint is the commit.** A step's files are kept in the blob store and then the checkpoint is appended to the
  `checkpoints` table, under the id the step's decision chose. A step that died before the append made nothing.
- **Saves thin out.** Once a checkpoint is served, the checkpoints the run made are thinned to `retention` (`Retention()`:
  the weights and trainer state of the newest two and of every twentieth by depth). Whatever is served, whatever any
  run starts from, and whatever `kept` says (the bookmarked checkpoints) keep theirs ([checkpoints](checkpoints.md#checkpoints)).
- **One loop at a time.** Starting takes the run's fence, which its checkpoints are appended under too. A loop that was
  replaced, and does not know it yet, has its next write refused. What it does outside the ledger it does only while
  its fence is the newest: it looks before it publishes a checkpoint, before it deletes files in the run's directory,
  and before it moves a bookmark (`made`), and stops (`Fenced`) once another loop took the run. Its trainer writes a
  step's files into `DIRECTORY/making/FENCE/MAKES`, a directory of the loop's own, renamed to `DIRECTORY/MAKES` once
  the checkpoint is appended: a loop replaced while its trainer ran never writes where its replacement does. A look
  and the action after it are two steps, so a loop replaced between them still acts once.
- **`groups` is how many groups this start plays**, those a stopped loop left unplayed among them; the loop ends
  once they are played and every one with something to train on has been in a step.

What is redone: a step that was in progress, and the episodes that were in flight in a runner that stopped with it
(a runner plays them again from the same start).

## Objectives

An objective is chosen by its **family**, the primary selector, and composed from **components**, orthogonal
settings (`rollout_train.objectives`, free of torch; [`rollout_objectives`](../../implementations/rollout-objectives.md)
computes it). A family fixes what a batch item is and the core term of the loss:

| Family | Batch item | Core term |
|---|---|---|
| `policy_gradient` | a segment with an advantage ([`Weighted`](../../guide/reference.md#weighted)) | the advantage times each sampled token's logprob, under the ratio, clipping and importance components |
| `preference` | a pair, chosen and rejected over a shared context ([`Pair`](../../guide/reference.md#pair)), or an example labelled desirable or undesirable ([`Labelled`](../../guide/reference.md#labelled)) | a function of each side's log-likelihood ratio to the reference (or its likelihood alone, for a loss with no reference) |
| `likelihood` | a segment with a weight | the weighted log-likelihood of its sampled tokens (supervised fine-tuning, imitation) |

Each component has a dotted key under `objective.` in a run's settings, the values it takes and the families that
accept it; validation refuses the rest. The numbers can change between steps; what shapes the loss cannot.

| Component | Values | Families |
|---|---|---|
| `advantage.baseline` | `group_mean`, `leave_one_out`, `none` | policy_gradient, likelihood |
| `advantage.scale` | `none`, `group_std` | policy_gradient, likelihood |
| `advantage.filter` | `none`, `equal_scores` (DAPO's dynamic sampling) | policy_gradient, likelihood |
| `ratio` | `token`, `segment` (the geometric mean of its tokens' ratios, GSPO), `none` (the logprob itself, REINFORCE) | policy_gradient |
| `clip.kind` | `none`, `ratio` (PPO), `weight` (the clipped ratio as a weight with no gradient, CISPO), `dual` (and no less than `clip.dual` times a negative advantage) | policy_gradient |
| `clip.low`, `clip.high`, `clip.dual` | numbers: the ratio within 1 - low .. 1 + high | policy_gradient |
| `importance.correction` | `none`, `untruncated`, `truncate` (TIS), `mask` (tokens outside `floor` .. `cap` dropped) | policy_gradient |
| `importance.level` | `token`, `segment` | policy_gradient |
| `importance.cap`, `importance.floor` | numbers | policy_gradient |
| `kl.target` | `none`, `reference`, `old` (the step's start) | policy_gradient |
| `kl.estimator` | `k1`, `k2`, `k3` | policy_gradient |
| `kl.placement` | `loss`, `reward` (taken from each token's advantage, with no gradient) | policy_gradient |
| `kl.coefficient`, `entropy.coefficient` | numbers | policy_gradient |
| `aggregate` | `token_mean`, `segment_mean`, `segment_sum`, `constant` (divided by `constant_tokens`, Dr. GRPO) | policy_gradient, likelihood |
| `constant_tokens` | a whole number | policy_gradient, likelihood |
| `reference` | `base` (the model trained over), `none` | policy_gradient (with a KL to it), preference |
| `preference.loss` | `sigmoid` (DPO), `hinge`, `square` (IPO), `margin` (SimPO), `odds_ratio` (ORPO), `kto` | preference |
| `preference.beta`, `preference.margin`, `preference.desirable`, `preference.undesirable` | numbers | preference |
| `preference.length_normalized` | true or false | preference |
| `likelihood.coefficient` | a number: the chosen side's mean negative logprob beside the preference loss (ORPO) | preference |

A component that follows from another follows where it is not given: a KL to the reference reads `reference = base`, a
preference loss with a reference reads it and one without (`margin`, `odds_ratio`) reads none, and an odds ratio is
length-normalized.

**Presets** are the literature's objectives, each a family and component values pinned to its paper (its test compares
the composed loss with a transcription of the paper's formula): `default` (Dr. GRPO's advantages, DAPO's clip-higher
(0.2, 0.28) and token mean, truncated importance sampling at 2), `reinforce`, `rloo`, `ppo_clip`, `grpo`, `dr_grpo`,
`dapo`, `gspo`, `cispo`, `sft`, `dpo`, `ipo`, `simpo`, `kto` and `orpo` ([the design](../../research/objectives-design.md#presets)
lists their values and sources). A run names one (`objective.preset`) and overrides any component
(`objective.kl.target = "reference"`, `objective.kl.coefficient = 0.01`); `resolved(preset, overrides)` is the
objective, refused (`ValueError`) for a component the family does not accept or a combination that means nothing.

A trainer's own settings can name an objective too: `objective = "policy_gradient"` is `default`, `"likelihood"` is
`sft`, and `ratio`, `clip_low`, `clip_high`, `segment_clip_low`, `segment_clip_high` and `truncate` are its components
(`from_trainer_settings`); a run's settings hold them as the `objective.*` keys they say.

## The algorithm

The loop asks two things of an [`Algorithm`](../../guide/reference.md#algorithm): how many episodes of one start it
compares (`group_size`), and what to train on from a group's episodes (`batch`). `batch` is given the episodes of
every outcome, the trainer's budget and a random number generator seeded by the group's number, and returns a
[`Batch`](../../guide/reference.md#batch): the items to train on, or why there are none, and notes to log with the
group. By default the loop takes the algorithm of the trainer's objective's family (`algorithm_for`: `Grpo` for a
policy gradient or a likelihood, `Preferences` for a preference loss); another is passed as `train(..., algorithm=...)`.

[`Grpo`](../../guide/reference.md#grpo) is group-relative optimisation, by the objective's advantage components:

| | |
|---|---|
| Advantages | An episode's score less a baseline: the group's mean (`group_mean`, Dr. GRPO), the mean of the others' (`leave_one_out`, RLOO), or none; then divided by the standard deviation of the group's scores (`group_std`, the sample's, as GRPO's implementations take it) or not. Every token the policy sampled in the episode gets it; with several model slots, every slot's, so a team is rewarded together. |
| Dynamic sampling | A group whose scores are all equal has nothing to teach and is skipped (`equal_scores`, DAPO; the default). So is a group with fewer than two episodes fit to train on, and one whose every advantage is zero. |
| Behaviour logprobs | A policy gradient does not train on a group with a segment whose turns were sampled without their exact tokens, nor, with an importance correction, without their behaviour logprobs; `skipped` says which channel's turns lacked what. A likelihood reads neither. |
| The fastest of the saturated | Episodes that reached everything their task has to give earned the same; the one that took the least scores a point more, and episodes that tie for fastest all do. The task says what saturated means and how long it took ([result conventions](episodes.md#result-conventions)); comparing across the group is done here. An episode that does not say its duration is not compared. `tie_break` turns this off. |
| Untrained slots | A segment of a slot that is not trained (a judge's, a fixed opponent's: `Segment.trained` is false) is never trained on. |
| What is trained on | Every segment of the episodes whose advantage is not zero, up to what the trainer can afford in a step (`Budget.segments`). Beyond that, segments are taken at even steps through the group, so that each episode and slot keeps its share, spread over its whole game. |

[`Preferences`](../../guide/reference.md#preferences) makes a group's pairs: its best episode (the first of the best,
by score with the bonus) preferred to its worst. Both sides start from the group's start, so their shared context is
the start's first observation; a side is every turn of its episode, and only the tokens the policy sampled count. With
`labelled` (KTO) each episode above the group's mean is desirable and each below it undesirable. A group whose scores
are all equal gives none. A preference loss reads no behaviour logprobs, so turns sampled without them (a provider
that returns none) are trained on.

## The curriculum

[`Curriculum`](../../guide/reference.md#curriculum) (`rollout.curriculum`) says which row to train on next. The loop
uses the environment's own when it has one (`environment.curriculum()`, `curriculum_of`), else the generic one below.

- **Weight.** A group-relative update learns from the differences between a group's episodes, so a row's weight is
  the share of its recent groups whose rewards differed (a moving average), plus a little for every unlocked row so
  that none is forgotten. Untried rows come first.
- **Pending rows.** A row whose group is still running comes after the others until that group is recorded:
  choosing it would be choosing on what was known before it.
- **Unlocking.** Rows unlock in the environment's order: the first `start` of them, and `reach` past the hardest one
  solved at least half the time. Whether a row was solved decides only what unlocks.
- **Rows that teach about others.** A row's `counts_for` names other rows its groups are evidence about too (the
  same situation with more help, say): a group counts for its own row and for each of them.
- **Rows that cannot be set up.** A group none of whose episodes completed counts for nothing at first: the next
  start may be one the row can be set up from. After `FAILED_GROUPS` such groups in a row, the row counts as tried,
  and as having taught nothing.
- **What it sees.** The task's own rewards, whatever the algorithm adds to them.
- **It is a fold.** A curriculum is rebuilt from the run's results when the loop starts: each goes to the row of
  its title, so records stay with their rows when an environment changes. The run's evals are folded in after them,
  in the order they ended. A choice is made with a random number generator seeded by the group's number, and is
  written down before it is acted on.
- **Evals, and gates on them.** `evaluated(suite, checkpoint, results, entry)` takes one entry of an eval the run made
  of its checkpoints into account ([evals during training](evals.md#evals-during-training)): the suite, the
  checkpoint that played it, one result per start of the entry, and the entry's environment (`module:name`). It is
  called once for each entry of the suite. The generic curriculum keeps the newest of each suite's entry in
  `evaluations`, by the suite and the environment, and decides nothing from it. An environment's own overrides it to
  open rows on how a checkpoint did:

```py
@dataclass
class Gated(Curriculum):
    passed: bool = False

    def evaluated(
        self, suite: str, checkpoint: str | None, results: Sequence[GroupResult], entry: str | None = None
    ) -> None:
        super().evaluated(suite, checkpoint, results, entry)
        self.passed = self.passed or (suite == "teams-stage-2" and solved_share(results) >= 0.6)

    def unlocked(self) -> list[Row]:  # stage 3 (rows 20 on) waits for the suite
        return super().unlocked() if self.passed else super().unlocked()[:20]
```

## The trainer

A [`Trainer`](../../guide/reference.md#trainer) takes a batch and makes new weights from given ones. It keeps
nothing between steps that it cannot be given again, so any trainer can take any step from any checkpoint.

- **`budget`** ([`Budget`](../../guide/reference.md#budget)) is what the trainer can take: the longest segment, and
  how many segments a step can afford. It comes from the trainer's hardware, and nothing above the trainer chooses
  it. The algorithm selects within it, and a deployment makes the longest segment its channel's longest turn
  ([limits](channels.md#limits)).
- **`objective`** is what it trains with ([objectives](#objectives)); a trainer that says none trains the `default`
  preset.
- **`step(batch, seed=..., parent=..., into=...)`** trains on the items of its objective's family (weighted
  segments, pairs or labelled examples), starting from [`Files`](../../guide/reference.md#files) (a checkpoint's weights and state on this
  machine; none means the base model). It leaves the new weights in `into/weights`, as engines load them, and what
  a later step starts from (an optimizer's state, say) in `into/state`. It returns its metrics.
- **`StepFailed`** means the step produced no weights: the weights are as they were, and a later step may succeed.
- **`changeable`** and **`change(settings)`**, where a trainer has them ([`Changeable`](../../guide/reference.md#changeable)):
  the settings it takes between steps, by its name for each, with their values now, and a call that has it take some
  of them from its next step on (raising `ValueError` for one it does not take). They change neither what its weights
  are nor its `budget`. `LoraTrainer` and `FullTrainer` take `learning_rate`, `tokens_per_step`, `max_kl`,
  `max_gradient_norm` and the numbers of their objective (`objective.clip.low`, `objective.kl.coefficient`, ...,
  named by the run settings' keys): each step runs in a process of its own, which reads them afresh (the optimizer's
  saved state is given the learning rate then).

[`Colocated`](../../guide/reference.md#colocated) wraps a trainer that shares an accelerator with the engines of
some channels. For each step it holds new requests back, waits for those in flight, puts the engines to sleep,
steps, wakes the engines and lets requests go on. A `guard` is called once the engines are asleep and raises if the
step should not start. It adds `waited_for_requests_seconds` and `update_seconds` to the step's metrics.

`LoraTrainer` is the trainer this repository gives: [LoRA trainer](../../implementations/rollout-lora.md).

## Changing a running run's settings

A run's settings are named by dotted key, as a profile's are, and are of two kinds (`rollout_train.settings`).
**Changeable** ones can change between two steps without breaking the run: `groups_per_step`, `max_lag` (written into what the channel should serve with each
checkpoint it serves, so runners elsewhere take it with that checkpoint), the evals it makes
(`evals.suite`: a suite by name, which follows its newest version, a version by id, or none for no evals;
`evals.every`; `evals.episodes`, none for the suite's own), and its trainer's (`trainer.NAME` for each of its
`changeable`, and the numbers of its objective by their own keys, `objective.kl.coefficient`). **Fixed** ones make what the run is: the model, the trainer's kind and what its weights are, the
adapter's rank and the trainer's other settings, the channels and their engines, how many episodes it plays at once,
and the groups and seed the loop was started with (`fixed(profile, trainer, …)`). `rollout train` writes both into the
run's start record (`settings`: `fixed`, and `changeable` with their values as it starts).

What someone wants of a run's changeable settings (its **desired settings**) is ordinary state beside the ledger,
changed in place and not appended: `settings.json` beside a ledger of files, the `run_settings` table in a database
ledger's database (`DatabaseDesiredSettings`); `desired_settings_of(ledger)` finds them, `want(run, settings)` replaces
the keys given and keeps the others. The monitor's run page writes them ([a run's settings](monitor.md#a-runs-settings)).

The loop (`train(desired=…, scheduled=…)`) reads them each time it is about to decide a step. Each one it has, with a
value it can take (`checked`: a whole number of 1 at least for `groups_per_step`, `evals.every` and
`evals.episodes`, which may also be none; 0 at least for `max_lag`), is taken in place of what it used (`applied`); its trainer is told its own (`change`), and one the
trainer refuses leaves the trainer's as they were, noted to the hooks as a `settings` note with the error. A change is
noted as a `settings` note with what changed. The step is decided with the settings then in effect, and its record in
`steps` says them (`settings`), with the version of the suite they name as its name points then (`suite_version`); a
step taken again after a stop is taken with those. Whether a step's checkpoint is evaluated is that step's evals, and
the version played is the one its record names: `scheduled(suite, every, episodes)` gives the schedule of a suite by
name (the version its name points to now) or of a version by id. `rollout train`'s resolves it as the profile's
`[evals]` suite is resolved (`suite_for`): the ledger's version, or else the environment's eval data of that name,
frozen on first use, whatever environments it plays; for a name neither has, or a suite whose environments do not all
load where the run is, there is none, and nothing is evaluated.
An edit of the suite (a new version, [versions](evals.md#versions)) is played from the next step decided. So a change made while a step is being taken applies from
the next one, and a run that is stopped takes it when it is started again.

## Pausing and resuming

A run can be paused and resumed two ways (`rollout_train.resuming`): in place, while its process stays; and, once it
is stopped, by starting it again in its own directory.

**Paused in place**, a run's process keeps beating and holding its engines and GPU, and nothing new starts:

- Its desired settings say `paused: true` (`rollout_train.settings.PAUSED`, beside the changeable ones, kept where
  they are: [changing a running run's settings](#changing-a-running-runs-settings)). `rollout pause RUN` and the
  monitor's **Pause** write it.
- The loop looks each time it is about to decide a group or a step (between groups: no step boundary is waited for).
  Paused, it decides neither: episodes already playing play out and are recorded, a step being taken is finished and
  served, and groups already decided wait. It looks again every `PAUSE_LOOK` seconds (1), and notes `paused` and
  `resumed` to its hooks (the feed).
- A runner looks each time it looks for work (every half second), whether or not it has room, and claims nothing of a
  paused run (`rollout_train.settings.paused`), nor of an eval a paused run's schedule asked for (`by`) or of a part of
  a paused eval (`part_of`): so a run's scheduled evals pause with it, and an eval launched on its own pauses the same
  way. Its beats say which of the runs it serves are paused (`paused`), and it beats at once when that changes: the
  monitor shows such a run **paused**.
- **Resume** (`rollout resume RUN`, the monitor's **Resume**) sets `paused` false while the run's process beats, and
  the loop and runners go on within a second.

**Stopped**, a run's process is gone and its machine free: a launch's **Stop** interrupts it at a group boundary and
it ends `stopped` ([launchers](../../guide/deploying.md#launchers)). Resuming a run that stopped, failed or was lost
asks a launcher to start it again (`rollout_train.launches`): a launch that names the run (`resumes`) and its
directory, which the launcher starts `rollout train` (or `rollout eval`) in, with what the run's last launch asked (a
run started by hand: its newest start's profile, environment, seed and groups a step). The directory names the run, so
it goes on from the ledger as any run started again does ([dying and starting again](#dying-and-starting-again)): no
step that made its checkpoint is taken again and no recorded episode is played again. A training run is asked for the
groups it had left: those its newest start was to play, less those it has played since. A launcher alive must offer
its profile (by the name its last launch used, else by path, else by file name) and its environments.

Resume refuses a run whose process is there (it beats and its newest start has not said how it ended) and is not
paused, one that finished, one a launch is going for already, an eval a run's schedule asked for (that run plays it),
and a part of an eval (its eval is resumed).

## The record

When the loop starts it appends to the run's `starts` table, under the number of the fence it took: the checkpoint it
starts from (`from`), the host, when, and what `train(started=…)` adds; `rollout train` adds the run's directory, the profile and, with
`--monitor URL`, where the monitor on that machine serves (`address`), as other machines reach it, and where the run's
blobs are (`blobs`, [rollouts](rollouts.md#what-runners-write)). A run started
again appends another. That is how a [monitor](monitor.md) over a shared ledger finds every run, and where each keeps
its episodes. With `settings`, its fixed and changeable settings ([changing a running run's
settings](#changing-a-running-runs-settings)).

For each group the run appends a [`Result`](../../guide/reference.md#result) to its `results` table when its last
episode ends: the rewards, `solved` and durations of the episodes fit to train on, how many episodes failed and why,
how many segments were recorded and how many the algorithm found to train on (`segments`; none, and `skipped` with
its reason, if it found none), its notes, and how many rows are unlocked. What the group's own record says is not
kept again: `results(ledger, run)` reads each result with its group's number, row (`task`, `title`) and the time from
its decision to its result (`rollout_seconds`). The result goes to the loop's `hooks` as a `result` note with the
group's number; each step goes as a `step` note with the checkpoint it made and its metrics, or its error; each checkpoint
served, as a `published` note ([watching](rollouts.md#watching)).

What was done with a group is read by joining: `trained(ledger, run)` gives, for each group a step covers, that step
and the checkpoint it made or why it failed (a [`Trained`](../../guide/reference.md#trained)); the checkpoint's record
has the trainer's metrics. The report and the [monitor](monitor.md) read `results(ledger, run)` and that join.

With the run's [episodes](rollouts.md#the-record) and its [checkpoints](checkpoints.md), that is the whole run:
every episode, what each step was trained on, and the weights and the trainer's state after it.

## Reporting

`rollout report DIRECTORY ENVIRONMENT` writes `progress.png` and `progress.md` into the run's directory: the climb through the
curriculum, every group's rewards, and what each update did. With `--watch` it does so after every group; with a
Discord webhook (`--webhook`, or `DISCORD_WEBHOOK_URL`) it posts both there. It needs the `report` extra.

## Trying it without a GPU

`rollout_train.testing` has public test doubles, so that an environment, an algorithm or a whole profile can be tried
with no model and no accelerator.

| Double | Stands for |
|---|---|
| [`ScriptedEngine`](../../guide/reference.md#scriptedengine) | an engine: it answers each request from a script, and keeps what it was asked and told |
| [`PlainRenderer`](../../guide/reference.md#plainrenderer) | a model family's token format: one token per character, readable in a failing test |
| `plain_channel(script)` | a channel over both |
| `scripted_engine`, `plain_renderer` | what a profile names as `engine` and `renderer`: `rollout_train.testing:scripted_engine`, `rollout_train.testing:plain_renderer` |

`tests/rollout_train/test_profile.py` opens a profile made of these, with a trainer that trains nothing, and runs
the loop over it. For tasks and agents alone, see [guide: testing](../../guide/testing.md).

## Imitation

`rollout_train.imitation` trains a model to do, without being told how, what it did when it was told. An
environment that guides its agents reports, in each episode's result, the guidance its prompts carried, word for
word and by kind (`info["guidance"]`, for example `way` and `teamwork`).

- **`without(segment, texts, renderer)`** cuts guidance out of a segment: the fewest tokens before its first sampled
  token whose text holds it, and which encode back to themselves, are decoded, the guidance taken out, and the rest
  encoded again; the sampled tokens stay as they were, and their spans move with them. (A segment's tokens were
  joined from pieces encoded apart, so a whole prompt need not encode back to itself; a stretch of plain text
  does.) A segment where no such stretch is found is left out.
- **`examples(ledger, run, blobs, renderer, kinds=...)`** reads a run's episodes for those that carried guidance
  of those kinds and solved their task, and gives their segments, cut, each weighted 1.
- **`imitate(checkpoints, trainer, examples, fence=..., run=..., start=..., base=..., directory=...)`** takes one
  step of a trainer whose objective is a likelihood, or a preference loss for a dataset's pairs or labelled examples
  ([LoRA trainer](../../implementations/rollout-lora.md), or the trainer of every weight), from the newest checkpoint
  the run made (else from `start`), and appends the checkpoint it makes as the run's (with no step). Its parents are the checkpoint it trained from, then the checkpoints that sampled
  its examples, where those are known. The checkpoint says whether its examples were `importance` data (every turn
  sampled with its exact tokens and behaviour logprobs) or `supervised` (`supervision`).

```bash
rollout imitate PROFILE [--directory RUN] [--without KIND ...] [--limit N] [--seed N]   # with the run stopped
rollout imitate PROFILE --dataset REF [--start REF] [--name NAME]                     # a dataset's examples
rollout imitate PROFILE … --resume-optimizer          # go on from the parent's trainer state (by default: afresh)
rollout imitate PROFILE … --learning-rate R --warmup N --passes N   # the step's schedule
```

It takes the run's fence, so the run must be stopped, and writes a start of `kind: imitation` that says its
examples' `supervision`. It reads the episodes
of the run in the directory for guidance of the kinds given (`way` by default), or, with `--dataset`, a
[dataset's](datasets.md) examples; steps the profile's trainer with the run's objective if it is a likelihood or a
preference preset (a preference preset for a dataset of pairs or labelled examples), else `sft`; and adds
`imitated_episodes`, `imitated_segments` and `optimizer_resumed` to the checkpoint's metrics. The step starts from
the parent's weights with its optimizer afresh, unless `--resume-optimizer`, on a schedule of its own: 1e-6 for
every weight or 1e-4 for an adapter, warmed up over 4 updates, with passes enough for 8 updates
([its schedule](datasets.md#its-schedule)). Started again, the training loop serves
the checkpoint imitation made (the run's newest) and trains on from it.
