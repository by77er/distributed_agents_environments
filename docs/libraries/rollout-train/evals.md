# Evals

For people who measure models: suites of environments, evals of any checkpoint or base model, and evals on a schedule
during training.

**Read first:** [Evaluate a model](../../evaluate/README.md). **Next:** [The monitor](monitor.md).

An **eval** plays one version of a suite with one checkpoint (or the base model), trains on nothing, and records how
each episode went. A suite may play several environments, each with settings of its own. Every checkpoint that plays the
same version plays the same starts in the same way, so checkpoints compare start for start. The code is
`rollout_train.evals`; the commands are `rollout suite` and `rollout eval`; the monitor's **Evals** page shows them
(and each checkpoint's or base model's evals over time), makes and edits suites, and asks for new evals.

## Suites

A **suite** is an eval configuration, by name: a list of **entries** (`SuiteEntry`), one for each environment it plays.
An environment is in a version once. Each entry says:

- the environment, as `module:name`, and its version when the entry was made;
- the starts every subject plays of it, chosen one of three ways: the environment's **eval data** of a name
  (`Environment.evals()`, [train and eval](rollouts.md#train-and-eval)); a start of each of some rows (every row, by
  default) for each of some seeds, the row's start drawn with `random.Random(seed)`; or starts given as they are;
- the episodes of each start an eval plays, unless it is asked for another number;
- optional sampling limits for its episodes: `thinking_tokens` and `answer_tokens` (none: the channel's own, which
  may be no budget, [limits](channels.md#limits)).

A version's starts are numbered from 1 across its entries, in order: the first entry's, then the next's. Training never
draws an environment's eval starts. An entry says whether every one of its starts is among them (`held_out`): always so
for eval data, and so for rows and seeds that happen to draw only eval starts. A version is held out where each of its
entries is.

Each entry is scored apart. Environments' rewards do not compare, so a suite has no score across its entries.

### Versions

A suite is kept in **versions**. A version is one record with an id of its own, `NAME@NUMBER`, written once under the
suite's fence (`suites/NAME`) and never changed or deleted. Editing a suite makes its next version. The suite's name
points to its newest version, as a bookmark points to a checkpoint ([bookmarks](checkpoints.md#bookmarks)): the
registry beside the ledger holds where it points (a file beside a ledger of files, the `suite_names` table of a
database ledger), and each edit moves it. A name the registry holds nothing for is its newest version in the ledger.

Wherever a suite is named, `NAME` is the version its name points to then, and `NAME@NUMBER` is that version for good.
Every eval records the exact version it played, so scores of two versions are never mixed: the monitor compares
subjects within a version, and charts each version apart.

```bash
uv run rollout suite list --environment minecraft_team.environment:environment --ledger L  # its eval data too
uv run rollout suite make teams-every-task --environment minecraft_team.environment:environment --ledger L
uv run rollout suite make words-v1 --environment E --rows chests,diamonds --seeds 1,2,3 --episodes 2 --ledger L
uv run rollout suite edit words-v1 --seeds 1,2,3,4 --thinking-tokens 512 --ledger L   # its next version
uv run rollout suite edit words-v1 --environment G --seeds 1 --answer-tokens 64 --ledger L  # an entry of G added
uv run rollout suite edit words-v1 --drop E --ledger L                                  # E's entry left out
```

`rollout suite make` makes a suite of one entry. It takes `--data NAME` for eval data of another name than the suite's,
and `--episodes`, `--thinking-tokens` and `--answer-tokens`. `rollout suite edit` changes the entry of `--environment`
(for a suite of one entry, that one by default), or adds one where the suite has none of that environment; `--drop`
leaves an entry out. It keeps what it is not told: the entry's starts, unless it is given rows, seeds or eval data, and
its episodes and limits.

`suite_entry(environment_name, environment, …)` makes an entry: its starts of `rows=` and `seeds=`, of `starts=`, or of
`eval_data=`, with `episodes=`, `thinking_tokens=` and `answer_tokens=`. It refuses a row the environment does not have,
eval data it does not have, no seeds or starts, or a count below 1. `make_suite(ledger, name, entries)` makes version 1.
`edit_suite(ledger, name, entries)` makes the next version and moves the name; an entry whose starts are those of the
current version's entry of its environment keeps how they were chosen. Both refuse no entries, or an environment in two.
`edit_suite` refuses an edit that changes nothing (the same environments, starts, episodes, limits and environment
versions), and, given `base=` (the version the edit was made from), an edit of another version than the newest.
`suite_for(ledger, reference, environment_name, environment)` is the version a reference says, from the ledger whatever
its environments, or else the environment's eval data of that name, frozen now as version 1 of a suite of that name; a
name neither has is refused. `suite_of(ledger, reference)` reads a version back, `versions_of(ledger, name)` every
version of a suite, and `suites_in(ledger)` every suite's name.

| Table | Key | Holds |
|---|---|---|
| `evaluations/SUITE/suite` | `suite` (version 1), then `2`, `3`, … | each version: when it was `made`, the version it was `edited_from`, and its `entries` in order, each its environment (`module:name`) and `environment_version`, how its starts were `chosen` (`eval data`, `rows and seeds`, `starts`) and the `eval_data` they are, its rows and seeds, whether it is `held_out`, its `episodes`, `thinking_tokens` and `answer_tokens`, and its `starts` in order: each the row's key (`task`) and title, the seed, and the start's `parameters` |

Two makers of one suite at once leave one of their suites whole. The second to take the fence shuts the first out
(`Fenced`); a maker whose record finds the suite's version 1 already there plays the one there, not its own.
`suite_for` makes the environment's eval data again after a `Fenced` until it finds a suite of the name, and plays that
one. Two editors at once make two versions, one after the other: an editor whose number another took tries the next
(and, given `base=`, is refused), and the name only moves forward, so it points to the later.

## An eval

```bash
uv run rollout eval words-v1 --checkpoint diamonds                         # its newest version, by a checkpoint
uv run rollout eval words-v1@2 --checkpoint diamonds --episodes 4
uv run rollout eval math --preset gsm8k-tinker                             # by the base model the preset's channel serves
uv run rollout eval math --preset gridworld-qwen3-0.6b --model Qwen/Qwen3-0.6B
```

`rollout eval SUITE` asks for an eval run on the cluster ([deploying](../../guide/deploying.md#asking-for-a-run)): its
settings say the suite (`eval.suite`), the checkpoint (`start`, `--checkpoint`), the episodes of each start
(`eval.episodes`, `--episodes`; none: each entry's own) and the channel it plays on (`channels.policy.*`). A preset's
settings an eval does not take (the trainer's) are left out of it. With `--checkpoint`, the channel's model, renderer,
providers and budgets are those of the run that made the checkpoint, unless the settings say others. The suite must be
one the ledger has: a name never becomes a suite by itself. Its environment is the suite's first entry's.

An eval is a run of its own, with its own id and name in the registry ([runs](checkpoints.md#runs)) and its own fence.
Its start record says `kind: eval`, the suite and the version (`suite_version`, by id), the checkpoint, and the
environment's version (`version`, as a training run's start says it) and description. A start that does not say
`suite_version` is read with the version its subject's record says. `--checkpoint` takes any
[reference](checkpoints.md#references): a bookmark, `RUN:STEP`, `RUN`, or a checkpoint's id, read once, when the eval
starts. Without one, the base model of its channel plays, and the eval records it as its subject (`kind: model`).

An entry's sampling limits travel in its episodes' binding (`SamplingParameters.thinking_tokens` and `answer_tokens`
of each recorded model, carried in each slot's [gateway key](gateway.md#keys)), and the gateway samples with them in
place of the channel's own, so entries of one eval take different limits on one channel. The eval's driver makes no
trainer, and serves the checkpoint on its channel's engines: on engine hosts of its own, which load it by its id from
what its serving records say, its files made by the bridges from the checkpoint's format to what the provider loads
([bridges](checkpoints.md#bridges)). Without a trainer, the channel's longest turn is what its engines accept. When it
ends, however it ends, it deletes what it fetched under its directory (`bases/`, `checkpoints/`).

**A hosted model.** An eval of a model behind a hosted API plays on a channel of an `api` provider:
`channels.policy.provider = "openai"` (or `"anthropic"`) and `channels.policy.model` one of its models, with no
renderer and no checkpoint ([hosted APIs](../../guide/cluster.md#hosted-apis)). Its driver starts no engines; the
gateway in it samples each turn through the provider's endpoint and records what it cost. Its estimated spend (the
whole eval's: `spend_of`, from the suite's starts, each entry's episodes and the budgets, at the model's catalog
prices) is said when it is checked, and `limits.spend` ends it, failed, once it spends that.

```bash
uv run rollout eval math --provider anthropic --model claude-sonnet-5-5 --set limits.spend=5 \
    --set channels.policy.thinking_tokens=2048 --set channels.policy.answer_tokens=1024
```

`evaluate(checkpoints, run=…, suite=…, subject=…, …)` does the work. It takes the entries' environments
(`environments=`, by `module:name`; any not given are imported), how an environment's episodes are bound (`binding=`, by
default every slot from the channel), and, for a suite of several entries, the run of each (`parts=`, by its number
from 1; by default `RUN-NUMBER`; an eval's driver registers each as `ID-NUMBER`, called `NAME-NUMBER`, and adds it to
the runs its runner plays).

A version of one entry is played in the eval's own run. A version of several is played in a run for each entry (its
**parts**), each with the entry's own plan: its program, and its binding with its tool sets, pools and limits. A runner
claims an entry's episodes only where it has what that entry's environment needs, as it does any run's
([rollouts](rollouts.md)), so a suite can mix environments no single runner has. The eval's own run then plays nothing;
it holds the eval's start, its subject and its results.

1. It writes each part's plan and start (`kind: eval`, the suite, `part_of`: the eval's run, `entry`: its number, the
   environment), the eval's start, and a record of who plays, the version and the parts.
2. If a checkpoint plays, it writes down that each part's channel serves it, with the files its bridges make
   ([bridges](checkpoints.md#bridges)), fetches them into the run's `checkpoints/` and publishes them on the
   channel. A checkpoint whose weights were released cannot play. With `publish=None`, the channel serves the
   checkpoint already (a training run's newest), and nothing is fetched.
3. It asks, in each part's `groups` table, for one group per start of its entry, `episodes` episodes each (the
   entry's, unless given). Episode runners claim and play them as they play any run's, starting with the runner the
   eval's driver starts.
4. As each group's episodes end, it records each episode's outcome under the start's number in the version, and the
   group's result in its part's `results`. The result's `skipped` is "an evaluation trains on nothing", so a part's
   page reads like any run's.
5. Once every start has been played, it records each entry's scores (`scores`): episodes played, solved where its
   environment's results say it, the share solved and the mean reward.
6. It returns how many episodes were played and solved, the mean reward, each start's `Result` (`results`), and each
   entry's environment, run, scores and results (`entries`), as first recorded.

| Table | Key | Holds |
|---|---|---|
| `evaluations/SUITE/EVAL/subject` | `subject` | who played (`kind`: `checkpoint` or `model`), the checkpoint's id, the base model, episodes a start (none where entries differ), who asked, the eval's run, the `version` it played, and its `parts`: each entry's `environment`, `run`, `episodes` and `limits` |
| `evaluations/SUITE/EVAL/subject` | `scores` | each entry's `environment`, `played`, `solved` (none where its results do not say), `share` and `reward` |
| `evaluations/SUITE/EVAL/results` | `START-EPISODE` | each episode (its start by number in the version): its `run_id`, reward, whether it solved the start, its duration, outcome and detail |

`EVAL` is the eval's run id, so two evals of one checkpoint are kept apart. An eval resumed goes on from where it was: groups it asked for and outcomes it recorded are not done twice. It prints the
version, how many episodes were solved of how many were played, and the mean reward.

## Made, edited and asked for from the page

The monitor's **Evals** page (`#/evals`) lists every suite and every eval, and its **New suite** form makes a suite:
its name and its entries. Each entry has its environment, picked from those the monitor knows (`GET /api/environments`:
offered by the cluster, started on by a run, or played by a suite; those the cluster offers first), how its starts
are chosen (eval data, rows and seeds, or starts, a row and a seed on each line), the episodes per start and the limits
(thinking and answer tokens, each left empty for the channel's own). Entries are added and removed. A suite's page (`#/evals/SUITE`) shows:

- the version its name points to: for one entry, its environment and version, how its starts were chosen, its rows,
  seeds, episodes per start and limits ("channel's" for a limit it leaves to the channel); for several, a table of
  its environments with the same;
- an **Edit** form, the same fields filled from that version (each entry's starts kept unless they are chosen again),
  which saves the next version;
- each subject's episodes at each start, with totals, a column group for each version played (newest first, marked
  off from the next), within a version of several environments a column group for each environment with each
  subject's total there, and a version picker that shows one version;
- two subjects of one version compared at the starts both played;
- a **Run this suite** form, which asks for an eval of a version.

Both forms post `POST /api/suites/NAME` with `{"entries": [...], "base"}`, each entry `{"environment", "chosen",
"eval_data", "rows", "seeds", "starts", "episodes", "thinking_tokens", "answer_tokens"}` (`chosen` is `eval data`,
`rows and seeds`, `starts`, or `same` for an edit that keeps that environment's entry's starts; `rows` null for every
row; `starts` a list of `{"task", "seed"}`, each drawn as the row's start with that seed). A body with no `entries` is
one entry. The monitor loads each environment (`GET /api/environments/NAME` says its version, rows and eval data, for
the forms; the form says "does not load here" for one it cannot), makes the version with `make_suite` or `edit_suite`,
and answers 409 for what they refuse, an environment that does not load on its machine, or seeds that are no whole
numbers.

The **Run this suite** form takes who plays (a base model the cluster's inference providers offer, the first by default;
a bookmark; or any checkpoint whose weights are kept, by where it came from and its short id), the version (the newest
by default), the episodes a start (the version's by default), a preset (by default one whose channel serves the base
model played), for a base model the provider it plays on (those that serve it, each said metered or scheduled), and
the eval's name. On a metered provider (a hosted API), the form shows the model's prices and the eval's estimated
spend (`POST /api/launches/check`), and takes a limit (`limits.spend`). It asks for an eval run (`POST /api/launches`, [launching a
run](monitor.md#launching-a-run)), with the preset's channel settings, the suite's version (`eval.suite`, by id), the
checkpoint (`start`) or the base model (`channels.policy.model`, on the provider chosen; else the preset's where it
serves that model, else one that does; a hosted API's channel names no renderer), the episodes and the limit. The monitor gives it the suite's first environment, checks it as it checks any run, and
submits its job ([launching runs](launching.md)). The suite's page shows the launch and how it goes.

A checkpoint's page (`#/checkpoint/ID`) has the same form the other way round, **Run an eval**: it takes the suite, its
version, the episodes per start, the preset and the name, and asks for the eval with the checkpoint as `start`; the
launches of evals of it follow, with how each goes. The page lists every eval the checkpoint had, by hand or by its
run's schedule (the suite and, for a suite with more than one version, the version played; the share solved where its
episodes say, the mean reward, the episodes, who asked for it and when), each opening the eval's own page
(`#/eval/RUN`: who played, the suite and version, who asked, its score, and how it did at each start of that version),
and charts each version's score along its line from the base model ([scores along a
line](monitor.md#scores-along-a-line)).

A base model's page (`#/base/NAME`, a root of the checkpoints' graph: every base model with history, and every one the
cluster offers) has the same **Run an eval** form, with the base model as `channels.policy.model`, and a link to start a
training run from it (not for a model only hosted APIs serve: it is played, never trained). The launches of evals of it follow, with how each goes.

The **Evals** page lists every checkpoint and base model that has had an eval, the one evaluated last first, and each
opens its history (`#/evals/checkpoint/ID`, `#/evals/model/NAME`; a checkpoint's page links to it): every eval it has
had, a card for each version of a suite it played, with each environment's share solved (else its mean reward) over
time and each eval's scores at each environment, who asked for it and when; and, for a checkpoint, its scores along
its line ([a subject's history](monitor.md#a-subjects-history)).

On a suite's page, within a version, a column stands for each subject: the base model first, then each run's
checkpoints under the run's name, runs in the order their checkpoints grew and within a run by depth. A start a
version does not have is struck through under its columns.

## Evals during training

A training run says the evals it makes of its own checkpoints as it makes them, in its settings (a preset, the New
run form, `--set`):

```toml
"evals.suite" = "words-held-out"  # the version its name points to as each step is decided; or words-held-out@1, for good
"evals.every" = 2               # the checkpoint of every second step (1 unless it says otherwise)
"evals.episodes" = 1            # episodes of each start (by default the version's)
```

`evals.suite` unset (or `null`) is no evals. A suite the ledger does not have is refused when the run is asked for (a
name never becomes a suite by itself), and so is one whose environment the cluster does not offer. The run's driver
finds the version and loads each entry's environment. A suite may play other environments than the
one trained on: the policy is the same, so each entry is played on the trained channel. It passes the loop a
[`Schedule`](../../guide/reference.md#schedule): the version, how often, how many episodes, the entries' environments,
and how an environment's program is bound to the trained channel and the run's tool sets and pools. Each entry's
limits travel in its binding. After a step whose number is a multiple of `every` makes its checkpoint and serves it, the
loop:

1. Registers the eval's run, `RUN-eval-STEP` (called `NAME-eval-STEP`: the same run each time it is asked for), and for
   a suite of several entries a run for each, `RUN-eval-STEP-NUMBER`, and adds them to the runs the driver's episode
   runner plays.
2. Calls `evaluate` with that checkpoint, the version the step names, and `publish=None`: the channel already serves
   it. The eval's start record says `by` (the training run) and `step`, and `from` is empty, so whether the
   checkpoint keeps its files is the run's retention's ([saves thin out](training.md#dying-and-starting-again)), not
   the eval's.
3. Waits until every start has been played. The next step is not taken until then, so the channel serves that
   checkpoint all that time. Training groups go on being played meanwhile, under that same checkpoint, and queue for
   the next step.
4. Writes the eval to the run's `evals` table, under the step's number: the suite, the version, the checkpoint, the
   eval's run, episodes played and solved, the mean reward, and each entry's environment, run and scores. Then it
   calls `curriculum.evaluated(suite, checkpoint, results, entry)` for each entry, with each of its starts' `Result`
   and its environment.

| Table | Key | Holds |
|---|---|---|
| `runs/RUN/evals` | step | the suite, the `version` played, the checkpoint, the eval's run (`run`), `played`, `solved`, `reward`, each entry's `environment`, `run`, `played`, `solved`, `share` and `reward` (`entries`), and when it ended (`at`) |

The eval is an eval like any other: it is listed on the **Evals** page, the suite's grid (under its version and its
run's name, with its step), the checkpoint's page and its history, with a page of its own.

The evals are among a run's changeable settings ([changing a running run's
settings](training.md#changing-a-running-runs-settings)): set when the run is launched (the **New run** form's suite,
every N steps and episodes per start), and changed while it runs from its page. Each step's record says the evals it was
decided with and the version of the suite they name as its name pointed then (`suite_version`); those decide whether its
checkpoint is evaluated and which version plays. An edit of the suite, or a change of the evals, made while a step is
taken applies from the next. A run's driver gives the loop `scheduled`, which resolves a suite by name or a version by
id from the ledger (`suite_of`). Started again, the loop folds each eval in `evals` into its curriculum, entry by
entry (reading each part's `results`), and if it died during an eval, it finishes that eval (the checkpoint is the run's
newest, served when the loop starts, and the version is the one its step names) before it decides anything else.

[`Curriculum.evaluated`](../../guide/reference.md#curriculum) keeps the newest eval of each suite's entry in
`curriculum.evaluations`, by the suite's name and the entry's environment: the checkpoint, and its result at each start
of that entry. Sampling does not read
it; a curriculum that gates rows on how a checkpoint did reads it there.

### Why in the loop, not as a launch

A run could instead ask for an eval run of each checkpoint. That fits a cluster: the eval runs on other engines while
training goes on. On one GPU, with the trainer colocated, it does not: a second run needs engines of its own beside the
run's, in memory sized for the run's engines and the trainer. The loop already serves the new checkpoint on engines that
are awake between steps, and its runner can play the eval's episodes beside the run's, so playing the suite there needs
no memory and no engines of its own. The cost is time: the next step waits for the eval (training groups are still
played meanwhile, so the engines stay busy). A run whose evals should not hold up its steps asks for eval runs instead,
from the page or with `rollout eval`.
