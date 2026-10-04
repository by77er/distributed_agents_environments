# Evals

An **eval** plays one version of a suite with one checkpoint (or the base model), trains on nothing, and records how
each episode went. Every checkpoint that plays the same version plays the same starts in the same way, so checkpoints
compare start for start. The code is `rollout_train.evals`; the commands are `rollout suite` and `rollout eval`; the
monitor's **Evals** page shows them, makes and edits suites, and asks for new evals.

## Suites

A **suite** is an eval configuration, by name:

- the environment, as `module:name`, and its version when the configuration was made;
- the starts every subject plays, chosen one of three ways: the environment's **eval data** of a name
  (`Environment.evals()`, [train and eval](rollouts.md#train-and-eval)); a start of each of some rows (every row, by
  default) for each of some seeds, the row's start drawn with `random.Random(seed)`; or starts given as they are;
- the episodes of each start an eval plays, unless it is asked for another number;
- optional sampling limits for the eval's channel: `thinking_tokens` and `answer_tokens` (none: the channel's own).

Training never draws the environment's eval starts. A suite says whether every one of its starts is among them
(`held_out`): always so for eval data, and so for rows and seeds that happen to draw only eval starts.

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
```

`rollout suite make` takes `--data NAME` for eval data of another name than the suite's, and `--episodes`,
`--thinking-tokens` and `--answer-tokens`. `rollout suite edit` keeps what it is not told: the starts of the version
the name points to, unless it is given rows, seeds or eval data, and that version's episodes and limits.

`suite_for(ledger, reference, environment_name, environment)` is the version a reference says, from the ledger, or
else the environment's eval data of that name, frozen now as version 1 of a suite of that name; a suite of another
environment, or a name neither has, is refused. `make_suite(ledger, name, environment_name, environment, …)` makes
version 1 (of `rows=` and `seeds=`, of `starts=`, or of `eval_data=`; with `episodes=`, `thinking_tokens=`,
`answer_tokens=`). `edit_suite(ledger, name, environment_name, environment, …)` makes the next version (its starts
chosen the same ways, or `same_starts=True` for those of the version the name points to) and moves the name.
`edit_suite` refuses another environment than the suite's, an edit that changes nothing (the same starts, episodes,
limits and environment version), and, given `base=` (the version the edit was made from), an edit of another version
than the newest. Both refuse a row the environment does not have, eval data it does not have, no seeds or starts, or
a count below 1. `suite_of(ledger, reference)` reads a version back, `versions_of(ledger, name)` every version of a
suite, and `suites_in(ledger)` every suite's name.

| Table | Key | Holds |
|---|---|---|
| `evaluations/SUITE/suite` | `suite` (version 1), then `2`, `3`, … | each version: its environment (`module:name`) and `environment_version`, when it was made, how its starts were `chosen` (`eval data`, `rows and seeds`, `starts`) and the `eval_data` they are, its rows and seeds, whether it is `held_out`, its `episodes`, `thinking_tokens` and `answer_tokens`, the version it was `edited_from`, and its `starts` in order: each the row's key (`task`) and title, the seed, and the start's `parameters` |

Two makers of one suite at once leave one of their suites whole. The second to take the fence shuts the first out
(`Fenced`); a maker whose record finds the suite's version 1 already there plays the one there, not its own.
`suite_for` makes the environment's eval data again after a `Fenced` until it finds a suite of the name, and plays that
one. Two editors at once make two versions, one after the other: an editor whose number another took tries the next
(and, given `base=`, is refused), and the name only moves forward, so it points to the later.

A suite made before suites had versions is its version 1: one record whose `version` is the environment's version, or,
older still, a record and its starts in a table of their own (`evaluations/SUITE/starts`, by number from 1). One held
out of training was the environment's eval data of the suite's name. An eval recorded before then played version 1.

## An eval

```bash
uv run rollout eval profile.toml words-v1 --checkpoint diamonds --directory EVAL          # its newest version
uv run rollout eval profile.toml words-v1@2 --checkpoint diamonds --episodes 4 --directory EVAL
uv run rollout eval profile.toml teams-every-task --environment minecraft_team.environment:environment --directory EVAL
```

With `--environment`, a suite not made yet is that environment's eval data of the name, frozen as the eval starts.
Without `--episodes`, each start is played as many times as the version says. An eval is a run of its own, with its own
id and name in the registry ([runs](checkpoints.md#runs)) and its own fence. Its start record says `kind: eval`, the
suite and the version (`version`, by id), the checkpoint, and the environment's version and description.
`--checkpoint` takes any [reference](checkpoints.md#references): a bookmark, `RUN:STEP`, `RUN`, or a checkpoint's id.
A bookmark is read once, when the eval starts. Without `--checkpoint`, the base model of the profile's trained channel
(or its first channel) plays.

`rollout eval` finds the version before it opens the profile: the version's sampling limits replace the trained
channel's (`thinking_tokens`, `answer_tokens`), and the subject's record says them. It then opens the profile without
its trainer (`Profile.open(training=False)`): no trainer is made, and the engines never sleep for one. What the
trainer's `start` would be is the checkpoint, so the trained channel's engines load what it is served over: the model;
a full checkpoint's files; or, for an adapter over a full checkpoint, that checkpoint's files, with the adapter loaded
over them. The engines keep the profile's sizes (`--set` changes them). Without a trainer, the channel's longest turn
is what its engines accept, not the trainer's longest segment. When it ends, however it ends, it deletes what it
fetched to serve the checkpoint (its directory's `bases/`, `checkpoints/` and `resharding/`): a full checkpoint's files
are a whole model's.

`evaluate(environment, checkpoints, run=…, suite=…, subject=…, …)` does the work:

1. It writes the run's plan (its program and binding, as a training run does), its start, and a record of who plays
   and the version.
2. If a checkpoint plays, it fetches the checkpoint's files into the run's `checkpoints/` (resharded into the
   channel's layout when the profile has one, [resharding](checkpoints.md#resharding)) and publishes them on the
   channel. A checkpoint whose weights were released cannot play. With `publish=None`, the channel serves the
   checkpoint already (a training run's newest), and nothing is fetched.
3. It asks for one group per start, `episodes` episodes each (the version's, unless given), in the run's `groups`
   table. Episode runners claim and play them as they play any run's ([rollouts](rollouts.md)), starting with the
   runner the eval's profile opens.
4. As each group's episodes end, it records each episode's outcome, and the group's result in the run's `results`.
   The result's `skipped` is "an evaluation trains on nothing", so the run's page reads like any run's.
5. It returns how many episodes were played and solved, the mean reward, and each start's `Result` (`results`), as
   first recorded.

| Table | Key | Holds |
|---|---|---|
| `evaluations/SUITE/EVAL/subject` | `subject` | who played (`kind`: `checkpoint` or `model`), the checkpoint's id, the base model, episodes a start, who asked, the eval's run, the `version` it played, and the sampling `limits` its channel was given, where it was given the version's |
| `evaluations/SUITE/EVAL/results` | `START-EPISODE` | each episode (its start by number in the version): its `run_id`, reward, whether it solved the start, its duration, outcome and detail |

`EVAL` is the eval's run id, so two evals of one checkpoint are kept apart. An eval started again, in the same
directory, goes on from where it was: groups it asked for and outcomes it recorded are not done twice. It prints the
version, how many episodes were solved of how many were played, and the mean reward.

## Made, edited and asked for from the page

The monitor's **Evals** page (`#/evals`) lists every suite and every eval, and its **New suite** form makes a suite:
its name, its environment, how its starts are chosen (eval data, rows and seeds, or starts, a row and a seed on each
line), the episodes per start and the limits. A suite's page (`#/evals/SUITE`) shows:

- the version its name points to: its environment and version, how its starts were chosen, its rows, seeds, episodes
  per start and limits;
- an **Edit** form, the same fields filled from that version (its starts kept unless they are chosen again), which
  saves the next version;
- each subject's episodes at each start, with totals, a column group for each version played (newest first, marked
  off from the next), and a version picker that shows one version;
- two subjects of one version compared at the starts both played;
- a **Run this suite** form, which asks a launcher to play a version.

Both forms post `POST /api/suites/NAME` with `{"environment", "chosen", "eval_data", "rows", "seeds", "starts",
"episodes", "thinking_tokens", "answer_tokens", "base"}` (`chosen` is `eval data`, `rows and seeds`, `starts`, or
`same` for an edit that keeps its starts; `rows` null for every row; `starts` a list of `{"task", "seed"}`, each drawn
as the row's start with that seed). The monitor loads the environment (`GET /api/environments/NAME` says its version,
rows and eval data, for the forms), makes the version with `make_suite` or `edit_suite`, and answers 409 for what they
refuse, an environment that does not load on its machine, or seeds that are no whole numbers.

The **Run this suite** form takes the checkpoint (the base model, a bookmark, or any checkpoint whose weights are kept,
by where it came from and its short id), the version (the newest by default), the episodes a start (the version's by
default), the profile, and the eval's name. It posts a launch of kind `eval` (`POST /api/launches`, `{"kind": "eval",
"suite", "profile", "name", "start", "episodes"}`). The monitor fills in the suite's environment and names the version
by id (a suite named by its name plays the version the name points to then; a suite not made yet is an environment's
eval data, and the launch says the `environment`), checks the launch as it checks a run's, and refuses an unknown suite
or version (404) or fewer than one episode a start (409). A launcher that offers the profile and the environment claims
it and starts

```bash
python -m rollout_train.cli eval PROFILE SUITE@N --directory RUNS/NAME-ID --name NAME --episodes N --environment E \
    --checkpoint REF
```

as a process of its own, or as a Ray job with `--ray` ([launchers](../../guide/deploying.md#launchers)). The suite's
page shows the launch and how it goes.

A checkpoint's page (`#/checkpoint/ID`) has the same form the other way round, **Run an eval**: it takes the suite
(by default the first a launcher alive can play), its version, the episodes per start, the profile and the name, and
posts the same launch with the checkpoint as `start`; the launches of evals of it follow, with how each goes. The page
lists every eval the checkpoint had, by hand or by its run's schedule (the suite and, for a suite with more than one
version, the version played; the share solved where its episodes say, the mean reward, the episodes, who asked for it
and when), each opening the eval's own page (`#/eval/RUN`: who played, the suite and version, who asked, its score,
and how it did at each start of that version), and charts each version's score along its line from the base model
([scores along a line](monitor.md#scores-along-a-line)).

On a suite's page, within a version, a column stands for each subject: the base model first, then each run's
checkpoints under the run's name, runs in the order their checkpoints grew and within a run by depth. A start a
version does not have is struck through under its columns.

## Evals during training

A training run says the evals it makes of its own checkpoints as it makes them: a suite, or none, said so. A profile's
`[evals]` table says them ([deploying](../../guide/deploying.md#the-file)), `--set evals.suite=…` changes them, and a
launch from the page says them in its settings:

```toml
[evals]
suite = "words-held-out"  # the version its name points to as each step is decided; or words-held-out@1, for good
every = 2               # the checkpoint of every second step (1 unless it says otherwise)
episodes = 1            # episodes of each start (by default the version's)
```

`suite = ""` (as `--set evals.suite=""`) is no evals. `rollout train` finds the suite with `suite_for`: the ledger's
version, or else the environment's eval data of that name, frozen now (a name neither has stops it before it starts).
It passes the loop a [`Schedule`](../../guide/reference.md#schedule): the version, its environment, how often, how
many episodes, and the binding of the suite's program to the trained channel. The loop plays it on the trained
channel, under that channel's sampling limits: a version's limits apply where an eval serves a channel of its own
(`rollout eval`, a launch). After a step whose number is a multiple of `every` makes its checkpoint and serves it, the
loop:

1. Registers the eval's run, `NAME-eval-STEP` (its id kept in `directory/evals/RUN-eval-STEP/run.json`, so the same
   run each time it is asked for), and adds it to the runs the profile's episode runner plays.
2. Calls `evaluate` with that checkpoint, the version the step names, and `publish=None`: the channel already serves
   it. The eval's start record says `by` (the training run) and `step`, and `from` is empty, so whether the
   checkpoint keeps its files is the run's retention's ([saves thin out](training.md#dying-and-starting-again)), not
   the eval's.
3. Waits until every start has been played. The next step is not taken until then, so the channel serves that
   checkpoint all that time. Training groups go on being played meanwhile, under that same checkpoint, and queue for
   the next step.
4. Writes the eval to the run's `evals` table, under the step's number: the suite, the version, the checkpoint, the
   eval's run, episodes played and solved, the mean reward. Then it calls `curriculum.evaluated(suite, checkpoint,
   results)` with each start's `Result`.

| Table | Key | Holds |
|---|---|---|
| `runs/RUN/evals` | step | the suite, the `version` played, the checkpoint, the eval's run (`run`), `played`, `solved`, `reward`, and when it ended (`at`) |

The eval is an eval like any other: it is listed on the **Evals** page, the suite's grid (under its version and its
run's name, with its step) and the checkpoint's page, with a page of its own.

The evals are among a run's changeable settings ([changing a running run's
settings](training.md#changing-a-running-runs-settings)): set when the run is launched (the **New run** form's suite,
every N steps and episodes per start), and changed while it runs from its page. Each step's record says the evals it
was decided with and the version of the suite they name as its name pointed then (`suite_version`); those decide
whether its checkpoint is evaluated and which version plays. An edit of the suite, or a change of the evals, made
while a step is taken applies from the next. `rollout train` gives the loop `scheduled`, which resolves a suite by name
or a version by id as the profile's is (`suite_for`). Started again, the loop folds each eval in `evals` into its
curriculum (reading the eval's `results`), and if it died during an eval, it finishes that eval (the checkpoint is the
run's newest, served when the loop starts, and the version is the one its step names) before it decides anything else.

[`Curriculum.evaluated`](../../guide/reference.md#curriculum) keeps the newest eval of each suite in
`curriculum.evaluations`, by the suite's name: the checkpoint, and its result at each start. Sampling does not read
it; a curriculum that gates rows on how a checkpoint did reads it there.

### Why in the loop, not as a launch

A run could instead ask for an eval launch of each checkpoint, which a launcher would start as `rollout eval`. That
fits a cluster: the eval runs on other engines while training goes on, and Ray places it. On one GPU, with the
trainer colocated, it does not: a second process needs engines of its own beside the run's, in memory sized for the
run's engines and the trainer, and a launcher must be running. The loop already serves the new checkpoint on engines
that are awake between steps, and its runner can play the eval's episodes beside the run's, so playing the suite
there needs no memory and no engines of its own. The cost is time: the next step waits for the eval (training groups
are still played meanwhile, so the engines stay busy). A run whose evals should not hold up its steps asks for
launches instead, from the page or with `rollout eval`.
