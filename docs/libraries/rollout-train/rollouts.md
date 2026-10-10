# Rollouts

Code: `rollout_train.rollouts` · See [episodes](episodes.md), [training](training.md),
[API reference](../../guide/reference.md#rollout_trainrollouts)

For people who design training: how a run asks for episodes in the ledger, and how runners claim, play and record them.

**Read first:** [Train a model](../../train/README.md). **Next:** [Episodes](episodes.md).

A run asks for episodes in the [ledger](checkpoints.md#the-ledger), and runners play them. The run decides what to play,
how often and how to group it; a runner knows no algorithm, and the run never learns where its episodes were played.
Several runners, on one machine or many, share the work of every run whose ledger they reach: which machine plays a
group's episodes is only a matter of where runners are.

```python
await plan(ledger, "train", Plan(program, binding), fence)        # how the run's episodes are played
await ledger.append(table("train", GROUPS), "1", {"parameters": row, "episodes": 4}, fence)
endpoints = GatewayEndpoints.of(gateway)                           # what the runs' recorded slots sample through
async with playing(EpisodeRunner("host/train", ledger, LocalRunner(gateway=endpoints), endpoints, blobs, places=6)):
    episodes = await episodes_of(ledger, blobs, "train", 1, 4)     # the group's four, once all have ended
```

The [training loop](training.md) writes the plan and the groups; a run's driver starts a runner beside it
([launching runs](launching.md#the-driver)). Everything goes through the ledger, so the run and its runners may be in one process or on different
machines.

## What a run writes

| Table | Keyed by | Holds |
|---|---|---|
| `runs/RUN/plans` | the fence the run's loop took | how its episodes are played from then on: a [`Plan`](../../guide/reference.md#plan), the program (each group's start is its row) and the [`RunBinding`](../rollout/README.md#run-specifications) |
| `runs/RUN/groups` | group | its row, the start each of its episodes is given (`parameters`), and how many episodes it asks for (`episodes`) |
| `runs/RUN/serving` | `CHANNEL/CHECKPOINT` | what each of its channels should serve from then on ([what a channel should serve](channels.md#what-a-channel-should-serve)): engine hosts load it into their servers, and runners ask for it by name |

An episode is `GROUP/EPISODE`, numbered from 1 within its group. A group with a result (`runs/RUN/results`) asks for
nothing more.

## What runners write

| Table | Keyed by | Holds |
|---|---|---|
| `runs/RUN/claims` | `GROUP/EPISODE/ATTEMPT` | the runner that plays that attempt, the number of its fence, when, and the run that plays it (`run_id`) |
| `runs/RUN/episodes` | `GROUP/EPISODE` | the episode's [`Record`](../../guide/reference.md#record), once it has ended, appended under [the episode's fence](#each-episodes-fence) |
| `runs/RUN/interrupted` | `GROUP/EPISODE/ATTEMPT` | an attempt cut short, and why: its runner closed, the claim lapsed while its runner was stopped, a sandbox of its was lost (its pool ended it or was started again, or the pod that served it is no longer the run's), another took the episode's fence before it was recorded, or its pool released its sandboxes |
| `runs/RUN/adopted` | `GROUP/EPISODE/ATTEMPT/FENCE` | an attempt whose run its runner, started again, took up under its new fence, appended under the episode's fence |

- **First append wins.** A claim is an append to a key no one has written, so two runners never play one attempt:
  the one whose append is refused looks for other work.
- **A claim holds while it is its episode's latest attempt and its runner keeps its fence and beats.** A runner takes
  the fence of `runners/NAME` when it starts. Its claims hold until a newer attempt of the episode is claimed, until it
  is started again (under the same name, its fence moves on) and does not adopt them, until it notes an attempt as
  interrupted, or, where runners beat ([heartbeats](#heartbeats)), until its newest beat is older than 90 seconds (its
  machine died, say). `Claims` holds a run's claims as read at one moment, and `Claims.holds(...)` is the rule;
  `holding(ledger, run, ...)` is the claims of a run that hold, which pools beside the ledger use too
  ([sandboxes](../rollout/sandboxes.md#in-training-a-lease-ends-with-its-claim)). An episode with no record and no
  claim that holds is open: the next claim is its next attempt. So at most one claim of an episode holds at a time,
  and a claim that lapsed never holds again beside a newer one. A runner does not wait on its own beat: its own claims
  hold for it while its fence is its own.
- **Every attempt that ends is an episode**, whatever its outcome: completed, failed (the program raised, or the run
  could not start), cancelled. The first record of an episode is its record, and only the attempt that holds the
  episode's fence can append it.
- **An episode's trajectories and its run's events go to the blob store**; the record names both
  ([the record](#the-record)). A run started by `rollout train` says in its `starts` record where that store is
  (`rollout_train.stores`: its kind and settings, never a credential), so that any machine can read a finished
  episode back.

## Each episode's fence

Each episode has a fence of its own, `runs/RUN/episodes/GROUP/EPISODE` (`episode_scope(run, episode)`), beside its
runner's.

| Who | Takes the episode's fence | And appends under it |
|---|---|---|
| The runner whose claim of an attempt was appended | at once, after the claim: only the winner of an attempt takes it | the episode's record, when the attempt ends |
| A runner started again, for each claim it adopts | before it reads its claims again | the adoption, then the record |
| A pool's keeper, ending a lapsed claim ([sandboxes](../rollout/sandboxes.md#in-training-a-lease-ends-with-its-claim)) | before it releases the claim's lease | the note that the attempt was cut short (`RELEASED`) |

Whoever took the fence last shuts out every attempt before it:

- **A runner whose claim lapsed records nothing over a newer attempt.** A runner that paused (a stop-the-world pause,
  a suspended machine, a partition) past its claim's lapse, while another claimed the episode again, finds its record
  refused (`Fenced`) when it resumes. It notes the attempt cut short (`SUPERSEDED`) and goes on.
- **An adoption and a new attempt never both stand.** A runner started again takes the episode's fence, then reads
  the claims: an attempt claimed before the take is seen, and the claim is not adopted; one claimed after takes the
  fence in turn, and the adoption (or the adopted run's record) is refused.
- **A runner's interrupts** stay under its own fence: noting an attempt cut short only ever ends its own claim.

Each fence is a row of the ledger's fences, one per episode claimed; the [monitor](monitor.md) leaves them out of the
fences it lists.

## A runner

[`EpisodeRunner(name, ledger, runner, recorder, blobs, places, ...)`](../../guide/reference.md#episoderunner) claims
open episodes and plays them on a [`Runner`](../rollout/README.md#runner). Its `recorder` is what the runs' recorded
slots sample through, the runner's too: the [gateway](gateway.md#a-runner-served-by-the-gateway)'s endpoints
([`Recorded`](../../guide/reference.md#recorded)).

- **`places`**: how many episodes it plays at once. When one ends it looks again at once; otherwise every `every`
  seconds.
- **Oldest group first.** Open episodes are claimed in the order their groups were decided, across every run it
  serves.
- **What it serves.** A run whose plan's recorded models its recorder can sample now (`reaches`), whose local
  imports are all among `imports` (the tool sets it has), and whose local pools are all among `pools` (the sandbox
  pools it has). With `runs`, those runs only. A channel whose engines are on other machines can be sampled while one
  of its servers has a checkpoint close enough to what the run says it should serve: the runner knows nothing of
  servers.
- **Room in the pools.** An episode whose program declares sandboxes is claimed only while their pools have room
  for them: each pool's `capacity()` is asked once a look, and what the runner claims is counted against it as it
  goes ([sandboxes](../rollout/sandboxes.md#in-training-a-lease-ends-with-its-claim)).
- **`guard`** is called before claiming, and raises to wait: a machine short of memory claims nothing until it has
  room.
- **An episode is played** as the run's program with the group's `parameters` as its row, labelled `run`, `group`
  and `episode`, with the claim's key (`RUN/GROUP/EPISODE/ATTEMPT`) as the run's lease: its sandboxes are leased
  under it, and their leases end with the claim. Before the run starts, the runner admits it to its recorder under
  the episode's fence (`admit(run_id, Attempt(run, fence, "GROUP/EPISODE", attempt))`): the gateway appends the run's
  turns under that fence, so an attempt taken over records nothing more. When it ends, the runner reads the run's
  segments from the gateway's turn store (`await sessions(run, run_id)`), assembles the [episode](episodes.md),
  stores it and appends its record under the episode's fence.
- **Closing** cancels what it plays, in the runner too, and notes each attempt whose run had started in
  `interrupted`: the episode is open again, for any runner with room. An attempt cancelled before its run started
  is noted nowhere; its claim lapses once its runner's fence moves on or its beats stop. Over a runner whose runs
  survive it (`resumes`), closing leaves its runs to be resumed.

### A runner started again

A runner started again under its name takes its fence anew (`prepare()`, which `serve` calls if it has not been): the
claims it held lapse, and their episodes are open again for any runner with room. Over a runner whose runs survive it
(`resumes`), it also adopts what it finds of its runs, before that runner is launched:

- **Adopted:** a run of its own claim that is still its episode's latest attempt, not cut short, its episode without
  a record, found by the claim's `run_id`; the claim may have been made under any fence the runner held before (one
  that died before adopting, or whose fence was taken twice, adopts its runs all the same). The runner takes the
  episode's fence, reads the claims again, and notes the adoption in `adopted`, keyed by its new fence and appended
  under the episode's; the claim holds again, and the run is admitted to the recorder under the episode's new fence.
  The runner follows the run to its end and records its episode, which trains like any other: the gateway kept what
  the run sampled before its runner stopped, and answers a sample asked for again under its effect id with the turn
  it recorded. A run that ended while its runner was stopped, unrecorded, is recorded now.
- **Cut short:** a run of its own claim that lapsed meanwhile (another runner took the episode up, say), still
  going. The attempt is noted in `interrupted` and the run cancelled; a pool beside the ledger refuses it its
  sandboxes and releases them.
- **Played again:** a run that fails because a sandbox of its is gone (`SandboxLost`: its pool ended its lease, was
  started again, or did not outlive the runner, or the pod that served it is no longer the run's: [sandboxes on a host
  pod](../../deploy/providers.md#sandboxes-on-a-host-pod)), adopted or not, is noted in `interrupted`, and the episode
  is played as a new attempt; its third such attempt is recorded failed, saying its sandbox was lost three times
  running.

`serve()` runs until cancelled; `async with playing(runner):` serves while a block runs.
[`episodes_of(ledger, blobs, run, group, count)`](../../guide/reference.md#episodes_of) waits until all `count`
episodes of a group have records and returns them, trajectories and all.

### Heartbeats

With `presence` (a `Presence`, `rollout_train.presence`), a runner beats when it
starts, before it claims anything, and every `beating` seconds (15) after; `beat()` beats at once besides. A beat holds what `about()` says of its machine (called
in a thread: it may measure), with its `places`, how many episodes it is `playing`, and, when it has pools, how full
each is (`pools`: `size`, `leased`, `free`), and, when any of the runs it serves is paused, which (`paused`: it claims
none of their episodes, and beats at once when that changes: [pausing and resuming](training.md#pausing-and-resuming)).
Each runner's newest beat
is kept with the measurements of its recent ones (240: an hour), so its machine can be shown from anywhere.

Beats are kept beside the ledger, as ordinary state changed in place, not appended: `presence.json` beside a ledger
of files, the `presence` table in a database ledger's database (`DatabasePresence`); `presence_of(ledger)` finds
them. A runner whose newest beat is older than `STALE` (90 seconds) is taken to be gone, and its claims lapse.

A beat's time is the store's, never the runner's. The store stamps a beat (`Beat.at`) when it keeps it, and says how
old it is when it is read (`Beat.age`), by the same clock; `alive(beat)` compares the age with `STALE`. A Postgres
database stamps and ages beats by its server's clock (`clock_timestamp()`), so a runner whose machine's clock is
behind or ahead of the reader's is judged by when it last beat all the same. A SQLite database (`julianday('now')`)
and a ledger of files serve one machine, whose clock every writer and reader shares. What every reader of the beats
judges by them (whether a claim holds, whether a pool's keeper ends a lease, whether the monitor shows a run running)
goes by the age.

A run's runner (`run/RUN`) says, in each beat: its host, the run it serves, the run's directory, its machine
(`rollout_train.machine`: memory, each GPU's memory and how busy, the disk the directory is on), and what each channel
serves (adapter and version) with what passed through it since the beat before (requests, tokens, tokens a second,
requests at once); for a routed channel, by its name within each run (`RUN/NAME`), what each of its servers would sample
from and how far behind that is; and what the run holds of Ray: its demand in all (`demand`: the driver's and its
placement group's CPUs, memory and GPUs), when it asked for it (`asked`) and when Ray reserved it (`reserved`); and,
while a step is being taken, how far it has got (`progress`: the trainer's
[`Progress`](training.md#how-far-a-step-has-got) with the step's number), beating at once when that moves on (at most
every 5 seconds). The [monitor](monitor.md) shows machines, inference throughput, what is served, the queue and the
step being taken from these beats. Before it
plays, while Ray has not given the run what it asked for, its driver beats under the same name (kind `run`) with what
it waits for, and its `demand` and `asked` (`reserved` none until its placement group is reserved). An engine host (kind `engines`) beats with the run it follows, its machine, and
what each channel's engines serve ([what a channel should serve](channels.md#what-a-channel-should-serve)); a gateway
replica (kind `gateway`) with where it listens, its machine and its channels ([the gateway](gateway.md#running-it)); a
pool on a machine of its own (kind `pool`) with how full it is. A pod on RunPod says besides what it is (`pod`: its
name, identity, role, whether it is ready, and the run it is ready for): an inference or host pod beats as an engine
host, a training pod with kind `trainer` ([GPU pods on RunPod](../../deploy/providers.md#gpu-pods-on-runpod)). The
monitor's [Machines page](monitor.md#the-machines) shows runners, waiting runs, pools, engine hosts and gateways, each
by its kind.

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

What a run trains on and an eval measures is an [`Environment`](../../guide/reference.md#environment):

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
uv run rollout env check minecraft_team.environment:environment --pools minecraft=minecraft_team.worlds:worlds
uv run rollout env check tests.rollout_train.rollouts.games:guessing --preset PRESET --groups 4   # with a model
```

`rollout env check ENVIRONMENT` says, a line for each, whether its rows build (keys and titles unique, `counts_for`
naming rows it has); whether its description and version say something; whether one seed draws one start and its eval
data is the same each time; how many of the starts drawn for training were eval starts, and were drawn again; and how
one episode went on the local runner with a scripted model (`--reply` is what it says each turn; `--row` the row; the
tool sets its program imports from `--tools NAME=module:factory` or a URL, and the pools of the sandboxes it declares
from `--pools KIND=module:factory` or a URL), with a reward in the described range and a result that
says what the description says it does.

With a model's settings (`--preset`, `--provider`, `--settings` or `--set`) it also asks for a check run on the
cluster ([deploying](../../guide/deploying.md#asking-for-a-run)), which plays `--groups` groups (4; each of
`--episodes` episodes, else the algorithm's group size) of the rows the environment's curriculum would choose first, on
its channel, served by its base model with nothing trained (start `kind: check`). A group whose
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
| `progress` | the run | a step being taken moved on by a whole percent or a phase, and once its trainer returned | `step`, and `progress`: the trainer's [`Progress`](training.md#how-far-a-step-has-got), none once it returned |

The [monitor](monitor.md)'s feed is one such hook.
