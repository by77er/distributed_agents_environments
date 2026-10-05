# Launching runs

For people who start runs and deploy the platform: how a run is asked for, the job it becomes, what its driver starts
and claims, and what a cluster offers.

**Read first:** [The training loop](training.md) and [the cluster config and run settings](../../guide/cluster.md).
**Next:** [Checkpoints, runs and the ledger](checkpoints.md).

A run is asked for with its settings ([run settings](../../guide/cluster.md#run-settings)) on a cluster
([the cluster config](../../guide/cluster.md)), wherever it is asked: the monitor's **New run** form, `rollout train`,
`rollout eval`, `rollout imitate`, `rollout env check`, `rollout resume`. One function starts every run's job, and the
job's driver builds the run from its settings and claims what it needs.

| Module | What it does |
|---|---|
| `rollout_train.launching` | A run's settings in layers (`settled`), the facts validation reads (`environment_facts`, `ledger_facts`), the check itself (`checked`), and what a cluster offers (`offers`) |
| `rollout_train.submitting` | `submit`: records a launch and starts its job, as a Ray job or a RayJob; `followed`, `stopped` |
| `rollout_train.launches` | The launches table: what was asked, the run it is, the job it became, its state |
| `rollout_train.jobs` | The job's entrypoint: the run built from the cluster config and its settings (`Run`), and the loop of its kind |
| `rollout_train.demand` | What a run's scheduled parts need (`demand`), the placement group that reserves them (`reserve`), and the pods of its Ray cluster (`pods`) |

## Asking for a run

`submit(settings, cluster, ledger, preset=…, resumes=…)`:

1. records a launch (`ask`): what was asked (`Asked`: the kind, the name, the settings as given, the preset they came
   from as `NAME@N`, and the run it resumes, if it resumes one), and the run it is: registered in the registry under its
   name, or the run it resumes. A name another run has is refused;
2. starts the run's job (`start`) where the cluster config says (`backend_of`): Ray's job API, or a RayJob on
   Kubernetes. A job that cannot be made fails the launch with why.

Whoever submits checks the settings first (`rollout_train.launching.checked`): the monitor refuses them with each
refusal tied to its setting, and the CLI prints them and exits 2. The driver checks them again before it claims
anything.

### The job

Every run's job runs

```bash
python -m rollout_train.jobs LAUNCH
```

in the interpreter its run starts in: its environment's own where the cluster config gives it one
(`[environments."NAME"] interpreter`, or a project's `.venv`), else `[ray] python` (`platform`: the `python` on the
job's `PATH`). Its runtime environment holds the cluster config as JSON (`ROLLOUT_CLUSTER_JSON`), so the job reads the
config it was submitted with. A run on a published environment ([publishing](../../guide/publishing.md)) is a job in
its version's Ray runtime environment, the config added to its environment variables: the driver imports the
environment from the version's source.

### A Ray job

On a cluster config without `[kubernetes]`, the job is a Ray job submitted to `[ray] jobs` (`RayJobs`): its submission
id `run-` and the launch's id in lowercase, its driver's CPUs as `entrypoint_num_cpus` ([what a run
needs](#what-a-run-needs)), and the launch, kind and name as its metadata.

### A RayJob

On a cluster config with `[kubernetes]`, the job is a RayJob custom resource (`RayJobResources`), made from the
template `[kubernetes] rayjob` names (a RayJob as YAML: its Ray cluster, image, volumes, retries) by `rendered`: its
name (`run-…`, as above), its namespace, its labels (`app.kubernetes.io/managed-by: rollout`, `rollout/launch`,
`rollout/kind`), its entrypoint and the driver's CPUs (`entrypointNumCpus`), its job id, its runtime environment
(`runtimeEnvYAML`) and its metadata; its Ray cluster sized from the run's demand (`sized`, below); and, with
`[kubernetes] queue`, Kueue's label (`kueue.x-k8s.io/queue-name`) and `suspend: true`. Everything else is the
template's. It is created through the API server
(`KubernetesApi`: `[kubernetes] api`, with the pod's service account token and CA), which the account must allow:
create, get, list, watch and delete on `rayjobs` in `ray.io`. KubeRay starts a Ray cluster for it, runs the driver
there, and removes the cluster when the job ends; when the driver or its pod is lost, the template's `backoffLimit`
submits it again, and the run goes on from the ledger. The chart's template is `deploy/chart/rollout/files/rayjob.yaml`
([Deploying](../../guide/deploying.md#on-kubernetes)).

`sized` sizes the Ray cluster from the run's demand. Its head pod asks Kubernetes for what Ray schedules on it and
room for Ray's own processes (`HEADROOM`: 1 CPU, 2 GiB), and Ray starts with its CPUs and GPUs (`rayStartParams`
`num-cpus`, `num-gpus`, and `resources` for custom ones); a pod that holds no GPU asks for none. The template's limits
are the most one pod may have: a memory or CPU limit below the request is raised to it, and a GPU limit is the pod's
whole GPUs. Where the whole run is more than one pod's worth, the head holds the driver, the trainer's bundle and the
bridge's, and each engine host's bundle is a worker pod (`workerGroupSpecs`, `engines-0`, …, made from the head's
template, as many replicas as hosts of that size).

### Kueue

With `[kubernetes] queue` (the chart's `kueue.enabled`), Kueue admits each run's RayJob whole: the RayJob is made
suspended in that LocalQueue, and Kueue lets it start once its ClusterQueue's quota holds every pod it asks for (its
head, its workers, and the pod KubeRay starts to submit the job), so two runs never hold half of what each needs.
While Kueue holds it, its launch says so: `followed` reads the RayJob's `spec.suspend` and notes
`waits for admission by Kueue (queue runs)`, followed by the reason Kueue's Workload gives (its `QuotaReserved`
condition: the quota it waits for), which the monitor's launch tile shows. Reading the Workload needs `get` and `list`
on `workloads` in `kueue.x-k8s.io`.

## The launches table

A launch is ordinary state beside the ledger, changed in place: `launches.json` beside a ledger of files, the
`launches` table in a database ledger's database.

| Field | Says |
|---|---|
| `asked` | `kind` (`train`, `eval`, `imitate`, `check`), `name`, `settings`, `preset`, `resumes` |
| `run` | The run it is, by id |
| `job` | Its job: a Ray job's submission id, or a RayJob's name |
| `backend` | `ray` or `kubernetes` |
| `state` | Below |
| `detail` | Why it failed, how it ended, or what its run waits for |

Every change of a launch's state compares and sets: it is made only if the launch is in a state its writer expects,
and may go where it is sent (`MOVES`):

| From | To |
|---|---|
| `asked` | `submitted`, `running`, `failed`, `stopped` |
| `submitted` | `running`, `stopping`, `stopped`, `ended`, `failed` |
| `running` | `stopping`, `stopped`, `ended`, `failed` |
| `stopping` | `stopped`, `ended`, `failed` |

A launch is `asked` once recorded, `submitted` once its job is made, `running` once its driver starts (a driver may
start before its submitter notes the job), `stopping` once a stop is asked for; it ends `ended`, `failed` or
`stopped`. A finished launch goes nowhere, so a stop asked for while the job starts stays.

The driver notes on its launch that it runs, what it waits for, and how it ended. `followed(launch, launches,
cluster)` reads the job's status for a launch that is going and notes what the driver could not: a job that waits
(with the job server's or KubeRay's reason), one that runs before its driver said so, one that ended, failed (with the
end of its output or its RayJob's message) or was stopped, and a RayJob that is gone. The monitor reads every launch
going so each time it reads the launches, and `rollout train` while it follows one. `stopped(launch, launches,
cluster)` asks a launch to stop: one whose job is not made yet is `stopped` at once; a job going is asked to stop (a
Ray job stopped, a RayJob deleted), and its driver notes its run stopped on the way out.

A launch record that names a profile is read as run settings (its environment, start, bookmark, groups,
groups a step and seed, an eval's suite and episodes, and the settings it changed); `claimed` reads as `submitted`.

## The driver

`rollout_train.jobs.driven(launch, cluster)` reads the launch, notes it `running`, and runs it; `ran(run)` is what it
runs, which a test calls on a `Run` built directly:

1. **The environment.** The run's environment is imported, where the cluster config offers it (a published one from
   its version, in the runtime environment the job was given). One the cluster does not offer is refused, not
   imported.
2. **The check.** The settings are checked against the cluster config with what the driver finds now: the environment
   it imported, the ledger, and what the Ray cluster has free (a note, never a refusal: an autoscaled cluster has more
   than its nodes now). A refusal ends the run: its start records its settings, its end says the refusals, and the
   launch fails with them.
3. **What it claims** (`Run.start`): first the run's placement group, reserved whole ([what a run
   needs](#what-a-run-needs)), then in it, as actors the job owns, so they go with it:
   - each channel on a `vllm` provider: an engine host per replica (`run/RUN/engine/CHANNEL/N`,
     [engine hosts](channels.md#engine-hosts)), bound to the run's channel, asking for one CPU and the provider's GPUs
     per replica (half of them where the trainer is colocated with them) in its bundle, started from the model's
     `options` with its `context` as `max_model_len`;
   - the trainer (a training or imitate run): an actor (`run/RUN/trainer`, `TrainerActor`) on the driver's node, since
     a step's files are handed to it by path; a scheduled trainer asks for its GPUs (half of them where it is colocated
     with the trained channel's engine hosts, which then sleep while it steps: `Colocated`) and one CPU in its bundle,
     and a metered one (Tinker's) for nothing. It is made by the
     trainer's `implementation` with the model (`trainer.model`, else the trained channel's; for a run that starts from
     full weights, or an adapter over them, those weights fetched here), the trainer settings it takes, the objective
     the settings resolve to, and Tinker's project where the config names one. `TrainerClient` is the `Trainer` the
     loop steps over the actor.

   While the group or any of these waits for Ray, the driver beats as `run/RUN` (kind `run`, with what it waits for:
   each part, what it asked for, and an actor's state where Ray says) and notes it on its launch (`waits for
   run/RUN/engine/policy/0 (1 GPU, 1 CPU), run/RUN/bridge (2 CPUs, 1 GiB)`), every two seconds, until each is
   ready.
4. **The channels.** Every channel the settings name is sampled through a gateway in the driver's process
   ([the gateway](gateway.md)): a channel on engine hosts, or on servers at addresses (`vllm-servers`,
   `runpod-inference`: their `via` or their addresses, reached as their auth says), is a routed channel, sampled by
   checkpoint name from what each run's serving records say (its evals' and their parts' too); a channel on Tinker
   is sampled by engines in the driver's process; a channel on an `api` provider is sampled through its provider's
   endpoint, with the key the driver's environment has (`ApiChannel`, [hosted APIs](gateway.md#hosted-apis)), its
   turns never trained on. The trained channel's longest turn is the trainer's longest segment
   (`trainer.segment_tokens`, else the trainer's in the config).
5. **The runner.** An episode runner in the driver's process (`run/RUN`), with `episodes_at_once` places, over a
   runner whose harnesses reach the gateway on this node (a free port on `127.0.0.1`); a pool of each kind of sandbox
   the environment's programs declare, from the cluster's `[sandboxes.KIND]` (its provider made with the run's
   directory, `size` and its settings; named `KIND@RUN`; its leases beside the ledger, with a keeper); the tool sets of
   the cluster's `[tools]`; the memory guards of `[guards]`; the monitor's feed in the run's directory.
6. **The loop of its kind**: `train`, `evaluate`, `imitate` or `check` ([training](training.md), [evals](evals.md),
   [datasets](datasets.md#a-step-on-a-dataset), [checking an environment](rollouts.md#checking-an-environment)), given
   what was built. The trained channel's files are made by the bridges from the trainer's format to what the channel's
   first provider loads, each a Ray task ([bridges](checkpoints.md#bridges)); none, where its files are served as they
   are. A scheduled eval is the run `RUN-eval-STEP` (called `NAME-eval-STEP`), and a part of an eval of several
   `ID-PART`, played by the same runner.
7. **The way out**: what it started is stopped in reverse, the actors ended, and the launch noted `ended`, `failed`
   (with why) or `stopped`.

An eval with `limits.spend` is bounded by it: once what the eval and its parts spent on hosted APIs reaches the limit
(each turn's `spend`, over what the eval's earlier starts recorded), the gateway samples no more on them for it, and
the driver ends it (`SpendReached`): the eval and its parts end `failed`, saying what was spent and the limit.

The run's directory is `[scratch]/runs/RUN` on the driver's node: the monitor's feed, the checkpoints in use, fetched
bases. Its start records, beside what the loop records: the environment (and the published version), the blob store,
the directory, the run settings, the cluster's name, the launch and the job.

A run's channels on a `vllm` provider get engine hosts of the run's own (`Run.hosted`).

## What a run needs

`rollout_train.demand.demand(settings, cluster)` says what a run's scheduled parts need, from its settings and the
cluster config alone. A part is metered or scheduled as its provider's or trainer's `allocation` says: a metered part
(Tinker's trainer and sampler, a hosted API) is bounded by spend, rate limits and its concurrency, and is not in the
demand; a metered trainer's actor asks Ray for nothing and runs on the driver's node.

| Part | Asks for | Where |
|---|---|---|
| The driver | 1 CPU for the loop and its gateway, 1 CPU for each runner of `[runners] places` episodes (`episodes_at_once`), each sandbox pool's `size × cpus`; 2 GiB and each pool's `size × memory_gib` | the job's entrypoint (its CPUs as `entrypoint_num_cpus`) |
| The trainer (a scheduled one) | 1 CPU and its `gpus` (half where it shares the trained channel's card) | a bundle on the driver's node |
| Each engine host (a scheduled `vllm` provider, per replica) | 1 CPU, a replica's GPUs, `[placement.engines]` | a bundle of its own, or the trainer's where they share a card |
| The bridge (a training run, an eval of a checkpoint) | the largest bridge of the chain the run may run: its `cpus` and `memory_gib`, or `[bridges."NAME"]`'s | a bundle of its own |

Bridges run one at a time (a checkpoint is bridged before the next is served, and a chain's bridges in turn), so one
bundle the size of the largest holds them all. Channels on servers elsewhere (`vllm-servers`, RunPod pods) and on
Tinker ask the run's Ray cluster for nothing.

The driver reserves the bundles as one placement group (`reserve`, named `run/RUN`) before it starts any part, so a
run starts only with all of it reserved and never waits half-placed for a task Ray cannot place. The group is `PACK`:
Ray puts its bundles on as few nodes as hold them, all on one node where one has room, and spreads engine hosts over
nodes only where no one node holds them (`STRICT_PACK` would refuse a run larger than a node; `SPREAD` would split
one that fits). The trainer's bundle names the driver's node (`node:IP`). Each actor and bridge task asks for exactly
what its part counted, in its bundle (`placed`); a share of GPUs above one in a bundle is rounded up to whole GPUs, as
Ray takes fractions of one GPU only. On one machine, the same group is reserved in the local Ray. The group is removed
when the run ends.

For the acceptance run's shape (a Tinker trainer, one engine host of `local-vllm` with one GPU, the
`peft-from-tinker` bridge, `episodes_at_once` 6 with 8 places a runner):

| Part | Asks for |
|---|---|
| The driver | 2 CPUs, 2 GiB |
| `engine/policy/0` | 1 GPU, 1 CPU |
| `bridge` | 2 CPUs, 1 GiB |
| The run's Ray cluster | 5 CPUs, 3 GiB, 1 GPU (its pod asks Kubernetes for 6 CPUs, 5 GiB, 1 GPU) |

Validation (`capacity`, [validation](../../guide/cluster.md#validation)) refuses a run whose Ray cluster would ask for
more than the cluster config's `[capacity]` (on Kubernetes with Kueue, the queue's quota, which the chart writes
there), with each number; one that fits but finds less free now waits, with a note.

## What a cluster offers

`offers(cluster, ledger, beats)` is what the New run form chooses from (`GET /api/offers`):

| Field | Holds |
|---|---|
| `cluster`, `kinds`, `submits` | The cluster's name, the kinds of run, and where jobs go (`ray` or `kubernetes`) |
| `environments` | The cluster config's (`environment`, `python`: `platform` or `project`), then every published version beside the ledger (`environment` as `NAME@VERSION`, `name`, `source`, `commit`, `imported`, `sandboxes`); each with the renderer `families` runs and presets on it named for their trained channel |
| `trainers` | Each trainer: `name`, `kind`, `produces`, `format`, `models`, `gpus`, `colocate_with`, `segment_tokens`, `cost`, `families` (the objective families it takes), `allocation` (`metered` or `scheduled`), `concurrency`, `weights` (`lora` or `full`: what it trains), and its `settings` (each `key`, `types`, `default`, `changeable`) |
| `inference` | Each provider: `name`, `kind`, `gpus`, `replicas`, `allocation`, `concurrency`, `capabilities`, `weights` (those it serves a run's checkpoints as: `lora` with adapters, `full` with full-weight reload), and its `models` (each `model`, `context`, `base`, `max_lora_rank`, `cost`, the `renderers` that say they render it, or the model it was quantized from, and their `families`; none for a provider that renders messages itself) |
| `pairs` | Each trainer and provider: the `bridge` chain's names, or none and why it is `refused` (no bridge, or a provider that cannot serve the trainer's weights) |
| `sandboxes` | Each pool's `size` and `provider` |
| `presets` | Each preset's newest version: `name`, `version`, `id`, `settings`, `note`, `saved` |
| `capacity` | The GPUs the machines that beat now have, and those idle (under a twentieth of their memory used), by machine; none where no beat says |
| `objectives` | The objective's `families`, its `presets` (each `name`, `family`, `source`, `says` and its family's `components` with their values) and the `components` (each `key`, `types`, `families` that accept it, `changeable`, `says`, `choices`, `least`, `above`) |
| `schema` | The keys a training run takes, each `key`, `types`, `default`, `changeable`, `says`, `choices`, `least` |

`examined(settings, cluster, ledger)` is `checked` with what it found beside the findings: the environment's facts, one
step's estimated spend on the run's metered parts (`spend_of`) and what the run trains (`weights_of`). The monitor's
`POST /api/launches/check` answers with it. Both the check and the submission first say what follows from the settings
(`completed`): a training run's `weights` (its trainer's, where the settings do not say), and each channel's renderer
where it names a model and no renderer and exactly one declared renderer renders that model (`with_renderers`). So a
run's start records both. The `renderer` rule refuses a channel sampling tokens whose model no renderer renders, or
several do with none said, or a renderer said that says it renders other models; a hosted API takes messages and
needs none.
