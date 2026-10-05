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
id `run-` and the launch's id in lowercase, one CPU for its driver, and the launch, kind and name as its metadata.

### A RayJob

On a cluster config with `[kubernetes]`, the job is a RayJob custom resource (`RayJobResources`), made from the
template `[kubernetes] rayjob` names (a RayJob as YAML: its Ray cluster, image, volumes, retries) by `rendered`: its
name (`run-…`, as above), its namespace, its labels (`app.kubernetes.io/managed-by: rollout`, `rollout/launch`,
`rollout/kind`), its entrypoint and the driver's CPU (`entrypointNumCpus`), its job id, its runtime environment
(`runtimeEnvYAML`) and its metadata. Everything else is the template's. It is created through the API server
(`KubernetesApi`: `[kubernetes] api`, with the pod's service account token and CA), which the account must allow:
create, get, list, watch and delete on `rayjobs` in `ray.io`. KubeRay starts a Ray cluster for it, runs the driver
there, and removes the cluster when the job ends; when the driver or its pod is lost, the template's `backoffLimit`
submits it again, and the run goes on from the ledger. The chart's template is `deploy/chart/rollout/files/rayjob.yaml`
([Deploying](../../guide/deploying.md#on-kubernetes)).

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
   it imported, the ledger, and the GPUs free in the Ray cluster (a note, never a refusal: an autoscaled cluster has
   more than its nodes now). A refusal ends the run: its start records its settings, its end says the refusals, and
   the launch fails with them.
3. **What it claims** (`Run.start`), asked of Ray as actors the job owns, so they go with it:
   - each channel on a `vllm` provider: an engine host per replica (`run/RUN/engine/CHANNEL/N`,
     [engine hosts](channels.md#engine-hosts)), bound to the run's channel, asking for the provider's GPUs per
     replica (half of them where the trainer is colocated with them), started from the model's `options` with its
     `context` as `max_model_len`;
   - the trainer (a training or imitate run): an actor (`run/RUN/trainer`, `TrainerActor`) on the driver's node, since
     a step's files are handed to it by path, asking for the trainer's GPUs (half of them where it is colocated with
     the trained channel's engine hosts, which then sleep while it steps: `Colocated`) and one CPU. It is made by the
     trainer's `implementation` with the model (`trainer.model`, else the trained channel's; for a run that starts from
     full weights, or an adapter over them, those weights fetched here), the trainer settings it takes, the objective
     the settings resolve to, and Tinker's project where the config names one. `TrainerClient` is the `Trainer` the
     loop steps over the actor.

   While any of these waits for Ray, the driver beats as `run/RUN` (kind `run`, with what it waits for: each actor, what
   it asked for, and its state where Ray says) and notes it on its launch (`waits for run/RUN/engine/policy/0 (1 GPU:
   pending creation)`), every two seconds, until each is ready.
4. **The channels.** Every channel the settings name is sampled through a gateway in the driver's process
   ([the gateway](gateway.md)): a channel on engine hosts, or on servers at addresses (`vllm-servers`,
   `runpod-inference`: their `via` or their addresses, reached as their auth says), is a routed channel, sampled by
   checkpoint name from what each run's serving records say (its evals' and their parts' too); a channel on Tinker
   is sampled by engines in the driver's process. A channel on an `api` provider is not sampled by a run. The trained
   channel's longest turn is the trainer's longest segment (`trainer.segment_tokens`, else the trainer's in the
   config).
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

The run's directory is `[scratch]/runs/RUN` on the driver's node: the monitor's feed, the checkpoints in use, fetched
bases. Its start records, beside what the loop records: the environment (and the published version), the blob store,
the directory, the run settings, the cluster's name, the launch and the job.

A run's channels on a `vllm` provider get engine hosts of the run's own (`Run.hosted`).

## What a cluster offers

`offers(cluster, ledger, beats)` is what the New run form chooses from (`GET /api/offers`):

| Field | Holds |
|---|---|
| `cluster`, `kinds`, `submits` | The cluster's name, the kinds of run, and where jobs go (`ray` or `kubernetes`) |
| `environments` | The cluster config's (`environment`, `python`: `platform` or `project`), then every published version beside the ledger (`environment` as `NAME@VERSION`, `name`, `source`, `commit`, `imported`, `sandboxes`) |
| `trainers` | Each trainer: `name`, `kind`, `produces`, `format`, `models`, `gpus`, `colocate_with`, `segment_tokens`, `cost`, `families`, and its `settings` (each `key`, `types`, `default`, `changeable`) |
| `inference` | Each provider: `name`, `kind`, `gpus`, `replicas`, `shared`, `capabilities`, and its `models` (each `model`, `context`, `base`, `max_lora_rank`, `cost`, the `renderers` runs and presets named for it so far and their `families`) |
| `pairs` | Each trainer and provider: the `bridge` chain's names, or none and why it is `refused` |
| `sandboxes` | Each pool's `size` and `provider` |
| `presets` | Each preset's newest version: `name`, `version`, `id`, `settings`, `note`, `saved` |
| `capacity` | The GPUs the machines that beat now have, and those idle (under a twentieth of their memory used), by machine; none where no beat says |
