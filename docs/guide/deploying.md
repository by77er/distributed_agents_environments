# Deploy with the rollout command

The `rollout` command on one machine: setting up a cluster of one machine, asking for runs and evals, what a run is
made of, engines elsewhere, the gateway, Ray, and the platform on Kubernetes. This page is for whoever runs the
platform from a shell.

**Read first:** [Choose a setup](../deploy/setups.md). **Next:** [The cluster config and run settings](cluster.md).


A cluster is described once, in its cluster config ([the cluster config and run settings](cluster.md)): where the
ledger and the blob store are, the Ray cluster runs' jobs go to (or the RayJob each is made from, on Kubernetes), the
inference providers and trainers it offers, its sandbox pools and the environments it offers. A run is described by
its settings, chosen in the monitor's **New run** form or on the command line, often starting from a preset. Asking for
a run records a launch and submits the run's job; the job's driver builds the run from the cluster config and its
settings, and asks Ray for what it needs ([launching runs](../libraries/rollout-train/launching.md)). The command is
`rollout` (`rollout_train.cli`).

## On one machine

```bash
uv sync --all-extras                                   # the platform, with vLLM, the trainers and Tinker's SDK
cp deploy/clusters/example.toml ~/.config/rollout/cluster.toml   # then edit it for the machine
uv run rollout cluster check                           # what of it does not resolve here
uv run ray start --head --node-ip-address 127.0.0.1 --dashboard-host 127.0.0.1 --num-gpus 1 --temp-dir ~/.cache/ray
uv run rollout preset load deploy/chart/rollout/files/presets --cluster   # the presets shipped with the platform
uv run rollout monitor --cluster                       # the page over the cluster's ledger: http://localhost:8765
uv run rollout train minecraft_team.environment:environment --preset minecraft-one-gpu --name team-8
```

Ray's head keeps its sessions and spilled objects in its temporary directory, on disk (`/tmp` may be memory); `ray
stop` stops it. Ray's workers run in this environment: a run tells Ray not to start them through `uv run`, which would
build each a fresh environment without the extras.

## Asking for a run

```bash
uv run rollout train ENVIRONMENT --preset NAME [--name NAME]          # a training run, submitted and followed
uv run rollout train ENVIRONMENT --trainer local-lora --provider local-vllm --model Qwen/Qwen3-0.6B \
    --renderer rollout_qwen:qwen3 --set trainer.rank=16 --groups 40   # its settings without a preset
uv run rollout train ENVIRONMENT --preset NAME --settings run.toml --set trainer.learning_rate=3e-5 --check
uv run rollout train ENVIRONMENT --preset NAME --detach               # return once its job is submitted
uv run rollout train ENVIRONMENT --preset NAME --here                 # run its job in this process
uv run rollout eval SUITE --checkpoint diamonds --episodes 4          # an eval, by the run that made the checkpoint
uv run rollout eval SUITE --preset NAME --model Qwen/Qwen3-0.6B       # an eval of a base model
uv run rollout imitate --dataset solved-teams --preset NAME --start team-7:31
uv run rollout env check ENVIRONMENT                                  # its checks, and a scripted episode, here
uv run rollout env check ENVIRONMENT --preset NAME --groups 4         # and groups played by a model, as a check run
uv run rollout resume team-8 --cluster                                # resume a run: in place, or submitted again
uv run rollout resume team-7 --preset minecraft-one-gpu --cluster        # a run whose start records no providers
uv run rollout pause team-8 --cluster                                 # nothing new starts; what plays plays out
```

A command that asks for a run builds its settings in layers, each over the last: `--preset NAME[@N]` (the settings a run
of its kind takes), `--settings FILE` (TOML or JSON, dotted keys or tables), `--set KEY=VALUE` (repeatable; the value
read as JSON, then TOML, then as text), then its flags. The flags of every such command:

| Flag | Sets |
|---|---|
| `--name` | the run's name (by default a free name after what it plays: `words`, `minecraft_team`, `math on 31`) |
| `--model`, `--provider`, `--renderer`, `--channel` | `channels.CHANNEL.model`, `.provider`, `.renderer` of `--channel` (`policy`); not `imitate`'s, which renders with `channels.policy.renderer` from its preset or `--set` |
| `--trainer` | `trainer.provider` (not `eval`'s or `env check`'s) |
| `--cluster [PATH or NAME]` | the cluster config (alone or left out: the one this process was handed, else `ROLLOUT_CLUSTER`, else `~/.config/rollout/cluster.toml`) |
| `--check` | check the settings and say each refusal (`refused: KEY: why`) and note, then stop: exit 0, or 2 where something refuses |
| `--detach` | return once the job is submitted |
| `--here` | record the launch and run its job in this process, on the cluster config's Ray (`[ray] address`) |

Their own flags: `train ENVIRONMENT` takes `--groups`, `--groups-per-step` and `--seed`; `eval SUITE` takes
`--checkpoint` (a bookmark, `RUN:STEP`, `RUN` or an id; its channel's model, renderer, providers and budgets are those
of the run that made it, unless the settings say others) and `--episodes`; `imitate` takes `--dataset`, `--start`,
`--limit`, `--seed`, `--learning-rate`, `--warmup`, `--passes` and `--resume-optimizer`; `env check ENVIRONMENT` takes
`--row`, `--reply`, `--tools NAME=WHERE`, `--pools KIND=WHERE` (for the scripted episode), and `--groups` (4) and
`--episodes` for a check run, which it asks for only where the settings name a model's channel (a preset, `--provider`,
`--settings`, `--set`).

Settings that refuse are said with the setting each is about, and nothing is asked for. Otherwise the run is
submitted ([launching runs](../libraries/rollout-train/launching.md#asking-for-a-run)) and, unless `--detach`, followed:
the command prints each change of its launch (`submitted: waits for …`, `running`, `ended`), and an interrupt asks it to
stop. It exits 0 once the run ended, 1 if it failed, 130 if it was stopped.

`rollout COMMAND --help` lists each command's options. The other commands:

```bash
uv run rollout checkpoints --cluster                                    # every checkpoint: where it came from
uv run rollout bookmark diamonds first:20 --cluster                     # name the checkpoint run first made at step 20
uv run rollout rename first "diamonds, unguided" --cluster              # call a run something else (its id stays)
uv run rollout preset list --cluster                                    # presets ([presets](cluster.md#presets))
uv run rollout preset save faster --from-run first --set trainer.learning_rate=1e-4 --cluster
uv run rollout suite make words-v1 --environment ENVIRONMENT --seeds 1,2,3 --cluster   # an eval configuration
uv run rollout suite edit words-v1 --seeds 1,2,3,4 --cluster                          # its next version
uv run rollout report RUN_DIRECTORY ENVIRONMENT --watch                 # charts; posted to DISCORD_WEBHOOK_URL if set
uv run rollout pool --kind minecraft --cluster --host 0.0.0.0 --port 8710   # [sandboxes.minecraft], served on a machine of its own
uv run rollout tools FACTORY --directory DATA --port 8700               # a tool set on a machine of its own
uv run rollout gateway --cluster --listen 0.0.0.0:8900                  # a replica of the cluster's gateway
uv run rollout cluster check                                            # what of the cluster config does not resolve here
```

`rename` names a run again, by its name or its id; `pause` and `resume` pause a run and resume it, in place or by
submitting it again ([pausing and resuming](../libraries/rollout-train/training.md#pausing-and-resuming)); `bookmark`
names a checkpoint by any reference, and `checkpoints` lists them all ([checkpoints, runs and the
ledger](../libraries/rollout-train/checkpoints.md#the-command-line)). `suite` makes, edits and lists suites
([evals](../libraries/rollout-train/evals.md)); `env check` checks an environment before anything trains on it
([checking an environment](../libraries/rollout-train/rollouts.md#checking-an-environment)); `report` and `imitate` are in
[reporting](../libraries/rollout-train/training.md#reporting) and [imitation](../libraries/rollout-train/training.md#imitation).

## What a run is made of

A run's job is `python -m rollout_train.jobs LAUNCH` ([the driver](../libraries/rollout-train/launching.md#the-driver)).
Its driver, from the cluster config and the run's settings:

- imports the environment, and checks the settings again with what it finds now;
- asks Ray for an engine host per replica of each channel on a `vllm` provider, and for the trainer, on its own node,
  colocated with the trained channel's engine hosts where the trainer's `colocate_with` names their provider;
- samples every channel through a gateway in its own process: engine hosts and servers elsewhere by checkpoint name,
  Tinker through engines in its process;
- plays the run's episodes with a runner in its own process, with the sandbox pools the environment's programs declare
  (from `[sandboxes]`: made in its process, or reached at their `url`), the tool sets of `[tools]` and the memory
  guards of `[guards]`;
- runs the loop of the run's kind, its bridges as Ray tasks.

While Ray has not given it what it asked for, the run waits: the driver beats as `run/RUN` saying what it waits for,
and its launch says it (the monitor shows both). Settings that can never run (a provider the cluster lacks, a rank above
a provider's, more GPUs than the cluster has) are refused before anything is asked for.

The run keeps its files on its driver's node under `[scratch]/runs/RUN`: the monitor's feed, the checkpoints in use,
fetched bases. Everything else is in the ledger and the blob store: a run on any node, a monitor anywhere.

## What can change while a run goes

Some of a run's settings can change between two steps without breaking it, and are taken from its next step on:
`groups_per_step`, `max_lag` (how many checkpoints behind the newest a turn of the trained channel may begin), the evals
(`evals.suite`, `evals.every`, `evals.episodes`; an edit of the suite, too, as a new version its name points to),
`limits.spend`, `share`, the numbers of the objective's components, and the settings its trainer takes between steps
(`trainer.learning_rate`, and for `rollout_lora`'s trainers `tokens_per_step`, `max_kl` and `max_gradient_norm`). The
rest are fixed when it starts. The monitor's run page changes the changeable ones and shows the fixed ones; what is
wanted is kept beside the ledger, and each step's record says the settings it used ([changing a running run's
settings](../libraries/rollout-train/training.md#changing-a-running-runs-settings)). A run submitted again starts from
its recorded settings and takes what is wanted at its next step.

## Runners

A run's driver plays its episodes with an [episode runner](../libraries/rollout-train/rollouts.md#a-runner) named
`run/RUN`, with `episodes_at_once` places and the run's pools. It claims episodes in the ledger and records them there,
with their trajectories and events in the blob store; it claims an episode only while the pools of its sandboxes have
room for them, and leases them under the claim. Started again, it takes its fence anew, and what it had claimed is open
to be played again ([a runner started again](../libraries/rollout-train/rollouts.md#a-runner-started-again)).

Each runner beats every 15 seconds ([heartbeats](../libraries/rollout-train/rollouts.md#heartbeats)): its host, its
machine's memory, GPUs and disk, what each channel serves and how fast, and how full its pools are. A runner that stops
beating for 90 seconds is taken to be gone. A run's start records where its blob store is (its kind and settings; a
store's credentials come from its environment and are never written), so a monitor anywhere reads the run's finished
episodes back.

## Engines on other machines

A channel's provider may be vLLM servers this cluster does not start (`kind = "vllm-servers"`, with their `addresses`
and an optional `via`), or pods on RunPod (`runpod-inference`). A run samples them by checkpoint name; a follower beside
each server loads what the run's serving records say. Three roles meet through the ledger, the heartbeats beside it and
the blob store, wherever each runs:

| Role | Does |
|---|---|
| The run's loop | writes down, under the run's fence, what each of its channels should serve: the checkpoint, its depth and kind, and the files the engines load ([what a channel should serve](../libraries/rollout-train/channels.md#what-a-channel-should-serve)) |
| A follower beside each server | loads what the run says from the blob store into its vLLM server, as an adapter named by the checkpoint's id, and beats with what it serves (`python -m rollout_train.pods.inference`, told the run, the channel, the model and the server's address) |
| The run's gateway | asks the servers (or the router in front of them) for the checkpoint the run says, by name, and records what they sample, so tokens and logprobs are recorded exactly as sampled |

### The servers

Each engine is a stock vLLM OpenAI-compatible server, started on its machine with the model, LoRA, the logprobs of the
distribution sampled from, and adapters loaded and unloaded while it runs:

```bash
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True uv run vllm serve Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000 \
    --enable-lora --max-lora-rank 32 --max-loras 2 --logprobs-mode processed_logprobs --max-model-len 8192 \
    --api-key "$ROLLOUT_ENGINES_TOKEN"            # (optional: TLS with --ssl-certfile, --ssl-keyfile, --ssl-ca-certs)
```

`--max-loras` must cover the adapters every run it serves keeps loaded: `max_lag + 1` each (2 here, for one run with
`max_lag = 1`).

A request names the checkpoint it samples from as its `model`: the base model by its own name, a LoRA checkpoint by
its id. It completes the prompt's token ids (`/v1/completions` with `return_token_ids` and `logprobs`), so the tokens
and their logprobs come back exactly as sampled, the stop token among them, and the answer names the model that
sampled it. A request carries the session (`session_id`, which a router may keep to one server) and a name of its own
(`request_id`: the effect, the attempt and the phase of the turn). Nothing else of the protocol is ours: any router
(vLLM's production stack, llm-d, SGLang's router, Dynamo), proxy or tunnel can stand in front of the servers.

A vLLM server serves full weights only under the name it was started with, so a full checkpoint (or an adapter over
one) is not served this way: a run that trains every weight serves its channel on engine hosts of its own (a `vllm`
provider).

### How stale a sample may be

Each turn asks for the checkpoint the run last wrote down, by name. Where the server (or the router) does not have it
yet, the turn asks for the newest one before it that it has, no more than `max_lag` checkpoints behind (1: the
checkpoint before, which a server serves while it loads the newest); a server that answers that it does not have a
model is asked for the one before, and for the newest again at the next look (every two seconds). A turn waits while no
server has a checkpoint close enough, up to five minutes. Every token is stamped with the depth of the checkpoint its
answer names, and the trainer's importance weight corrects for the difference, as it does for the turns of an episode
that spanned a step. A runner claims a run's episodes only while a server has a checkpoint of each of its channels close
enough.

### Directly or through a router or proxy

A provider's `addresses` alone sends to the servers directly; `via` sends everything to one URL that passes requests
on. What makes either safe:

- **Requests carry what they need.** The checkpoint is the request's `model`; the session and the request's name
  travel with it. Nothing between needs to know more than vLLM's API, or keep a session to a connection.
- **Answers are checked.** The answer names the model that sampled it: the gateway refuses one that names another, and
  samples the turn again, so a proxy that sends a request to the wrong model, or answers from a cache, cannot make a
  recorded version wrong. What a server has is judged from its own listing (`/v1/models`) and answers, never from the
  proxy.
- **A retry is safe.** A request sent again through a proxy may be sampled again by the server; the gateway keeps the
  answer it was given, under the turn's effect, once.
- **Authentication is the provider's `auth`.** A bearer token (vLLM's `--api-key`, or the proxy's own) or mutual TLS
  with the cluster's CA and client certificate; a server reached by its address alone is known by the SPIFFE identity
  its certificate carries ([auth](cluster.md#inference-providers)).

### Pods on RunPod

Pods rented on RunPod run an image of their own: `deploy/images/inference` (a stock vLLM server, the follower beside
it, Envoy in front), `deploy/images/trainer` (the training service a run reaches with
`rollout_train.pods.RemoteTrainer`), or `deploy/images/host` (both on one GPU). A run leases the pods its RunPod
providers give it when it starts and releases them when it ends; a released pod stays warm for the next run, and
`rollout pods reap` deletes what no run holds. Each pod is reached at a public TCP port over mutual TLS, with
certificates from the cluster's step-ca, and reaches the cluster through the ledger service with a token of its own.
What a deployment provides is in [GPU pods on RunPod](../deploy/providers.md#gpu-pods-on-runpod); how the pieces fit
and the security model in [RunPod pods as inference and training providers](../research/runpod-providers.md).

## The gateway

Each run's driver samples its channels through a gateway in its own process, over the run's engine hosts, servers and
engines, and serves it on its node for the harnesses its runner starts ([a runner served by the
gateway](../libraries/rollout-train/gateway.md#a-runner-served-by-the-gateway)).

`rollout gateway --cluster` serves a replica of the cluster's [gateway](../libraries/rollout-train/gateway.md): it
samples every channel a run's start names on the cluster's providers whose servers answer vLLM's API at their endpoints
(`vllm-servers`, and a `vllm` provider's `listen`) or are the pods the run leases (`runpod-inference`, `runpod-host`),
for programs and harnesses that hold a signed
key, and records every turn in the cluster's ledger and blob store before it replies. Replicas keep no session, so as
many as wanted stand behind one proxy, and any of them may stop at any moment.

- **Keys.** The secrets are read from `[gateway] keys_file` or `keys_env`, else from the environment
  (`ROLLOUT_GATEWAY_KEYS`, or a file named by `ROLLOUT_GATEWAY_KEYS_FILE`), and never from the ledger. Whoever mints
  keys holds the same secrets.
- **Where it listens.** `--listen HOST:PORT`, else `[gateway] listen`.
- **TLS and proxies.** A proxy in front terminates TLS and checks its own credentials; `--proxied` names the addresses
  whose `X-Forwarded-*` headers are trusted. `--certificate` and `--private-key` serve TLS from the replica itself.
- **Health.** `/healthz` answers while the process serves; `/readyz` answers 200 once the ledger and the blob store
  answer, and 503 otherwise.

## Ray

Every run is a Ray job. On one machine, the job goes to the job server `[ray] jobs` names, on the head `ray start`
started: Ray places its driver on a node with a CPU free and supervises it. The driver joins the Ray cluster at
`[ray] address` (`auto`: the one the job runs on; name its GCS address where a machine runs more than one Ray cluster),
and asks it for its actors: the trainer with its GPUs on the driver's node, an engine host with its share of a GPU per
replica; a bridge runs as a Ray task asking for the CPUs and memory it declares, or what `[bridges."NAME"]` says
([bridges](../libraries/rollout-train/checkpoints.md#bridges)). The actors are the job's: Ray ends them when the job
ends, so a stopped or failed run leaves no engine behind. `ray job list --address http://127.0.0.1:8265` lists the runs'
jobs.

## On Kubernetes

The chart `deploy/chart/rollout` runs the platform in one namespace, by role: the stores (Postgres and an S3 gateway),
a long-lived Ray cluster (KubeRay: a head, and a GPU group and a CPU group the autoscaler starts from zero), where the
monitors check environments imported from git; the gateway and the monitors (Deployments, each behind a Service and an
Ingress); and the cluster config, the RayJob template and the presets in a ConfigMap every pod mounts at
`/etc/rollout`.

- **Each run is a RayJob** with a Ray cluster of its own: the cluster config's `[kubernetes]` names the namespace and
  `files/rayjob.yaml`, the template (a head pod of the platform's image, sized from what the run needs within the
  template's limits, `backoffLimit` retries, removed `ttlSeconds` after it ends). With `kueue.enabled`, Kueue admits
  it whole once its queue's quota has room ([Kueue](../deploy/helm.md#kueue)). A monitor makes it when a run is asked
  for from its page, with its account (`templates/rbac.yaml`: create, get, list, watch and delete on `rayjobs`), reads
  its status, and deletes it to stop the run.
- **The presets** in `files/presets` are saved beside the ledger by a hook Job at every install and upgrade (`rollout
  preset load /etc/rollout/presets --cluster`): a new version only where a preset's newest says otherwise.
- **The gateway** runs `rollout gateway --cluster`.

Its cluster config names the ledger and the blob store at their addresses in the cluster, and every pod mounts one
volume for run directories and other local state. On a single node it can run in K3s; `deploy/k3s/README.md` installs it
(its Volumes section says where the volumes are kept, how they are kept when their claims go, and to back them up
before uninstalling K3s).

## Stopping

A run can be paused instead: its driver stays, with its engines and GPU, and starts nothing new until it is resumed
(`rollout pause`, `rollout resume`, or the monitor's buttons). A run stopped, failed or lost is resumed by submitting
it again with its recorded settings ([pausing and resuming](../libraries/rollout-train/training.md#pausing-and-resuming)).

A run asked to stop (the monitor's **Stop**, an interrupt to `rollout train`, a termination) stops what it started: its
episodes, its tool sets, a step in progress, and its actors with its job.
