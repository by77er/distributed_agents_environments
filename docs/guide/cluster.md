# The cluster config and run settings

Code: `rollout_train.cluster`, `rollout_train.run_settings`, `rollout_train.validation` · See
[launching runs](../libraries/rollout-train/launching.md), [API reference](reference.md#rollout_traincluster)

For whoever describes a cluster or asks for runs on one: every section of the cluster config, the kinds of provider
and trainer, bridges, every run setting, presets, and the one check of a run.

**Read first:** [Choose a setup](../deploy/setups.md). **Next:**
[Launching runs](../libraries/rollout-train/launching.md).

A cluster is described once, in a cluster config: where the ledger and the blob store are, the inference providers
and trainers it offers, its sandbox pools, the environments it offers and the Python each runs in. A run is described
by its settings: the environment, the trainer, the provider of each channel, and the numbers each takes. Settings can
be saved as presets, and one pure function checks a run's settings against a cluster.

This page describes what these modules provide:

- `rollout_train.cluster`: the cluster config, found and read strictly into a `Cluster`;
- `rollout_train.providers`: the kinds of inference provider and trainer, their capabilities, how each is reached;
- `rollout_train.bridges`: how a checkpoint in one format becomes files a provider loads, and the bridges' tasks;
- `rollout_train.run_settings`: the schema of a run's settings, how they are given, and their recorded copy;
- `rollout_train.presets`: named, versioned settings kept beside the ledger;
- `rollout_train.validation`: `check`, with its rule table.

They are declarations, but for the bridges' tasks: none of the rest starts, imports or reaches anything. What opens
and reaches things from them:

- `rollout_train.stores.Stores.open(cluster)` opens the ledger and the blob store the config names
  ([the stores](#the-stores));
- `rollout_train.bridges.bridged` and `on_ray` run a chain of bridges, here or as Ray tasks ([bridges](#bridges));
- `rollout_train.launching` gathers what validation reads and checks a run's settings
  ([asking for a run](#asking-for-a-run));
- `rollout_train.submitting.submit` records a launch and starts the run's job, and `rollout_train.jobs` is what that
  job runs ([launching](../libraries/rollout-train/launching.md));
- `rollout cluster check` reads the config and says what of it does not resolve on this node;
- the commands over a ledger (`rename`, `bookmark`, `pause`, `resume`, `checkpoints`, `suite`, `dataset`, `merge`,
  `preset`) take the cluster's with `--cluster`, in place of `--ledger`;
- `rollout preset` lists, shows, saves, loads and deletes [presets](#presets);
- the commands that ask for runs (`train`, `eval`, `imitate`, `env check`) take run settings in layers and the
  cluster config ([the command line](deploying.md#asking-for-a-run)).

## The cluster config

### Where it is

`rollout_train.cluster.find` looks in this order:

1. `--cluster PATH` or `--cluster NAME`, where `NAME` is `~/.config/rollout/clusters/NAME.toml`;
2. the `ROLLOUT_CLUSTER` environment variable, a path or a name;
3. `~/.config/rollout/cluster.toml`.

A file it names that is not there is an error, which says where it looked. `load(path)` reads the file into a
`Cluster`. An unknown key is an error, and so are:

- an unknown kind;
- a trainer whose `colocate_with` is not a `vllm` provider;
- a provider reached with no auth at an address that is not this machine;
- a provider reached over mutual TLS when the cluster has no `[tls]`.

A run's job is handed the config it was submitted with, as JSON in `ROLLOUT_CLUSTER_JSON`; `located` reads that
first, and finds the file as above otherwise.

`deploy/clusters/example.toml` is a config for one machine with one 16 GB GPU: SQLite, files, vLLM engines, the LoRA
trainer, Tinker, the Minecraft worlds, the gridworld and GSM8K in its own Python. `deploy/chart/rollout/files/cluster.toml`
is the chart's, for a Kubernetes cluster ([on Kubernetes](deploying.md#on-kubernetes)).

```toml title="cluster.toml"
name = "home"                                 # Ray namespace rollout-home; what runs record as where they ran

[ray]
address = "auto"                              # the Ray cluster a run's driver joins (auto: the one Ray finds)
jobs = "http://127.0.0.1:8265"                # its job server: where runs' jobs are submitted
temp_dir = "~/.cache/ray"                     # on disk: /tmp may be memory

[ledger]
url = "sqlite:///~/.cache/rollout/ledger.db"  # or url_env = "ROLLOUT_LEDGER_URL", for a URL with a password

[blobs]
kind = "files"                                # or "module:name" of a store, with its settings beside it
directory = "~/.cache/rollout/blobs"

[gateway]
url = "http://127.0.0.1:8830"
keys_file = "~/.config/rollout/gateway.keys"  # or keys_env

[guards]
runs_gib = 6
training_gib = 4

[inference.local-vllm]
kind = "vllm"
gpus = 1
[inference.local-vllm.models."Qwen/Qwen3.5-4B"]
context = 8192
options = { gpu_memory_utilization = 0.78, max_num_seqs = 20, max_lora_rank = 64 }

[inference.tinker]
kind = "tinker"                               # auth vendor: Tinker's SDK reads TINKER_API_KEY
[inference.tinker.models."Qwen/Qwen3.5-4B"]
context = 65536
cost = { input = 0.33, cached_input = 0.066, output = 1.005 }   # dollars per million tokens

[trainers.local-lora]
kind = "lora"
gpus = 1
colocate_with = "local-vllm"                  # the engines sleep while it steps
models = ["Qwen/Qwen3.5-4B"]
segment_tokens = 8000

[trainers.tinker-lora]
kind = "tinker"
models = ["Qwen/Qwen3.5-4B"]
segment_tokens = 32768
cost = { train = 0.737 }                      # dollars per million tokens trained

[sandboxes.minecraft]
provider = "minecraft_team.worlds:worlds"
size = 6

[environments."rollout_verifiers.environments:gsm8k"]
project = "~/Code/distributed_agents_environments/implementations/rollout-verifiers"
```

### Every section

| Section | Fields | Notes |
|---|---|---|
| top | `name` | Required; lowercase letters, digits and `-` |
| `[ray]` | `address` (`auto`), `jobs`, `temp_dir`, `memory_threshold`, `python` (`platform`) | `address`: the Ray cluster a run's driver joins, as its GCS's `host:port`; `auto` is the one Ray finds (in a Ray job, the cluster the job runs on). Name it where a machine runs more than one Ray cluster. `jobs`: the job server runs' jobs are submitted to. `python`: the interpreter a run's job starts in (`platform`: `python` on the job's `PATH`) |
| `[kubernetes]` | `namespace`, `rayjob`, `api` (`https://kubernetes.default.svc`), `queue` | With it, each run's job is a RayJob made from the template `rayjob` names (relative to the config file's directory), sized from the run's demand, in `namespace`, through the API server `api` with the pod's service account ([launching](../libraries/rollout-train/launching.md#a-rayjob)). `queue`: the Kueue LocalQueue that admits each RayJob whole; it is made suspended, and starts once admitted ([Kueue](../libraries/rollout-train/launching.md#kueue)) |
| `[capacity]` | `cpus`, `memory_gib`, `gpus` | The most the cluster schedules for one run (with Kueue, the queue's quota). A run whose Ray cluster would ask for more is refused, with the numbers ([what a run needs](../libraries/rollout-train/launching.md#what-a-run-needs)); each is unbounded where it is not said |
| `[ledger]` | `url`, or `url_env` / `url_file`; `token_env` / `token_file`; `public` | A URL holding a password is refused: name it instead. `url` is a database's, or the ledger service's (`https://…`), which needs the platform's token (`token_env`). The ledger service checks tokens against the same token. `public`: where pods outside the cluster reach the ledger service ([the ledger over HTTP](../libraries/rollout-train/checkpoints.md#the-ledger-over-http)) |
| `[blobs]` | `kind` (`files` or `module:name`), `directory` or the store's settings; `access_key_id_env`, `secret_access_key_env` | A setting that looks like a credential is refused. The store's key is its environment's, or read from the two variables it names |
| `[stores.NAME]` | as `[blobs]`, and `reader = { access_key_id_env, secret_access_key_env }` | A blob store beside the default: an R2 bucket that RunPod's pods reach. A run whose trainer or servers are RunPod's writes its blobs to the store its RunPod providers name (`store`); a reader finds each blob in the store its reference names. `reader` names the read-only key inference pods are given; trainer pods get the store's own |
| `[scratch]` | `directory` | Node-local |
| `[tls]` | `ca`, `certificate`, `key`, `identity` (`spiffe://rollout/gateway`) | The cluster's CA, and the client certificate it presents; paths |
| `[gateway]` | `url`, `listen`, `replicas`, `keys_file` / `keys_env`, `lifetime` | |
| `[monitor]` | `listen`, `feed_episodes` | |
| `[runners]` | `places` | |
| `[guards]` | `runs_gib`, `training_gib` | |
| `[inference.NAME]` | `kind`, `auth`, `gpus`, `replicas`, `allocation`, `concurrency`, `models`, and the kind's own | Below |
| `[trainers.NAME]` | `kind`, `auth`, `models`, `segment_tokens`, `gpus`, `colocate_with`, `cost`, `costs`, `allocation`, `concurrency`, and the kind's own | Below |
| `[sandboxes.KIND]` | `provider`, `python`, `size`, `cpus`, `memory_gib`, `pools`, the provider's settings | |
| `[tools.NAME]` | `url`, `auth` | Tool sets served elsewhere |
| `[environments."NAME"]` | `python = "platform"` or `project = PATH`; `interpreter` | A relative project is from the config file's directory. A run on it starts in `interpreter`, by default `PROJECT/.venv/bin/python` for a project and the platform's for `python = "platform"` |
| `[placement.ROLE]` | `resources` | Roles: gateway, monitor, runners, pools, engines, trainers, workers, bridges |
| `[bridges."NAME"]` | `cpus`, `memory_gib` | Overrides what a bridge declares |

A model a provider offers (`models."MODEL"`) has a `context`, and optionally a `base` (the model it was quantized
from), a `cost` table (dollars per million tokens by class: `input`, `cached_input`, `output`, `thinking`; or
`hour`; cached input is priced as input and thinking as output where the table does not say) and `options` (what its
engines are started with; `max_lora_rank` is the highest adapter rank it loads, and a training run's engines are
started with room for its own adapters only, `trainer.rank` times the bridge's rank factor, since a larger rank costs
cache; vLLM takes the next rank it accepts (1, 8, 16, 32, 64, 128, 256, 320 or 512); a hosted API's model says what it
takes, [below](#hosted-apis)).

A trainer's `cost` is dollars per million tokens trained (`train`, every token of each trained segment: its prompts and
what was sampled) or per `hour`. Where the price depends on the model, `costs` gives each model its own table
(`costs = { "Qwen/Qwen3.5-9B" = { train = 1.463 } }`), and `cost` covers the rest. A training run's estimated spend
(`spend_of`) is one step's on its metered parts: every trained token at a metered trainer's price for the model, and
the sampled and prompt tokens at the dearest metered provider's prices, prompts uncached. An eval's is the whole
eval's: every episode of the suite's starts, each turn's thinking and answer budgets sampled and its prompt read at the
dearest metered provider of the channel it plays. Their tokens are the episodes (for a step, the groups it waits for times
the episodes of a group), times the turns an episode plays and the samples a turn takes, as the environment's
description says them (`turns`, `samples_per_turn`: every agent of a team samples each turn), each sample its
thinking and answer budgets and its `prompt_tokens`. A training run's step on RunPod's pods costs each pod's `price` for
as long as a step of the same trainer and model took here lately (a host's pod once); where no such run made three
checkpoints, the estimate says it is not known yet.

**Metered or scheduled.** Each inference provider and trainer says how it is allocated (`allocation`), by default as
its kind is: `tinker` and `api` are `metered`; `vllm`, `vllm-servers`, `runpod-inference`, `runpod-host`,
`runpod-trainer`, `lora` and `full` are `scheduled`. RunPod's pods are leased by the runs that use them, outside the
capacity rule and Kueue's quota; a provider's `max_pods` caps them. A metered one is always available: a run is bounded by its `limits.spend`, the provider's rate
limits and its `concurrency` (requests sent at once; none: unbounded), which only a metered one takes. A scheduled
one is capacity a run is placed on: its GPUs count toward the capacity rule. One that asks for the cluster's GPUs is
scheduled.

### Secrets

A secret appears only by name: a key `NAME_env` (an environment variable) or `NAME_file` (a file). A key that looks
like a secret but holds a value (`api_key = "sk-…"`, `token = "…"`) is refused, and so is a ledger URL with a
password. So the parsed `Cluster` holds no secret: its `repr` and its JSON (`Cluster.described`, which `parsed` reads
back for a job or an actor) are safe to show. A `Secret` is resolved where it is used, at the moment it is needed
(`Secret.resolve`). `Cluster.secrets()` lists every reference. `inspect(cluster)` says, on this node, which
references do not resolve (by name, never by value) and which environment projects have no `uv.lock`.

### The stores

`Stores.open(cluster)` opens, on this node:

- **the ledger** `[ledger]` names, a database (`DatabaseLedger`): `sqlite:///…` (`~` is the home directory) or
  `postgresql://…`; or the ledger service (`HttpLedger`), `https://…`, with the platform's token. Where the config
  names the URL (`url_env`, `url_file`), it is read there and then; a name that is not set here is an error that says
  the name, and the URL itself is never printed;
- **the blob store** `[blobs]` names: files in `directory`, or `kind = "module:name"` called with the table's other
  settings. An S3 store (or any S3-compatible service, such as versitygw or R2) is
  `kind = "rollout_s3:S3BlobStore"` with `bucket`, and optionally `prefix`, `endpoint_url` and `region`; its
  credentials come from the node's `AWS_*` environment, or from the variables it names (`access_key_id_env`,
  `secret_access_key_env`), never from the config. `Stores.open(cluster, store=NAME)` writes to `[stores.NAME]`
  instead. Every blob reference says its store in its URI (`s3://BUCKET/PREFIX…`); a checkpoint's files are read from
  the store that holds them, which a run's start names, so a process that reads them holds every store's key.

`Stores.location` is where the blob store is, as any process opens it: what a run's start records. The stores beside
the ledger are reached through it: `checkpoints`, `registry` and `presets`.

```bash
rollout cluster check                      # the config found as above: its providers, trainers, pools, environments,
                                           # and each secret or project that does not resolve here (exit 1 if any)
rollout cluster check --cluster lab        # ~/.config/rollout/clusters/lab.toml
rollout checkpoints --cluster              # a command over a ledger, on the cluster's ledger
rollout bookmark diamonds first:20 --cluster lab
```

## Inference providers

| Capability | `vllm` | `vllm-servers` | `tinker` | `api` | `runpod-inference`, `runpod-host` |
|---|---|---|---|---|---|
| token-exact | yes | yes | yes | no | yes |
| sampled-token logprobs | yes | yes | yes | no | yes |
| prompt logprobs | yes | yes | not until checked live | no | yes |
| top-k logprobs | up to `max_logprobs` (20) | up to `max_logprobs` | not until checked live | no | up to `max_logprobs` |
| honours sampling | yes | yes | yes | no | yes |
| adapters by name | yes | yes | its own | no | yes |
| full-weight reload | yes | no | no | no | no |
| streaming | no | no | no | yes | no |
| loads | `peft`, `full` | `peft` | `tinker` | nothing | `peft` |
| bills | nothing | nothing | tokens | tokens | hours |
| allocation | scheduled | scheduled | metered | metered | scheduled (leased pods) |
| auth | `none`, `bearer`, `mtls` (default `none`) | `none`, `bearer`, `mtls` (must say) | `vendor` | `vendor`, `bearer` (default `vendor`: the key `api_key_env` names) | `mtls` (each pod's identity from its heartbeat) |
| its own fields | `engine` (what makes its engines, `module:name`: `rollout_vllm:VllmEngine` unless said), `listen`, `max_logprobs` | `addresses`, `via`, `loader`, `max_logprobs` | `project` / `project_env` | `endpoint`, `base_url`, `api_key_env` / `api_key_file` | its pods' table ([GPU pods on RunPod](../deploy/providers.md#the-providers-table)): `image`, `gpu_types`, `gpu_count`, `cloud`, `regions`, `price`, `max_pods`, `idle_stop`, `start_timeout`, `volume_gb`, `container_disk_gb`, `store`, `step_ca`, `secrets`, `api_key_env`, `memory_fraction` (and `sleep`, on a host), `max_logprobs` |

Tinker's prompt and top-k logprobs are declared as its SDK says (`Capabilities.unchecked`): the SDK takes prompt
logprobs and a top k at prompt and sampled positions, whose width Tinker's server bounds without the SDK saying how
far (declared as 20). Nothing relies on them, and `TinkerEngine` does not ask for them, until a live test confirms
them.

`max_logprobs` is the top-k a provider declares. A `vllm` provider's engines are started with it (`VllmEngine`'s
`max_logprobs`, which vLLM caps a request's top k at), so a model's `options` do not set it. A `vllm-servers`
provider's servers must have been started with at least as many (`--max-logprobs`, 20 unless given); a RunPod
provider's pods are started with it.

**RunPod's pods.** A `runpod-inference` pod serves a run's channel; a `runpod-host` pod serves it and takes the steps of
a `runpod-trainer` whose `colocate_with` names the host, on one GPU. A run leases its pods when it starts (warm ones
first), renews them, and releases them when it ends; a released pod stays warm for `idle_stop` seconds, and
`rollout pods reap` deletes what no run holds ([GPU pods on RunPod](../deploy/providers.md#gpu-pods-on-runpod)). A run
is refused more pods than a provider's `max_pods`, pods without `step_ca`, or a cluster whose pods cannot reach the
ledger service (`[ledger] public` and `token_env`).

**Auth.** `auth` is a kind or a table: `auth = "none"`, `auth = { kind = "bearer", token_env = "ENGINES_TOKEN" }`,
`auth = { kind = "vendor", key_env = "OPENAI_API_KEY" }`, `auth = { kind = "mtls", identity = "spiffe://…" }`.
`Auth.connection(tls)` gives the client's connection settings (`rollout_train.inference.remote.Connection`):

| Kind | Server verified by | Host name or identity | Client certificate | Token |
|---|---|---|---|---|
| `mtls` | the cluster's CA (`[tls] ca`) | the SPIFFE identity, where said (a pod's from its heartbeat); else the host name | `[tls] certificate`, `key` | |
| `bearer` | the system's CAs, or the cluster's with `trust = "cluster"` | the host name | | `token_env` / `token_file` |
| `vendor` | the vendor's SDK | | | the SDK reads its own key |
| `none` | only for addresses on this machine | | | |

**Several providers.** A channel may name several providers (`channels.NAME.providers`), shared by a routing rule
(`channels.NAME.routing`): `spill` fills the first and sends the rest to the next; `weighted` shares turns by
`channels.NAME.weights`. A hosted API shares a channel with no other provider.

### Hosted APIs

A provider of the kind `api` is a hosted model's API: OpenAI's Responses API (`endpoint = "rollout_openai:hosted"`)
or Anthropic's Messages API (`endpoint = "rollout_anthropic:hosted"`, [rollout-anthropic](../implementations/rollout-anthropic.md)).
It is metered, takes messages and returns text, with no exact tokens and no behaviour logprobs: what it samples is
never trained on. It serves evals of its models and the slots of a run that are not trained (a judge, a fixed
opponent); validation refuses it for a trained channel or one following it, and asks no renderer of its channels.

```toml
[inference.anthropic]
kind = "api"
endpoint = "rollout_anthropic:hosted"
api_key_env = "ANTHROPIC_API_KEY"             # the key, named: read where the channel samples, when it does
concurrency = 16                              # requests at once, from each gateway (and each run's driver)
[inference.anthropic.models."claude-sonnet-5-5"]
context = 1000000
cost = { input = 2.0, cached_input = 0.20, output = 10.0 }   # dollars per million tokens
options = { max_output_tokens = 128000, thinking = "adaptive", sampling = false, forced_tool_choice = false }
```

- **The key** is `api_key_env` (or `api_key_file`), else the auth's key or token. It is read the first time a channel
  on the provider samples; a provider whose key is not set refuses each turn with the variable's name, and
  `rollout cluster check` says it is missing. The chart reads the keys from the Secret `providers`
  ([Helm](../deploy/helm.md#provider-keys)).
- **`base_url`** is where the API is reached, where it is not the vendor's own (a proxy, a compatible server).
- **A model's catalog entry** gives its `context`, its prices (`cost`) and what it takes (`options`):
  `max_output_tokens`, the most one reply writes (by default its context); for an Anthropic model, how it thinks
  (`thinking`: `adaptive`, steered by an `effort`; `budget`, a number of thinking tokens; none), whether it takes
  temperature and top-p (`sampling`) and a forced tool choice (`forced_tool_choice`); for an OpenAI model, the
  `reasoning_effort` a turn is sampled with where its binding says none.
- **A turn** is sampled with the binding's temperature and top-p and its thinking and answer budgets (else the
  channel's): the budgets added up are the most it writes. Its record keeps its reply (text, tool calls, usage) and no
  tokens (`sampled_with` empty), with what it cost: its usage (input, cached input, output, thinking tokens) at the
  model's prices.
- **Rate limits and an overloaded API** (429, 5xx, 529) are asked again with backoff (`retry-after` honoured);
  credentials refused or a request rejected end the episode as failed, with the reason.
- **An eval's `limits.spend`** ends it once what it spent on hosted APIs reaches the limit, stopped with that reason.

The example config (`deploy/clusters/example.toml`) and the chart's offer `openai` and `anthropic`, their current
models priced from each vendor's pricing page, with the date the prices were checked.

## Trainers

| | `lora` | `full` | `tinker` | `runpod-trainer` |
|---|---|---|---|---|
| produces | `lora` | `full` | `lora` | as the trainer it runs (`trainer = "lora"` or `"full"`) |
| format | `peft` | `full` | `tinker` | `peft` or `full` |
| objective families | `policy_gradient`, `preference`, `likelihood`, `distillation` | the same | the same | the same |
| reference | yes (the adapter switched off) | when asked (`trainer.frozen_reference`: a frozen copy) | no | as the trainer it runs |
| entropy | yes | yes | no | as the trainer it runs |
| logprobs of tokens not sampled (distillation's top-k form) | yes | yes | no | as the trainer it runs |
| scores given tokens | yes | yes | yes | yes |
| starts from | `peft`, `full` | `full` | `tinker` | as the trainer it runs |
| auth | `none` | `none` | `vendor` | `mtls` |
| allocation | scheduled | scheduled | metered | scheduled (a leased pod: its own, or a `runpod-host`'s it names in `colocate_with`) |
| settings | `rollout_lora.settings:LoraSettings`, less `frozen_reference` | the same, less `rank` | `rollout_tinker.settings:TinkerSettings`, less `project` and `weights` | `LoraSettings` |

`settings_of(kind)` reads a trainer's settings from its dataclass, without importing the trainer or torch: each field
is `trainer.FIELD`, with its type and default, changeable when its module's `CHANGEABLE` names it. A trainer's
`objective` is not one of them: the run's `objective.*` settings say it.

A `lora`, `full` or `tinker` trainer may name its `implementation` (`module:name`), what makes the trainer in place of
its kind's, called as the kind's is: with the model and the settings it takes (`rollout_train.testing:ScriptedTrainer`
in tests, or an imported environment's own). A `vllm` provider's `engine` does the same for its engines.

## Bridges

| From | To | Bridge | Task |
|---|---|---|---|
| `tinker` | `tinker` | `none` | none: the checkpoint's own files |
| `tinker` | `peft` | `peft-from-tinker` | `rollout_tinker.bridges:peft`; 2 CPUs and the network; rank × 3 for Qwen3.5 |
| `peft` | `peft` | `verbatim` | `rollout_train.bridges:verbatim` |
| `full` | `full` | `full-reload` | `rollout_train.bridges:verbatim` |
| `peft` | `full` | `merge-quantize` | `rollout_lora.bridges:merge_quantize`; 8 CPUs, 48 GiB; only with `channels.NAME.bridge = "merge-quantize"`; full weights a provider quantizes as it loads them |
| `peft`, `full` | `tinker` | refused | Tinker samples only checkpoints Tinker trained: there is no upload |
| `full` | `peft` | refused | full weights are not an adapter |

`path(source, loads, wanted=…)` finds the cheapest chain; `rank_factor(chain, model)` is how many times the trained
rank the provider sees; `format_of(files)` reads a checkpoint's formats from its files. `bridged` runs a chain in
the calling process and `on_ray` runs each bridge as a Ray task, each noted in the ledger under `CHECKPOINT@BRIDGE`
([bridges](../libraries/rollout-train/checkpoints.md#bridges)).

## Run settings

A run's settings are dotted keys. Fixed ones are decided when the run starts; changeable ones are taken from its next
step on. `KEYS` is the schema: each key's type, default, whether it is changeable and which kinds of run (`train`,
`eval`, `imitate`, `check`) take it.

| Key | Default | Says |
|---|---|---|
| `kind` | `train` | The kind of run |
| `name` | | The run's name (never kept in a preset) |
| `environment` | | `module:name` |
| `groups`, `seed`, `episodes_at_once` | 100, 0, 6 | |
| `group_size` | the objective's | Episodes of each group a training run plays |
| `start`, `bookmark` | | The checkpoint it starts from; a bookmark it carries |
| `trainer.provider`, `trainer.channel`, `trainer.model` | , `policy`, the trained channel's model | The trainer, the trained channel, what it trains over |
| `weights` | what the trainer makes | `lora` or `full`: whether it trains a LoRA or full weights. It decides which trainers (Tinker trains LoRAs only), which providers (a LoRA needs adapters at the run's rank; full weights need full-weight reload, never Tinker's sampler) and which bridges fit |
| `trainer.FIELD` | the trainer's | Its own settings: `rank`, `segment_tokens`, `learning_rate`, `max_kl`, … |
| `objective.preset` | `default` | The objective's preset: `default`, `reinforce`, `rloo`, `ppo_clip`, `grpo`, `dr_grpo`, `dapo`, `gspo`, `cispo`, `sft`, `dpo`, `ipo`, `simpo`, `kto`, `orpo`, `on_policy_distillation`, `distillation`, `mopd`, `mopd_top_k` |
| `objective.COMPONENT` | the preset's | Each component ([objectives](../libraries/rollout-train/training.md#objectives)): `objective.clip.kind`, `objective.kl.target`, `objective.preference.loss`, …; its numbers (`objective.clip.low`, `objective.kl.coefficient`, `objective.preference.beta`, …) changeable. A distillation's teachers are `objective.distillation.teachers`, a table of channels by route (`{"*" = "teacher"}`), and the top-k it reads `objective.distillation.top_k` |
| `channels.NAME.provider` or `.providers` | | What samples the channel |
| `channels.NAME.routing`, `.weights` | `spill` | How several providers share its turns |
| `channels.NAME.model`, `.renderer` | | An imitate run renders its dataset's examples with its channel's renderer |
| `channels.NAME.thinking_tokens`, `.answer_tokens` | | Budgets per turn |
| `channels.NAME.replicas` | the provider's | |
| `channels.NAME.bridge` | `auto` | Or `merge-quantize` |
| `channels.NAME.mode` | | `fixed` or `follows`; the trained channel serves what the run trains |
| `channels.NAME.checkpoint` | | What a fixed channel serves (none: the base model) |
| `channels.NAME.follows`, `.lag` | , 0 | The channel a following channel follows, and how many of its serving records behind |
| `slots.SLOT` | the trained channel, for a trained slot | The channel a program's slot samples; a slot that is not trained (a judge, a fixed opponent) has no default |
| `self_judging` | false | Whether a judge may be bound to a channel serving the run's own checkpoints |
| `eval.suite`, `eval.episodes` | | An eval's suite and episodes |
| `check.episodes` | | Episodes of each group a check plays (none: a group's size) |
| `imitation.dataset`, `.limit`, `.passes`, `.warmup`, `.resume_optimizer`, `.without` | | Supervised steps |
| `groups_per_step` | 4 | Changeable |
| `max_lag` | 1 | Changeable |
| `evals.suite`, `evals.every`, `evals.episodes` | , 1, | Changeable |
| `limits.spend` | | Changeable: dollars. The run ends, stopped, once it spends this: its turns on hosted APIs (an eval's with its parts') and its pods' hours at their price. A training run whose one step is estimated above it is refused |
| `limits.hours` | | Hours: the run ends, stopped, once it has run this long, its pods released; its RayJob is stopped half an hour later in any case (`activeDeadlineSeconds`) |

Settings are given in layers, each over the last (`layered`): the schema's defaults, a preset, a file
(`from_file`: TOML or JSON, dotted keys or tables; JSON's `null` unsets a key), then the flags (`from_flags`:
`--set KEY=VALUE`, the value read as JSON, then TOML, then as text; `shortcuts` for `--model`, `--provider`,
`--renderer`, `--trainer`). `RunSettings` answers each key with its default where it was not given, and holds the
trainer settings that name the objective (`trainer.objective`, `trainer.ratio`, `trainer.clip_low`,
`trainer.clip_high`, `trainer.segment_clip_low`, `trainer.segment_clip_high`, `trainer.truncate`) as the `objective.*`
keys they say, where those are not given: `trainer.objective = "policy_gradient"` is the `default` preset and
`"likelihood"` is `sft`. `objective_in(settings)` is the objective they resolve to. `recorded(settings,
trainer_settings, preset)` is what a run's start records: a full copy, fixed and changeable, the resolved objective of a
run that trains (`objective`), and the preset version they came from. `diff(before, after)` says what changed, key by
key.

### Asking for a run

A run is asked for with its kind, its name, its environment, its settings and a preset
(`rollout_train.launching.settled`): the preset's settings that a run of its kind takes (so a training run's preset
serves an eval of the same channels), then the settings given, then its kind and name. `checked(settings, cluster,
ledger)` checks them against the cluster with the facts gathered now: the environment's (`environment_facts`: imported
here, its programs' sandboxes and slots; a published version's sandboxes as its record says; none for an environment
in a project's Python, which this process does not import) and the ledger's (`ledger_facts`: the checkpoints the
settings name and their formats, the suites, the names other runs have). The monitor checks a run when it is asked
for, the CLI before it submits it, and the run's driver again before it claims anything
([launching](../libraries/rollout-train/launching.md)).

The run's start records its settings as it runs (`run_settings`: `recorded`, with the settings its trainer declares
and the preset it came from). A training run that does not say `weights` is asked for with its trainer's
(`with_weights`), so its start says what it trains; the monitor reads a start that does not say it as its trainer's.

## Presets

A preset (`Preset`) is named run settings, in versions: each save is the next version, and nothing is changed in
place. `NAME` is the newest version and `NAME@N` one version. Two saves at once make two versions. Deleting a preset
appends a version that says so: its name then points to nothing, and its earlier versions stay readable for the runs
that name them. `presets_of(ledger)` gives the store beside a ledger: `FilePresets` (a file per version, in
`presets/` beside a ledger of files) or `DatabasePresets` (the `presets` table of a database ledger's database).

```bash
rollout preset load deploy/chart/rollout/files/presets --cluster   # each NAME.toml as preset NAME (a new version only
                                                             # where its newest says otherwise)
rollout preset list --cluster                                # every preset's newest version
rollout preset show minecraft-one-gpu@3 --cluster            # one version's settings, a key a line
rollout preset save minecraft-one-gpu --from-run team-8 --set trainer.learning_rate=3e-5 --note "slower" --cluster
rollout preset save gsm8k-tinker --settings run.toml --cluster   # a file of settings (each source over the last)
rollout preset delete minecraft-one-gpu --cluster            # its name points to nothing; its versions stay
```

`--from-run RUN` copies the run settings the run's newest start records, less its name. The presets shipped with the
platform are files of settings in `deploy/chart/rollout/files/presets` (`minecraft-one-gpu`, `minecraft-tinker`,
`gridworld-qwen3-0.6b`, `gsm8k-tinker`), each commented with why its numbers are what they are; the chart loads them at
every install and upgrade, and on one machine `rollout preset load` does.

## Validation

`check(settings, cluster, environment, ledger)` returns a list of `Finding`s, each with its rule, the key it is about
and a reason. It is pure: what it needs beyond the settings and the cluster is passed in as facts, gathered
beforehand (`EnvironmentFacts`, `LedgerFacts`). A finding whose `refuses` is false is a note: the run waits, or
something could not be estimated. `refusals(findings)` keeps the ones that refuse.

| Rule | Refuses when |
|---|---|
| `settings` | a key the kind does not take; a wrong type or a value out of range; a required key missing; a channel that contradicts itself (`provider` and `providers`, `channels.NAME.weights` without `weighted`, a mode on the trained channel, following nothing); a slot naming no channel; a slot the program declares and the run does not bind (one that is not trained has no default); a channel a slot samples without a provider or a model; a judge bound to the trained channel, or one following it, without `self_judging` (`rollout_train.slots`) |
| `providers` | the trainer or a channel's provider is not offered; a hosted API sharing a channel with another provider |
| `auth` | a provider is reached with no auth away from this machine |
| `capabilities` | a provider of the trained channel is a hosted API, whatever the objective (it returns text: no exact tokens or behaviour logprobs, so nothing it samples is trained on); is not token-exact, for a policy gradient; or returns no sampled-token logprobs or does not honour sampling, for one with an importance correction. A preference loss and a likelihood read neither |
| `bridge` | no bridge from the trainer's format (or a checkpoint's) to what a provider of the channel loads |
| `weights` | a trainer that makes the other kind than the run's `weights`; a provider serving the run's checkpoints that cannot serve them (a LoRA without adapters; full weights, or a LoRA merged by `merge-quantize`, without full-weight reload); a checkpoint a fixed channel serves on a provider that cannot |
| `models` | `trainer.model` not among the trainer's; a channel's model not among its provider's; a channel serving the run's checkpoints with a model that is neither `trainer.model` nor quantized from it; a start trained over another model |
| `rank` | `trainer.rank` times the bridge's rank factor above the provider model's `max_lora_rank` |
| `segment` | `trainer.segment_tokens` above the trainer's here, or above the trained channel's context |
| `start` | the start does not exist or was released; a full-weight trainer from an adapter (merge it first); Tinker from a checkpoint Tinker did not make; a local trainer from a Tinker checkpoint (bridge it first) |
| `objective` | a component the objective's family does not accept, or a combination that means nothing (a clip with no ratio, a KL to the reference with none, a reference for a loss that compares likelihoods alone, a k1 KL penalty in the loss); a family the trainer does not take; a policy gradient for an imitate run; a reference the trainer cannot give (Tinker: none; the full-weight trainer: only with `trainer.frozen_reference`); an entropy bonus on a trainer without entropies; the top-k form of distillation on a trainer that gives the sampled tokens' logprobs only (Tinker). A policy gradient without an importance correction while turns may begin behind the newest checkpoint (`max_lag` above 0): a note |
| `evals` | a suite that does not exist (a name never becomes a suite by itself), a version it lacks, a suite's environment not offered |
| `distillation` | a distillation (or a policy gradient's distillation term) with no teachers, a teacher channel without a provider, or no route for the environment the run plays (routes for some of its rows only: a note); a teacher's provider without prompt logprobs, with fewer top logprobs than `objective.distillation.top_k`, or with them only unchecked (Tinker); the trainer does not score; the teacher's renderer family differs |
| `environment` | not offered, does not load, needs a sandbox kind with no pool or a tool set not served |
| `capacity` | more CPUs, memory or GPUs than `[capacity]` gives one run, counting the run's scheduled parts and room for Ray's own processes; more GPUs than the cluster has (more than are free: a note, it waits) |
| `spend` | a training run's `limits.spend` below one step's estimated cost on its metered parts (`spend_of`); a note where it cannot be estimated yet, or where the run uses metered parts and sets no `limits.spend`. For an eval, notes only: no limit on a metered provider, or a limit below the eval's estimate (it ends early) |
| `name` | not a name, or taken |

```python
import tomllib

from rollout_train.cluster import parsed
from rollout_train.run_settings import from_flags, layered, shortcuts
from rollout_train.validation import EnvironmentFacts, LedgerFacts, SuiteFacts, check, refusals

cluster = parsed(tomllib.loads("""
name = "home"
[ledger]
url = "sqlite:///~/.cache/rollout/ledger.db"
[inference.local-vllm]
kind = "vllm"
gpus = 1
[inference.local-vllm.models."Qwen/Qwen3.5-4B"]
context = 8192
options = { max_lora_rank = 64 }
[inference.tinker]
kind = "tinker"
[inference.tinker.models."Qwen/Qwen3.5-4B"]
context = 65536
[trainers.tinker-lora]
kind = "tinker"
models = ["Qwen/Qwen3.5-4B"]
segment_tokens = 32768
[environments."rollout_verifiers.environments:gsm8k"]
python = "platform"
"""))
gsm8k = "rollout_verifiers.environments:gsm8k"
settings = layered(
    shortcuts(model="Qwen/Qwen3.5-4B", provider="local-vllm", trainer="tinker-lora"),
    from_flags([f"environment={gsm8k}", "trainer.rank=16", "evals.suite=math", "limits.spend=2"]),
)
ledger = LedgerFacts(suites={"math": SuiteFacts("math", 1, frozenset({gsm8k}))})
assert refusals(check(settings, cluster, EnvironmentFacts(gsm8k), ledger)) == []

too_big = layered(settings.values, from_flags(["trainer.rank=32"]))  # Tinker's q, k, v joined: 96 > 64
(finding,) = refusals(check(too_big, cluster, EnvironmentFacts(gsm8k), ledger))
assert finding.rule == "rank" and "is 96, above provider local-vllm's max_lora_rank 64" in finding.reason
```
