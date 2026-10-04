# The cluster config and run settings

A cluster is described once, in a cluster config: where the ledger and the blob store are, the inference providers
and trainers it offers, its sandbox pools, the environments it offers and the Python each runs in. A run is described
by its settings: the environment, the trainer, the provider of each channel, and the numbers each takes. Settings can
be saved as presets, and one pure function checks a run's settings against a cluster.

This page describes what these modules provide:

- `rollout_train.cluster`: the cluster config, found and read strictly into a `Cluster`;
- `rollout_train.providers`: the kinds of inference provider and trainer, their capabilities, how each is reached;
- `rollout_train.bridges`: how a checkpoint in one format becomes files a provider loads;
- `rollout_train.run_settings`: the schema of a run's settings, how they are given, and their recorded copy;
- `rollout_train.presets`: named, versioned settings kept beside the ledger;
- `rollout_train.validation`: `check`, with its rule table.

They are declarations: none of them starts, imports or reaches anything. What opens and reaches things from them:

- `rollout_train.stores.Stores.open(cluster)` opens the ledger and the blob store the config names
  ([below](#the-stores));
- `rollout cluster check` reads the config and says what of it does not resolve on this node;
- the commands over a ledger (`rename`, `bookmark`, `pause`, `resume`, `checkpoints`, `suite`, `dataset`, `merge`)
  take the cluster's with `--cluster`, in place of `--ledger`.

The commands that start runs still take profiles ([Deploying](deploying.md)); the
[runtime design](../research/runtime-design.md) says how they move onto these.

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

`deploy/clusters/example.toml` is a config for one machine with one 16 GB GPU: SQLite, files, a vLLM pool, the LoRA
trainer, Tinker, the Minecraft worlds and GSM8K in its own Python.

```toml title="cluster.toml"
name = "home"                                 # Ray namespace rollout-home; what runs record as where they ran

[ray]
address = "auto"                              # the head this machine runs
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
pool = { adapter_slots = 4 }                  # runs share it: each run's live checkpoints are adapters side by side
[inference.local-vllm.models."Qwen/Qwen3.5-4B"]
context = 8192
options = { gpu_memory_utilization = 0.78, max_num_seqs = 20, max_lora_rank = 64 }

[inference.tinker]
kind = "tinker"                               # auth vendor: Tinker's SDK reads TINKER_API_KEY
[inference.tinker.models."Qwen/Qwen3.5-4B"]
context = 65536
cost = { input = 0.0, output = 0.0 }          # dollars per million tokens

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
cost = { train = 0.0 }

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
| `[ray]` | `address` (`auto`), `jobs`, `temp_dir`, `memory_threshold`, `python` | |
| `[ledger]` | `url`, or `url_env` / `url_file` | A URL holding a password is refused: name it instead |
| `[blobs]` | `kind` (`files` or `module:name`), `directory` or the store's settings | A setting that looks like a credential is refused |
| `[scratch]` | `directory` | Node-local |
| `[tls]` | `ca`, `certificate`, `key`, `identity` (`spiffe://rollout/gateway`) | The cluster's CA, and the client certificate it presents; paths |
| `[gateway]` | `url`, `listen`, `replicas`, `keys_file` / `keys_env`, `lifetime` | |
| `[monitor]` | `listen`, `feed_episodes` | |
| `[launcher]` | `at_once` | |
| `[runners]` | `places`, `durable`, `database_url` or `database_url_env` | |
| `[guards]` | `runs_gib`, `training_gib` | |
| `[inference.NAME]` | `kind`, `auth`, `gpus`, `replicas`, `models`, and the kind's own | Below |
| `[trainers.NAME]` | `kind`, `auth`, `models`, `segment_tokens`, `gpus`, `colocate_with`, `cost`, and the kind's own | Below |
| `[sandboxes.KIND]` | `provider`, `python`, `size`, `cpus`, `memory_gib`, `pools`, the provider's settings | |
| `[tools.NAME]` | `url`, `auth` | Tool sets served elsewhere |
| `[environments."NAME"]` | `python = "platform"` or `project = PATH` | A relative project is from the config file's directory |
| `[placement.ROLE]` | `resources` | Roles: gateway, monitor, launcher, runners, pools, engines, trainers, workers, bridges |
| `[bridges."NAME"]` | `cpus`, `memory_gib` | Overrides what a bridge declares |

A model a provider offers (`models."MODEL"`) has a `context`, and optionally a `base` (the model it was quantized
from), a `cost` table (dollars per million tokens by class: `input`, `cached_input`, `output`, `thinking`; or
`hour`) and `options` (what its engines are started with; `max_lora_rank` is the highest adapter rank it loads).

### Secrets

A secret appears only by name: a key `NAME_env` (an environment variable) or `NAME_file` (a file). A key that looks
like a secret but holds a value (`api_key = "sk-…"`, `token = "…"`) is refused, and so is a ledger URL with a
password. So the parsed `Cluster` holds no secret: its `repr` and its JSON (`Cluster.described`, which `of_json` reads
back for a job or an actor) are safe to show. A `Secret` is resolved where it is used, at the moment it is needed
(`Secret.resolve`). `Cluster.secrets()` lists every reference. `inspect(cluster)` says, on this node, which
references do not resolve (by name, never by value) and which environment projects have no `uv.lock`.

### The stores

`Stores.open(cluster)` opens, on this node:

- **the ledger** `[ledger]` names, a database (`DatabaseLedger`): `sqlite:///…` (`~` is the home directory) or
  `postgresql://…`. Where the config names the URL (`url_env`, `url_file`), it is read there and then; a name that is
  not set here is an error that says the name, and the URL itself is never printed;
- **the blob store** `[blobs]` names: files in `directory`, or `kind = "module:name"` called with the table's other
  settings. An S3 store (or any S3-compatible service, such as versitygw) is
  `kind = "rollout_s3:S3BlobStore"` with `bucket`, and optionally `prefix`, `endpoint_url` and `region`; its
  credentials come from the node's `AWS_*` environment, never from the config.

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

| Capability | `vllm` | `vllm-servers` | `tinker` | `api` | `runpod-inference` |
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
| auth | `none`, `bearer`, `mtls` (default `none`) | `none`, `bearer`, `mtls` (must say) | `vendor` | `vendor`, `bearer` (must say) | `mtls` (each pod's identity from its heartbeat) |
| its own fields | `engine`, `listen`, `max_logprobs`, `pool` | `addresses`, `via`, `loader`, `max_logprobs`, `pool` | `project` / `project_env` | `endpoint` | `image`, `gpu_types`, `pods`, `idle_stop`, `volume_gb`, `secrets`, `step_ca`, `max_logprobs`, `pool`, `api_key_env` |

Tinker's prompt and top-k logprobs are declared as its SDK says (`Capabilities.unchecked`): nothing relies on them
until a live test confirms them.

**Auth.** `auth` is a kind or a table: `auth = "none"`, `auth = { kind = "bearer", token_env = "ENGINES_TOKEN" }`,
`auth = { kind = "vendor", key_env = "OPENAI_API_KEY" }`, `auth = { kind = "mtls", identity = "spiffe://…" }`.
`Auth.connection(tls)` gives the client's connection settings (`rollout_train.inference.remote.Connection`):

| Kind | Server verified by | Host name or identity | Client certificate | Token |
|---|---|---|---|---|
| `mtls` | the cluster's CA (`[tls] ca`) | the SPIFFE identity, where said (a pod's from its heartbeat); else the host name | `[tls] certificate`, `key` | |
| `bearer` | the system's CAs, or the cluster's with `trust = "cluster"` | the host name | | `token_env` / `token_file` |
| `vendor` | the vendor's SDK | | | the SDK reads its own key |
| `none` | only for addresses on this machine | | | |

**Shared pools.** A `vllm`, `vllm-servers` or `runpod-inference` provider is a pool runs share (`SharedPool`): its
servers hold each bound run's live checkpoints as adapters side by side. `pool = { adapter_slots = N, max_runs = M }`
limits it. A run needs `max_lag + 1` slots for each channel it serves there, and its rank must fit the model's
`max_lora_rank`. Turns are shared among a pool's runs by each run's `share`.

**Several providers.** A channel may name several providers (`channels.NAME.providers`), shared by a routing rule
(`channels.NAME.routing`): `spill` fills the first and sends the rest to the next; `weighted` shares turns by
`channels.NAME.weights`.

## Trainers

| | `lora` | `full` | `tinker` | `runpod-trainer` |
|---|---|---|---|---|
| produces | `lora` | `full` | `lora` | as the trainer it runs (`trainer = "lora"` or `"full"`) |
| format | `peft` | `full` | `tinker` | `peft` or `full` |
| objectives | `policy_gradient/token`, `policy_gradient/segment`, `likelihood` | the same | the same | the same |
| scores given tokens | yes | yes | yes | yes |
| starts from | `peft`, `full` | `full` | `tinker` | as the trainer it runs |
| auth | `none` | `none` | `vendor` | `mtls` |
| settings | `rollout_lora.settings:LoraSettings` | the same, less `rank` | `rollout_tinker.settings:TinkerSettings`, less `project` and `weights` | `LoraSettings` |

`settings_of(kind)` reads a trainer's settings from its dataclass, without importing the trainer or torch: each field
is `trainer.FIELD`, with its type and default, changeable when its module's `CHANGEABLE` names it.

## Bridges

| From | To | Bridge | Task |
|---|---|---|---|
| `tinker` | `tinker` | `none` | none: the checkpoint's own files |
| `tinker` | `peft` | `peft-from-tinker` | `rollout_tinker.bridges:peft`; 2 CPUs and the network; rank × 3 for Qwen3.5 |
| `peft` | `peft` | `verbatim` | `rollout_train.resharding:verbatim` |
| `full` | `full` | `full-reload` | `rollout_train.resharding:verbatim` |
| `peft` | `full` | `merge-quantize` | `rollout_lora.bridges:merge_quantize`; 8 CPUs, 48 GiB; only with `channels.NAME.bridge = "merge-quantize"` |
| `peft`, `full` | `tinker` | refused | Tinker samples only checkpoints Tinker trained: there is no upload |
| `full` | `peft` | refused | full weights are not an adapter |

`path(source, loads, wanted=…)` finds the cheapest chain; `rank_factor(chain, model)` is how many times the trained
rank the provider sees; `format_of(files)` reads a checkpoint's formats from its files.

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
| `start`, `bookmark` | | The checkpoint it starts from; a bookmark it carries |
| `trainer.provider`, `trainer.channel`, `trainer.model` | , `policy`, the trained channel's model | The trainer, the trained channel, what it trains over |
| `trainer.FIELD` | the trainer's | Its own settings: `rank`, `segment_tokens`, `learning_rate`, `objective`, `ratio`, … |
| `channels.NAME.provider` or `.providers` | | What samples the channel |
| `channels.NAME.routing`, `.weights` | `spill` | How several providers share its turns |
| `channels.NAME.model`, `.renderer` | | |
| `channels.NAME.thinking_tokens`, `.answer_tokens` | | Budgets per turn |
| `channels.NAME.replicas` | the provider's | |
| `channels.NAME.bridge` | `auto` | Or `merge-quantize` |
| `channels.NAME.mode` | | `fixed` or `follows`; the trained channel serves what the run trains |
| `channels.NAME.checkpoint` | | What a fixed channel serves (none: the base model) |
| `channels.NAME.follows`, `.lag` | , 0 | The channel a following channel follows, and how many checkpoints behind |
| `slots.SLOT` | | The channel a program's slot samples |
| `distill.channel`, `distill.k` | | A teacher's channel; top-k matched (none: the teacher scores) |
| `eval.suite`, `eval.episodes` | | An eval's suite and episodes |
| `check.episodes` | 1 | |
| `imitation.dataset`, `.limit`, `.passes`, `.warmup`, `.resume_optimizer`, `.without` | | Supervised steps |
| `groups_per_step` | 4 | Changeable |
| `max_lag` | 1 | Changeable |
| `evals.suite`, `evals.every`, `evals.episodes` | , 1, | Changeable |
| `limits.spend` | | Changeable: dollars; the run ends once its estimated spend reaches it |
| `share` | 1 | Changeable: its weight in a shared pool's fair shares |

Settings are given in layers, each over the last (`layered`): the schema's defaults, a preset, a file
(`from_file`: TOML or JSON, dotted keys or tables; JSON's `null` unsets a key), then the flags (`from_flags`:
`--set KEY=VALUE`, the value read as JSON, then TOML, then as text; `shortcuts` for `--model`, `--provider`,
`--renderer`, `--trainer`). `RunSettings` answers each key with its default where it was not given.
`recorded(settings, trainer_settings, preset)` is what a run's start records: a full copy, fixed and changeable,
and the preset version they came from. `diff(before, after)` says what changed, key by key.

## Presets

A preset (`Preset`) is named run settings, in versions: each save is the next version, and nothing is changed in
place. `NAME` is the newest version and `NAME@N` one version. Two saves at once make two versions. Deleting a preset
appends a version that says so: its name then points to nothing, and its earlier versions stay readable for the runs
that name them. `presets_of(ledger)` gives the store beside a ledger: `FilePresets` (a file per version, in
`presets/` beside a ledger of files) or `DatabasePresets` (the `presets` table of a database ledger's database).

## Validation

`check(settings, cluster, environment, ledger)` returns a list of `Finding`s, each with its rule, the key it is about
and a reason. It is pure: what it needs beyond the settings and the cluster is passed in as facts, gathered
beforehand (`EnvironmentFacts`, `LedgerFacts`). A finding whose `refuses` is false is a note: the run waits, or
something could not be estimated. `refusals(findings)` keeps the ones that refuse.

| Rule | Refuses when |
|---|---|
| `settings` | a key the kind does not take; a wrong type or a value out of range; a required key missing; a channel that contradicts itself (`provider` and `providers`, `weights` without `weighted`, a mode on the trained channel, following nothing); a slot naming no channel |
| `providers` | the trainer or a channel's provider is not offered |
| `auth` | a provider is reached with no auth away from this machine |
| `capabilities` | a provider of the trained channel is not token-exact, returns no sampled-token logprobs, or does not honour sampling |
| `bridge` | no bridge from the trainer's format (or a checkpoint's) to what a provider of the channel loads |
| `weights` | adapters for a provider without adapters; full weights for one without full-weight reload |
| `models` | `trainer.model` not among the trainer's; a channel's model not among its provider's; a channel serving the run's checkpoints with a model that is neither `trainer.model` nor quantized from it; a start trained over another model |
| `rank` | `trainer.rank` times the bridge's rank factor above the provider model's `max_lora_rank` |
| `segment` | `trainer.segment_tokens` above the trainer's here, or above the trained channel's context |
| `start` | the start does not exist or was released; a full-weight trainer from an adapter (merge it first); Tinker from a checkpoint Tinker did not make; a local trainer from a Tinker checkpoint (bridge it first) |
| `objective` | `trainer.objective` with `trainer.ratio` not among the trainer's objectives |
| `evals` | a suite that does not exist (a name never becomes a suite by itself), a version it lacks, a suite's environment not offered |
| `distillation` | the teacher's provider lacks prompt logprobs (or top-k with k at least `distill.k`), or has them only unchecked; the trainer does not score; the teacher's renderer family differs |
| `environment` | not offered, does not load, needs a sandbox kind with no pool or a tool set not served |
| `capacity` | more GPUs than the cluster has (more than are free: a note, it waits) |
| `pools` | more adapter slots than a shared pool has (more than are free, or the pool full of runs: a note, it waits) |
| `spend` | `limits.spend` below one step's estimated cost (`estimated_spend`) |
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
