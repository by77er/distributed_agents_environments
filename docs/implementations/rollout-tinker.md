# Tinker trainer and engine

Code: `rollout_tinker` · See [Thinking Machines' API](../research/thinking-machines.md), [training](../libraries/rollout-train/training.md),
[channels and engines](../libraries/rollout-train/channels.md), [checkpoints](../libraries/rollout-train/checkpoints.md),
[LoRA trainer](rollout-lora.md)

`TinkerTrainer` implements the [`Trainer`](../guide/reference.md#trainer) protocol and `TinkerEngine` the
[`Engine`](../guide/reference.md#engine) protocol on Thinking Machines' hosted API, Tinker. The trainer takes LoRA steps
there with the [LoRA trainer](rollout-lora.md)'s objective, expressed as Tinker's losses; the engine samples there,
token ids in, ids and logprobs out. Neither uses this machine's GPU. The loop, the gateway, the ledger, the renderer and
the objective are the platform's own: Tinker holds the weights and does the arithmetic.

## Installing

`implementations/rollout-tinker` is a uv project of its own, with its own lock, outside the workspace. Tinker's
`tinker-cookbook`, whose weight tools turn Tinker's adapters into PEFT's, pins `transformers<=5.5.4`, and the
platform's vLLM and Gemma renderers need 5.10 or later. The project pins `tinker==0.32.0` and `tinker-cookbook==0.5.7`,
runs the cookbook on the platform's `transformers` (5.17.0, by an override: its weight tools read a model's
configuration and safetensors headers only, and the tests run them there), and holds `torch`, `vllm` and `openai` at
the platform's versions. It depends on the workspace's packages by path.

```bash
cd implementations/rollout-tinker
uv sync                 # the trainer, the engine, the Minecraft environment, the Qwen renderers, the tests
uv sync --group local   # with vLLM, to serve Tinker's adapters on this machine (`weights = "peft"`)
uv run pytest tests     # on a fake Tinker: no key, no network
```

Commands run from there (`uv run rollout ...`) have Tinker's SDK; the workspace's environment does not.

## The key

The SDK reads `TINKER_API_KEY`, else the key `tinker auth login` stores in `~/.tinker/credentials.json`
(`uv run tinker auth login` from the project). `TINKER_PROJECT_ID`, or the `project` setting, puts the sessions in a
Tinker project; a project's id is not a secret. Nothing in `rollout_tinker` prints or records the key, and the text of
an error is cleared of it before it reaches a failed step's record. Every process that opens the profile's channel or
trainer needs the key, in its environment or its user's credentials file.

## Billing errors are fatal

Tinker bills a prepaid balance. With its automatic top-up off, a call it refuses for billing (HTTP 402: the balance ran
out, or billing is not set up) will be refused again until someone adds money, so it is never retried:

- the SDK's own pause on a 402 (it otherwise asks again every five seconds for up to an hour) is turned off in every
  session `rollout_tinker` opens (`service.no_billing_pause`);
- the engine raises `Unpaid` (a `ModelEndpointError`: the gateway answers it as the endpoint failing) for that turn,
  and for every later one without calling Tinker again;
- the trainer raises `Unpaid` rather than `StepFailed`, so the run stops instead of going on to a step that would be
  refused too.

No call of the SDK, documented or in its REST client, reads the balance. `RestClient.get_billing_usage` (`tinker billing usage`) gives usage by hour, each
row with an estimated cost at list prices, up to several hours late: what has been spent, not what is left.

## In a profile

```toml
[channels.policy]
model = "Qwen/Qwen3.5-9B"             # Tinker's id; the renderer loads the same model's tokenizer from the Hub
renderer = "rollout_qwen:qwen35"
engine = "rollout_tinker:TinkerEngine"
engines = [{ max_model_len = 8192 }]

[trainer]
kind = "rollout_tinker:TinkerTrainer"
channel = "policy"
rank = 32
learning_rate = 1e-4
segment_tokens = 8000
segments_per_step = 384
tokens_per_step = 16384
```

`environments/minecraft/profiles/tinker.toml` is the Minecraft environment's, on the full `Qwen/Qwen3.5-9B`:

```bash
cd implementations/rollout-tinker
uv run rollout train ../../environments/minecraft/profiles/tinker.toml minecraft_team.environment:environment --groups 30
```

### Evals through a gateway

An environment that cannot share this project's environment (verifiers' pins its own `openai` and `mcp`) is played by
a runner in its own project, sampling at Tinker through a gateway served from here
([a gateway elsewhere that hosts channels](../libraries/rollout-train/gateway.md#a-gateway-elsewhere-that-hosts-channels)).
One profile serves both: `rollout gateway` here starts the `TinkerEngine` channel and records every turn; the runner,
with `[gateway] url`, starts no engine and samples there under signed keys. The channel serves the base model.
`implementations/rollout-verifiers/examples/gsm8k_tinker.toml` is GSM8K's, on `Qwen/Qwen3.5-9B`:

```bash
uv run rollout gateway ../rollout-verifiers/examples/gsm8k_tinker.toml     # from here
```

and the eval from `implementations/rollout-verifiers` ([GSM8K](rollout-verifiers.md#gsm8k)). Its first run,
`gsm8k-tinker-base` on 2026-10-04, played the suite `math` (`gsm8k-test-100`, one episode each, 1,024 tokens of
thinking and 512 of answer) in four minutes, eight episodes at once:

| | |
|---|---|
| Solved | 89 of 100 (mean reward 0.89) |
| Turns | 100, one an episode, each in two phases: the thinking closed by force at 1,024 tokens in nearly every turn, and 12 answers cut at 512 |
| Tokens | 126,142 sampled (1,261 a turn), each with a finite logprob; 210 forced, with none; 125,588 of prefill over both phases |
| Exact | every prompt is our renderer's rendering of its task, the ids Tinker sampled from |
| Cost at list prices | $0.27 to $0.33, by how much of each second phase's prefix Tinker cached |

Qwen3.5-9B thinks past 1,024 tokens on most GSM8K problems, so the cap, not the problem, ends its thinking.

`colocated`, `training_gib` and `reshard` do nothing for a remote trainer: there is nothing on this machine to share,
and a step's files are pointers. An engine's options are `max_model_len` (the longest turn; Tinker's context for
`Qwen/Qwen3.5-9B` is 64K), `project` and `service`.

### Settings

`TinkerSettings` (`rollout_tinker/settings.py`) is the one place they are written. They have `LoraSettings`' names where they
mean the same, so a profile switches trainers by changing `kind`.

| Setting | What it sets |
|---|---|
| `rank` | The adapter's rank |
| `learning_rate` | AdamW's rate (1e-4). Tinker scales its adapters by its own `lora_alpha / rank`. Its cookbook assumes an alpha of 32, half our scale at rank 32, so twice `LoraTrainer`'s rate would move the weights as far (*unverified*: the live test records the alpha an archive says) |
| `clip_low`, `clip_high`, `segment_clip_low`, `segment_clip_high`, `truncate`, `ratio`, `objective` | As `LoraSettings` |
| `tokens_per_step` | Sampled tokens per optimizer step (65,536). A step that is one optimizer step needs no pass for where it starts (below) |
| `max_kl`, `max_gradient_norm`, `passes`, `warmup_updates`, `segment_tokens`, `segments_per_step` | As `LoraSettings` |
| `strict_kl` | Read each minibatch's distance before its update is sent (true): two of Tinker's clock cycles a minibatch. False sends both at once, and a stop at `max_kl` comes one minibatch late |
| `beta1`, `beta2`, `eps` | Adam's, as torch's AdamW has them (0.9, 0.999, 1e-8; Tinker's own defaults are 0.95 and 1e-12) |
| `train_unembed` | Also adapt the output layer (false) |
| `weights` | `pointer` (a step's weights name its Tinker checkpoints) or `peft` (and hold the adapter itself, below) |
| `project` | A Tinker project's id |

`service` (trainer and engine) is what calls Tinker: by default a session the SDK opens; `module:name` of what makes
another, as the tests name `rollout_tinker.testing:fake_service`.

The trainer takes some settings between steps, as the LoRA trainer does (`Changeable`): `learning_rate`, the clips,
`truncate`, `tokens_per_step`, `max_kl`, `strict_kl` and `max_gradient_norm`. The next step reads them, on the same
client.

## A step

A step takes the same minibatches as the [LoRA step](rollout-lora.md#the-step): segments longer than `segment_tokens`
or with nothing sampled left out and counted, the rest shuffled by the step's seed and cut where a minibatch reaches
`tokens_per_step` sampled tokens, `passes` times. Each segment is one of Tinker's `Datum`s, its input the tokens but the
last and its targets the tokens but the first, so a sampled token at position *t* is row *t* - 1; every other row
(a prompt, a tool's result, a token the gateway forced) carries zeros and adds nothing to the loss.

Tinker's losses take one reference logprob per token, where ours has two: the logprob at the step's start (`old`) and
the one it was sampled at (`behavior`). Folding the importance weight `w = min(exp(old - behavior), truncate)` and the
minibatch's units `U` into the advantages makes each loss ours, in value and in gradient:

| Objective, ratio | Optimizer steps | Tinker | Reference; advantage |
|---|---|---|---|
| `policy_gradient`, `token` | one | `cispo`, its ratio clipped to 0 .. `truncate` | behaviour; A/U |
| `policy_gradient`, `token` | several | a forward pass for `old`, then `ppo`, clipped to 1 - `clip_low` .. 1 + `clip_high` | old; A·w/U |
| `policy_gradient`, `segment` | any | a forward pass for `old` if several, then a custom loss that calls `rollout_lora.objectives.terms` | |
| `likelihood` | any | `cross_entropy`, weights A/U | |

With one optimizer step `old` is the logprob now, so no forward pass is needed. A custom loss is computed here from the
logprobs Tinker returns, and Tinker then takes a pass on a linear stand-in with that loss's gradient: a forward pass
more than a built-in loss. Each minibatch's statistics are `terms` of the logprobs its forward-backward returns (the
policy before that update), so the metrics are the LoRA step's.

**Where a step starts.**

- No parent: a new LoRA run on the model, its optimizer fresh (warmed up over `warmup_updates`).
- A parent with its trainer state: Tinker's `create_training_client_from_state_with_optimizer` on the state it names.
  When the parent is the state this trainer's last step saved, that step's client goes on instead (no new training run).
- A parent's weights without its state (`rollout imitate`, unless `--resume-optimizer`): its weights, a fresh optimizer.
- A parent not trained on Tinker: the step fails. No call takes an adapter trained elsewhere into a Tinker run.

**The stop at `max_kl`.** A minibatch that finds the policy further than `max_kl` from where the step began stops the
pass. Its gradient was accumulated where no call clears it, so that client is not used again: the next step resumes
the saved state.

**Failures.** Any error of the step (Tinker's, or a batch it refuses) raises `StepFailed` with the error's type and
message: the weights stay as they were and the run goes on. A step retried after a crash makes a new checkpoint id, so
its Tinker checkpoints have new names.

**Metrics.** The LoRA step's (`loss`, `clip_fraction`, `mean_ratio`, `kl_floor`, `mean_mismatch`, `mean_weight`,
`truncated_fraction`, `kl_moved`, `tokens`, `segments`, `segments_too_long`, `optimizer_steps`, `stopped_at_max_kl`,
`seconds`, ...), and `billed_tokens`: every token Tinker's training passes processed, prompts included.
`state/minibatches.jsonl` has each update's line, with what Tinker's `optim_step` reported (`optimizer_*`); a
`gradient_norm` is reported when that includes one.

## What a checkpoint holds

| Path | Holds |
|---|---|
| `weights/tinker.json` | `sampler` (`tinker://RUN/sampler_weights/ID`, what engines sample), `state` (`tinker://RUN/weights/ID`, weights and Adam's state), `base_model`, `rank` |
| `state/tinker.json` | `state` again, `sampler`, the SDK's version |
| `state/minibatches.jsonl` | What each update did |
| `weights/adapter_config.json`, `weights/adapter_model.safetensors` | With `weights = "peft"`: the adapter, in PEFT's layout |

Tinker's checkpoints are named after the checkpoint's id. The pointer files are blobs like any weights, so a checkpoint
copied to another machine points at the same remote checkpoints, and the ledger needs nothing new. Releasing a
checkpoint (retention) deletes its files from the blob store, not its checkpoints at Tinker, which are billed for
storage until deleted: `uv run tinker checkpoint delete --run-id RUN` (or by path) deletes them.

**`weights = "peft"`.** After saving, the step downloads the sampler checkpoint's archive (Tinker's own names), turns
it into PEFT's layout with `tinker_cookbook.weights.build_lora_adapter`, in a process that cannot see the GPU, and keeps
it beside the pointer. Qwen3.5's linear-attention layers hold one `in_proj_qkv` projection where Tinker adapts
`in_proj_q`, `in_proj_k` and `in_proj_v` apart, and vLLM loads only the joined name: the three are joined into one
adapter (A stacked and B block-diagonal, three times the rank; at the same rank when they share one A), which is the
same update. `rollout_tinker.weights.ranks(directory)` says the largest rank, which an engine's `max_lora_rank` must
reach. Then:

- `rollout_vllm:VllmEngine` serves it, as the commented profile in `tinker.toml` shows;
- `rollout merge CHECKPOINT --base Qwen/Qwen3.5-9B` folds it into the model, a full checkpoint of its own;
- the blob store holds the adapter, so the policy outlives the model's retirement at Tinker.

A rank-32 adapter of `Qwen/Qwen3.5-9B` without the output layer has 86.5 million parameters (Tinker's cookbook counts
them), about 0.35 GB in float32.

On this machine's 16 GB card (measured with a synthetic adapter in Tinker's layout, converted so), vLLM loads the
joined adapter and samples from it. The full model does not fit at an 8,192-token context even in FP8: its weights take
10.8 GiB, leaving 0.25 GiB of cache at 0.88 of the card, not one turn. The 4-bit checkpoint one-gpu.toml serves does:
85,000 tokens of cache at 0.78 of the card with a rank-32 adapter, 15,600 with `max_lora_rank = 128` (for q, k and v
joined at rank 96). Serving an adapter trained over bfloat16 on 4-bit weights widens the gap `kl_floor` and
`mean_mismatch` measure.

## Serving

`TinkerEngine.generate` sends the prompt's ids with the channel's stop tokens (none sends no stop list, so that the
end of text still stops it) and returns the sampled ids, a logprob for each, and `length` or `stop`. `load_adapter(name,
path)` reads the pointer in `path` (a checkpoint's weights) and makes a sampling client for its sampler checkpoint;
requests that name the adapter sample from it at once. The channel keeps the one before for turns in flight, as with
vLLM's adapters. `sleep` and `wake` do nothing, `processes` is empty, and `load_weights` refuses full weights: Tinker
serves adapters over its own models.

Tinker's sampling is tuned for throughput, and a turn there may take longer than on the local engine while a world
keeps running. The run's turn latency (the monitor's turn timings) is worth comparing with a local run's first.

## Costs

Prices (dollars per million tokens, re-checked on 2026-10-04 against
[Models & Pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/index.md)):

| Model | Prefill | Cached prefill | Sample | Train |
|---|---|---|---|---|
| `Qwen/Qwen3.5-9B` | 0.66 | 0.132 | 1.995 | 1.463 |
| `Qwen/Qwen3.5-4B` | 0.33 | 0.066 | 1.005 | 0.737 |

Storage is $0.10 per GB-month. "Train" is a forward and backward pass; a forward-only pass of a training client is
billed at the same price.

**An estimate for the Minecraft profile, from curriculum-9's records** (the ledger's checkpoints, and the inference
counters of the 14 groups its log has them for):

- curriculum-9 took 31 steps over 72 groups (2.3 groups a step); 22 steps were full, 384 segments each. It trained
  9,408 segments, 258 sampled tokens each on average (96,000 a full step).
- A group made 1,486 requests: 4,861 prompt tokens and 170 sampled tokens each.
- A trained segment is taken as its last turn's prompt and what was sampled, about 5,100 tokens, and training is taken
  to be billed on every token of a datum, prompts included (*unverified*; `billed_tokens` counts them so).

| Item | Tokens | Cost |
|---|---|---|
| Play, a group: prefill | 7.2M | $4.77 uncached; $1.72 if 80% hit the cache |
| Play, a group: sampling | 0.25M | $0.50 |
| Training, a full step, one optimizer step (`cispo`) | 1.97M | $2.88 |
| Training, a full step, several (`ppo`, and the forward pass for `old`) | 3.94M | $5.75 |
| **A full step, with its 2.3 groups of play** | | **about $8 (cached, one update) to $18 (uncached, several)** |
| **A run as long as curriculum-9 (72 groups, 31 steps)** | 520M prefill, 18M sampled, 48M trained | **about $230 to $520** |

Play costs more than training: every turn is played, while training takes 384 of a step's turns (equal-score groups
are skipped). Prompts are the lever: 4,861 tokens a turn, against 170 sampled. Whether turns hit Tinker's prefix cache,
and how often (a new sampler checkpoint each step starts its cache afresh), is to be read from the first run's billing
(`tinker billing usage`, or `RestClient.get_billing_usage`).

## Testing

`rollout_tinker.testing.FakeService` stands in for Tinker: a bigram model (a fixed table of logits by the token before,
plus a learnable table) with training runs, checkpoints, sampling and archives, its losses computed as Tinker's
documentation writes them. The project's tests show, with no network:

- the substitutions are exact: a step through the fake moves the model as `rollout_lora.step.PolicyStep` moves the same
  model, and its metrics are the same, for each row of the table above, with two passes and warm-up, and at a stop at
  `max_kl`; going on from a parent's state (the live client, or a new one) equals the LoRA step going on with its
  optimizer; a parent's weights alone start a fresh optimizer;
- a datum's rows: the shift by one, spans across turns, forced tokens left out;
- the engine's contract, and a published version sampled at once through the channel and the gateway;
- `weights = "peft"`: the cookbook's conversion (on the platform's transformers) and the joining of q, k and v, on a
  tiny model laid out as Qwen3.5, and `rollout merge` folding the result in exactly;
- a profile naming the trainer and the engine, the loop playing groups and stepping on the fake.

`tests/test_live.py` is opt-in (`-m tinker`) and skipped without a key. On `Qwen/Qwen3.5-4B` it checks the tokenizer
against our renderer, the sampling contract, the sampler's logprobs against a training pass's, a step's round trip to a
new sampler, and the adapter downloaded, converted and served by this machine's vLLM; it deletes what it made and
writes what it found to `~/.cache/rollout/tinker-smoke.json`. It costs well under ten cents:

```bash
cd implementations/rollout-tinker
flock ~/.cache/rollout/gpu.lock uv run --group local pytest -m tinker -s tests/test_live.py
```
