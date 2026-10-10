# Hosted training: Thinking Machines' Tinker, and Prime Intellect for contrast

**Status: built**, as `rollout_tinker`; the few parts still proposed are marked. A design note: see [Design
notes](README.md) for the others.

See [training](../libraries/rollout-train/training.md), [channels and engines](../libraries/rollout-train/channels.md),
[checkpoints](../libraries/rollout-train/checkpoints.md), [deploying](../guide/deploying.md),
[LoRA trainer](../implementations/rollout-lora.md), [the checkpoint graph](policy-dag.md)

This page asks how this system trains and samples through Thinking Machines' hosted API (Tinker) as naturally as it
does through `LoraTrainer` and `VllmEngine`, and compares Prime Intellect's offerings. The design below is implemented
as `rollout_tinker` ([Tinker trainer and engine](../implementations/rollout-tinker.md); [the code](#the-code) says where
it settled what this page left open). What remains proposed is marked so: a sweeper for released checkpoints, and
teachers of another base for distillation.

The external facts were checked on 2026-10-03 against the vendors' own documentation, SDK source and package indexes,
and each cites its source; Tinker's were re-checked against the SDK's source (0.32.0) and its prices on 2026-10-04. Anything not confirmed there is marked *unverified*, and the list at the end collects them.
Vendors change prices, model lists and APIs often (Tinker retired 22 models in June 2026), so re-check before acting.

## The answer in brief

- **Tinker fits our seams almost one to one.** Its sampler takes token ids and returns token ids with per-token
  logprobs, so our renderer stays in charge of the token format and the recorder's segments stay exact. Its trainer
  takes per-token `logprobs` and `advantages` in a `Datum`. It offers built-in `ppo`, `cispo` and
  `importance_sampling` losses, plus `forward_backward_custom` for any loss over logprobs. Checkpoints are named
  `tinker://` paths, and a new sampler checkpoint can be sampled from at once.
- **Our model is on Tinker.** `Qwen/Qwen3.5-9B` is available with a 64K context. It is LoRA only, with a default rank
  of 32, our setting. Training costs $1.463 per million tokens, sampling $1.995 and prefill $0.66 ($0.132 when
  cached).
- **Our objective maps onto the built-in losses exactly**, in four of five configurations, once importance weights
  and normalisation are folded into the advantages ([the objective on Tinker](#the-objective-on-tinker)). The segment
  ratio (GSPO) with more than one optimizer step per step needs the custom-loss path, which costs more.
- **It needed no change to `Trainer`, `Engine`, `Channel`, the checkpoints or the loop.** A checkpoint's weights
  directory holds a small pointer file naming its Tinker checkpoints (optionally beside a downloaded PEFT
  (parameter-efficient fine-tuning) adapter). The engine reads the pointer when the channel publishes. A run
  switches backends by naming a `tinker` trainer and inference provider of the cluster config in its settings.
- **Prime Intellect offers no equivalent API today.** Its shared hosted LoRA training stops taking new runs on
  2026-10-05. Its hosted training runs its own rollouts and loss through `verifiers` environments. Its useful pieces
  for us are GPU pods and the open-source `prime-rl` ([Prime Intellect, for contrast](#prime-intellect-for-contrast)).

## Tinker as it is today (verified)

### The product and the SDK

- Tinker trains LoRA adapters on open-weight models, from a training loop that runs on your own CPU machine. It is
  "LoRA fine-tuning, not full fine-tuning", and "you can download the weights of your trained model to use outside of
  Tinker" ([overview](https://tinker-docs.thinkingmachines.ai/tinker/index.md)).
- Install with `pip install tinker`. PyPI's current version is 0.32.0, which requires Python 3.11 or later
  ([PyPI](https://pypi.org/project/tinker/)). The API reference was generated from 0.30.4. The SDK is Apache-2.0
  ([GitHub](https://github.com/thinking-machines-lab/tinker)).
- Recipes, renderers and weight-export utilities are in `tinker-cookbook`, now at 0.5.7
  ([GitHub](https://github.com/thinking-machines-lab/tinker-cookbook)).
- General availability (no waitlist) was announced on 2025-12-12
  ([news](https://thinkingmachines.ai/news/tinker-general-availability/)).
- **Auth.** The `TINKER_API_KEY` environment variable, or `tinker auth login`, which stores a key pasted from the
  console in `~/.tinker/credentials.json`. `TINKER_PROJECT_ID` scopes sessions to a project
  ([quickstart](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/index.md),
  [changelog](https://tinker-docs.thinkingmachines.ai/changelog/index.md) for SDK 0.29.0 and 0.22.4).
- **Resources.** An organization holds projects. A project holds sessions, one per `ServiceClient`. A session holds
  training runs, one per `TrainingClient`, and sampling clients. A training run holds checkpoints. A project role,
  "Project Sampler", may sample from checkpoints without seeing how they were trained
  ([data model](https://tinker-docs.thinkingmachines.ai/tinker/data-model/index.md)).

### Primitives

All calls are on three clients made from `tinker.ServiceClient()`
([ServiceClient](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/serviceclient/index.md),
[TrainingClient](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/trainingclient/index.md),
[SamplingClient](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/index.md)). Each has an
`_async` variant.

**ServiceClient**

| Call | Does |
|---|---|
| `create_lora_training_client(base_model, rank=32, seed=None, train_mlp=True, train_attn=True, train_unembed=True, ...)` | Starts a new training run |
| `create_training_client_from_state(path)` | Resumes from a checkpoint's weights only |
| `create_training_client_from_state_with_optimizer(path)` | Resumes from a checkpoint's weights and Adam state |
| `create_sampling_client(model_path=... \| base_model=...)` | Samples from a checkpoint or from the base model |
| `create_rest_client()` | Manages checkpoints and sessions |
| `close(status)` | Finishes the session |

**TrainingClient**

| Call | Does |
|---|---|
| `forward(data, loss_fn)` | Forward pass only, without gradients |
| `forward_backward(data, loss_fn, loss_fn_config=None)` | Accumulates gradients |
| `forward_backward_custom(data, fn)` | Accumulates gradients for a loss you compute from the logprobs |
| `optim_step(AdamParams(...))` | Applies the accumulated gradients. AdamW; weight decay defaults to 0 |
| `save_state(name, ttl_seconds=None, overwrite=False)` | Saves weights and optimizer state: `tinker://RUN/weights/NAME` |
| `save_weights_for_sampler(name)` | Saves weights to sample from: `tinker://RUN/sampler_weights/NAME` |
| `save_weights_and_get_sampling_client()` | Both at once: saves the weights and returns a sampler |
| `get_tokenizer()` | The model's tokenizer |
| `get_info()` | The run's details |

**SamplingClient**

| Call | Does |
|---|---|
| `sample(prompt: ModelInput, num_samples, sampling_params, include_prompt_logprobs=False, topk_prompt_logprobs=0, topk_sample_logprobs=0, target_prompt_logprobs=None)` | Samples completions |
| `compute_logprobs(prompt)` | The prompt tokens' logprobs |

`SamplingClient` is picklable and safe to share between processes.

**Types**
([AdamParams](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/types/adamparams/index.md),
[SamplingParams](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/types/samplingparams/index.md),
[SampledSequence](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/types/sampledsequence/index.md))

| Type | Holds |
|---|---|
| `Datum` | `model_input: ModelInput` and `loss_fn_inputs: dict[str, tensor]`. Position *i* of the input predicts token *i*+1 |
| `AdamParams` | `learning_rate`, `beta1`, `beta2`, `eps`, `weight_decay`, and `grad_clip_norm` (0 means no clipping) |
| `SamplingParams` | `max_tokens`, `seed`, `stop` (a string, strings, or token ids; `[]` disables even EOS), `temperature`, `top_k`, `top_p` |
| `SampledSequence` | `tokens` (ids), `logprobs` (one per generated token), `stop_reason`, `sequence_id` |
| `ForwardBackwardOutput` | `loss_fn_outputs`: one dict per datum, including the learner's `logprobs`. Also `metrics` |

**Losses**

The [losses page](https://tinker-docs.thinkingmachines.ai/tinker/losses/index.md) defines them. Each loss is a sum
over tokens, not a mean.

| Loss | Takes | Computes |
|---|---|---|
| `cross_entropy` | `target_tokens`, `weights` (float weights allowed, (N, K) for top-K distillation) | −Σ w · log p |
| `importance_sampling` | `target_tokens`, `logprobs` (the sampler's), `advantages` | −Σ (p/q)·A |
| `ppo` | `target_tokens`, `logprobs`, `advantages` | −Σ min((p/q)·A, clip(p/q)·A) |
| `cispo` | `target_tokens`, `logprobs`, `advantages` | −Σ sg(clip(p/q))·log p·A |
| `dro` | `target_tokens`, `logprobs`, `advantages` | Σ of a quadratic penalty on log(p/q) |

- The ratio in every loss is *p/q*: the learner against the logprobs you pass in.
- The RL losses accept exactly those three keys. "Passing an extra key such as `weights` is rejected by the server",
  and an advantage of 0 removes a position from the loss.
- `ppo` takes `loss_fn_config={"clip_low_threshold": ..., "clip_high_threshold": ...}`, which are absolute bounds on
  the ratio. `cispo` defaults to 0.0 and 4.0.
- The documentation recommends a Kullback-Leibler (KL) divergence penalty go into the reward, not the loss.
- **Custom losses.** `forward_backward_custom` runs a forward pass, calls your function on the client, then runs a
  `forward_backward` on a linear surrogate with the same gradient. It takes "1.5x as many FLOPs" and "up to 3x as long
  (wall time)" ([custom](https://tinker-docs.thinkingmachines.ai/tinker/losses/custom/index.md)). Since SDK 0.30.1 it
  reports your loss as `loss:sum`.

### Sampling

- Prompts are token ids (`ModelInput.from_ints`), so no server-side chat template touches them.
- The sampler returns token ids and their logprobs (the cookbook's `TinkerTokenCompleter` asserts the logprobs are
  present). It returns a `stop_reason`, whose values include `"stop"` and `"length"` (the cookbook tests
  `stop_reason == "length"`).
- A stop token is part of the returned tokens. The cookbook's `parse_response_for_stop_token` expects "zero or one
  occurrence of the stop token", where zero means the length ran out
  ([renderers/base.py](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/renderers/base.py),
  [completers.py](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/completers.py)).
- The sampler can also score: `include_prompt_logprobs`, top-k logprobs at prompt or sampled positions, and the
  logprobs of chosen ids at chosen positions (`target_prompt_logprobs`, SDK 0.30.0). Distillation teachers would need
  these ([checkpoint graph](policy-dag.md#distillation)).

### Checkpoints and weights

- **Two kinds** ([checkpoints](https://tinker-docs.thinkingmachines.ai/tinker/howto/checkpoints/index.md)):
  - `SAMPLER_WEIGHTS`, from `save_weights_for_sampler`, can be sampled from.
  - `WEIGHTS`, from `save_state`, holds the weights and the Adam state, and can start or resume a training run.
- **Storage** costs $0.10 per GB per month. `ttl_seconds` must be between one hour and ten years, or none.
- **RestClient**
  ([RestClient](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/restclient/index.md)):

  | Call | Does |
  |---|---|
  | `list_checkpoints(run)`, `list_user_checkpoints()` | List checkpoints |
  | `delete_checkpoint_from_tinker_path(path)` | Deletes a checkpoint |
  | `set_checkpoint_ttl_from_tinker_path(path, ttl_seconds)` | Changes a checkpoint's expiry |
  | `get_checkpoint_archive_url_from_tinker_path(path)` | A signed URL to a tar of a sampler checkpoint |
  | `publish_checkpoint_from_tinker_path(path)` | Lets other Tinker users load it |
  | `get_billing_usage(start, end)` | Usage, with estimated costs since SDK 0.30.2 |

  `ServiceClient.copy_weights(path)` copies a checkpoint into another project.
- **The archive** holds `adapter_model.safetensors` and `adapter_config.json` with Tinker's own keys. The live test's,
  of a rank-32 `Qwen/Qwen3.5-4B` adapter, holds 496 float32 tensors named as a plain text model's layers
  (`base_model.model.model.layers.0.linear_attn.in_proj_q.lora_A.weight`: q, k and v apart, no `language_model`), and
  says `lora_alpha` 32 and `target_modules` `all-linear`.
  `tinker_cookbook.weights.build_lora_adapter(base_model, adapter_path, output_path)` remaps them to PEFT names for
  vLLM or SGLang. `build_hf_model` merges them into a full model instead
  ([PEFT adapter](https://tinker-docs.thinkingmachines.ai/cookbook/deployment/lora-adapter/index.md)). The CLI
  equivalent is `tinker checkpoint download PATH --output DIR`.
- **Import.** We found no call that uploads an adapter trained elsewhere into a Tinker training run (*unverified*
  that none exists).

### Models and prices

[Models & Pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/index.md) lists 29 models,
and [`models.json`](https://tinker-docs.thinkingmachines.ai/tinker/models.json) is the machine-readable contract.
Prices are dollars per million tokens. Cached prefill is 80% off. A forward-only pass from a training client is
billed at the training price.

| Tinker ID | Context | Prefill (cached) | Sample | Train | Relevance |
|---|---|---|---|---|---|
| `Qwen/Qwen3.5-9B` | 64K | 0.66 (0.132) | 1.995 | 1.463 | The base of minecraft-one-gpu's `cyankiwi/Qwen3.5-9B-AWQ-4bit`, in BF16 |
| `Qwen/Qwen3.5-4B` | 64K | 0.33 (0.066) | 1.005 | 0.737 | A cheaper relative for tests |
| `Qwen/Qwen3-8B` | 32K | 0.195 (0.039) | 0.60 | 0.44 | `rollout_qwen:qwen3`'s family; the cheapest |
| `Qwen/Qwen3.6-35B-A3B` | 64K | 0.54 (0.108) | 1.335 | 1.177 | A larger mixture-of-experts model, at about the 9B's cost |

The list also has Thinking Machines' own Inkling models, Nemotron 3 and 3.5, GLM-5.3, Kimi-K2.6, Qwen3.8-27B,
Qwen3.5-397B, gpt-oss and DeepSeek-V3.1. Prices rose on 2026-07-17, and models retire with notice: Qwen3-32B,
Qwen3-30B-A3B and the Llama models went on 2026-06-12
([deprecations](https://tinker-docs.thinkingmachines.ai/tinker/model-deprecations/index.md)).

### Limits, latency and inference endpoints

- **Shared training.** A training run shares a worker pool that does a forward-backward and an optimizer step per
  "clock cycle". Submit `forward_backward` and `optim_step` together, or a step costs three cycles, and pipeline
  batches. "Even if training with a small batch, you'll still see the same step time as a large batch"
  ([under the hood](https://tinker-docs.thinkingmachines.ai/tinker/under-the-hood/index.md)).
- **No client timeouts.** The SDK retries transient failures and detects stuck requests. "Don't" wrap requests in
  your own timeouts and retries. Latency varies with load: a request "that normally takes a minute can legitimately
  take much longer".
- **Concurrency.** A `SamplingClient` allows 2,000 requests in flight by default, set by the server (SDK 0.22.4).
  Rate limits surface as `RateLimitError` (HTTP 429). No per-account rate limits are published (*unverified*).
- **Throughput.** One worked trace in the docs shows about 1,500 to 1,900 generated tokens per second at 80 to 110
  requests in flight ([session metrics](https://tinker-docs.thinkingmachines.ai/tinker/session-metrics/index.md)).
  That is an example, not a guarantee.
- **OpenAI- and Anthropic-compatible endpoints.** They are in beta, "meant for testing and internal use with low
  internal traffic". The model is named by a sampler path. "For inference within your training runs (e.g. RL), we
  recommend using Tinker's standard sampling client"
  ([OpenAI-compatible](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/index.md)).
- **Serverless inference** is in beta, for Inkling only.

## Where a hosted backend plugs in (the repository as read)

This section is the repository as it was read for this design; [training](../libraries/rollout-train/training.md)
and [channels and engines](../libraries/rollout-train/channels.md) describe the code now. What the loop asks of
training and serving is already behind four seams, each implementation picked by `module:name`.

| Seam | Protocol (as written) | Today |
|---|---|---|
| Trainer | `Trainer.budget: Budget`; `async step(batch: Sequence[Weighted], *, seed: int, parent: Checkpoint \| None, into: Path) -> Step`, leaving `into/weights` and `into/state`, or raising `StepFailed` (`rollout_train/trainer.py`) | `LoraTrainer`: each step in a fresh GPU process, from the parent's adapter and `optimizer.pt` |
| Engine | `max_model_len`; `async generate(prompt, *, max_tokens, temperature, top_p, stop_token_ids, adapter) -> Generation(tokens, logprobs, finish_reason)`; `load_adapter(name, path)`, `remove_adapter`, `sleep`, `wake`, `processes`, `close` (`rollout_train/inference/channel.py`) | `VllmEngine`, with logprobs of the processed (post-temperature) distribution |
| Channel | `publish(adapter, path, version)` loads the adapter on every engine and keeps the one before for turns in flight. `Recorder.publish(channel, adapter, path, version)` is what the loop's `publish` calls | Unchanged by this proposal |
| Policies | `Policies.add(fence, policy, number, weights=Path, state=Path, ...)` keeps every file under the paths as a blob (`Manifest`) and appends the version. `files(manifest, directory)` puts them back on any machine. `thin(fence, policy, Retention)` releases old versions' blobs | Unchanged by this proposal |

The other parts behave as follows.

- **The loop** (`loop.py`):
  - Before a step, it materialises the parent's files (`Checkpoint(weights, state)`).
  - It calls `trainer.step(..., into=directory / "POLICY@N")`, so `into.name` is the new version's name.
  - It adds the version from `into/weights` and `into/state`.
  - It serves the version: `publish(channel, version.name, str(local weights directory), version.number)`.
  - It thins the policy.
- **The objective** (`rollout_objectives/terms.py`, composed from the components `rollout_train.objectives` declares)
  is a pure torch function of one segment. It takes logprobs now (`logprobs`), at the step's start (`old`), when
  sampled (`behavior`) and under the reference:
  - The importance weight is `old / behavior`, truncated at the cap (the `default` preset's).
  - The PPO ratio is `logprobs / old`, clipped to `1 - clip.low .. 1 + clip.high`.
  - A segment ratio is the geometric mean (GSPO).
  - `likelihood` is for imitation; a preference loss is of pairs or labelled examples.
- **The step** (`rollout_objectives/step.py`):
  - It computes `old` for every segment first.
  - It takes shuffled minibatches of about `tokens_per_step` sampled tokens, one optimizer step each.
  - The loss is divided by the minibatch's units (tokens, or segments for GSPO).
  - Gradients are clipped to `max_gradient_norm`.
  - The pass stops before a minibatch's update once the policy is `max_kl` from where it began.
- **The recorder**:
  - It renders prompts with the channel's renderer (`rollout_qwen:qwen35`, a Hugging Face chat template).
  - It samples through the channel.
  - It keeps segments: tokens, the spans the policy sampled with their weights version, and behaviour logprobs.
  - Forced tokens (a thinking block closed for lack of budget) are not in any span, so they are never trained on.
- **`Colocated`** pauses the channels and puts their engines to sleep while a trainer that shares their GPU steps.
  The profile's `training_gib` guard goes with it.

## The mapping

| Our concept | Tinker | Where the semantics differ |
|---|---|---|
| `Trainer.step(batch, seed, parent, into)` | `create_lora_training_client` (no parent) or `create_training_client_from_state_with_optimizer(parent's state path)`; then per minibatch `forward_backward_async` + `optim_step_async`; then `save_state(name)` and `save_weights_for_sampler(name)` | The weights stay remote. `into/weights` holds a pointer, and optionally a downloaded PEFT adapter ([versions](#versions-when-the-weights-are-remote)). A live client may be reused across steps when the parent is the state it last saved, an optimisation the protocol allows |
| `Weighted(segment, advantage)` | One `Datum`: `model_input = tokens[:-1]`, `target_tokens = tokens[1:]`, and for each sampled position *t* the row *t*−1 carries its behaviour (or old) logprob and its advantage. Other rows are 0 | Exact. Multi-turn segments with several spans are one datum. Forced tokens get advantage 0, which masks them |
| Advantages and importance weights | Per-token `advantages` and `logprobs` | The built-in losses take *one* reference logprob. Our factored objective is expressed by passing `old` as the reference and folding the truncated weight and 1/units into the advantages ([the objective](#the-objective-on-tinker)) |
| `old` (π at the step's start) | `forward_async(data, "cross_entropy")` on the parent, reading `loss_fn_outputs[i]["logprobs"]` | Billed at the training price, the same as a training pass. Not needed when a step is a single optimizer step |
| GSPO segment ratio | `forward_backward_custom_async(data, fn)`, with `fn` calling `rollout_objectives.terms.terms` on the returned logprobs | 1.5x the FLOPs, and up to 3x the wall time |
| `likelihood` (imitation) | `cross_entropy` with `weights = advantage / units` on sampled rows | Exact |
| `tokens_per_step`, `max_gradient_norm`, `learning_rate` | Minibatches are ours; `AdamParams(learning_rate, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, grad_clip_norm=...)` | Tinker's LoRA scaling (`lora_alpha`) is its own, not our `2 × rank`, so learning rates need re-tuning |
| `max_kl` stop | KL from each `forward_backward` output (the logprobs before that minibatch's update) against `old` | Exact if we await the forward-backward before submitting `optim_step` (two clock cycles per minibatch), or one minibatch late if pipelined. No call clears accumulated gradients (SDK 0.32.0), so after a stop the live client is not used again |
| `Budget(segment_tokens, segments)` | The model's context (64K for `Qwen3.5-9B`); no published batch limit (`get_server_capabilities()` reports per-model `max_context_length`) | Chosen for cost, not memory |
| `Checkpoint`, version files | `tinker://RUN/weights/NAME` (state), `tinker://RUN/sampler_weights/NAME` (sampling) | Named deterministically after the version, so a step retried after a crash overwrites its own checkpoint (`overwrite=True`) |
| `Channel.publish` → `Engine.load_adapter(name, path)` | `create_sampling_client_async(model_path=sampler path)`, kept under the adapter's name | The previous adapter's client is kept, so a turn in flight finishes on the weights it started with |
| `Engine.generate` | `sample_async(ModelInput.from_ints(prompt), 1, SamplingParams(max_tokens, temperature, top_p, stop=stop_token_ids))` | Returns ids, logprobs and `stop_reason` (`"length"` maps to `length`, the rest to `stop`). Whether logprobs are of the post-temperature distribution is *unverified*; it is moot at our default temperature and `top_p` of 1.0 |
| Renderer and tokenizer | None on the sampling path: token ids in and out | Our renderer stays the authority. Agreement with Tinker's tokenizer (`get_tokenizer()`) must be checked once per model |
| `Engine.sleep`, `wake`, `processes`; `Colocated`; `training_gib` | None | Moot: nothing on this machine to free. `colocated` must be false, or every step would pause play for nothing |
| Retention, release | `delete_checkpoint_from_tinker_path`, `ttl_seconds` | Releasing a checkpoint deletes the pointer blobs but not the remote checkpoints. A sweeper is proposed ([retention](#retention-proposed)) |
| Weights for local serving | `get_checkpoint_archive_url_from_tinker_path` or `tinker_cookbook.weights.download`, then `build_lora_adapter` | MBs to hundreds of MBs per version, per step (size *unverified*) |
| Throughput, `Channel.take()` | Session metrics and a Perfetto trace in the console | Our counters keep working: they count at the channel |
| Costs | Per token: prefill, sample, train; storage per GB-month | No GPU of our own is needed. Play continues during steps |

## The objective on Tinker

Notation per sampled token: `p` is now (with gradient), `old` is at the step's start, `b` is behaviour (what the
engine recorded), `A` is the segment's advantage, `U` is the minibatch's units, and `w = min(exp(old − b), truncate)`.
Ours is

```text
token ratio:   loss = Σ_tokens −w · min(r·A, clip(r, 1−lo, 1+hi)·A) / U,     r = exp(p − old)
```

**Several minibatches, token ratio.** Tinker's `ppo` computes `−Σ min(ρ·A′, clip(ρ, L, H)·A′)` with `ρ = exp(p − q)`.
Pass `q = old`, `A′ = A·w/U`, `L = 1 − clip_low` and `H = 1 + clip_high`. Because `w/U > 0`, each term equals
`−(w/U)·min(r·A, clip(r)·A)`, which is ours in value and in gradient (w is a constant in both). It needs `old`, which
costs one forward pass of the batch at the training price.

**One minibatch (the step is one optimizer step), token ratio.** Then `old = p` in value, `r = 1`, and our gradient
is `−w·A/U·∇p`. Tinker's `cispo` with `q = b`, `L = 0` and `H = truncate` (a large number when `truncate` is None)
has gradient `−clip(exp(p − b), 0, truncate)·A/U·∇p`, which is the same. No `old` pass is needed, and the
forward-backward's own output logprobs are `old`, so `kl_floor`, `mean_mismatch` and `truncated_fraction` come out
exactly.

**Segment ratio (GSPO).**

- With one minibatch, `importance_sampling` with `q = old` and `A′ = w_seg·A/(n_seg·U)` gives our gradient. It needs
  the `old` pass.
- With several minibatches, the clip acts on the segment's ratio, which no built-in loss expresses. Use
  `forward_backward_custom`, whose function calls `rollout_objectives.terms.terms` unchanged (pure torch, on CPU).
- Inside that function the KL check can return a zero loss and skip `optim_step`. A zero gradient would still move
  Adam through its momentum, so skipping is necessary.

**Likelihood.** Use `cross_entropy` with `weights = A/U` on the sampled rows.

| Objective, ratio | Optimizer steps per step | Tinker calls | Extra cost |
|---|---|---|---|
| `policy_gradient`, `token` | 1 | `cispo` (q = behaviour) | none |
| `policy_gradient`, `token` | many | `forward` (old), then `ppo` per minibatch (q = old) | one forward pass |
| `policy_gradient`, `segment` | 1 | `forward` (old), then `importance_sampling` | one forward pass |
| `policy_gradient`, `segment` | many | `forward` (old), then `forward_backward_custom` per minibatch | forward pass, plus custom (1.5x FLOPs, up to 3x time) |
| `likelihood` | any | `cross_entropy` | none |

How many optimizer steps a step should make is now a cost and behaviour choice, not a memory one.

- **Local.** The default (`tokens_per_step = 4096`) makes dozens of optimizer steps per step.
- **On Tinker.** Each optimizer step is at least one clock cycle. Tinker's LoRA primer warns that LoRA "is less
  tolerant of large batch sizes than full fine-tuning"
  ([LoRA primer](https://tinker-docs.thinkingmachines.ai/tinker/lora-primer/index.md)), which argues against a
  single giant minibatch.
- **Recommendation.** Start with a few minibatches per step (`tokens_per_step` of about 32K to 64K sampled tokens)
  and the `ppo` path. Compare it with a single minibatch on `cispo` as an experiment.

## The code

`rollout_tinker` ([Tinker trainer and engine](../implementations/rollout-tinker.md)) implements milestone 1 and
`weights = "peft"`: `TinkerTrainer`, `TinkerEngine`, pointer files, a fake service for tests, and the live smoke test of
milestone 0. Neither `Trainer`, `Engine`, `Channel`, `Checkpoints` nor the loop changed. Where it settles what the
proposal left open:

- **In the workspace, with the SDK alone.** `implementations/rollout-tinker` is a workspace member (the `tinker`
  extra); `tinker==0.32.0` added eight small packages to the workspace's lock and moved none. The archive's renaming
  into PEFT's layout is ours (`rollout_tinker.weights.peft_adapter`): `tinker-cookbook` pins `transformers<=5.5.4`,
  which the workspace's vLLM and Gemma renderers exclude. On a synthetic archive with the live test's 496 names and
  247.6 MB, it writes what the cookbook's `build_lora_adapter` (0.5.7) wrote, byte for byte.
- **The objective is shared, not copied.** The package depends on `rollout-lora` for `Objective`, `terms`,
  `minibatches` and `sampled`; importing them loads none of its GPU code.
- **Names.** A checkpoint's id (sixteen letters) names its Tinker checkpoints, so whether Tinker accepts `@` no longer
  matters. `save_weights_for_sampler` takes no `overwrite`; a retried step has a new id.
- **`weights/tinker.json` names the training state as well as the sampler checkpoint**, so a step given a parent's
  weights without its state (`rollout imitate` starts its optimizer afresh) loads them with
  `create_training_client_from_state` and a fresh optimizer, warmed up.
- **The segment ratio** takes the custom loss with one optimizer step too (its logprobs are `old`), rather than
  `importance_sampling` after a forward pass: the same cost.
- **Adam.** `AdamParams`' defaults are 0.95 and 1e-12, not torch's; the trainer passes torch's (`beta1` 0.9, `beta2`
  0.999, `eps` 1e-8).
- **Qwen3.5's q, k and v.** Renamed, the PEFT adapter keeps Tinker's `in_proj_q`, `in_proj_k` and `in_proj_v`, and
  vLLM 0.30 adapts Qwen3.5's linear attention only as `in_proj_qkv` (packed with `in_proj_z`). The conversion joins the
  three: A stacked and B block-diagonal (three times the rank), or B stacked at the same rank when they share one A.

The tests compare a step through the fake, whose losses are the documented formulas, with `PolicyStep` on the same
model, for each row of [the table above](#the-objective-on-tinker): the models move the same, and the metrics agree.

### Versions when the weights are remote

| Path under `into` | Holds |
|---|---|
| `weights/tinker.json` | `{"sampler": "tinker://RUN/sampler_weights/ID", "state": "tinker://RUN/weights/ID", "base_model": "Qwen/Qwen3.5-9B", "rank": 32}` |
| `weights/adapter_config.json`, `adapter_model.safetensors` | Only with `weights = "peft"`: the adapter in PEFT's layout, which `VllmEngine` loads and `rollout merge` folds in |
| `state/tinker.json` | `{"state": ..., "sampler": ..., "sdk": "0.32.0"}` |
| `state/minibatches.jsonl` | What each minibatch did, as `LoraTrainer` writes it |

The checkpoints, their manifests and the ledger are unchanged: the pointers are blobs like any weights.

### A preset

The preset `minecraft-tinker` (`deploy/chart/rollout/files/presets/minecraft-tinker.toml`) trains and samples the full
`Qwen/Qwen3.5-9B` at Tinker with minecraft-one-gpu's settings where they still apply, and says the local-serving
alternative in its comments.

### Retention (proposed)

`Checkpoints.thin` releases a checkpoint by deleting its blobs. For a Tinker checkpoint those are the pointer files, so
the remote checkpoints remain and keep costing storage. They are named after checkpoint ids, so a sweeper could
reconcile them:

- list `RestClient.list_user_checkpoints()`;
- map each checkpoint's name back to a checkpoint;
- delete, with `delete_checkpoint_from_tinker_path`, those of released checkpoints, and those of no checkpoint once
  older than a day (steps that died before their checkpoint was appended).

TTLs alone do not fit: retention keeps some checkpoints indefinitely, and the trainer cannot know at save time which.
Until then, `tinker checkpoint delete` deletes them by path or by run.

## Mixed setups

| Trains | Serves | Works | What to know |
|---|---|---|---|
| Tinker | Tinker | Yes | Pointers only; no weight transfer |
| Tinker | Local vLLM | Yes, with `weights = "peft"` | Each step downloads the archive and converts it (renamed, then q, k and v joined). See below |
| Local `LoraTrainer` | Tinker | No | No call imports an adapter into Tinker (*unverified*). `TinkerTrainer` raises `StepFailed` for such a parent instead of silently starting over |
| Tinker, then local `LoraTrainer` | Local vLLM | Through a merge | A Tinker adapter has other layers and ranks than `LoraTrainer`'s; `rollout merge` folds it into the model, and a run starts from that full checkpoint |
| Tinker | A frontier model, beside it | Yes, unrelated | Other slots of a run can be bound to `DirectModel` endpoints as today |

**Training remotely and serving with local vLLM** (measured on the 16 GB card with a synthetic adapter in Tinker's
layout, converted as `weights = "peft"` converts one; no Tinker call):

- vLLM 0.30 loads the converted adapter, its q, k and v joined into `in_proj_qkv`, and it changes what is sampled.
  `max_lora_rank` must reach the adapter's largest rank: 32 when the three share one A, 96 (so 128) when each has its
  own. Which Tinker sends is *unverified*; the smoke test records its names.
- The full `Qwen/Qwen3.5-9B` does not fit at an 8,192-token context. Quantized to FP8 as it loads, its weights take
  10.8 GiB (the embeddings and the output layer stay bfloat16, 2 GiB each), and at 0.88 of the card (the most that
  was free beside 1.5 GiB other programs held) with 1,024-token batches, 0.25 GiB is left for the cache: not one turn.
- The 4-bit checkpoint (`cyankiwi/Qwen3.5-9B-AWQ-4bit`) fits: at 0.78 of the card, 85,000 tokens of cache with a
  rank-32 adapter, 15,600 with `max_lora_rank = 128`. It widens the gap between behaviour and trainer logprobs (the
  adapter was trained over bfloat16); the truncated importance weight absorbs it, and `kl_floor` and `mean_mismatch`
  measure it.
- Tinker adapts the unembedding by default; the trainer asks it not to (`train_unembed=False`), and whether vLLM serves
  an `lm_head` adapter is *unverified*.

**Distillation teachers** from the [checkpoint graph](policy-dag.md#distillation) gain an option. A teacher of another
base (`Qwen/Qwen3.5-397B-A17B`, say) can score the student's tokens with `target_prompt_logprobs` or top-k prompt
logprobs, since the Qwen3.5 family shares a tokenizer (*unverified* across sizes; to check as for the student).

## Gaps and risks

- **Tokenizer and renderer agreement.** Our renderer encodes with the tokenizer of the channel's `model`. For a
  Tinker channel that is `Qwen/Qwen3.5-9B` from the Hugging Face Hub, which should equal Tinker's `get_tokenizer()`.
  This should be checked once per model (vocabulary hash, and a rendered conversation encoded both ways), not
  assumed. Tinker pins tokenizer revisions for some models (SDK 0.21.0 did so for Kimi-K2.6). Sampling never applies
  a template, so nothing else can drift.
- **Behaviour against trainer numerics.** Tinker's sampler and trainer are separate systems. The factored objective
  exists for this, and the first live run should report `kl_floor`. Whether sampled logprobs are post-temperature is
  *unverified*; at our defaults (temperature and `top_p` of 1.0) they are the same.
- **Latency in real-time worlds.** Tinker is tuned for throughput. A turn that takes long while a Paper server keeps
  ticking changes the game, and Tinker asks us not to cut requests short. Measure turn latency against the local
  engine before trusting results, and check that no recorder or runner timeout sits on the sampling path.
- **Costs scale with tokens, not hours.** From curriculum-9's records (31 steps over 72 groups; a group made 1,486
  requests of 4,861 prompt and 170 sampled tokens; a full step trained 384 segments of about 5,100 tokens), a full step
  with its 2.3 groups of play costs about $8 (80% of prefill cached, one optimizer step) to $18 (none cached, several
  steps and the `old` pass), and a run as long as curriculum-9 about $230 to $520. Play costs more than training:
  every turn is played, at 4,861 prompt tokens. The assumptions and the table are in
  [costs](../implementations/rollout-tinker.md#costs). Storage adds $0.10 per GB-month per kept checkpoint.
- **What the API gives, from the SDK's source (0.32.0).**
  - No call clears accumulated gradients: after a stop at `max_kl` the trainer does not use that client again.
  - `create_training_client_from_state*` makes a new LoRA training client, and so a new training run, and loads the
    state into it. Reusing the live client avoids it in steady state.
  - `forward_backward_custom` runs a training client's `forward` and then its `forward_backward`: both training passes,
    billed at the training price.
  - `optim_step` returns `metrics`, a dictionary whose keys are not documented: whether it holds the gradient norm is
    *unverified*.
  - LoRA rank limits and per-account rate limits are not published (*unverified*).
- **Model churn.** Models retire with a few months' notice. A policy's versions are tied to its base, so a retired
  base strands a remote policy unless its adapters were downloaded (`weights = "peft"` keeps a local copy in the blob
  store).
- **Secrets.**
  - The key comes from `TINKER_API_KEY` or `~/.tinker/credentials.json` and never appears in run settings, a ledger
    record or a log line.
  - `[trainer] project` holds a project id, which is not a secret.
  - Errors are reduced to their type and message before they reach `StepFailed` (whose text the ledger keeps).
  - A run's driver (which samples a Tinker channel) and its trainer actor need the key in their environment.
  - A narrower credential for runners (a Project Sampler) would limit what a leaked key can do. Per-role API keys are
    *unverified*.
- **Vendor lock-in.** It is limited by design: the ledger, the recorder's segments, the renderer and the objective
  stay ours. With `weights = "peft"`, the adapters are ours too.

## Prime Intellect, for contrast

These facts were verified by a parallel investigation of Prime Intellect's documentation (`docs.primeintellect.ai`,
its `llms-full.txt` and OpenAPI spec), the live inference model list, and the `prime-rl` (v0.9.0), `verifiers` and
`prime` repositories, with the key points re-checked against the downloaded docs. The sources are listed at the end.

- **Hosted Training is environment-driven.**
  - Lab is the Environments Hub, Hosted Training and Hosted Evaluations, built on `prime-rl` and `verifiers`. A run
    names `[[env]]` entries (verifiers environments), and Prime plays the rollouts, computes advantages and applies
    its loss.
  - The OpenAPI spec has no `forward_backward` or `optim_step` style endpoint. You cannot submit your own rollouts,
    advantages or behaviour logprobs.
  - Every Hosted Training page now says: "Shared Hosted Training for LoRA runs will stop accepting new runs on
    October 5, 2026". The replacement, dedicated runs, is documented as full fine-tuning in closed beta (64 GPUs per
    run), configured with `prime-rl`'s own TOML.
  - The legacy LoRA price for `Qwen/Qwen3.5-9B` was $0.20 input, $0.60 output and $0.60 training per million tokens.
    Its configuration exposed `lora_alpha` but no rank.
- **Prime Inference.**
  - It is an OpenAI-compatible endpoint at `https://api.pinference.ai/api/v1`, with `Authorization: Bearer
    $PRIME_API_KEY`. It routes 128 models, 6 of them on Prime's own hardware, among them `Qwen/Qwen3.5-9B` at $0.18
    input and $0.54 output.
  - Whether it accepts token-id prompts or returns token ids and per-token logprobs is *unverified*, so it cannot
    back our recorder.
  - Adapters from Hosted Training deploy in "a few minutes" and are named `BASE:<adapter_id>`. There is no instant
    swap, and no upload of an adapter trained elsewhere.
- **Compute.** GPU pods (`prime availability list`, `prime pods create`) and CPU sandboxes (`prime-sandboxes`, $0.02
  per vCPU-hour; GPU sandboxes "coming soon").
- **`prime-rl`, open source.** It has three processes:
  - **Inference:** vLLM with a token-in `/inference/v1/generate` route, plus `/update_weights` and
    `/load_lora_adapter`.
  - **Orchestrator:** owns the verifiers environments and the advantages.
  - **Trainer:** FSDP2. It reads packed batches from files or ZMQ, and broadcasts weights to `broadcasts/step_N`
    behind a marker handshake.

  Further details:
  - Off-policy samples are bounded by `max_off_policy_steps` (8 by default).
  - Its default loss masks on the absolute probability difference, with a squared log-ratio penalty.
  - `[trainer.loss] type = "custom"` takes trainer logprobs, inference logprobs, advantages and masks, so our
    objective could be written for it.
  - Its wire type `TrainingSample` (token ids, mask, logprobs, temperatures, per-token advantages) is close to our
    `Weighted`.
- **`verifiers`.** v1 has tasksets and harnesses (v0's `MultiTurnEnv` and `ToolEnv` are deprecated). A custom
  `Harness.launch(ctx, trace, runtime, endpoint, ...)` could run our program against Prime's interception server.
  Tokens and logprobs would then be rebuilt by Prime's renderers from chat-completions traffic, not recorded by ours.
  Whether hosted environments can reach our tool servers or a Minecraft server is *unverified* (the docs warn some
  external APIs may not be reachable). The other direction, verifiers environments played here with our recorder
  recording the tokens, is [Prime Intellect's verifiers](prime-compat.md).

| | Tinker | Prime Intellect |
|---|---|---|
| Who runs the loop | We do: primitives over HTTP | Prime does (hosted), or we self-host `prime-rl` |
| Our rollouts, advantages, behaviour logprobs | Yes, per token in a `Datum` | Not to the hosted service. Yes to a self-hosted `prime-rl` trainer, through its batch files |
| Our loss | Built-in `ppo`, `cispo`, `importance_sampling`, or a custom loss | Self-hosted only (`type = "custom"`) |
| Token-in sampling with logprobs | Yes (`SamplingClient`) | Self-hosted `prime-rl` vLLM route; hosted inference *unverified* |
| A new adapter sampled at once | Yes: a new `SamplingClient` from the sampler path | Self-hosted: vLLM reloads in place. Hosted: minutes per deployment |
| `Qwen3.5-9B` | Yes, LoRA, 64K | Inference yes; hosted LoRA training ends 2026-10-05 |
| GPUs we manage | None | Pods, if we self-host |

**How Prime's pieces would plug in.**

| Seam | Piece | Verdict |
|---|---|---|
| Trainer | A `prime-rl` trainer behind our `Trainer` | Possible but heavy. `Weighted` becomes `TrainingSample`; we write its batch files, acknowledge its handshake and copy `broadcasts/step_N` into the blob store. Its trainer is a long-running lockstep process, not a call per step, and its internal formats are young (v0.9.0) |
| Engine | `prime-rl`'s vLLM server | Natural: token-in generation and `/load_lora_adapter`. Its sleep and wake must be checked |
| Compute | GPU pods running our own `LoraTrainer` and `VllmEngine` | The lowest-risk way to use Prime: nothing in our design changes, and it scales the one-GPU preset up |
| Training | Hosted training through a verifiers-wrapped environment | Not recommended. It closes to LoRA in two days. It would also hand the loop, the advantages, the loss, the staleness policy and token recording to Prime, giving up the ledger, the recorder's exact segments, `ID@N` versions and our renderer |

Prime Intellect is a source of GPUs and open-source parts. Tinker is the managed service that keeps our loop.

## Milestones

**0. The live smoke test** (`tests/rollout_tinker/test_live.py`, `ROLLOUT_TINKER=1`, skipped without a key)
runs on `Qwen/Qwen3.5-4B`, the same family at half the price, for well under ten cents:

1. **Tokenizer agreement.** `rollout_qwen.qwen35("Qwen/Qwen3.5-4B")` renders a conversation with tools; Tinker's
   tokenizer encodes its text to the same ids, and the vocabularies hash the same.
2. **Sampling contract.** A sample with our stop ids returns `stop` or `length`, a logprob per token, and the stop
   token among the tokens.
3. **Numerics.** A training client's `forward` recomputes the sampled tokens' logprobs; the mean absolute difference
   is the `mean_mismatch` to expect.
4. **Round trip.** One `cispo` step on four segments, its sampler checkpoint loaded by `TinkerEngine` and sampled; the
   archive's configuration and names are recorded.
5. **Local serving.** The same with `weights = "peft"`: the adapter served by this machine's vLLM.
6. **Cleanup.** Each test deletes the checkpoints it made.

**1. `rollout_tinker`**: the engine, the trainer with every row of [the table](#the-objective-on-tinker),
`likelihood` for `rollout imitate`, pointer files, the live client, `StepFailed`, and `weights = "peft"`. Its tests
run on a fake service (a bigram whose losses are the documented formulas), with no network:

- a step through the fake moves the model as `PolicyStep` does, with the same metrics, for each row of the table, with
  passes, warm-up and a stop at `max_kl`;
- going on from a parent's state (on the live client or a new one) equals the LoRA step going on with its optimizer, a
  parent's weights alone start a fresh optimizer, and a parent not trained on Tinker is refused;
- a datum's alignment, the engine's contract and a publish through `Channel` and `Recorder`;
- the conversion with q, k and v joined, and `rollout merge` folding it in exactly, on a tiny model laid out as
  Qwen3.5;
- a run naming `rollout_tinker`'s trainer and engine, the loop playing groups and stepping.

**First real run.** A short Minecraft run on `Qwen/Qwen3.5-9B` (the preset `minecraft-tinker`, five
groups, two steps), audited before anything longer:

- Check the launch flags.
- Check the share of actions that fail.
- Check that `kl_floor` is small and that the steps move the policy (`kl_moved`, `clip_fraction` above zero).
- Compare turn latency and cost per step with the estimates above (`billed_tokens`, and Tinker's billing usage).

**2. Proposed.**

- The sweeper ([retention](#retention-proposed)).
- Teachers of another base for distillation.

## Unverified points

- Whether there is an API to import an adapter trained elsewhere.
- What `optim_step`'s metrics hold (the gradient norm?).
- LoRA rank limits.
- Per-account rate limits.
- Whether sampled logprobs are post-temperature.
- Whether vLLM serves an unembedding adapter (vLLM 0.30's Qwen3.5 maps `lm_head` adapters onto its output embeddings;
  not run). The trainer does not adapt the unembedding (`train_unembed=False`).
- What training is billed on: every token of a datum (assumed), or the trained rows only.
- How often turns hit Tinker's prefix cache.
- Checkpoint sizes: 86.5 million parameters for a rank-32 adapter of the 9B model without the output layer, in
  float32 in an archive; the Adam state beside it.
- Whether the Qwen3.5 tokenizer is the same across sizes (for teachers).
- For Prime Intellect: whether Prime Inference supports token-id prompts and logprobs; whether dedicated runs accept
  LoRA; whether hosted environments can reach outside services.
- The "$150 free credits" for new Tinker users, reported by a third-party blog
  ([beam.cloud](https://www.beam.cloud/blog/tinker-model-pricing)), is not on Thinking Machines' own pages and is
  left out above.

## Sources

**Thinking Machines (Tinker)**

- [Whole documentation in one file](https://tinker-docs.thinkingmachines.ai/llms-full.txt) (index: `llms.txt`)
- [Overview](https://tinker-docs.thinkingmachines.ai/tinker/index.md)
- [Quickstart](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/index.md)
- [SDK cheatsheet](https://tinker-docs.thinkingmachines.ai/tinker/sdk-cheatsheet/index.md)
- [Models & Pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/index.md) and
  [models (JSON)](https://tinker-docs.thinkingmachines.ai/tinker/models.json)
- [Model deprecations](https://tinker-docs.thinkingmachines.ai/tinker/model-deprecations/index.md)
- [Data model & permissions](https://tinker-docs.thinkingmachines.ai/tinker/data-model/index.md)
- [Loss functions](https://tinker-docs.thinkingmachines.ai/tinker/losses/index.md), `.../losses/ppo/index.md`,
  `.../losses/cispo/index.md`, `.../losses/importance-sampling/index.md`, `.../losses/custom/index.md`
- [LoRA primer](https://tinker-docs.thinkingmachines.ai/tinker/lora-primer/index.md)
- [Checkpoints](https://tinker-docs.thinkingmachines.ai/tinker/howto/checkpoints/index.md)
- [OpenAI-compatible inference](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/index.md)
- [Under the hood](https://tinker-docs.thinkingmachines.ai/tinker/under-the-hood/index.md)
- [Session metrics](https://tinker-docs.thinkingmachines.ai/tinker/session-metrics/index.md)
- [API reference](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/serviceclient/index.md),
  `.../trainingclient/index.md`, `.../samplingclient/index.md`, `.../restclient/index.md`, `.../types/index.md`
- [PEFT export](https://tinker-docs.thinkingmachines.ai/cookbook/deployment/lora-adapter/index.md)
- [Changelog](https://tinker-docs.thinkingmachines.ai/changelog/index.md)
- [SDK](https://github.com/thinking-machines-lab/tinker) and [tinker on PyPI](https://pypi.org/project/tinker/)
- [Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) (`tinker_cookbook/completers.py`,
  `tinker_cookbook/renderers/base.py`)
- [Announcements](https://thinkingmachines.ai/news/announcing-tinker/) and
  [tinker general availability](https://thinkingmachines.ai/news/tinker-general-availability/)

**Prime Intellect**

- [What Lab is, and the LoRA shutdown notice](https://docs.primeintellect.ai/hosted-training/what-is-lab)
- [Models and pricing](https://docs.primeintellect.ai/hosted-training/models-and-pricing)
- [Advanced configs](https://docs.primeintellect.ai/hosted-training/advanced-configs)
- [Full fine-tuning](https://docs.primeintellect.ai/hosted-training/full-finetuning)
- [Volumes](https://docs.primeintellect.ai/hosted-training/volumes)
- [Dedicated runs API](https://docs.primeintellect.ai/api-reference/training/create-dedicated-run)
- [Inference](https://docs.primeintellect.ai/inference/overview) and
  [adapter deployments](https://docs.primeintellect.ai/inference/adapter-deployments)
- [Sandboxes](https://docs.primeintellect.ai/sandboxes/overview)
- [prime-rl](https://docs.primeintellect.ai/prime-rl/overview), `.../prime-rl/training`, `.../prime-rl/algorithms`,
  `.../prime-rl/advanced`, and [prime-rl on GitHub](https://github.com/PrimeIntellect-ai/prime-rl) (v0.9.0)
- [verifiers](https://docs.primeintellect.ai/verifiers/overview) and
  [harnesses](https://docs.primeintellect.ai/verifiers/v1/harnesses)
