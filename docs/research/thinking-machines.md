# Hosted training: Thinking Machines' Tinker, and Prime Intellect for contrast

See [training](../libraries/rollout-train/training.md), [channels and engines](../libraries/rollout-train/channels.md),
[checkpoints](../libraries/rollout-train/checkpoints.md), [deploying](../guide/deploying.md),
[LoRA trainer](../implementations/rollout-lora.md), [the policy graph](policy-dag.md)

**A proposal.** This page asks how this system could train and sample through Thinking Machines' hosted API (Tinker)
as naturally as it does through `LoraTrainer` and `VllmEngine`, and compares Prime Intellect's offerings. Nothing
proposed here is implemented: there is no `rollout_tinker` package. Sections say which parts describe the repository
as it is and which parts are proposed.

The external facts were checked on 2026-10-03 against the vendors' own documentation, SDK source and package indexes,
and each cites its source. Anything not confirmed there is marked *unverified*, and the list at the end collects them.
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
- **The design needs no change to `Trainer`, `Engine`, `Channel`, `Policies` or the loop.** A version's weights
  directory holds a small pointer file naming its Tinker checkpoint (optionally beside a downloaded PEFT adapter). The
  engine reads the pointer when the channel publishes. A profile switches backends by naming `rollout_tinker`
  classes.
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
- The documentation recommends a KL penalty go into the reward, not the loss.
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
  these ([policy graph](policy-dag.md#distillation)).

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
- **The archive** holds `adapter_model.safetensors` and `adapter_config.json` with Tinker's own keys.
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
| `Qwen/Qwen3.5-9B` | 64K | 0.66 (0.132) | 1.995 | 1.463 | The base of our profile's `cyankiwi/Qwen3.5-9B-AWQ-4bit`, in BF16 |
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

## Where a hosted backend plugs in (the repository as it is)

What the loop asks of training and serving is already behind four seams. A profile picks every implementation
by `module:name`.

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
- **The objective** (`rollout_lora/objectives.py`) is a pure torch function of one segment. It takes logprobs now
  (`logprobs`), at the step's start (`old`), and when sampled (`behavior`):
  - The importance weight is `old / behavior`, truncated at `truncate`.
  - The PPO ratio is `logprobs / old`, clipped to `1 - clip_low .. 1 + clip_high`.
  - A segment ratio is the geometric mean (GSPO).
  - `likelihood` is for imitation.
- **The step** (`rollout_lora/step.py`):
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
| GSPO segment ratio | `forward_backward_custom_async(data, fn)`, with `fn` calling `rollout_lora.objectives.terms` on the returned logprobs | 1.5x the FLOPs, and up to 3x the wall time |
| `likelihood` (imitation) | `cross_entropy` with `weights = advantage / units` on sampled rows | Exact |
| `tokens_per_step`, `max_gradient_norm`, `learning_rate` | Minibatches are ours; `AdamParams(learning_rate, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, grad_clip_norm=...)` | Tinker's LoRA scaling (`lora_alpha`) is its own, not our `2 × rank`, so learning rates need re-tuning |
| `max_kl` stop | KL from each `forward_backward` output (the logprobs before that minibatch's update) against `old` | Exact if we await the forward-backward before submitting `optim_step` (two clock cycles per minibatch), or one minibatch late if pipelined. No call clears accumulated gradients (*unverified*), so after a stop the live client is discarded |
| `Budget(segment_tokens, segments)` | The model's context (64K for `Qwen3.5-9B`); no published batch limit (`get_server_capabilities()` reports per-model `max_context_length`) | Chosen for cost, not memory |
| `Checkpoint`, version files | `tinker://RUN/weights/NAME` (state), `tinker://RUN/sampler_weights/NAME` (sampling) | Named deterministically after the version, so a step retried after a crash overwrites its own checkpoint (`overwrite=True`) |
| `Channel.publish` → `Engine.load_adapter(name, path)` | `create_sampling_client_async(model_path=sampler path)`, kept under the adapter's name | The previous adapter's client is kept, so a turn in flight finishes on the weights it started with |
| `Engine.generate` | `sample_async(ModelInput.from_ints(prompt), 1, SamplingParams(max_tokens, temperature, top_p, stop=stop_token_ids))` | Returns ids, logprobs and `stop_reason` (`"length"` maps to `length`, the rest to `stop`). Whether logprobs are of the post-temperature distribution is *unverified*; it is moot at our default temperature and `top_p` of 1.0 |
| Renderer and tokenizer | None on the sampling path: token ids in and out | Our renderer stays the authority. Agreement with Tinker's tokenizer (`get_tokenizer()`) must be checked once per model |
| `Engine.sleep`, `wake`, `processes`; `Colocated`; `training_gib` | None | Moot: nothing on this machine to free. `colocated` must be false, or every step would pause play for nothing |
| Retention, release | `delete_checkpoint_from_tinker_path`, `ttl_seconds` | `Policies.thin` deletes the pointer blobs but not the remote checkpoints. A sweeper is needed ([retention](#retention)) |
| Weights for local serving | `get_checkpoint_archive_url_from_tinker_path` or `tinker_cookbook.weights.download`, then `build_lora_adapter` | MBs to hundreds of MBs per version, per step (size *unverified*) |
| Throughput, `Channel.take()` | Session metrics and a Perfetto trace in the console | Our counters keep working: they count at the channel |
| Costs | Per token: prefill, sample, train; storage per GB-month | No GPU of our own is needed. Play continues during steps |

## The objective on Tinker

Notation per sampled token: `p` is now (with gradient), `old` is at the step's start, `b` is behaviour (what the
engine recorded), `A` is the segment's advantage, `U` is the minibatch's units, and `w = min(exp(old − b), truncate)`.
Ours is

```
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
  `forward_backward_custom`, whose function calls `rollout_lora.objectives.terms` unchanged (pure torch, on CPU).
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

## Proposed code

### A package

`implementations/rollout-tinker`, import name `rollout_tinker`, depending on `rollout`, `rollout-train`, `tinker`
and `torch` (CPU, for the custom loss and the equivalence tests). It would join the workspace's default dependencies,
since it needs no GPU.

**Sharing the objective with `rollout_lora`.** The trainer reuses `Objective`, `terms`, `minibatches` and `sampled`,
which are pure torch and pure Python. Importing them from `rollout_lora` would drag its GPU dependencies
(`flash-linear-attention`, `accelerate`) into a CPU-only package. Two choices:

- move them into a small torch-only package that both trainers depend on;
- or make `rollout-lora`'s heavy dependencies an extra.

Either keeps `tests/test_layers.py` satisfied.

### Versions when the weights are remote

A version's files stay files. Neither `Policies`, `Manifest`, the loop nor `Channel` changes.

| Path under `into` | Holds |
|---|---|
| `weights/tinker.json` | `{"sampler": "tinker://RUN/sampler_weights/NAME", "base_model": "Qwen/Qwen3.5-9B", "rank": 32}` |
| `weights/adapter_config.json`, `adapter_model.safetensors` | Only with `weights = "peft"`: the downloaded adapter remapped by `build_lora_adapter`, which `VllmEngine` loads as it is |
| `state/tinker.json` | `{"state": "tinker://RUN/weights/NAME", "training_run": "RUN", "sdk": "0.32.0"}` |
| `state/minibatches.jsonl` | What each minibatch did, as `LoraTrainer` writes it |

How the pieces fit:

- `NAME` is the version's name (`into.name`, `POLICY_ID@N`, with `@` replaced if Tinker refuses it in names, which
  is *unverified*).
- `Policies.add` keeps the pointer files as blobs, as it keeps an adapter today. A fork from a Tinker version works
  as a fork does now: the child's first step resumes from the parent's state path.
- The ledger records where every version's weights are without a schema change.
- A later refinement could give `Manifest` an explicit `remote` field, so that a reader does not have to open a file
  to know the weights are elsewhere. It is not needed to start.

### `TinkerTrainer`

```py
# implementations/rollout-tinker/src/rollout_tinker/trainer.py (proposed)
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rollout_train.trainer import Budget, Checkpoint, Step, StepFailed, Weighted, WEIGHTS, STATE

POINTER = "tinker.json"


@dataclass(frozen=True)
class TinkerSettings:
    """LoraSettings' names where they mean the same, so a profile switches by changing `kind`."""
    rank: int = 32
    learning_rate: float = 2e-5          # re-tuned: Tinker's lora_alpha is not ours
    clip_low: float = 0.2
    clip_high: float = 0.28
    segment_clip_low: float = 3e-4
    segment_clip_high: float = 4e-4
    truncate: float | None = 2.0
    tokens_per_step: int = 65_536
    max_kl: float | None = 0.02
    strict_kl: bool = True               # await each forward-backward before its optim_step
    max_gradient_norm: float = 1.0
    segment_tokens: int | None = 32_768  # at most the model's context on Tinker (64K for Qwen3.5-9B)
    segments_per_step: int | None = None
    objective: str = "policy_gradient"
    ratio: str = "token"
    train_unembed: bool = False          # vLLM's support for an lm_head adapter is unverified
    weights: str = "pointer"             # or "peft": also download the adapter for local engines
    project: str | None = None           # a Tinker project id (not a secret); else TINKER_PROJECT_ID


class TinkerTrainer:
    """A `Trainer` whose weights live at Thinking Machines. A step resumes the parent's training state (or starts a
    LoRA run on `model`), trains, saves a state and a sampler checkpoint named after the version, and leaves pointers
    to them in `into`. A client from the last step is reused when the parent is the state it saved."""

    def __init__(self, model: str, *, service: "ServiceFactory | None" = None, **settings: Any) -> None:
        self.settings = TinkerSettings(**settings)
        self.budget = Budget(self.settings.segment_tokens, self.settings.segments_per_step)
        self._model = model
        self._service = service or connected(self.settings.project)  # reads the key from the environment
        self._live: tuple[str, "tinker.TrainingClient"] | None = None
        self._lock = asyncio.Lock()

    async def step(self, batch: Sequence[Weighted], *, seed: int, parent: Checkpoint | None, into: Path) -> Step:
        async with self._lock:
            try:
                client = await self._client(parent, seed)
                metrics, clean = await passed(client, batch, self.settings, seed)  # old, then minibatches
                name = checkpoint_name(into.name)
                state = await settled(client.save_state_async(name, overwrite=True))
                sampler = await settled(client.save_weights_for_sampler_async(name))
            except tinker.TinkerError as error:  # (never the key: the SDK's errors carry no credentials)
                self._live = None
                raise StepFailed(f"tinker: {type(error).__name__}: {error}") from error
            await written(into, sampler.path, state.path, self._model, self.settings)  # pointers; PEFT if asked
            self._live = (state.path, client) if clean else None  # (a stop at max_kl may leave gradients behind)
            return Step(metrics)

    async def _client(self, parent: Checkpoint | None, seed: int) -> "tinker.TrainingClient":
        if parent is None:
            return await self._service.create_lora_training_client_async(
                base_model=self._model, rank=self.settings.rank, seed=seed,
                train_unembed=self.settings.train_unembed,
            )
        state = pointer(parent.state, "state") if parent.state else None
        if state is None:
            raise StepFailed("the parent was not trained on Tinker: a Tinker step cannot start from its files")
        if self._live is not None and self._live[0] == state:
            return self._live[1]
        return await self._service.create_training_client_from_state_with_optimizer_async(state)
```

How `passed` works:

1. It drops segments longer than `segment_tokens`, or with nothing sampled, and counts them.
2. It shuffles with `seed`.
3. It builds one `Datum` per segment.
4. If the objective needs `old` ([the table above](#the-objective-on-tinker)), it runs `forward_async` once and
   checks that every behaviour logprob is finite, as `PolicyStep` does.
5. For each minibatch it submits the loss call and `optim_step_async(AdamParams(...))`. It submits both together
   unless `strict_kl` requires the forward-backward's result first.
6. Its metrics mirror `PolicyStep.step`'s (`loss`, `clip_fraction`, `mean_ratio`, `kl_floor`, `mean_mismatch`,
   `mean_weight`, `truncated_fraction`, `kl_moved`, `tokens`, `segments`, `segments_too_long`, `optimizer_steps`,
   `stopped_at_max_kl`, `seconds`).
7. It adds Tinker's own measures: the tokens billed per meter and their estimated cost, and the optimizer's metrics
   when `OptimStepResponse.metrics` reports a gradient norm (*unverified* that it does).

`settled` awaits either an `APIFuture` or a direct result. The docs show both for the `_async` variants.

### `TinkerEngine`

```py
# implementations/rollout-tinker/src/rollout_tinker/engine.py (proposed)
from rollout_train.inference import Generation


class TinkerEngine:
    """An `Engine` that samples at Thinking Machines: the base model, or the sampler checkpoint an adapter's pointer
    names. Token ids in, ids and logprobs out; no chat template is applied there."""

    processes: Sequence[int] = ()

    def __init__(self, model: str, *, max_model_len: int = 32_768, project: str | None = None,
                 service: "ServiceFactory | None" = None) -> None:
        self.max_model_len = max_model_len
        self._model = model
        self._service = service or connected(project)
        self._base: "tinker.SamplingClient | None" = None
        self._adapters: dict[str, "tinker.SamplingClient"] = {}

    async def generate(self, prompt: Sequence[int], *, max_tokens: int, temperature: float, top_p: float,
                       stop_token_ids: Sequence[int], adapter: str | None) -> Generation:
        client = self._adapters[adapter] if adapter is not None else await self._base_client()
        response = await client.sample_async(
            prompt=tinker.ModelInput.from_ints(list(prompt)), num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                                                  stop=list(stop_token_ids)),
        )
        sequence = response.sequences[0]
        tokens, logprobs = list(sequence.tokens), sequence.logprobs
        if logprobs is None or len(logprobs) != len(tokens):  # (it could not be trained on)
            raise RuntimeError("the sampler returned a token without its logprob")
        return Generation(tokens, list(logprobs), "length" if sequence.stop_reason == "length" else "stop")

    async def load_adapter(self, name: str, path: str) -> None:
        """`path` is a version's weights directory: its pointer names the sampler checkpoint."""
        self._adapters[name] = await self._service.create_sampling_client_async(model_path=pointer(Path(path), "sampler"))

    async def remove_adapter(self, name: str) -> None:
        self._adapters.pop(name, None)

    async def sleep(self) -> None: ...   # nothing on this machine to free
    async def wake(self) -> None: ...
    def close(self) -> None: ...
```

**How publishing swaps the sampler.** The loop's `serve` calls `Recorder.publish`, which calls `Channel.publish`,
which calls `TinkerEngine.load_adapter("POLICY@N", ".../POLICY@N/weights")`. That reads `tinker.json` and makes a
`SamplingClient` for the sampler path. From then on, requests that name `POLICY@N` sample from it. The channel keeps
the previous version's client for turns in flight and drops the one before, as it does with vLLM adapters. Every
sampled span is still stamped with the version.

### A profile

```toml
# Train and serve at Thinking Machines; play (the worlds, the runners) on this machine. TINKER_API_KEY in the
# environment, never in this file.
directory = "~/.cache/rollout/runs/team-tinker"
feed_runs = 80
episodes_at_once = 12             # no GPU to share: what the machine's memory for Paper servers allows

[ledger]
kind = "rollout_train.database:DatabaseLedger"
url = "sqlite:///~/.cache/rollout/ledger.db"

[channels.policy]
model = "Qwen/Qwen3.5-9B"         # Tinker's id; also what rollout_qwen:qwen35 loads the tokenizer from
renderer = "rollout_qwen:qwen35"
engine = "rollout_tinker:TinkerEngine"
engines = [{ max_model_len = 32768 }]
thinking_tokens = 1024
answer_tokens = 400

[trainer]
kind = "rollout_tinker:TinkerTrainer"
channel = "policy"
colocated = false                 # nothing to share
rank = 32
learning_rate = 2e-5
segment_tokens = 16000
segments_per_step = 384
tokens_per_step = 65536

[tools]
minecraft = "minecraft_team.worlds:tools"

[memory]
runs_gib = 6
```

The same profile with `engine = "rollout_vllm:VllmEngine"` and `weights = "peft"` under `[trainer]` trains remotely
and serves locally ([mixed setups](#mixed-setups)).

**What does not apply.**

- `colocated` and `training_gib` do nothing.
- `Colocated` is never used. Wrapping a remote trainer in it would pause play for every step and gain nothing.
- `TinkerEngine.processes` is empty, so `engine.json` lists nothing.
- `Platform.start` needs no change: it calls `named(kind)(model, **settings)` and `named(engine)(model, **options)`
  as it does now.

### Retention

`Policies.thin` releases a version by deleting its blobs. For a Tinker version those are the pointer files, so the
remote checkpoints remain and keep costing storage. Checkpoints are named after versions, so a sweeper can reconcile
them:

- `rollout_tinker.sweep(policies, policy)` (proposed) lists `RestClient.list_user_checkpoints()`.
- It maps each checkpoint's name back to a version.
- It deletes, with `delete_checkpoint_from_tinker_path`:
  - the checkpoints of released versions;
  - checkpoints of no version at all once they are older than a day (steps that died before their version was
    appended).

It could run after each `thin`, from a hook, or as `rollout tinker sweep`. TTLs alone do not fit: `Retention` keeps
every twentieth version indefinitely, and the trainer cannot know at save time which versions will be kept.

## Mixed setups

| Trains | Serves | Works | What to know |
|---|---|---|---|
| Tinker | Tinker | Yes | Pointers only; no weight transfer |
| Tinker | Local vLLM | Yes, with `weights = "peft"` | Each step downloads the archive and remaps it (`build_lora_adapter`). See below |
| Local `LoraTrainer` | Tinker | No | No call imports an adapter into Tinker (*unverified*). `TinkerTrainer` raises `StepFailed` for such a parent instead of silently starting over |
| Tinker | A frontier model, beside it | Yes, unrelated | Other slots of a run can be bound to `DirectModel` endpoints as today |

**Training remotely and serving with local vLLM.**

- `max_lora_rank` must be at least the Tinker rank.
- Tinker adapts the unembedding by default. Whether vLLM serves an `lm_head` adapter is *unverified*, so the
  proposed default is `train_unembed = false`.
- Serving the adapter over the AWQ 4-bit checkpoint (`cyankiwi/Qwen3.5-9B-AWQ-4bit`), rather than the BF16 base it
  was trained on, works but widens the gap between behaviour and trainer logprobs. The truncated importance weight
  absorbs it, and `kl_floor` and `mean_mismatch` measure it. Serving `Qwen/Qwen3.5-9B` in BF16 avoids it, if the GPU
  holds it.

**Distillation teachers** from the [policy graph](policy-dag.md#distillation) gain an option. A teacher of another
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
- **Costs scale with tokens, not hours.** An estimate for one step at the one-GPU profile's scale assumes:
  - 384 segments of about 4,000 tokens, 1,000 of them sampled;
  - about eight turns per segment, so about 16,000 prompt tokens re-sent per segment;
  - 80% of prefill hitting the cache.

  | Item | Tokens | Cost |
  |---|---|---|
  | Training | 1.5M | $2.25 |
  | The `old` pass | 1.5M | $2.25 |
  | Sampled tokens | 0.38M | $0.77 |
  | Prefill | 6.1M | $1.45 (cached) to $4.05 |
  | Total, trained segments only | | about $7 to $9 |

  Play also produces segments that are never trained on (equal-score groups are skipped), so more like $10 per step,
  or about $1,000 per 100 steps. The single-minibatch `cispo` path saves the `old` pass. Real numbers should come from
  a past run's `inference` notes (prompt and generated tokens per minute) before committing. Storage adds $0.10 per
  GB-month per kept checkpoint.
- **What the API does not give.** These points are *unverified*:
  - a way to clear accumulated gradients, which matters after a `max_kl` stop;
  - the gradient norm in `optim_step`'s metrics;
  - LoRA rank limits (the cookbook uses 32 to 128);
  - published rate limits;
  - billing of `forward_backward_custom`'s extra forward pass (probably at the training price);
  - whether `create_training_client_from_state*` makes a new training run each time (reusing the live client avoids
    it in steady state).
- **Model churn.** Models retire with a few months' notice. A policy's versions are tied to its base, so a retired
  base strands a remote policy unless its adapters were downloaded (`weights = "peft"` keeps a local copy in the blob
  store).
- **Secrets.**
  - The key comes from `TINKER_API_KEY` or `~/.tinker/credentials.json` and never appears in a profile, a ledger
    record or a log line.
  - `[trainer] project` holds a project id, which is not a secret.
  - Errors are reduced to their type and message before they reach `StepFailed` (whose text the ledger keeps).
  - Every machine that opens the profile's channel (each episode runner's) needs the key in its environment.
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
  external APIs may not be reachable).

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
| Compute | GPU pods running our own `LoraTrainer` and `VllmEngine` | The lowest-risk way to use Prime: nothing in our design changes, and it scales the one-GPU profile up |
| Training | Hosted training through a verifiers-wrapped environment | Not recommended. It closes to LoRA in two days. It would also hand the loop, the advantages, the loss, the staleness policy and token recording to Prime, giving up the ledger, the recorder's exact segments, `ID@N` versions and our renderer |

Prime Intellect is a source of GPUs and open-source parts. Tinker is the managed service that keeps our loop.

## A first milestone

The milestones run in order: first a live smoke test, then the smallest integration, then the rest.

**0. Live smoke script.** About an hour, under $0.10 (*estimate*: a few thousand tokens at $2 or less per million).
An opt-in script or test (`-m tinker`, skipped without `TINKER_API_KEY`) on `Qwen/Qwen3.5-4B`, the same family at
half the price. It checks:

1. **Tokenizer agreement.** `rollout_qwen.qwen35("Qwen/Qwen3.5-4B")` renders a conversation with tools. Its token
   ids are compared with Tinker's `get_tokenizer()` encoding, and the vocabularies are hashed.
2. **Sampling contract.** It samples 64 tokens with our stop ids. It checks that the stop token is included,
   `stop_reason` is `stop` or `length`, and there is one logprob per token.
3. **Numerics.** It recomputes the sampled tokens' logprobs with a training client's `forward`, and reports the mean
   absolute difference (the expected `mean_mismatch`).
4. **Round trip.** It runs one `cispo` step on four segments, then `save_state`, `save_weights_for_sampler`, a new
   `SamplingClient` and a sample.
5. **Cleanup.** It deletes the checkpoints.

**1. `rollout-tinker` with the token-ratio objective, serving from Tinker only.** About two days.

- `TinkerEngine`.
- `TinkerTrainer` with the `cispo` (one minibatch) and `ppo` (several) paths, plus `likelihood`, so that
  `rollout imitate` works.
- Pointer files, the live-client reuse, `StepFailed` mapping and deterministic checkpoint names.

**Tests without a network.** `TinkerTrainer` and `TinkerEngine` take a `service` factory, and a fake service stands
in for Tinker.

- **The fake.** Its "model" is a learnable bias over a small vocabulary (log-softmax per position, in torch). Its
  `forward`, `forward_backward` (the documented `ppo`, `cispo`, `importance_sampling` and `cross_entropy` formulas),
  `optim_step` (AdamW on the bias), `save_state`, `save_weights_for_sampler` and `sample` all act on that bias.
- **Tests on top of it:**
  - Datum alignment: the shift by one, spans across several turns, forced tokens masked.
  - **Equivalence**: a step through the fake gives the same update as `PolicyStep`'s `terms` on the same tiny model,
    for each row of [the table](#the-objective-on-tinker). This is the test that proves the substitutions exact.
  - Resume from a parent's state, reuse of the live client, refusal of a parent without a pointer.
  - Engine contract and publish swap through `Channel` and `Recorder`, with `rollout_train.testing`'s
    `PlainRenderer`.
  - A whole profile opened with `rollout_tinker` names and the fake, running the loop for two groups, as
    `tests/rollout_train/test_profile.py` does with a scripted engine.

These are package-local tests, run on their own.

**First real run.** A short Minecraft run (five groups, two steps) on `Qwen/Qwen3.5-9B`, audited before anything
longer:

- Check the launch flags.
- Check the share of actions that fail.
- Check that `kl_floor` is small and that the steps move the policy (`kl_moved`, `clip_fraction` above zero).
- Compare turn latency and cost per step with the estimates above.

**2. Afterwards.**

- `weights = "peft"` for local serving.
- The sweeper.
- The segment ratio through `forward_backward_custom`.
- Teachers of another base for distillation.

## Unverified points

- Whether Tinker accepts `@` in checkpoint names.
- Whether there is an API to import an adapter trained elsewhere.
- Whether accumulated gradients can be cleared.
- Whether `optim_step` reports the gradient norm.
- LoRA rank limits.
- Per-account rate limits.
- How the custom loss's extra forward pass is billed.
- Whether resuming from a state makes a new training run.
- Whether sampled logprobs are post-temperature.
- Whether vLLM serves an unembedding adapter.
- Checkpoint sizes for a rank-32 adapter of the 9B model, with and without its Adam state.
- The cost estimate's token counts.
- Whether the Qwen3.5 tokenizer is the same across sizes (for teachers).
- For Prime Intellect: whether Prime Inference supports token-id prompts and logprobs; whether dedicated runs accept
  LoRA; whether hosted environments can reach outside services.
- The "$150 free credits" for new Tinker users, reported by a third-party blog
  ([beam.cloud](https://www.beam.cloud/blog/tinker-model-pricing)), is not on Thinking Machines' own pages and is
  left out above.

## Sources

**Thinking Machines (Tinker)**
- Whole documentation in one file: https://tinker-docs.thinkingmachines.ai/llms-full.txt (index: `llms.txt`)
- Overview: https://tinker-docs.thinkingmachines.ai/tinker/index.md
- Quickstart: https://tinker-docs.thinkingmachines.ai/tinker/quickstart/index.md
- SDK cheatsheet: https://tinker-docs.thinkingmachines.ai/tinker/sdk-cheatsheet/index.md
- Models & Pricing: https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/index.md and
  https://tinker-docs.thinkingmachines.ai/tinker/models.json
- Model deprecations: https://tinker-docs.thinkingmachines.ai/tinker/model-deprecations/index.md
- Data model & permissions: https://tinker-docs.thinkingmachines.ai/tinker/data-model/index.md
- Loss functions: https://tinker-docs.thinkingmachines.ai/tinker/losses/index.md, `.../losses/ppo/index.md`,
  `.../losses/cispo/index.md`, `.../losses/importance-sampling/index.md`, `.../losses/custom/index.md`
- LoRA primer: https://tinker-docs.thinkingmachines.ai/tinker/lora-primer/index.md
- Checkpoints: https://tinker-docs.thinkingmachines.ai/tinker/howto/checkpoints/index.md
- OpenAI-compatible inference: https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/index.md
- Under the hood: https://tinker-docs.thinkingmachines.ai/tinker/under-the-hood/index.md
- Session metrics: https://tinker-docs.thinkingmachines.ai/tinker/session-metrics/index.md
- API reference: https://tinker-docs.thinkingmachines.ai/tinker/api-reference/serviceclient/index.md,
  `.../trainingclient/index.md`, `.../samplingclient/index.md`, `.../restclient/index.md`, `.../types/index.md`
- PEFT export: https://tinker-docs.thinkingmachines.ai/cookbook/deployment/lora-adapter/index.md
- Changelog: https://tinker-docs.thinkingmachines.ai/changelog/index.md
- SDK: https://github.com/thinking-machines-lab/tinker and https://pypi.org/project/tinker/
- Cookbook: https://github.com/thinking-machines-lab/tinker-cookbook (`tinker_cookbook/completers.py`,
  `tinker_cookbook/renderers/base.py`)
- Announcements: https://thinkingmachines.ai/news/announcing-tinker/ and
  https://thinkingmachines.ai/news/tinker-general-availability/

**Prime Intellect**
- What Lab is, and the LoRA shutdown notice: https://docs.primeintellect.ai/hosted-training/what-is-lab
- Models and pricing: https://docs.primeintellect.ai/hosted-training/models-and-pricing
- Advanced configs: https://docs.primeintellect.ai/hosted-training/advanced-configs
- Full fine-tuning: https://docs.primeintellect.ai/hosted-training/full-finetuning
- Volumes: https://docs.primeintellect.ai/hosted-training/volumes
- Dedicated runs API: https://docs.primeintellect.ai/api-reference/training/create-dedicated-run
- Inference: https://docs.primeintellect.ai/inference/overview and
  https://docs.primeintellect.ai/inference/adapter-deployments
- Sandboxes: https://docs.primeintellect.ai/sandboxes/overview
- prime-rl: https://docs.primeintellect.ai/prime-rl/overview, `.../prime-rl/training`, `.../prime-rl/algorithms`,
  `.../prime-rl/advanced`, and https://github.com/PrimeIntellect-ai/prime-rl (v0.9.0)
- verifiers: https://docs.primeintellect.ai/verifiers/overview and https://docs.primeintellect.ai/verifiers/v1/harnesses
