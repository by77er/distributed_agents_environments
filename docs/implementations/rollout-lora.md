# LoRA trainer

Code: `rollout_lora`

`LoraTrainer` implements the [`Trainer`](../guide/reference.md#trainer) protocol: it trains a LoRA adapter over a
4-bit checkpoint, the same file the engines serve, and writes each step's adapter where engines load it. What the
[training loop](../libraries/rollout-train/training.md) asks of a trainer is defined there; this page is what this
one does. The package is installed with `uv sync --all-extras` and needs an NVIDIA GPU.

## In a profile

```toml
[trainer]
kind = "rollout_lora:LoraTrainer"
channel = "policy"
colocated = true
rank = 32
segment_tokens = 8000
```

A [profile](../guide/deploying.md) calls `LoraTrainer(model, **settings)` with the model of the channel it trains.
Every key of `[trainer]` other than `kind`, `channel`, `policy` and `colocated` is a setting.

## Settings

[`LoraSettings`](../guide/reference.md#lorasettings) is the one place the settings and their defaults are written.

| Setting | What it sets |
|---|---|
| `rank` | The adapter's rank. Its scaling (`alpha`) is twice the rank |
| `learning_rate` | AdamW's learning rate. Adam moves a weight by at most this much per optimizer step |
| `clip_low`, `clip_high` | A token's ratio to its logprob at the step's start is clipped to `1 - clip_low` .. `1 + clip_high` |
| `ratio` | `token` (a ratio for each token, PPO) or `segment` (one for each segment, the geometric mean of its tokens', GSPO) |
| `segment_clip_low`, `segment_clip_high` | With `ratio = "segment"`, the segment's ratio is clipped to `1 - segment_clip_low` .. `1 + segment_clip_high` |
| `truncate` | The most a token's importance weight may be: its logprob at the step's start against the one it was sampled at (`None`: not truncated) |
| `tokens_per_step` | Sampled tokens per optimizer step. How far a step moves the policy is set by how many optimizer steps its tokens make |
| `max_kl` | The pass stops when the policy has moved this far from where the step began, in nats per token (`None`: never) |
| `max_gradient_norm` | Gradients are clipped to this norm before each optimizer step |
| `segment_tokens` | The longest segment a step can hold on its GPU (`None`: any) |
| `segments_per_step` | How many segments a step can afford (`None`: any number) |
| `layer_inputs_on_host` | Keep each layer's input in pinned system memory between the forward and backward passes ([activations](#activations)) |
| `mlp_rows` | Run each layer's MLP over this many tokens at a time when it is computed again, and in passes without a gradient (`None`: the whole segment) |
| `objective` | `policy_gradient` (the weighted, clipped policy gradient of [the step](#the-step)) or `likelihood` (the sampled tokens' log-likelihood, for [imitation](../libraries/rollout-train/training.md#imitation)) |

`segment_tokens` and `segments_per_step` are the trainer's [`Budget`](../guide/reference.md#budget). An open profile
gives `segment_tokens` to the trained channel as its longest turn, so that every sampled turn can be trained on.

## Files

A step is told where its files go (`into`) and leaves:

| Path under `into` | Holds |
|---|---|
| `weights/` | The adapter, in PEFT's layout (`adapter_config.json`, `adapter_model.safetensors`), which vLLM loads as it is. Weights are saved as float32: the next step starts from this file, and updates are smaller than bfloat16 resolves |
| `state/optimizer.pt` | The optimizer's state after the step |
| `state/minibatches.jsonl` | What each minibatch of the step did: segments, tokens, loss, clipped share, KL estimate, gradient norm |

The trainer keeps nothing of its own between steps: a step starts from the adapter and the optimizer's state of
the version it is given, so any `LoraTrainer` can take any step of any policy. A run keeps each step's files as a
[version](../libraries/rollout-train/policies.md) of the policy it trains.

## A fresh process per step

`step` runs in a spawned process (`rollout_lora.worker`). The process loads the policy onto the GPU, loads the
previous step's adapter and the optimizer's state, makes one pass over the batch, saves both and exits.

| Why | |
|---|---|
| Memory | Exiting frees the GPU and system memory. A trainer kept in system memory between steps takes 9 GB, next to a sleeping engine and whatever the environments run |
| Isolation | vLLM's client libraries change how transformers builds models in the process that imports them. A separate process loads the checkpoint unaffected |
| Cleanup | The process ends when its parent dies (`rollout.processes.end_with_parent`), and is terminated when the step is cancelled |

The learning rate is the settings' on every step, whatever the saved optimizer state carries. A step whose process
fails or exits without a result raises [`StepFailed`](../guide/reference.md#stepfailed) with the process's traceback.

## The memory bound

The process may use the GPU memory that is free when it starts, less `MEMORY_MARGIN`, and no more
(`torch.cuda.set_per_process_memory_fraction`). Some drivers let a process spill past the card into system memory,
where a step crawls instead of failing; the bound turns that into an out-of-memory error. A minibatch that runs out
of memory is dropped whole, its gradient cleared, and counted in `minibatches_out_of_memory`; the pass goes on with
the next. Segments longer than `segment_tokens` are left out before the pass and counted in `segments_too_long`.

## Activations

With gradient checkpointing a layer keeps only its input for the backward pass and computes itself again there.
Two things still grow with a segment's length, and `rollout_lora.activations` takes each off the GPU's peak:

- **The layer inputs**, a quarter of a megabyte a token for Qwen3.5-9B (2 GiB at 8,000 tokens). With
  `layer_inputs_on_host` they wait for the backward pass in pinned system memory (`HostStore`, whose buffers are
  reused from segment to segment).
- **The MLP of a layer computed again**, whose intermediate activations are the largest part of the layer's peak.
  With `mlp_rows` it runs over the segment in pieces. A layer's output is `residual + mlp(norm(residual))`, so when
  the layer is computed again for its backward pass the MLP's output is not needed: it returns at once, and when the
  gradient arrives each piece is computed with gradients and takes its backward step before the next. Nothing is
  computed more often than without it.

Both give the same logprobs and gradients. A pass without a gradient stores nothing.

## The policy

| Part | What `rollout_lora.policy` and `rollout_lora.quantized` do |
|---|---|
| Weights | The checkpoint's packed int4 weights stay packed. `Int4Linear` dequantizes a layer's weight inside the matrix multiply, in the forward and again in the backward pass, so at most one layer's bfloat16 weight exists at a time |
| Adapter | LoRA on every attention, linear-attention and MLP projection of the language model (`TARGETS`). `B` starts at zero: a new adapter changes nothing |
| Left off the GPU | A vision tower is dropped. The token embedding table is memory-mapped from the checkpoint file and only a segment's rows are read |
| Output layer | Run only at the sampled positions, `LOGIT_ROWS` at a time, each chunk recomputed in the backward pass. Peak memory is one chunk's logits, whatever the share of sampled tokens |
| Activations | Gradient checkpointing over the transformer |

## The step

`rollout_lora.step.PolicyStep` takes a step. Which segments are in the batch, and each one's advantage, is the
[algorithm's](../libraries/rollout-train/training.md) business; the loss is a function from `rollout_lora.objectives`,
by the settings' name, which any trainer in torch can call.

Three logprobs of each sampled token meet in it:

| Logprob | Computed by | When |
|---|---|---|
| behavior | the engine | while sampling, under whichever version was served then (recorded in the segment) |
| start | the trainer, without a gradient | at the start of the step, on the weights the step starts from |
| now | the trainer, with a gradient | in each minibatch, as the step updates the weights |

Behavior and start differ because the data came from elsewhere: an older version, and the engine computing
differently from the trainer. Start and now differ by how far the step has moved the policy.

1. The trainer computes every sampled token's logprob at the start (the recorded spans; forced tokens and prompts are
   not trained on). A segment that does not fit the GPU here is left out and counted.
2. The segments are shuffled with the step's `seed` and cut into minibatches of about `tokens_per_step` sampled
   tokens. A last minibatch of less than half that joins the one before.
3. For each token, the importance weight `w = min(start / behavior, truncate)` corrects for where it was sampled; it is
   a constant. The ratio `r = now / start` is 1 when the step begins. The loss is
   `-w · min(r · advantage, clip(r) · advantage)`, the upper clip wider than the lower, a mean over the minibatch's
   tokens. With `ratio = "segment"` the ratio and the weight are each one for the segment (the geometric means of its
   tokens'), clipped as a whole, and the loss is a mean over each segment's tokens and then over the minibatch's
   segments. There is no KL penalty.
4. Before a minibatch's optimizer step, its estimate of KL(start ‖ now) on the sampled tokens is compared with
   `max_kl`. If it is more, the pass stops without that step.
5. Otherwise gradients are clipped to `max_gradient_norm` and AdamW steps, with no weight decay.

A sampled token whose recorded logprob is not finite fails the step. Under `objective = "likelihood"` there is no
first pass, and steps 3 and 4 are replaced: the loss is the negative log-likelihood of each sampled token times its
segment's advantage, with no weight, ratio, clip or stop at `max_kl`, and recorded logprobs are not read.

## Metrics

A step returns these. The training loop keeps them with the version the step made, and sends them to the job in its
`step` note ([the record](../libraries/rollout-train/training.md#the-record)). Under the likelihood objective the
weight, ratio, clip, mismatch and KL metrics are zero.

| Metric | Meaning |
|---|---|
| `kl_floor` | KL(behavior ‖ start), estimated on the sampled tokens: how far the data is from the policy the step starts from (the engine's and the trainer's difference, and how stale the segments are) |
| `mean_mismatch` | The mean absolute difference between the start and behavior logprobs |
| `mean_weight`, `truncated_fraction` | The mean importance weight, and the share of tokens whose weight was truncated |
| `kl_moved` | KL(start ‖ now) as the last stepped minibatch found it: how far the step moved the policy |
| `stopped_at_max_kl` | 1 if the pass stopped at `max_kl` |
| `loss`, `clip_fraction`, `mean_ratio` | The loss (per token, or per segment), the share of tokens whose ratio was clipped, the mean ratio |
| `gradient_norm` | Before clipping, the mean over optimizer steps |
| `optimizer_steps` | Minibatches stepped on |
| `tokens`, `segments` | Sampled tokens and segments trained on |
| `segments_given`, `segments_too_long`, `longest_segment_tokens` | What the batch held, how many were left out for their length, and the longest one kept |
| `minibatches_out_of_memory`, `start_out_of_memory` | Minibatches dropped, and segments left out of the first pass |
| `start_seconds`, `seconds` | The first pass, and the whole step |
| `peak_gpu_gib`, `free_gpu_gib` | GPU memory reserved at the peak, and free when the process started |

## Measurements

One RTX 5080 (16 GB), `cyankiwi/Qwen3.5-9B-AWQ-4bit`, rank 32.

| What | Result |
|---|---|
| GPU memory with the policy loaded | 6.6 GiB |
| Peak GPU memory in a step | 10.7 GiB with segments of 5,000 tokens, 12.4 GiB with 8,000 |
| Time per segment | 4 to 8 s |
| A step of 384 turns of the [Minecraft swarm](../products/minecraft-swarm.md) | About 40 optimizer steps, about half an hour |
| Trainer logprobs against vLLM's | A mean difference of 0.016 per token, with and without an adapter |
| Without the memory bound, under Windows | A step of 96 turns that needed more than the card ran 13 minutes without finishing and left the host 0.6 GB of free memory |
| Allocator | `expandable_segments` saves about 0.7 GiB at the peak |

## Tests

`tests/rollout_lora/` needs torch and is collected only when it is installed. It covers the adapter's file format,
the direction of the update, what each minibatch did, forced tokens, segments left out, the last minibatch, the KL
stop, a missing logprob, the likelihood objective, the importance weight and its truncation, the token clip, and the
segment ratio and its gradient.
