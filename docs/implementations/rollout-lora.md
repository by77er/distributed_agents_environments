# LoRA trainer

For whoever trains on their own GPU: the LoRA (low-rank adaptation) and full-weight trainers, their settings, the memory
bound, and measurements.

**Read first:** [the trainer protocol](../libraries/rollout-train/training.md#the-trainer). **Next:** [Objectives in
torch](rollout-objectives.md).

Code: `rollout_lora`

`LoraTrainer` implements the [`Trainer`](../guide/reference.md#trainer) protocol: it trains a LoRA adapter over the
checkpoint the engines serve (a 4-bit image-text checkpoint, the same file, or a text model in bfloat16), and writes
each step's adapter where engines load it. What the
[training loop](../libraries/rollout-train/training.md) asks of a trainer is defined there; this page is what this
one does. Its step and its objective are [`rollout_objectives`](rollout-objectives.md)'; this package gives them a
policy on the GPU, with its reference and entropies. The package is installed with `uv sync --all-extras` and needs an
NVIDIA GPU.

## In a profile

```toml
[trainer]
kind = "rollout_lora:LoraTrainer"
channel = "policy"
colocated = true
rank = 32
segment_tokens = 8000
```

A [profile](../guide/deploying.md) calls `LoraTrainer(model, **settings)` with the model of the
[channel](../libraries/rollout-train/channels.md) it trains. Every key of `[trainer]` other than `kind`, `channel`,
`start`, `bookmark` and `colocated` is a setting. A run's `objective.*` settings
([objectives](../libraries/rollout-train/training.md#objectives)) reach it as its `objective`.

### Every weight

`FullTrainer` trains every weight of a text model, with the same settings (`rank` is not used) and the same fresh
process per step. The weights are kept in float32 and the forward pass runs in bfloat16 (autocast); each step leaves
`weights/` in the model's own layout (float32 safetensors, with the configuration saying `bfloat16`, which is what vLLM
loads them as, and the tokenizer) and the optimizer's state in `state/`. A step starts from its parent's weights and
state, or from the model for the first. It refuses an image-text model. Qwen3-0.6B's step of 4 segments of 600 tokens
peaks under 14 GiB on a 16 GB card; its optimizer's state is about 5 GB. Its reference, for an objective that reads one,
is a frozen copy of the model trained over, in bfloat16 beside the policy, which it holds only when asked
(`frozen_reference = true`); without it, validation refuses such an objective.

```toml
[trainer]
kind = "rollout_lora:FullTrainer"
channel = "policy"
colocated = true
learning_rate = 1e-6
```

### Merging

`rollout_lora.merge.merge(base, adapter, into)` folds an adapter in PEFT's layout into a model (a name or a
directory) and writes the merged model to `into`, reading the base's safetensors files one at a time: each adapted
layer's weight becomes W + (alpha / rank) · B · A, computed in float32 and stored in the base's dtype, and every
other weight is copied, with the configuration and tokenizer. It refuses an adapter whose layers the base lacks or
whose update has another shape. It is what `rollout merge` calls by default
([full weights and merges](../libraries/rollout-train/checkpoints.md#full-weights-and-merges)).

## Settings

[`LoraSettings`](../guide/reference.md#lorasettings) is the one place the settings and their defaults are written: a
policy step's ([`StepSettings`](rollout-objectives.md#settings), which the [Tinker trainer](rollout-tinker.md) takes
too), and these:

| Setting | What it sets |
|---|---|
| `rank` | The adapter's rank. Its scaling (`alpha`) is twice the rank |
| `frozen_reference` | For `FullTrainer`: hold a frozen copy of the model trained over as the reference (false). An adapter's reference is the model with the adapter switched off |

The step's settings (`learning_rate`, `tokens_per_step`, `max_kl`, `max_gradient_norm`, `passes`, `warmup_updates`,
`segment_tokens`, `segments_per_step`, `objective`) are [`rollout_objectives`'](rollout-objectives.md#settings). The
objective's components can be named by this trainer's own settings too (`ratio`, `clip_low`, `clip_high`,
`segment_clip_low`, `segment_clip_high`, `truncate`, and `objective = "policy_gradient"` or `"likelihood"`), which say
the `default` or `sft` preset and its components.

`segment_tokens` and `segments_per_step` are the trainer's [`Budget`](../guide/reference.md#budget). An open profile
gives `segment_tokens` to the trained channel as its longest turn, so that every sampled turn can be trained on.

## Files

A step is told where its files go (`into`) and leaves:

| Path under `into` | Holds |
|---|---|
| `weights/` | The adapter, in PEFT's layout (`adapter_config.json`, `adapter_model.safetensors`), which vLLM loads as it is. Weights are saved as float32: the next step starts from this file, and updates are smaller than bfloat16 resolves |
| `state/optimizer.pt` | The optimizer's state after the step |
| `state/minibatches.jsonl` | What each minibatch of the step did: segments, tokens, loss, clipped share, KL estimate, learning rate, gradient norm |

The trainer keeps nothing of its own between steps: a step starts from the adapter and the optimizer's state of
the checkpoint it is given, so any `LoraTrainer` can take any step from any checkpoint. A run keeps each step's files as a
[checkpoint](../libraries/rollout-train/checkpoints.md), named by its id, whose parent is the checkpoint the step began from.

## A fresh process per step

`step` runs in a spawned process (`rollout_lora.worker`). The process loads the policy onto the GPU, loads the
previous step's adapter and the optimizer's state, takes the step's passes over the batch, saves both and exits.

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

## The policy

| Part | What `rollout_lora.policy` and `rollout_lora.quantized` do |
|---|---|
| Weights | The checkpoint's packed int4 weights stay packed. `Int4Linear` dequantizes a layer's weight inside the matrix multiply, in the forward and again in the backward pass, so at most one layer's bfloat16 weight exists at a time |
| Adapter | LoRA on every attention, linear-attention and MLP projection of the language model (`TARGETS`). `B` starts at zero: a new adapter changes nothing |
| Reference | `Policy.reference`: the logprobs with every LoRA layer switched off (`rollout_lora.layers.adapter_off`), without a gradient: the model trained over, at no memory cost |
| Entropy | `Policy.logprobs_and_entropy`: each sampled position's entropy beside its logprob, from the same chunk of logits |
| Left off the GPU | A vision tower is dropped. The token embedding table is memory-mapped from the checkpoint file and only a segment's rows are read |
| Output layer | Run only at the sampled positions, `LOGIT_ROWS` at a time, each chunk recomputed in the backward pass. Peak memory is one chunk's logits, whatever the share of sampled tokens |
| Activations | Gradient checkpointing over the transformer |

## The step

The worker takes [`rollout_objectives.step.PolicyStep`](rollout-objectives.md#the-step) on the policy: the plan of
minibatches, where the step starts (each sampled token's logprob on the weights it starts from, and the reference's
where the objective reads it), each minibatch's loss as the objective composes it, the stop at `max_kl` and AdamW's
update. Which items are in the batch, and each one's advantage, is the
[algorithm's](../libraries/rollout-train/training.md) business. The step's metrics are
[`rollout_objectives`'](rollout-objectives.md#metrics), with the worker's `peak_gpu_gib` (GPU memory reserved at the
peak) and `free_gpu_gib` (free when the process started); the training loop keeps them with the checkpoint the step
made, and sends them to its hooks in its `step` note ([the record](../libraries/rollout-train/training.md#the-record)).

## Measurements

One RTX 5080 (16 GB), `cyankiwi/Qwen3.5-9B-AWQ-4bit`, rank 32.

| What | Result |
|---|---|
| GPU memory with the policy loaded | 6.6 GiB |
| Peak GPU memory in a step | 10.7 GiB with segments of 5,000 tokens, 12.4 GiB with 8,000 |
| Time per segment | 4 to 8 s |
| A step of 384 turns of the [Minecraft team](../products/minecraft-team.md) | About 40 optimizer steps, about half an hour |
| Trainer logprobs against vLLM's | A mean difference of 0.016 per token, with and without an adapter |
| Without the memory bound, under Windows | A step of 96 turns that needed more than the card ran 13 minutes without finishing and left the host 0.6 GB of free memory |
| Allocator | `expandable_segments` saves about 0.7 GiB at the peak |

## Tests

`tests/rollout_lora/` needs torch and is collected only when it is installed. It covers the adapter's file format and
the adapter switched off (the model it was added to), and the settings a trainer takes between steps (its objective's
numbers among them); the step itself is [`rollout_objectives`'](rollout-objectives.md#tests). `test_merge.py` covers
merging on the CPU. `test_small_on_gpu.py` runs only when asked
(`-m live`), with nothing else on the card: on Qwen3-0.6B (`ROLLOUT_SMALL_MODEL` names another) it trains an
adapter, takes two steps of every weight, and checks that a merged adapter gives what the adapter gave (an adapter
that moved logprobs by 2.6 on average, merged, is 0.06 from it: bfloat16 rounds part of a small update away). Its
steps write gigabytes, so give it `--basetemp` on disk, not `/tmp`.
