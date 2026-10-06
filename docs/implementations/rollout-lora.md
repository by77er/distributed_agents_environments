# LoRA trainer

For whoever trains on their own GPUs: the LoRA (low-rank adaptation) and full-weight trainers, on one GPU and on
several of one machine, their settings, the memory bound, and measurements.

**Read first:** [the trainer protocol](../libraries/rollout-train/training.md#the-trainer). **Next:** [Objectives in
torch](rollout-objectives.md).

Code: `rollout_lora`

`LoraTrainer` implements the [`Trainer`](../guide/reference.md#trainer) protocol: it trains a LoRA adapter over the
checkpoint the engines serve (a 4-bit image-text checkpoint, the same file, or a text model in bfloat16), and writes
each step's adapter where engines load it. What the
[training loop](../libraries/rollout-train/training.md) asks of a trainer is defined there; this page is what this
one does. Its step and its objective are [`rollout_objectives`](rollout-objectives.md)'; this package gives them a
policy on the GPU, with its reference and entropies, and on several GPUs shards it over them
([several GPUs](#several-gpus)). The package is installed with `uv sync --all-extras` and needs an NVIDIA GPU.

## In a cluster config

A `lora` trainer of the [cluster config](../guide/cluster.md#trainers) is `LoraTrainer`; a run's settings name it and
give its settings:

```toml
[trainers.local-lora]                 # the cluster config
kind = "lora"
gpus = 1
colocate_with = "local-vllm"          # the engines sleep while it steps
models = ["cyankiwi/Qwen3.5-9B-AWQ-4bit", "Qwen/Qwen3-0.6B"]
segment_tokens = 8000
```

```toml
"trainer.provider" = "local-lora"     # a run's settings, or a preset
"trainer.rank" = 32
"trainer.segment_tokens" = 8000
```

A run's driver calls `LoraTrainer(model, **settings)` in its trainer actor, with the model it trains and its
`trainer.*` settings less `provider`, `channel` and `model`. A run's `objective.*` settings
([objectives](../libraries/rollout-train/training.md#objectives)) reach it as its `objective`. The trainer steps on the
GPUs its actor is given (`CUDA_VISIBLE_DEVICES`, which Ray sets from the provider's `gpus`), or as many as `gpus` says
(`LoraTrainer(model, gpus=4, ...)`; a training pod's `ROLLOUT_TRAINER_GPUS`).

### Every weight

`FullTrainer` trains every weight of a text model, with the same settings (`rank` and `whole_base` are not used),
on one GPU in the same fresh process per step. The weights are kept in float32 and the forward pass runs in bfloat16
(autocast); on one GPU each step leaves `weights/` in the model's own layout (float32 safetensors, with the configuration saying `bfloat16`, which is what vLLM
loads them as, and the tokenizer) and the optimizer's state in `state/`. A step starts from its parent's weights and
state, or from the model for the first. It refuses an image-text model. Qwen3-0.6B's step of 4 segments of 600 tokens
peaks under 14 GiB on a 16 GB card; its optimizer's state is about 5 GB. Its reference, for an objective that reads one,
is a frozen copy of the model trained over, in bfloat16 beside the policy, which it holds only when asked
(`frozen_reference = true`); without it, validation refuses such an objective. Sixteen bytes a weight is more than one
GPU holds above a few billion parameters: on several GPUs the weights, gradients and Adam's moments are sharded over
them, and each step leaves a serving copy in bfloat16 ([several GPUs](#several-gpus)).

```toml
[trainer]
kind = "rollout_lora:FullTrainer"
channel = "policy"
colocated = true
learning_rate = 1e-6
```

### Merging

`rollout_lora.merge.merge(base, adapter, into)` folds an adapter in the layout of PEFT (parameter-efficient fine-tuning)
into a model (a name or a directory) and writes the merged model to `into`, reading the base's safetensors files one at
a time: each adapted layer's weight becomes W + (alpha / rank) · B · A, computed in float32 and stored in the base's
dtype, and every other weight is copied, with the configuration and tokenizer. It refuses an adapter whose layers the
base lacks or whose update has another shape. It is what `rollout merge` calls by default ([full weights and
merges](../libraries/rollout-train/checkpoints.md#full-weights-and-merges)).

## Settings

[`LoraSettings`](../guide/reference.md#lorasettings) is the one place the settings and their defaults are written: a
policy step's ([`StepSettings`](rollout-objectives.md#settings), which the [Tinker trainer](rollout-tinker.md) takes
too), and these:

| Setting | What it sets |
|---|---|
| `rank` | The adapter's rank. Its scaling (`alpha`) is twice the rank |
| `frozen_reference` | For `FullTrainer`: hold a frozen copy of the model trained over as the reference (false). An adapter's reference is the model with the adapter switched off |
| `state_every` | On several GPUs: write the trainer's full state (the optimizer's, and full weights in float32) every this many steps since its processes loaded; none: every step for an adapter, every 10 for full weights. On one GPU every step writes it |
| `whole_base` | For an adapter on several GPUs: each GPU holds the whole frozen model, gathered once (true), or its share, each layer gathered as it computes (false: a model too large for one GPU); none: whole where the model's files take at most half of the smallest GPU's memory |

The step's settings (`learning_rate`, `tokens_per_step`, `max_kl`, `max_gradient_norm`, `passes`, `warmup_updates`,
`segment_tokens`, `segments_per_step`, `objective`) are [`rollout_objectives`'](rollout-objectives.md#settings). The
objective's components can be named by this trainer's own settings too (`ratio`, `clip_low`, `clip_high`,
`segment_clip_low`, `segment_clip_high`, `truncate`, and `objective = "policy_gradient"` or `"likelihood"`), which say
the `default` or `sft` preset and its components.

`segment_tokens` and `segments_per_step` are the trainer's [`Budget`](../guide/reference.md#budget). A run's driver
gives `segment_tokens` to the trained channel as its longest turn, so that every sampled turn can be trained on.

## Files

A step is told where its files go (`into`) and leaves:

| Path under `into` | Holds |
|---|---|
| `weights/` | The adapter, in PEFT's layout (`adapter_config.json`, `adapter_model.safetensors`), which vLLM loads as it is. Weights are saved as float32: the next step starts from this file, and updates are smaller than bfloat16 resolves |
| `state/optimizer.pt` | The optimizer's state after the step (on several GPUs, an adapter's every `state_every` steps), in one GPU's layout whatever the number of GPUs |
| `state/minibatches.jsonl` | What each minibatch of the step did: segments, tokens, loss, clipped share, Kullback-Leibler (KL) divergence estimate, learning rate, gradient norm |
| `state/held.txt` | On several GPUs: the name the trainer's processes gave what they hold after the step |
| `state/shards/` | On several GPUs, a full-weight trainer's full state every `state_every` steps: the float32 weights and the optimizer's state, as PyTorch's distributed checkpoint writes them, a file for each tensor's shard |

On one GPU the trainer keeps nothing of its own between steps: a step starts from the adapter and the optimizer's
state of the checkpoint it is given, so any `LoraTrainer` can take any step from any checkpoint. On several it keeps
its policy and optimizer, and still leaves every file a later step starts from: steps on one GPU and on several go on
from each other's files. A run keeps each step's files as a
[checkpoint](../libraries/rollout-train/checkpoints.md), named by its id, whose parent is the checkpoint the step began from.

## A fresh process per step

On one GPU, `step` runs in a spawned process (`rollout_lora.worker`). The process loads the policy onto the GPU, loads the
previous step's adapter and the optimizer's state, takes the step's passes over the batch, saves both and exits.

| Why | |
|---|---|
| Memory | Exiting frees the GPU and system memory. A trainer kept in system memory between steps takes 9 GB, next to a sleeping engine and whatever the environments run |
| Isolation | vLLM's client libraries change how transformers builds models in the process that imports them. A separate process loads the checkpoint unaffected |
| Cleanup | The process ends when its parent dies (`rollout.processes.end_with_parent`), and is terminated when the step is cancelled |

The learning rate is the settings' on every step, whatever the saved optimizer state carries. A step whose process
fails or exits without a result raises [`StepFailed`](../guide/reference.md#stepfailed) with the process's traceback.

## Several GPUs

On more than one GPU of a machine, the trainer runs a process per GPU under torchrun (`rollout_lora.workers`), started
at its first step by `rollout_lora.resident.Workers` and kept between steps, each holding its shard of the policy and
of the optimizer (`rollout_lora.sharded`, PyTorch's FSDP2). Every step is handed to all of them, with the settings the
trainer has then (so a setting changed between steps reaches every process), and each answers; rank 0's answer is the
step's metrics.

| Part | What it does |
|---|---|
| Loading | Each process loads the policy onto the CPU and shards it onto its GPU, the processes taking turns, so the machine holds one unsharded copy at a time. A new adapter is drawn alike in every process (each keeps its share of the same one) |
| Sharding | Each decoder layer is a unit of `fully_shard`; the rest of the model (the output layer, the last norm, embeddings held on the GPU) is the root's, the policy's `Scorer`, which computes the hidden states and the output layer's chunks in one call, so the output layer is gathered once a call. For an adapter, where each GPU holds the whole frozen model (`whole_base`), a layer's adapter matrices are one unit, gathered for each pass, and the frozen layers are gathered once and kept; else each layer's frozen weights are gathered with its adapter's as it computes. The frozen output layer is kept once gathered. Full weights, their gradients and Adam's moments are sharded in float32, and a frozen reference beside them |
| Precision | Weights gathered in bfloat16 (an adapter's too), gradients reduced in float32 (`MixedPrecisionPolicy`), activations checkpointed as on one GPU |
| The step | `rollout_objectives.step.PolicyStep` shared among the processes ([a step on several GPUs](rollout-objectives.md#a-step-on-several-gpus)): every process makes the same plan, each computes its share of each minibatch's segments, balanced by tokens, and the gradients are summed, so the update is the one a single GPU makes of the same minibatch |
| Files | Rank 0 writes the step's files from tensors every process gathers in the same order: the adapter in PEFT's layout (float32) or the full weights' serving copy (bfloat16 safetensors in files of at most 4 GB, an index, the configuration and tokenizer); `state/optimizer.pt` for an adapter and `state/shards/` (every process writes its shards) for full weights, every `state_every` steps since the processes loaded; `state/minibatches.jsonl` and `state/held.txt` every step |
| Going on | A step whose parent's `state/held.txt` names what the processes hold goes on from their memory and reads none of its files (a training pod fetches only that file, [`Resident`](../guide/reference.md#resident)). Otherwise they load the parent: the adapter or the weights, and the optimizer's state from `state/shards/` (read back by however many processes there are now) or `state/optimizer.pt`; a parent with neither starts the optimizer afresh, and full weights then from the bfloat16 serving copy |
| Failures | A failure in one process leaves the others waiting at a collective: the step raises `StepFailed` with its traceback, and every process is ended; the next step starts others and loads its parent. A step shared among processes does not leave out a minibatch that runs out of memory: the step fails |
| Ending | `close()` ends the processes (a training pod closes its trainer when its lease is released or another run takes the pod); they also end when the trainer's process does, when their connection to it closes, and when a step is cancelled |
| Memory | Each process may use the GPU memory free when it started, less `MEMORY_MARGIN`, as on one GPU |

A step's metrics add `gpus`, `whole_base`, `loaded_from_files` (1 where the processes loaded the parent's files rather
than going on from memory) and `full_state` (1 where the step wrote the trainer's full state); `peak_gpu_gib` is the
largest of any process's, `free_gpu_gib` the least.

What each GPU needs is estimated by `rollout_train.memory` (the check's `memory` rule, [validation](../guide/cluster.md#validation)),
in GiB a GPU for a 9B and a 4B model of Qwen3.5's shapes, segments of 8,192 tokens (the same on 80 and 96 GB cards;
"no" where it is more than the card):

| Model | Weights | 1 GPU | 2 | 4 | 8 |
|---|---|---|---|---|---|
| 9B | adapter | 26 | 34 (whole on each) | 29 | 27 |
| 9B | full | 150 (no) | 91 (no on 80 GB) | 55 | 37 |
| 4B | adapter | 15 | 19 | 16 | 15 |
| 4B | full | 75 | 44 | 27 | 18 |

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

Qwen3-0.6B, rank 8, on the same card, a step sharded with FSDP2 on one GPU (`test_sharded_on_gpu.py`) against the
fresh process, from the same adapter and optimizer's state:

| What | Fresh process | Sharded, one GPU |
|---|---|---|
| Loss | -0.0621 | -0.0633 |
| Gradient norm | 1.4858 | 1.4854 |
| KL moved | 0.00345 | 0.00344 |
| The two steps' adapters apart | | 3.5% of what the step moved them |
| A step of 12 segments of 400 to 900 tokens, after the first | 6.7 s | 6.8 s (the model sharded), 8.0 s (whole on the GPU); the first step 13 to 14 s |

## Tests

`tests/rollout_lora/` needs torch and is collected only when it is installed. It covers the adapter's file format and
the adapter switched off (the model it was added to), and the settings a trainer takes between steps (its objective's
numbers among them); the step itself is [`rollout_objectives`'](rollout-objectives.md#tests). `test_merge.py` covers
merging on the CPU. `test_small_on_gpu.py` runs only when asked
(`-m live`), with nothing else on the card: on Qwen3-0.6B (`ROLLOUT_SMALL_MODEL` names another) it trains an
adapter, takes two steps of every weight, and checks that a merged adapter gives what the adapter gave (an adapter
that moved logprobs by 2.6 on average, merged, is 0.06 from it: bfloat16 rounds part of a small update away). Its
steps write gigabytes, so give it `--basetemp` on disk, not `/tmp`.

`test_sharded.py` takes the trainers' steps on two and three processes on the CPU (gloo) with a tiny random Qwen3,
against the step on one: an adapter with its model sharded and whole, a step from files one process wrote, a
full-weight trainer's state written by two processes and read by three, a step that fails and the next that starts
the processes again. `test_sharded_on_gpu.py` (`-m live`) takes a sharded step on one GPU against the fresh process. A
step on several GPUs (NCCL) is taken only where a machine has them.
