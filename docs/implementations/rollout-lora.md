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
policy on the GPU, with its reference and entropies, in a process per GPU ([processes](#processes)), and on several
GPUs shards it over them ([several GPUs](#several-gpus)). The package is installed with `uv sync --all-extras` and
needs an NVIDIA GPU.

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
(`LoraTrainer(model, gpus=4, ...)`; a training pod's `ROLLOUT_TRAINER_GPUS`). A trainer that shares its GPU with the
trained channel's engines (`colocate_with`, or a `runpod-host` pod whose vLLM sleeps) is made with `colocated=True`:
its processes end after each step ([processes](#processes)).

### Every weight

`FullTrainer` trains every weight of a text model, with the same settings (`rank` and `whole_base` are not used). The
weights, their gradients and Adam's two moments are kept in float32 and sharded with FSDP2 on any number of GPUs, one
included, each unit's weights gathered in bfloat16 for the forward and backward passes, so that one GPU computes as
several do ([several GPUs](#several-gpus)). Each step leaves `weights/` as a serving copy (bfloat16 safetensors in
files of at most 4 GB, an index, the configuration saying `bfloat16`, and the tokenizer), and its full state (the
float32 weights and the optimizer's state) in `state/shards/` every `state_every` steps, every step by default. A step
starts from its parent's full state, or from the model for the first, or from its parent's serving copy where it is
given no state (its optimizer afresh). It refuses an image-text model. Qwen3-0.6B's step of 4 segments of 600 tokens
peaks under 14 GiB on a 16 GB card; its full state is about 7 GB. Its reference, for an objective that reads one, is a
frozen copy of the model trained over, in bfloat16 beside the policy, which it holds only when asked
(`frozen_reference = true`); without it, validation refuses such an objective. Sixteen bytes a weight is more than one
GPU holds above a few billion parameters: on several GPUs each holds its share.

```toml
[trainers.local-full]                 # the cluster config
kind = "full"
gpus = 1
models = ["Qwen/Qwen3-0.6B"]
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
| `state_every` | How often a trainer whose processes are kept between steps writes its full state (an adapter's optimizer; full weights' float32 weights and optimizer), counted since its processes loaded. 1 (the default): every step, so that any trainer goes on from any checkpoint as the one that made it would. More: the steps between leave it out (for full weights, about 12 bytes a weight less to write), and a step from one of them goes on only from the processes that hold it; once they are gone (a failed step, a restart, another run taking the pod) it fails rather than go on from less, and a run started from the newest checkpoint with its full state goes on ([processes](#processes)). A trainer beside an engine writes it every step |
| `whole_base` | For an adapter on several GPUs: each GPU holds the whole frozen model, gathered once (true), or its share, each layer gathered as it computes (false: a model too large for one GPU); none: whole where the model's files take at most half of the smallest GPU's memory, its memory by its name as the memory estimate counts it (`rollout_train.memory.holds_whole_base`) |

The step's settings (`learning_rate`, `tokens_per_step`, `max_kl`, `max_gradient_norm`, `passes`, `warmup_updates`,
`segment_tokens`, `segments_per_step`, `pack_tokens`, `share_prefixes`, `objective`) are
[`rollout_objectives`'](rollout-objectives.md#settings). The
objective's components can be named by this trainer's own settings too (`ratio`, `clip_low`, `clip_high`,
`segment_clip_low`, `segment_clip_high`, `truncate`, and `objective = "policy_gradient"` or `"likelihood"`), which say
the `default` or `sft` preset and its components.

`segment_tokens` and `segments_per_step` are the trainer's [`Budget`](../guide/reference.md#budget). A run's driver
gives `segment_tokens` to the trained channel as its longest turn, so that every sampled turn can be trained on.

## Files

A step is told where its files go (`into`) and leaves:

| Path under `into` | Holds |
|---|---|
| `weights/` | The adapter, in PEFT's layout (`adapter_config.json`, `adapter_model.safetensors`), which vLLM loads as it is, in float32: the next step starts from this file, and updates are smaller than bfloat16 resolves. Full weights: the serving copy, in bfloat16 |
| `state/optimizer.pt` | An adapter's optimizer's state after the step (every `state_every` steps), in one process's layout whatever the number of GPUs |
| `state/shards/` | Full weights' full state (every `state_every` steps): the float32 weights and the optimizer's state, as PyTorch's distributed checkpoint writes them, a file for each tensor's shard, read back by however many processes there are |
| `state/minibatches.jsonl` | What each minibatch of the step did: segments, tokens, loss, clipped share, Kullback-Leibler (KL) divergence estimate, learning rate, gradient norm |
| `state/held.txt` | Where the processes are kept between steps: the name they gave what they hold after the step |

A step's full state is what a later step goes on from as the trainer that made it would: any `LoraTrainer` (or
`FullTrainer`), on any number of GPUs, kept or not, takes any step from a checkpoint that has it, which is every
checkpoint while `state_every` is 1. A checkpoint given without its state (a supervised step's by default) starts the
optimizer afresh from its weights: an adapter's in float32, full weights' from their bfloat16 serving copy. A run keeps each step's files as a
[checkpoint](../libraries/rollout-train/checkpoints.md), named by its id, whose parent is the checkpoint the step began
from.

## Processes

A trainer steps in a process per GPU, under torchrun (`rollout_lora.workers`), which `rollout_lora.resident.Workers`
starts at its first step. Every step is handed to all of them, with the settings the trainer has then (so a setting
changed between steps reaches every process), and each answers; rank 0's answer is the step's metrics. Whether they
are kept between steps depends on whether the trainer shares its GPU:

| Trainer | Its processes |
|---|---|
| With its GPUs to itself (a `runpod-trainer`; a `lora` or `full` trainer without `colocate_with`) | Kept between steps ([`Resident`](../guide/reference.md#resident)): a step from the checkpoint they made last goes on from what they hold, and loads nothing. On Qwen3-0.6B that step takes 1.3 s, where a fresh process takes 9 s, most of it loading the model |
| Beside an engine (`colocated`) | Ended after each step, which frees the GPU's memory and the machine's for the engine: a trainer kept in system memory between steps takes 9 GB, next to a sleeping engine and whatever the environments run. vLLM's client libraries, which change how transformers builds models in the process that imports them, are never in the trainer's processes |

| Part | What it does |
|---|---|
| Loading | Torch's generator is seeded alike first (`SEED`): a new adapter is the same in every process and every trainer. On one process an adapter's policy is loaded onto the GPU as it is; on several each process loads it onto the CPU and shards it onto its GPU, the processes taking turns, so the machine holds one unsharded copy at a time. Full weights are sharded on any number ([several GPUs](#several-gpus)). What the processes held is dropped, and its memory freed, before another parent is loaded |
| Going on | A step whose parent's `state/held.txt` names what the processes hold goes on from their memory and reads none of its files (a training pod fetches only that file). Otherwise they load the parent: the adapter or the weights, and the optimizer's state from `state/shards/` or `state/optimizer.pt`; a parent without a state starts the optimizer afresh. A parent whose state left the full state out (`state_every` above 1), and which the processes do not hold, is refused with `StepFailed` before any process is asked: going on would start its optimizer afresh, and full weights from their bfloat16 copy |
| Learning rate | The settings' on every step, whatever the saved optimizer state carries |
| Failures | A failure in one process leaves the others waiting at a collective: the step raises [`StepFailed`](../guide/reference.md#stepfailed) with its traceback, and every process is ended; the next step starts others and loads its parent |
| Ending | `close()` ends the processes (a training pod closes its trainer when its lease is released or another run takes the pod); they also end when the trainer's process does, when their connection to it closes, and when a step is cancelled. What they hold is nothing once torchrun has ended (`holding`) |
| Memory | Each process may use the GPU memory free when it started, less `MEMORY_MARGIN` ([the memory bound](#the-memory-bound)) |

A step's metrics add `gpus`, `whole_base`, `loaded_from_files` (1 where the processes loaded the parent's files rather
than going on from memory) and `full_state` (1 where the step wrote the trainer's full state); `peak_gpu_gib` is the
largest of any process's, `free_gpu_gib` the least.

## Several GPUs

On more than one GPU of a machine, each process holds its shard of the policy and of the optimizer
(`rollout_lora.sharded`, PyTorch's FSDP2); full weights are sharded so on one GPU too.

| Part | What it does |
|---|---|
| Sharding | Each decoder layer is a unit of `fully_shard`; the rest of the model (the output layer, the last norm, embeddings held on the GPU) is the root's, the policy's `Scorer`, which computes the hidden states and the output layer's chunks in one call, so the output layer is gathered once a call. For an adapter, each layer's adapter matrices are a unit of their own, gathered for each pass; where each GPU holds the whole frozen model (`whole_base`), the frozen layers are gathered once and kept, else each layer's frozen weights are gathered as it computes. The frozen output layer is kept once gathered. Full weights, their gradients and Adam's moments are sharded in float32, and a frozen reference beside them |
| Precision | An adapter is kept and computed in float32 and its frozen model in bfloat16, as on one GPU: sharded on one process, its step is bitwise the step on the policy as it is (on the CPU). Full weights' units are gathered in bfloat16, on one GPU as on several, so the forward and backward passes run in bfloat16 (the residual stream and the norms too). Gradients are reduced in float32 (`MixedPrecisionPolicy`), and activations checkpointed as on one GPU |
| Reductions | An adapter whose frozen model is whole on each GPU reduces its gradients once a minibatch, in its last pass (`gradient_sync`): the passes before keep theirs in each process, in float32. Elsewhere each pass reduces its own: full weights' gradients kept to a minibatch's last pass would be on every GPU unsharded |
| The step | `rollout_objectives.step.PolicyStep` shared among the processes ([a step on several GPUs](rollout-objectives.md#a-step-on-several-gpus)): every process makes the same plan and the same packs, each computes its share of each pass's packs (balanced by their count, then their tokens), and the gradients are summed, so the update is the one a single GPU makes of the same minibatch. The policy's `idle` passes keep a process with fewer packs in step with the others, and its `clip_gradients` clips by the norm over every shard |
| Files | Rank 0 writes the step's files from tensors every process gathers in the same order: the adapter in PEFT's layout (float32) or the full weights' serving copy; `state/optimizer.pt` for an adapter and `state/shards/` (every process writes its shards) for full weights, every `state_every` steps since the processes loaded; `state/minibatches.jsonl`, and `state/held.txt` where the processes are kept, every step |
| Failures | A step shared among processes does not leave out a minibatch that runs out of memory: the step fails |

What each GPU needs is estimated by `rollout_train.memory` (the check's `memory` rule, [validation](../guide/cluster.md#validation)),
in GiB a GPU for a 9B and a 4B model of Qwen3.5's shapes, segments (and packs) of 8,192 tokens (the same on 80 and 96 GB cards;
"no" where it is more than the card):

| Model | Weights | 1 GPU | 2 | 4 | 8 |
|---|---|---|---|---|---|
| 9B | adapter | 28 | 36 (whole on each) | 32 | 29 |
| 9B | full | 164 (no) | 92 (no on 80 GB) | 56 | 38 |
| 4B | adapter | 16 | 21 | 18 | 17 |
| 4B | full | 80 | 45 | 28 | 19 |

## The memory bound

Each process may use the GPU memory that is free when it starts, less `MEMORY_MARGIN`, and no more
(`torch.cuda.set_per_process_memory_fraction`). Some drivers let a process spill past the card into system memory,
where a step crawls instead of failing; the bound turns that into an out-of-memory error. On one process, a minibatch
that runs out of memory is dropped whole, its gradient cleared, and counted in `minibatches_out_of_memory`; the pass
goes on with the next. Full weights are sharded with FSDP2 on one GPU too, and a backward pass that ends part way leaves
gradients on the weights it gathered that it never reduced to their shards (the output layer's, the last norm's, a
layer's half done): the policy's `recover` drops them and resets FSDP's state of the pass (`reset_iter_state`), else
the next minibatch would add them to its own. A pack of the step's start that runs out of memory runs again a segment at a time, and an item with a
segment that runs out alone is left out and counted in `start_out_of_memory`. Segments longer than `segment_tokens`
are left out before the pass and counted in `segments_too_long`. A pack holds at most `pack_tokens` (by default
`segment_tokens`) row tokens, and its activations are those of a segment as long as its row whatever prefixes it shares
(nothing is copied for a branch), so a pass needs about the memory of the longest segment alone; the memory estimate
(`rollout_train.memory`, [several GPUs](#several-gpus)) counts activations for the larger of the two, and for a model
with linear attention a state for each run of a pack beside each chunk's.

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
| Packs | `Policy.packed`: every segment of a pack in one pass, the logprobs each has alone ([packs](#packs)) |

## Packs

`rollout_lora.packing.prepare` readies a model for packs (`rollout_objectives.packing`) where its decoder is one of
`PACKABLE` (Llama, Qwen2, Qwen3, and Qwen3.5's text model, whose layers are softmax attention and a gated delta rule)
with no sliding window and no rotary scaling that reads the sequence's longest position (`dynamic`, `longrope`: in a
pack, the row's), and the policy says so (`packing`); any other model runs one segment at a time, and a step's `packed`
metric is 0.
`Policy.packed` and `FullPolicy.packed` run a pack's row through the decoder and the output layer in one call of the
policy's `Scorer`, so on several GPUs a pack is one pass of the sharded model. Each layer keeps the segments apart:

| Layer | In a pack |
|---|---|
| Softmax attention | The attention function `PACKED`: each root on its own (`scaled_dot_product_attention`, causal), then every branch token in one call over the row's keys and values, seeing its prefix and its own branch's tokens up to itself; each key and value head serves its group of query heads as it is (`enable_gqa`), and nothing is copied for a branch. On the GPU that call is FlexAttention, compiled once a process for rows of any length, with a block mask of 128 tokens a side that skips the blocks no branch token sees (a pass without a gradient runs the kernels a pass with one does, so that both round alike); on the CPU, `scaled_dot_product_attention` with the mask whole. Rotary positions are each token's position in its segment. Without a pack it is transformers' `sdpa` |
| Qwen3.5's gated delta rule | The short convolution reads, for each run, the tokens before it in its segment (zeros for a root; for a branch, its prefix's last tokens, as many as the convolution's width less one). The recurrence runs over the roots, each from a zero state, then over the branches, each from the state its prefix ended in: on the GPU flash-linear-attention's kernel takes the runs' boundaries (`cu_seqlens`, and a copy on the CPU so that it reads none back from the GPU) and the branches' starting states and passes the gradient back through them; on the CPU transformers' own recurrence runs a run at a time |
| Every other layer | Each token on its own (projections, norms, MLPs), whatever is beside it in the row |

A shared prefix is computed once: its keys, values and recurrent state are those every segment under it computes
alone, and its backward pass adds up every branch's gradient. Every model `PACKABLE` names shares prefixes. What the
layers build from a pack (the branches' block mask, the convolution's indices) is built once a pass, in its `Layout`,
which FSDP passes to each layer as it is.

## The step

Each process takes [`rollout_objectives.step.PolicyStep`](rollout-objectives.md#the-step) on the policy: the plan of
minibatches, where the step starts (each sampled token's logprob on the weights it starts from, and the reference's
where the objective reads it), each minibatch's loss as the objective composes it, the stop at `max_kl` and AdamW's
update. Which items are in the batch, and each one's advantage, is the
[algorithm's](../libraries/rollout-train/training.md) business. The step's metrics are
[`rollout_objectives`'](rollout-objectives.md#metrics), with the processes' `peak_gpu_gib` (GPU memory reserved at
the peak) and `free_gpu_gib` (free when they started); the training loop keeps them with the checkpoint the step
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

Packs on the same card: one step of gridworld-like turns (a system prompt of 1,000 tokens every turn shares, an
observation of about 500, a reply of about 600 sampled tokens; `tokens_per_step` 4,096, `objective = "default"`) from
the same adapter, one segment at a time, in packs, and in packs with shared prefixes, after a small step each way (the
kernels compiled and tuned):

| Model | Segments | `pack_tokens` | One at a time | Packed | Packed, prefixes shared |
|---|---|---|---|---|---|
| Qwen3-0.6B, rank 32 | 300 | 16,384 | 130 s, 9,600 tokens/s | 108 s, 11,500 tokens/s | 67 s, 18,400 tokens/s (43% of the tokens shared) |
| `cyankiwi/Qwen3.5-9B-AWQ-4bit`, rank 32 | 32 | 8,000 | 95 s, 1,240 tokens/s | 78 s, 1,520 tokens/s | 51 s, 2,340 tokens/s (37% shared) |

The three agree to bfloat16's rounding: the first minibatch's loss is -0.01681 each way on Qwen3-0.6B, and -0.01456 to
-0.01457 on Qwen3.5; the step's gradient norm is within 1%.

Peak activations of one forward and backward pass over one pack of about 8,190 tokens, and its time (a 32-rank adapter;
the pass's peak less what the policy held before it):

| Pack | Qwen3-0.6B | Qwen3.5-9B AWQ |
|---|---|---|
| One segment of 8,192 | 1.29 GiB, 1.1 s | 5.24 GiB, 7.0 s |
| 8 segments of 1,024 | 1.29 GiB, 0.9 s | 5.24 GiB, 6.9 s |
| A prefix of 2,048 and 15 branches of 409 | 1.29 GiB, 1.1 s | 5.27 GiB, 8.6 s |
| A prefix of 4,096 and 30 branches of 136 | 1.29 GiB, 1.2 s | 5.30 GiB, 7.1 s |
| A prefix of 7,168 and 40 branches of 25 | 1.29 GiB, 1.2 s | 5.31 GiB, 7.4 s |

FlexAttention compiles its kernels once a process for each kind of pack (its branches' tokens one block of queries, or
several), about 5 s each.

A step that learns nothing (a learning rate of 0) on `cyankiwi/Qwen3.5-9B-AWQ-4bit`, 16 turns under a shared prompt of
1,000 tokens in 4 minibatches: every minibatch's KL from the step's start is 0, and the mean ratio exactly 1.

Qwen3-0.6B, rank 8, on the same card (`test_resident_on_gpu.py`), with PyTorch's deterministic algorithms: a step of
6 segments of 400 to 900 tokens in a fresh process, the same step again, the same step in a kept process, and that
process's next step, from memory, against the same step from its files, in a fresh process. Each pair is bitwise
alike. A step in a fresh process took 9 s, the kept process's first 7.3 s, its next, from memory, 1.3 s.

## Tests

`tests/rollout_lora/` needs torch and is collected only when it is installed. It covers the adapter's file format and
the adapter switched off (the model it was added to), and the settings a trainer takes between steps (its objective's
numbers among them); the step itself is [`rollout_objectives`'](rollout-objectives.md#tests). `test_merge.py` covers
merging on the CPU. `test_small_on_gpu.py` runs only when asked
(`-m live`), with nothing else on the card: on Qwen3-0.6B (`ROLLOUT_SMALL_MODEL` names another) it trains an
adapter, takes steps of every weight (its serving copy and full state written, its next step from memory, a step from
its weights alone), and checks that a merged adapter gives what the adapter gave (an adapter
that moved logprobs by 2.6 on average, merged, is 0.06 from it: bfloat16 rounds part of a small update away). Its
steps write gigabytes, so give it `--basetemp` on disk, not `/tmp`.

`test_packing.py` takes tiny random Qwen3, Llama and Qwen3.5 models on the CPU: a pack gives each segment the
logprobs, entropies, reference logprobs and logprobs of given tokens it has alone, its prefix shared or not; changing
one segment of a pack changes no other's (no attention, convolution or recurrent state crosses a boundary); the
branches' block mask holds the blocks of the whole mask that FlexAttention's own would; a model with `dynamic` or
`longrope` rotary scaling is not packed; on weights a step leaves alone, every minibatch's ratios are exactly 1 in
bfloat16; a step in packs takes the losses, minibatches and gradients of a step one segment at a time for each objective family (a policy
gradient with and without a KL to the reference and an entropy bonus, a segment ratio, a likelihood, pairs, labelled
examples, both forms of distillation); the first minibatch's start folded into it is the start computed apart; and two
processes (FSDP2 over gloo) step in packs, with shared prefixes, as one does. `test_packing_on_gpu.py` (`-m live`)
takes Qwen3.5's packs on the GPU, where flash-linear-attention's kernels and FlexAttention run: logprobs and gradients
against each segment alone, a step against one segment at a time, and every minibatch's ratios exactly 1 in bfloat16
on weights a step leaves alone.

`test_sharded.py` takes the trainers' steps on one, two and three processes on the CPU (gloo) with a tiny random Qwen3,
against the step on one: an adapter with its model sharded and whole, a step from files one process wrote, a
full-weight trainer's state written by two processes and read by three, a step after the processes ended that goes on
from its parent's full state (or is refused where the parent left it out), a step that fails and the next that starts
the processes again; one process kept between steps and one ended after each, their steps alike; what the processes
hold is nothing once torchrun is gone, and is dropped before another parent is loaded; an adapter's float32 units
sharded on one process step bitwise as the policy does, its gradients reduced once a minibatch are those reduced
after each pass, and full weights sharded on one process drop a minibatch that runs out of memory part way through its
backward pass with nothing of it left in the next minibatch's gradient (`sharded_alone.py`, under torchrun). `test_resident_on_gpu.py` (`-m live`) takes a kept process's
steps on one GPU against fresh processes'. A step on several GPUs (NCCL) is taken only where a machine has them.
