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
sequence_tokens = 8000
```

A [profile](../guide/deploying.md) calls `LoraTrainer(model, directory, **settings)` with the model of the channel it
trains and the run's directory. Every key of `[trainer]` other than `kind`, `channel` and `colocated` is a setting.

## Settings

[`LoraSettings`](../guide/reference.md#lorasettings) is the one place the settings and their defaults are written.

| Setting | What it sets |
|---|---|
| `rank` | The adapter's rank. Its scaling (`alpha`) is twice the rank |
| `learning_rate` | AdamW's learning rate. Adam moves a weight by at most this much per optimizer step |
| `clip_low`, `clip_high` | The probability ratio is clipped to `1 - clip_low` .. `1 + clip_high` |
| `tokens_per_step` | Sampled tokens per optimizer step. How far a step moves the policy is set by how many optimizer steps its tokens make |
| `max_kl` | The pass stops when the policy has moved this far, in nats per token (`None`: never) |
| `max_gradient_norm` | Gradients are clipped to this norm before each optimizer step |
| `sequence_tokens` | The longest sequence a step can hold on its GPU (`None`: any) |
| `sequences_per_step` | How many sequences a step can afford (`None`: any number) |

`sequence_tokens` and `sequences_per_step` are the trainer's [`Budget`](../guide/reference.md#budget). An open profile
gives `sequence_tokens` to the trained channel as its longest turn, so that every sampled turn can be trained on.

## Files

A step is told where its files go (`into`) and leaves:

| Path under `into` | Holds |
|---|---|
| `weights/` | The adapter, in PEFT's layout (`adapter_config.json`, `adapter_model.safetensors`), which vLLM loads as it is. Weights are saved as float32: the next step starts from this file, and updates are smaller than bfloat16 resolves |
| `state/optimizer.pt` | The optimizer's state after the step |
| `state/minibatches.jsonl` | What each minibatch of the step did: sequences, tokens, loss, clipped share, KL estimate, gradient norm |

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
the next. Sequences longer than `sequence_tokens` are left out before the pass and counted in `sequences_too_long`.

## The policy

| Part | What `rollout_lora.policy` and `rollout_lora.quantized` do |
|---|---|
| Weights | The checkpoint's packed int4 weights stay packed. `Int4Linear` dequantizes a layer's weight inside the matrix multiply, in the forward and again in the backward pass, so at most one layer's bfloat16 weight exists at a time |
| Adapter | LoRA on every attention, linear-attention and MLP projection of the language model (`TARGETS`). `B` starts at zero: a new adapter changes nothing |
| Left off the GPU | A vision tower is dropped. The token embedding table is memory-mapped from the checkpoint file and only a sequence's rows are read |
| Output layer | Run only at the sampled positions, `LOGIT_ROWS` at a time, each chunk recomputed in the backward pass. Peak memory is one chunk's logits, whatever the share of sampled tokens |
| Activations | Gradient checkpointing over the transformer |

## The step

One pass over the batch (`rollout_lora.step.ClippedPolicyGradient`). Which sequences are in the batch, and each
one's advantage, is the [algorithm's](../libraries/rollout-train/training.md) business.

1. The sequences are shuffled with the step's `seed` and cut into minibatches of about `tokens_per_step` sampled
   tokens. A last minibatch of less than half that joins the one before.
2. For each sequence, the policy computes the logprob of every token the policy sampled (the recorded spans). Forced
   tokens and prompts are not trained on.
3. The loss is PPO's clipped objective against the behavior logprobs recorded while sampling: for each token,
   `-min(ratio · advantage, clip(ratio) · advantage)`, where `ratio` is the probability now over the probability
   then. The upper clip is wider than the lower. The loss is a mean over the minibatch's tokens. There is no KL
   penalty.
4. Before a minibatch's optimizer step, its estimate of KL(behavior ‖ policy) on the sampled tokens is compared with
   the first minibatch's. If it is more than `max_kl` beyond it, the pass stops without that step.
5. Otherwise gradients are clipped to `max_gradient_norm` and AdamW steps, with no weight decay.

A sampled token whose recorded logprob is not finite fails the step.

## Metrics

A step returns these. The training loop keeps them with the version the step made, and as `update` in the group's iteration.

| Metric | Meaning |
|---|---|
| `kl_floor` | The first minibatch's KL estimate, before any optimizer step: the engine's and the trainer's numerical difference, and how stale the sequences are |
| `kl_moved` | The last stepped minibatch's estimate less the floor: how far the step moved the policy |
| `stopped_at_max_kl` | 1 if the pass stopped at `max_kl` |
| `loss`, `clip_fraction`, `mean_ratio` | Per token: the loss, the share of tokens whose ratio was clipped, the mean ratio |
| `mean_mismatch` | The mean absolute difference between the trainer's logprob and the recorded one |
| `gradient_norm` | Before clipping, the mean over optimizer steps |
| `optimizer_steps` | Minibatches stepped on |
| `tokens`, `sequences` | Sampled tokens and sequences trained on |
| `sequences_given`, `sequences_too_long`, `longest_sequence_tokens` | What the batch held, how many were left out for their length, and the longest one kept |
| `minibatches_out_of_memory` | Minibatches dropped |
| `seconds` | The pass |
| `peak_gpu_gib`, `free_gpu_gib` | GPU memory reserved at the peak, and free when the process started |

## Measurements

One RTX 5080 (16 GB), `cyankiwi/Qwen3.5-9B-AWQ-4bit`, rank 32.

| What | Result |
|---|---|
| GPU memory with the policy loaded | 6.6 GiB |
| Peak GPU memory in a step | 10.7 GiB with sequences of 5,000 tokens, 12.4 GiB with 8,000 |
| Time per sequence | 4 to 8 s |
| A step of 384 turns of the [Minecraft swarm](../products/minecraft-swarm.md) | About 40 optimizer steps, about half an hour |
| Trainer logprobs against vLLM's | A mean difference of 0.016 per token, with and without an adapter |
| Without the memory bound, under Windows | A step of 96 turns that needed more than the card ran 13 minutes without finishing and left the host 0.6 GB of free memory |
| Allocator | `expandable_segments` saves about 0.7 GiB at the peak |

## Tests

`tests/rollout_lora/` needs torch and is collected only when it is installed. It covers the adapter's file format,
the direction of the update, forced tokens, sequences left out, the last minibatch, the KL stop and a missing logprob.
