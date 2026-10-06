# vLLM engine

For whoever deploys or extends inference: vLLM as an engine, its options, adapters by name, sleep and wake, and
measurements.

**Read first:** [Channels and engines](../libraries/rollout-train/channels.md). **Next:** [LoRA
trainer](rollout-lora.md).

Code: `rollout_vllm`

`VllmEngine` implements the [`Engine`](../guide/reference.md#engine) protocol on vLLM: tokens in; tokens, their
logprobs and a finish reason out, or the logprobs of tokens it is given ([scoring](#scoring)). A [channel](../libraries/rollout-train/channels.md) holds one or more of them and
knows nothing else about vLLM. The package is installed with `uv sync --all-extras` and needs Linux and an NVIDIA GPU.

## In a cluster config

A `vllm` provider of the [cluster config](../guide/cluster.md#inference-providers) runs `VllmEngine` in each engine
host it starts for a run's channel, with the options of the model the channel serves:

```toml
[inference.local-vllm]
kind = "vllm"
gpus = 1
[inference.local-vllm.models."cyankiwi/Qwen3.5-9B-AWQ-4bit"]
context = 8192                    # max_model_len, unless options say one
options = { gpu_memory_utilization = 0.78, max_num_seqs = 20, max_lora_rank = 96 }
```

An engine host makes `VllmEngine(model, max_model_len=context, **options, max_logprobs=...)`. The defaults are in the
[reference](../guide/reference.md#vllmengine).

| Option | What it sets |
|---|---|
| `gpu_memory_utilization` | The share of the GPU's memory the engine takes while awake: the weights, and the rest is its cache |
| `max_model_len` | The most tokens, prompt and completion together, that the engine accepts. The channel reads it as `Engine.max_model_len` |
| `max_num_seqs` | How many requests are sampled at once; further requests queue |
| `max_num_batched_tokens` | How many tokens one scheduling step of vLLM processes |
| `max_lora_rank` | The highest adapter rank the engine loads; at least the trainer's `rank` |
| `max_loras` | How many adapters one batch may mix. A channel keeps its run's `max_lag + 1` loaded (two by default: the one it samples from and the one before it), and an engine host serving several runs holds each one's side by side, so this is at least the sum of their windows |
| `language_model_only` | Load only the language model of a multimodal [checkpoint](../libraries/rollout-train/checkpoints.md) |
| `speculative` | vLLM's speculative decoding, as its `speculative_config`: for Qwen3.5, whose checkpoints carry a multi-token prediction layer, `{ method = "qwen3_5_mtp", num_speculative_tokens = 2 }`. The draft is not adapted with the channel's LoRA (low-rank adaptation), so fewer drafted tokens are accepted as the policy moves from the base; what is sampled keeps the target model's distribution. Whether the logprobs it returns are the target model's, as the trainer needs, is to be checked before training on them. Off unless given |
| `quantization` | vLLM's quantization of a checkpoint as it loads: `fp8` turns a bfloat16 checkpoint's linear layers into FP8 (the embeddings and the output layer stay bfloat16). Off unless given |
| `seed` | The engine's sampling seed |
| `max_logprobs` | The most tokens a request may ask for at each position with their logprobs (`top`), as vLLM caps them: 20 unless given. An engine host started from a cluster's `vllm` provider is given the provider's `max_logprobs`, the top-k it declares |
| `reasoning` | `parser`, `open` and `close`: vLLM's reasoning parser for the model and the renderer's thinking open and forced close, as its reasoning config, so that the engine bounds a request's thinking itself (`Engine.bounds_thinking`, `thinking_budget`). Off unless given |

The engine always loads the model as bfloat16, with LoRA and sleep mode enabled.

## What a request returns

A request is a list of prompt tokens, sampling parameters, stop tokens and the name of an adapter (or none, for the
base model). The engine returns the sampled tokens, special tokens and the stop token included, and one logprob per
token. The logprobs are of the distribution the token was sampled from, after temperature (vLLM's
`processed_logprobs`). A sampled token that comes back without its logprob raises `RuntimeError`: it could not be
trained on. The finish reason is `length` when the token limit ended the sample and `stop` otherwise.

A request may also ask for the `top` most likely tokens at each sampled position (`top`, up to `max_logprobs`), as a
teacher's own samples carry them for distillation: `Generation.top_tokens` holds them, most likely first, and
`Generation.top_logprobs` their logprobs, of the same distribution as the sampled token's.

## Scoring

`score(tokens, start=, end=, top=, adapter=)` gives the logprobs the model, or the adapter named, gives the tokens at
positions `start` to `end` of `tokens` (`end` none: to the end), each given the tokens before it, and the `top` most
likely tokens at each position with their logprobs. Nothing is sampled. It is how a teacher scores a student's tokens.

```python
scores = await engine.score(tokens, start=len(prompt), top=20, adapter=None)
scores.logprobs        # one per position from start: the logprob of the token there
scores.top_tokens      # per position, the 20 most likely tokens, most likely first
scores.top_logprobs    # their logprobs
```

- **How.** One request of `tokens` up to `end` with vLLM's prompt logprobs (`prompt_logprobs = top`) and
  `max_tokens = 1`: vLLM generates at least one token, which is dropped. The first position cannot be scored, having
  nothing before it, so `start` is at least 1. The sequence must leave room for that one token: fewer than
  `max_model_len` tokens.
- **Which distribution.** The model's own, at temperature 1: prompt logprobs skip the sampling parameters, whatever
  `logprobs_mode` says.
- **Its cost.** vLLM reads no prefix cache for a request with prompt logprobs, so every position up to `end` is
  computed again. It computes their logits 1,024 positions at a time with a fused top-k log-softmax, so a long sequence
  takes no more GPU memory than a short one: about 0.6 GiB beyond the engine's own share for Qwen3's vocabulary of
  152K ([measurements](#measurements)).
- **What it refuses.** A range that does not lie within `tokens`, or a `top` above `max_logprobs`: `ValueError`. A
  scored token that comes back without its logprob raises `RuntimeError`.

## Adapters by name

`load_adapter(name, path)` registers a LoRA adapter, a directory in the layout of PEFT (parameter-efficient
fine-tuning), under a name. A request that names it samples from it. `remove_adapter(name)` unloads it. The
[channel](../libraries/rollout-train/channels.md) decides which adapters are loaded and when, as it publishes weights.

## Full weights

`load_weights(path)` reads a full checkpoint's files into the model the engine holds (vLLM's `reload_weights`) and
clears the prefix cache; the engine serves them from then on, and reads them again on waking. The files are in the
model's own layout, as the [full-weight trainer](rollout-lora.md#every-weight) or a merge writes them. Where the
output layer is tied to the embeddings (Qwen3-0.6B), vLLM warns that `ParallelLMHead` failed to load: it shares the
embeddings, and its logprobs after a reload match an engine started on the same files.

## Sleep and wake

A trainer that shares the GPU needs it free while it steps: [`Colocated`](../guide/reference.md#colocated) puts the
channel's engines to sleep for the step ([the trainer](../libraries/rollout-train/training.md#the-trainer)).

| Call | What the engine does |
|---|---|
| `sleep()` | Discards its prefix cache and sleeps at vLLM's level 2: the weights are dropped, not parked in system memory |
| `wake()` | Reads the weights again (the model's, or the full checkpoint it serves), then allocates the cache |

## The engine core process

vLLM's engine core runs in a process of its own, a child of the process that made the `VllmEngine`. vLLM starts it
with `spawn`, so an entry point that makes an engine guards itself with `if __name__ == "__main__":`.

`close()` shuts the core down. A process that is killed outright cannot call it, and a core left behind holds its GPU.
`VllmEngine.processes` therefore lists the core's process ids: the children of this process whose name begins with
`ENGINE_PROCESS`, found with `rollout.processes.children`. An engine host runs in a Ray actor its run's job owns, so
Ray ends the actor, and with it the core, when the job ends. `rollout.processes.note_processes` and `end_orphans` write
such ids to a file and end the noted processes that still run, for a process whose engines outlive it otherwise.

## Measurements

One RTX 5080 (16 GB), vLLM 0.30, `cyankiwi/Qwen3.5-9B-AWQ-4bit`.

| What | Result |
|---|---|
| Weights on the GPU | 8 GiB |
| `sleep()` | 0.2 s |
| `wake()`, with the checkpoint in the file cache | 3 s |
| System memory while asleep | 3.3 GiB. With the weights parked in system memory instead, 11.2 GiB |
| Throughput on short contexts | 88 tokens/s for one stream, 730 tokens/s for sixteen at once |
| Cache at `gpu_memory_utilization = 0.72` | 63,000 tokens. Sixteen agents of the [Minecraft team](../products/minecraft-team.md) need more: requests queue, and a group samples 240 tokens/s |
| `gpu_memory_utilization = 0.85` | The card is full, and a turn of the Minecraft team takes minutes instead of 10 s |

Scoring, on the same card with `Qwen/Qwen3-0.6B` (`max_num_batched_tokens = 4096`, `top = 20`), measured by the
engine core's allocator:

| Sequence | GPU memory beyond the engine's own share | Time |
|---|---|---|
| 512 tokens | 0.15 GiB | 0.15 s |
| 2,048 tokens | 0.6 GiB | 0.09 s |
| 4,096 tokens | 0.6 GiB | 0.17 s |
| 8,000 tokens | 0.6 GiB | 0.64 s |

Scores against a forward pass of the same checkpoint in transformers, bfloat16 on the same GPU, over 1,500 tokens of
text with `top = 20` (`tests/rollout_vllm/test_scoring_on_gpu.py`):

| | `Qwen/Qwen3-0.6B` | `Qwen/Qwen3-1.7B` |
|---|---|---|
| Mean absolute difference of the tokens' logprobs | 0.038 | 0.035 |
| Largest difference | 0.52 | 0.37 |
| Top-20 tokens shared, on average over positions | 98.2% | 98.1% |
| Top-20 tokens shared at the worst position | 17 of 20 | 18 of 20 |
| Positions whose top 20 are the same tokens | 66% | 65% |
| Positions 1,200 to 1,300 scored again in a shorter request: mean absolute difference | 0.032 | 0.025 |
| A sample's logprobs (temperature 1) against scoring what it sampled: mean absolute difference | 0.027 | 0.032 |

The two compute in bfloat16 with kernels of their own, so they are close but not equal. So does vLLM itself across
requests of different shapes.
