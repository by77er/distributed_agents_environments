# vLLM engine

Code: `rollout_vllm`

`VllmEngine` implements the [`Engine`](../guide/reference.md#engine) protocol on vLLM: tokens in; tokens, their
logprobs and a finish reason out. A [channel](../libraries/rollout-train/channels.md) holds one or more of them and
knows nothing else about vLLM. The package is installed with `uv sync --all-extras` and needs Linux and an NVIDIA GPU.

## In a profile

A [profile](../guide/deploying.md) names the engine for a channel and gives one table of options per replica:

```toml
[channels.policy]
model = "cyankiwi/Qwen3.5-9B-AWQ-4bit"
renderer = "rollout_qwen:qwen35"
engine = "rollout_vllm:VllmEngine"
engines = [{ gpu_memory_utilization = 0.78, max_model_len = 8192, max_num_seqs = 20 }]
```

Each entry of `engines` is passed to `VllmEngine(model, **entry)`. An entry may be empty; the defaults are in the
[reference](../guide/reference.md#vllmengine).

| Option | What it sets |
|---|---|
| `gpu_memory_utilization` | The share of the GPU's memory the engine takes while awake: the weights, and the rest is its cache |
| `max_model_len` | The most tokens, prompt and completion together, that the engine accepts. The channel reads it as `Engine.max_model_len` |
| `max_num_seqs` | How many requests are sampled at once; further requests queue |
| `max_num_batched_tokens` | How many tokens one scheduling step of vLLM processes |
| `max_lora_rank` | The highest adapter rank the engine loads; at least the trainer's `rank` |
| `max_loras` | How many adapters one batch may mix. A channel keeps its run's `max_lag + 1` loaded (two by default: the one it samples from and the one before it), and an engine host serving several runs holds each one's side by side, so this is at least the sum of their windows |
| `language_model_only` | Load only the language model of a multimodal checkpoint |
| `speculative` | vLLM's speculative decoding, as its `speculative_config`: for Qwen3.5, whose checkpoints carry a multi-token prediction layer, `{ method = "qwen3_5_mtp", num_speculative_tokens = 2 }`. The draft is not adapted with the channel's LoRA, so fewer drafted tokens are accepted as the policy moves from the base; what is sampled keeps the target model's distribution. Whether the logprobs it returns are the target model's, as the trainer needs, is to be checked before training on them. Off unless given |
| `quantization` | vLLM's quantization of a checkpoint as it loads: `fp8` turns a bfloat16 checkpoint's linear layers into FP8 (the embeddings and the output layer stay bfloat16). Off unless given |
| `seed` | The engine's sampling seed |

The engine always loads the model as bfloat16, with LoRA and sleep mode enabled.

## What a request returns

A request is a list of prompt tokens, sampling parameters, stop tokens and the name of an adapter (or none, for the
base model). The engine returns the sampled tokens, special tokens and the stop token included, and one logprob per
token. The logprobs are of the distribution the token was sampled from, after temperature (vLLM's
`processed_logprobs`). A sampled token that comes back without its logprob raises `RuntimeError`: it could not be
trained on. The finish reason is `length` when the token limit ended the sample and `stop` otherwise.

## Adapters by name

`load_adapter(name, path)` registers a LoRA adapter, a directory in PEFT's layout, under a name. A request that names
it samples from it. `remove_adapter(name)` unloads it. The [channel](../libraries/rollout-train/channels.md) decides
which adapters are loaded and when, as it publishes weights.

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
`ENGINE_PROCESS`, found with `rollout.processes.children`. An open profile writes them to `engine.json` in the run's
directory with `rollout.processes.note_processes`. The next process to open a profile over that directory calls
`rollout.processes.end_orphans`: if the process that wrote the file is gone, each noted process that still runs under
its noted name is killed.

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
