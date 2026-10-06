# Scaling: more models, parallel topologies, and choosing GPUs

**Status: phase 4 built in part, the rest proposed.** Built: FSDP2 in `rollout_lora` for `lora` and `full` on the GPUs
of one machine (a process per GPU under torchrun, sharded state, the serving copy and the adapter gathered by rank 0),
a resident trainer on several GPUs that keeps its state between steps and writes its full state every
`trainer.state_every` steps, a provider's `gpus` (local) and `gpu_count` (RunPod) above one, and the check's `memory`
rule from a model's files ([LoRA trainer](../implementations/rollout-lora.md#several-gpus)). Not built: everything
else here, among it MoE expert LoRA with grouped kernels, the model card and the planner (phases 1 and 3), engines on
several GPUs (phase 2), weights pushed to engines over NCCL or CUDA IPC (phase 5), several nodes (phase 6), and
tensor, pipeline and expert parallel training and Megatron (phase 7). The rest of the note was read against main
`2b93527`. A design note: see [Design notes](README.md) for the others.

Three questions: what it takes to train and serve many more model families and sizes; how training and inference
spread over several GPUs and machines, and how weights move between them; and how a run's GPUs (type and count, for
the trainer and the engines) can be chosen for it from a model, a workload, a time target and a budget. The roles of
[the architecture](../architecture/overview.md) do not change for any of it. A trainer still takes a step from a
batch and a parent's files and leaves a checkpoint's files; an engine replica still serves what a run's serving
record says, by checkpoint name. How many GPUs and machines a trainer or a replica spans, and how they talk to each
other, is the business of the provider's implementation and of the placement the run's demand asks for.

Code read for this note: `rollout_train.providers`, `.demand`, `.validation`, `.jobs`, `.bridges`, `.serving`,
`.following`, `.inference.hosts`, `.colocated`, `.recorder.renderers`; `rollout_vllm.engine`; `rollout_lora`
(`policy`, `full`, `layers`, `worker`, `merge`); `rollout_tinker.weights`; `rollout_qwen`; `rollout_runpod.api`;
`deploy/chart/rollout/files/cluster.toml` and `rayjob.yaml`. Figures from outside are dated and their sources listed
at the end.

## The answer in brief

| Question | Recommendation |
|---|---|
| What a new model family needs | A package of its own in `implementations/` that declares two things through entry points: its renderer (as now) and an **architecture**: which modules LoRA adapts, how its names map to PEFT's, vLLM's and Tinker's, where its decoder and vision tower are, how its experts are laid out. Trainers and bridges read the architecture instead of constants. A conformance kit of tests says when a family is supported |
| What the platform knows about a model | A **model card** computed from the checkpoint's `config.json` (parameters, active parameters, layers, attention layout, KV bytes per token, vocabulary, modality, native precision), cached, overridable in the cluster config. Validation, the planner and bridges read it |
| MoE and LoRA | Adapt every expert, each at the run's rank divided by the experts active a token (as Tinker does), where both the trainer and the provider's engine support it; else attention and shared experts only. The architecture says which groups each side adapts, and validation refuses a pair that disagrees |
| Very large models | LoRA on models above about 200B total parameters goes to Tinker first; the platform's own GPUs take them when the planner says a scheduled gang is cheaper for a sustained run, and only after single-node sharding is built |
| Inference parallelism | Tensor parallel inside one node first (`tensor_parallel` on a model offer, a replica's GPUs equal to it), then expert parallel for MoE inside a node. Pipeline parallel across nodes only for a model that does not fit one node of the largest GPUs at FP8. Data parallel stays what replicas are |
| Training parallelism | FSDP2 (fully sharded data parallel) across the GPUs of one node for `lora` and `full`, in the platform's own trainer, so objectives stay one implementation. Megatron-style tensor, pipeline and expert parallel through Megatron-Bridge, as a trainer kind of its own, only when a run needs more than one node or a large MoE trained in full |
| Gangs | A part that spans nodes gets a placement group of its own (one bundle per node) inside the run's own Ray cluster, which Kueue admitted whole; on Kubernetes a worker group per spanning part, kept in one network block by Kueue's topology-aware scheduling |
| Moving weights | The serving record keeps naming files in the blob store: that is what lets followers anywhere load a checkpoint, and what a crash recovers from. Inside a gang, a full-weight trainer may also push the new weights straight into its engines (CUDA IPC on a shared GPU, NCCL between GPUs), and the engine host then reports the checkpoint as held. LoRA adapters always move as files |
| Choosing GPUs | A planner: a pure function from the model card, the run's settings, the environment's numbers and a catalog of GPU types and prices to a ranked list of plans (provider, GPU type, count and shape for the trainer and the engines; step time and cost per step, broken down). It is advisory: the New run form offers its plans, a plan fills the settings it names, and the check judges whatever settings are asked for |
| Prices | A GPU catalog file in the repository for the hardware facts; RunPod's prices and stock read from its API by a periodic task into dated records beside the ledger; Tinker's from its `models.json`; each plan says how old its prices are |

## What the platform does now

The parts the three questions touch, as main has them:

| Part | Now | Code |
|---|---|---|
| LoRA trainer | On one GPU, one process per step (`device_map={"": "cuda"}`); the model is loaded from its files at the start of every step. On several GPUs of one machine, a process per GPU kept between steps, the frozen model sharded or whole on each, the adapter sharded (FSDP2). LoRA wraps every linear layer whose name ends in one of `TARGETS` (Qwen3.5's projections, linear attention included) inside `language_model` (an image-text model) or `layers`; the vision tower is dropped; the embedding table stays in its file | `rollout_lora.policy`, `.layers`, `.worker` |
| Full-weight trainer | On one GPU, the same process per step; weights, gradients and Adam's two moments in FP32 on the one device (16 bytes a weight: Qwen3-0.6B takes about 10 GiB); image-text models refused; every step saves the whole model in FP32. On several GPUs, all of it sharded over them and kept between steps; each step saves a BF16 serving copy, and the full state every `trainer.state_every` steps (PyTorch's distributed checkpoint) | `rollout_lora.full`, `.sharded`, `.resident`, `.workers` |
| Engines | `VllmEngine`: one vLLM `AsyncLLM` per engine host, started with a fixed set of arguments (BF16, LoRA on, `max_loras`, sleep mode, `language_model_only`, optional `quantization`): no tensor, pipeline, data or expert parallel. Full weights are loaded from a path with `collective_rpc("reload_weights")`; sleep drops the weights (level 2) and waking reads them again | `rollout_vllm.engine` |
| A run's gang | One `PACK` placement group: a bundle per engine replica, one for the trainer on the driver's node (a step's files are handed to it by path), one for the largest bridge. A bundle must fit on one node; more than one GPU is rounded up to whole GPUs. On Kubernetes the RayJob's head pod holds the driver, trainer and bridge, and each engine replica's bundle can be a worker pod | `rollout_train.demand`, `.submitting` |
| Weights from trainer to engines | Files: the trainer writes a checkpoint into the blob store, the loop appends a serving record naming the files (or what a bridge made of them), and each follower fetches and loads them. A colocated trainer shares one GPU with the engines, which sleep while it steps | `rollout_train.serving`, `.following`, `.colocated` |
| Renderers | `qwen35`, `qwen3` (`rollout_qwen`) and `gemma4` (`rollout_gemma`), found through the `rollout.renderers` entry point and chosen by the `renders` pattern of each; a renderer is a chat template plus a `ToolCallFormat` and a `ThinkingFormat`. The gateway renders and parses itself: engines see tokens only | `rollout_train.recorder.renderers` |
| Bridges | `verbatim`, `full-reload`, `peft-from-tinker` (Tinker's names remapped to PEFT's, Qwen3.5's q, k and v joined into `in_proj_qkv`, rank factor 3), `merge-quantize` | `rollout_train.bridges`, `rollout_tinker.weights`, `rollout_lora.bridges` |
| Providers | `vllm`, `vllm-servers`, `tinker`, `api`, `runpod-inference`, and the trainers `lora`, `full`, `tinker`, `runpod-trainer`. A RunPod provider names GPU types and `gpu_count` (a `runpod-trainer`'s pod steps on all of them; a host's has one). In progress elsewhere: leases with scale to zero, `runpod-host`, `max_pods` | `rollout_train.providers`, `rollout_runpod` |
| What a model is | A model offer per provider: `context`, `base` (what a quantized model was made from), `max_lora_rank`, `cost`, engine `options`. Nothing says a model's parameter count, layers, attention layout or memory | `ModelOffer` |
| Capacity and spend | The capacity rule counts GPUs (not their type or memory) against `[capacity]`; the `memory` rule estimates what each of a trainer's GPUs holds from the model's files and refuses one that cannot fit; `spend_of` counts metered parts only. In progress elsewhere: a pod's hourly price times how long a step of its trainer and model took lately | `rollout_train.validation` |

## 1. More models

### What a model touches

| Part | What it must know about the model | Where that lives now | Breaks when |
|---|---|---|---|
| Renderer | The chat template, tool-call and thinking formats, stop tokens | One function per family (`renders`) | A family whose format a chat template plus a tool-call parser cannot express (gpt-oss's channels) |
| Segments | Whether a conversation re-rendered after a turn begins with the tokens sampled | Decided at run time by `segments_of`: an edited context starts a new segment | Never wrong, but a template that drops earlier thinking makes every turn its own segment, so the prompt is trained once per turn: the planner has to know |
| LoRA trainer | Which modules to adapt, where the decoder and the embeddings are, how experts are laid out | `TARGETS`, `EMBEDDING`, `within` in `rollout_lora.policy` | Any family whose names differ from Qwen3.5's; any MoE whose experts are one fused 3-D parameter rather than linear layers (they are skipped silently) |
| Full-weight trainer | Memory: 16 bytes a weight on one device | — | Anything above about 4B parameters on an 80 GB GPU |
| Objectives | The vocabulary (each row of logits is a vocabulary wide: `LOGIT_ROWS` at a time) | — | Fine up to the 256K vocabularies of current models; noted for memory |
| Engine | Whether vLLM runs the architecture, adapts the module types LoRA trained, serves LoRA on MoE, takes the quantization | `VllmEngine`'s fixed arguments | A model that needs more than one GPU; LoRA on experts; options the engine does not pass |
| Bridges | Name maps between trainer, PEFT, vLLM and Tinker; fused projections; rank factors | `rollout_tinker.weights` (Qwen3.5 only), `rank_factors` per bridge | Any other family on Tinker served locally |
| Tinker | Its catalog and its own LoRA targets | The cluster config's `[inference.tinker.models]` | Models it retires (Qwen3-32B, Qwen3-30B-A3B and the Llama models went on 2026-06-12) |
| Catalog | Context, base, rank, cost | Model offers | Nothing says size or memory |
| Evals | The renderer and budgets of the channel played | Suites are model-agnostic | A family's thinking length differs (budgets per family) |
| Planner | Parameters, active parameters, layers, attention layout, KV per token | — | Does not exist |

### Shapes

**Dense** models (Qwen3-8B and -32B, Llama 3.3 70B, Qwen3.6-27B, Qwen3.8-27B) need nothing new beyond their names
and their sizes: LoRA on q, k, v, o, gate, up and down, a renderer, and above about 30B parameters more than one GPU
(section 2).

**Hybrid attention** (Qwen3.5, Qwen3-Next) interleaves linear-attention layers (Gated DeltaNet) with full gated
attention, three linear layers to one full (`full_attention_interval = 4`). Only the full-attention layers keep a KV
cache that grows with the sequence; each linear-attention layer keeps a fixed state per sequence (value heads × 128 ×
128: about 25 MB a sequence for Qwen3.5-9B, 94 MB for the 397B, at BF16). So a long context costs much less memory
than in a dense model of the same size (32 KiB a token for Qwen3.5-9B against 144 KiB for Qwen3-8B), and the memory
model counts layer by layer. The linear layers' projections have names of their own (`in_proj_qkv`, `in_proj_z`,
`out_proj`), which the trainer already adapts and Tinker splits (the rank factor 3). How well the engine's prefix cache
works for hybrid layers decides how much of a multi-turn prompt is computed again each turn; the planner takes the
cache hit rate it measures rather than assuming one. Qwen3.5 also brings a tokenizer of its own (248,320 tokens), not
Qwen3's: families are not stable across versions.

**Mixture of experts** (Qwen3.5-35B-A3B and -397B-A17B, Qwen3-235B-A22B, gpt-oss, DeepSeek-V3, Kimi K2, GLM): the
MLP of each layer is a router and many experts, of which a few run per token. Three things change:

- **Memory and compute come apart.** Memory follows the total parameters, compute the active ones. A 35B-A3B model
  needs the memory of a 35B model and computes like a 3B one; decoding at the batch sizes of RL touches nearly every
  expert each step, so it reads nearly all the weights.
- **LoRA on experts is a different kernel.** Recent transformers versions hold a layer's experts as fused 3-D
  parameters, not linear layers, so `add_lora` skips them; adapting them takes a grouped matrix multiply in the
  trainer (PEFT's `target_parameters`, Unsloth's grouped kernels, Megatron-Bridge's expert LoRA) and MoE LoRA in the
  engine (vLLM's fused MoE LoRA kernel, multi-adapter MoE serving since 0.15; the platform pins 0.30). An adapter on
  every expert at the full rank is large: rank 32 on all 128 experts of each of Qwen3-235B's 94 layers is about 6.5B
  parameters, 26 GB in FP32. Tinker gives each expert a rank of the total divided by the experts active per token,
  which keeps the adapter near the size of a dense model's. The choices are in [trainers](#trainers).
- **Routing differs between trainer and engine.** The same tokens can be routed to different experts in the
  trainer's forward pass and the engine's (different kernels, precision, batch), which widens the gap between
  behaviour and trainer logprobs. The importance weight corrects it as it corrects staleness; GSPO's sequence-level
  ratio (`gspo`) was proposed partly for this, as an alternative to replaying the engine's routing in the trainer,
  and Miles (a fork of slime) samples and trains MoE in FP8 end to end for the same reason. `kl_floor` and
  `mean_mismatch` measure the gap per step.

**Vision-language** models (Qwen3.5 is image-text throughout; Qwen3.8-27B, Gemma) are served and trained as text
models today: `language_model_only` in the engine, the vision tower dropped in the trainer. Images in turns would need
the renderer to place image tokens, the gateway to send the image beside the token prompt, the engine to take
`multi_modal_data`, and the trainer to run the vision tower. Nothing needs this until an environment sends images.

**Very large** models (70B dense, 235B to 400B MoE, and DeepSeek-V3.1's 671B, Kimi K2's 1T and Inkling's 975B) are
mostly a question of section 2 and section 3: they never fit one GPU, and the largest ship in FP8 (DeepSeek-V3,
Kimi K2), which a trainer has to dequantize or train over.

### Tokenizers, chat and tool-call formats

The gateway renders prompts and parses replies itself; engines see token ids and return token ids. So vLLM's chat
templates and tool parsers do not matter here, and every family needs a renderer that is exact.

| Family | Tool calls | Thinking | Vocabulary | Renderer here |
|---|---|---|---|---|
| Qwen3 | Hermes JSON in `<tool_call>`, results in `<tool_response>` | `<think>`, the model opens it; the template drops earlier thinking | 151,936 | `qwen3` |
| Qwen3.5, 3.6 | XML: `<tool_call><function=NAME><parameter=P>…` | On by default, the prompt opens it | 248,320 | `qwen35` |
| Gemma 4 | Special tokens of its own for calls, results and a string delimiter | In a `thought` channel, dropped between turns, kept through tool calls | 262,144 | `gemma4` |
| gpt-oss | Harmony: channels `analysis`, `commentary` (tools, addressed `to=functions.NAME`) and `final`; tools declared as a TypeScript-like namespace | The `analysis` channel; dropped after a final answer, kept across tool calls | 201,088 | New: a harmony renderer (OpenAI's `openai-harmony` library renders and parses), not a chat template plus a parser |
| DeepSeek V3.1, V3.2 | Special tokens (`<｜tool▁calls▁begin｜>` … `<｜tool▁sep｜>` JSON) | `<think>`, opened by the prompt | 129,280 | New: a token-level tool-call format |
| Kimi K2 | Sections of special tokens, call ids `functions.NAME:IDX` | `<think>` (thinking variants) | 163,840 | New; the call id must be rendered back exactly |
| GLM-4.5, 4.6 (GLM-5) | `<tool_call>NAME` with `<arg_key>` and `<arg_value>` pairs | `<think>` | 151,552 (154,880) | New: an XML-like format |
| Llama 3.1, 3.3 | JSON in the assistant turn; built-in tools behind a python tag token; results in the `ipython` role | None | 128,256 | New: JSON calls, which `JsonToolCalls` nearly fits |
| MiniMax M2.5 | `<minimax:tool_call><invoke name=…>` | Interleaved thinking, kept in history | 200,064 | New |

Formats that keep or drop reasoning between turns in their own ways (harmony, Gemma 4, MiniMax) decide whether a
conversation trains as one segment or one per turn, and so how many tokens a step trains.

What makes a renderer exact enough to train on, checked by a **conformance kit** each family's package runs
(`rollout_train.testing.renderer_kit`, proposed):

1. Prompts equal the family's chat template's output, token for token, for a fixed set of conversations: plain turns,
   thinking, tool calls with nested arguments, tool results, several tool calls in one turn, a system prompt, an empty
   reply.
2. Parsing a rendered reply gives back the canonical message (text, reasoning, tool calls with arguments of their
   schema's types).
3. Rendering after a turn either extends the sampled tokens or not, and the kit says which: that decides one segment
   per conversation or one per turn.
4. The stop tokens end every reply the template writes, and the thinking end token is one token where the renderer
   says so.
5. The tokenizer's revision is pinned in the run's start (Tinker pins revisions too: its SDK 0.21.0 did so for
   Kimi-K2.6).

### Trainers

**Targets from the architecture.** `TARGETS` becomes a declaration per architecture:

```python
@dataclass(frozen=True)
class Architecture:
    """What the platform knows about a family of checkpoints beyond `config.json`: declared by the family's package
    (`rollout.architectures` entry point), matched by `config.json`'s `architectures` or `model_type`."""

    name: str                                   # "qwen3_5", "qwen3_moe", "llama", "gpt_oss", "deepseek_v3"
    matches: tuple[str, ...]                    # config `architectures` it covers
    decoder: str                                # "model.language_model.layers" or "model.layers"
    embedding: str                              # the token embedding's tensor name
    groups: Mapping[str, tuple[str, ...]]       # LoRA target groups: {"attention": (...), "mlp": (...),
                                                #  "linear_attention": (...), "experts": (...), "shared_experts": (...)}
    experts: Literal["none", "modules", "fused"] = "none"   # how a layer's experts are held
    fused: Mapping[str, Mapping[str, tuple[str, ...]]] = {} # per consumer ("vllm", "tinker"): joined name -> parts
    renames: Mapping[str, str] = {}             # PEFT name -> consumer name, per consumer, where they differ
    vision: str | None = None                   # the vision tower's module path, if any
```

`rank_factor` of a bridge comes from `fused` (three parts joined is three times the rank), so the Qwen3.5 special case
in `rollout_tinker.weights` becomes data. A model with no architecture entry for the chosen trainer is refused at
validation (a new rule, `architecture`), not when a step fails.

**MoE LoRA.** Four choices, each declared by the trainer and by the provider's model offer as the groups it adapts or
loads:

| Choice | Adapter size (Qwen3-235B-A22B, rank 32) | Trainer | Engine | Learning |
|---|---|---|---|---|
| Attention (and shared experts) only | about 0.2B parameters | Works today: plain linear layers | Any LoRA-capable engine | Weaker: Thinking Machines' "LoRA Without Regret" found attention-only LoRA well behind LoRA on all layers, MLP included |
| Every expert its own adapter, rank `r` | about 6.5B parameters | A grouped-GEMM LoRA over fused experts (PEFT's `target_parameters`, Unsloth's kernels, Megatron-Bridge with `share_expert_adapters = false`) | vLLM's MoE LoRA | The adapter is no longer small: transfer, `max_loras` and memory |
| Every expert its own adapter, rank `r / k` (k experts active a token) | about 0.8B parameters | As above | vLLM's MoE LoRA | Tinker's choice, after "LoRA Without Regret"; PEFT recommends a rank divided by the experts too |
| One adapter shared by the experts | as attention-only plus one MLP's | Easy | Not supported by vLLM | Not recommended: no engine serves it |

Recommendation: adapt every expert at a rank of `r / k`, as Tinker does, where the provider's engine supports MoE
LoRA for the architecture; else attention and shared experts. Validation compares the trainer's groups with what the
provider loads (a rule `lora_targets`). vLLM has two layouts for expert adapters (2-D, one key per expert as Megatron
writes them; 3-D, stacked as PEFT writes them), and declaring the wrong one "silently produce[s] garbage outputs", so
the architecture says which the trainer writes, and the conformance kit loads a trained adapter into vLLM and compares
its logits with the trainer's. vLLM's MoE LoRA also needs its Triton experts kernel: on a model where it picks
FlashInfer's CUTLASS experts instead, adapters fail to load (seen on Qwen3.5-122B-A10B with vLLM 0.17), so the model
offer pins the kernel where needed.

**Full weights.** Sixteen bytes a weight is the floor for Adam in mixed precision; the only ways past one GPU are
sharding (section 2) or a smaller optimizer state (8-bit Adam: about 10 bytes a weight), which changes the numerics of
a recipe and is left out. Two habits of today's full-weight trainer do not survive large models:

- **A fresh process per step that loads the parent's files.** At 70B that is 280 GB of FP32 weights and 560 GB of
  optimizer state read and written every step.
- **The whole model saved in FP32 every step.** It goes to the blob store, so every follower fetches four bytes a
  weight to serve two.

So a full-weight trainer above a few billion parameters should be **resident**: it keeps its FP32 weights and
optimizer state in GPU memory between steps, writes the serving copy (BF16) every step, and writes its full state to
durable storage every few steps. A crash then loses at most those few steps: each checkpoint says whether it holds
the trainer's full state or only its serving copy, and a resumed run goes on from the newest that holds the state
(decision 1 below).

### Engines

vLLM runs every family named in this note. LoRA is narrower: dense models take it throughout; vLLM serves several
adapters on MoE models (gpt-oss, Qwen3 MoE, DeepSeek, Llama MoE) since 0.15, and 0.22 brought one Triton kernel for
MoE LoRA that serves the 2-D and 3-D layouts together. Exceptions are found per model, not per family: Tinker refuses
to export DeepSeek-V3.1 adapters for vLLM, and Qwen3.5-122B-A10B's adapters failed to load where vLLM chose
FlashInfer's experts kernel. LoRA over AWQ, GPTQ and bitsandbytes bases is long supported; over FP8 and NVFP4 MoE
bases it is not documented, so a model offer claims it only once the conformance kit has passed on it. What changes in
`VllmEngine`:

- **Options pass through.** The engine takes the arguments of an allow-list from the model offer's `options`
  (`tensor_parallel_size`, `pipeline_parallel_size`, `data_parallel_size`, `enable_expert_parallel`,
  `kv_cache_dtype`, `quantization`, `max_cpu_loras`, `enforce_eager`, ...) instead of a fixed signature, so a new
  knob needs no code.
- **More than one GPU per engine.** With tensor parallel inside one node the engine host asks for that many GPUs and
  vLLM starts a worker per GPU in its own processes (its multiprocessing backend), within the actor's GPUs.
- **Quantized serving.** A model offer's `base` already says an adapter trained over the base can be served on a
  quantized copy. FP8 (weights and activations) is the natural default for 70B and up on Hopper and Blackwell; AWQ
  and GPTQ 4-bit run anywhere; NVFP4 on Blackwell. Full weights can be quantized as they load each step (vLLM's
  `quantization = "fp8"`), which `merge-quantize` already relies on. The precision of each provider's model is part
  of its offer, so a turn records what it was sampled at.

**SGLang** is the other engine RL frameworks use: slime and Miles use only it; verl, AReaL, ROLL, SkyRL and NeMo-RL
offer it beside vLLM. It has the same three ways to take new weights (`update_weights_from_disk`, `_from_tensor` for a
colocated trainer, `_from_distributed` over NCCL), runtime adapter loading, MoE LoRA (`FusedMoEWithLoRA`, extended to
quantized MoE bases, gpt-oss, Qwen3-30B-A3B and DeepSeek-V3's MLA in April 2026), and strong expert-parallel serving
(DeepEP, Mooncake). It would be one more `Engine` implementation (`rollout_sglang`) with the same capabilities to
declare. Not now: vLLM serves every model here and has the native weight-transfer API below; the platform's following,
sleeping and LoRA loading are built on it. Revisit when a model a run needs serves markedly better on SGLang.

### Bridges and checkpoints

What one checkpoint weighs, and what moving it costs:

| Model | BF16 weights | FP8 | Full trainer state (FP32 weights and Adam, 12 bytes a weight) | LoRA rank 32, FP32 | BF16 at 1 GB/s |
|---|---|---|---|---|---|
| Qwen3.5-9B | 18 GB | 9 GB | 108 GB | about 0.3 GB | 18 s |
| Qwen3.5-35B-A3B | 70 GB | 35 GB | 420 GB | about 1.3 GB (experts at rank `r/k`) | 70 s |
| Llama 3.3 70B | 141 GB | 73 GB | 850 GB | about 1.7 GB | 2.4 min |
| Qwen3-235B-A22B | 470 GB | 239 GB | 2.8 TB | about 3.3 GB (experts at `r/k`) | 8 min |
| Qwen3.5-397B-A17B | 807 GB | about 400 GB | 4.8 TB | — | 13 min |
| DeepSeek-V3 (671B) | about 1.37 TB | 689 GB (native) | 8 TB | — | 23 min |
| Kimi K2 (1T) | about 2.05 TB | 1.03 TB (native) | 12.4 TB | about 10 GB (experts at `r/k`) | 34 min |

The serving copy goes to every follower each step; the trainer's state only to durable storage, and only every few
steps once the trainer is resident. An adapter is small at every size; full weights stop being cheap to publish
through the blob store somewhere around 30B.

- **Name maps from the architecture.** The trainer writes PEFT names; each consumer's names come from the
  architecture's `renames` and `fused`. One generic bridge (`peft-for`) replaces per-family code.
- **Adapters on experts.** PEFT writes expert adapters on the fused 3-D parameters (`mlp.experts.gate_up_proj`,
  `mlp.experts.down_proj`), Megatron-Bridge one per expert or one shared per expert-parallel rank; vLLM loads both
  layouts when told which. The architecture declares the layout and the bridge converts where trainer and engine
  differ. Tinker's export refuses adapters for families vLLM cannot serve with LoRA (DeepSeek-V3.1 today), which have
  to be merged into full weights: the `tinker → peft` bridge refuses them the same way, and `merge-quantize` is the
  path.
- **Sharded checkpoints.** A sharded trainer writes its state in shards (PyTorch's distributed checkpoint, one file per
  rank). Its serving copy is gathered into ordinary safetensors (the `full` format) as it is written, so nothing
  downstream sees shards. A Megatron trainer writes Megatron's layout; its serving copy is converted to Hugging Face
  names by Megatron-Bridge, inside the trainer or as a bridge (`hf-from-megatron`).
- **Pre-quantized serving copies.** A `quantize-fp8` bridge could write block-FP8 files so engines load them without
  quantizing; only worth it when quantizing at load becomes a measurable part of a step.

### The base model catalog

A **model card** is what the platform derives from a checkpoint's `config.json` (and its safetensors index, for the
exact parameter count):

```python
@dataclass(frozen=True)
class ModelCard:
    model: str
    revision: str
    architecture: str               # matched Architecture.name
    parameters: float               # total, from the safetensors index
    active: float                   # per token (MoE: shared parts + routed experts' share)
    layers: int
    full_attention_layers: int      # layers with a growing KV cache
    kv_bytes_per_token: int         # at BF16, summed over those layers (MLA: its latent width)
    state_bytes_per_sequence: int   # linear-attention and sliding-window state, fixed per sequence
    hidden: int
    vocabulary: int
    experts: int | None
    active_experts: int | None
    modality: Literal["text", "image-text"]
    native_dtype: str               # "bfloat16", "fp8" (block), "mxfp4"
    tied_embeddings: bool
```

`rollout_train.models.card(model, revision)` computes it (pure, from files fetched once and kept in the blob store by
model and revision); the cluster config's `[models."NAME"]` table may override any field and holds what is the
model's rather than a provider's: the renderer runs default to, the revision runs pin. Which provider offers which
model stays in the providers' tables. Validation (memory, LoRA groups), the planner, the New run form (a model's size
beside its name) and bridges read the card.

### Evals

Suites are lists of environments and do not change. A new family needs its renderer to pass the kit, budgets of its
own (thinking lengths differ several times between families), and a base-model score per suite before any training,
so a checkpoint's eval history starts from the base. Very large models that are not trained here (judges, opponents,
eval subjects) are reached through hosted APIs or Tinker's sampler, as now.

### How this maps onto the roles

| Role or piece | Change |
|---|---|
| Model family packages (`implementations/`) | Declare a renderer and an architecture through entry points; run the conformance kit in their tests. `rollout_qwen` declares `qwen3_5`, `qwen3`, `qwen3_moe`; new packages for other families (`rollout_gpt_oss`, `rollout_llama`, `rollout_deepseek`) as runs need them |
| `rollout-train` | `ModelCard` and `card()`; the `Architecture` declaration and its lookup; validation rules `architecture`, `lora_targets`, `memory`. It never names a family |
| Trainers | Read targets, decoder and embedding paths from the architecture |
| Bridges | One `peft-for` bridge driven by the architecture's names; `rank_factor` from `fused` |
| Providers | A model offer says its precision and the LoRA groups its engine loads |
| Engines | Options passed through from an allow-list |

## 2. More topologies

### Where it breaks now

- An engine host asked for two GPUs gets two, and vLLM uses one: `VllmEngine` takes no tensor-parallel size.
- A trainer spans the GPUs of one machine at most (FSDP2): nothing shards it across machines, or by tensor, pipeline or
  expert.
- A bundle of a placement group must fit on one node, so nothing can span nodes; the trainer's bundle is pinned to the
  driver's node because a step's files are handed over by path.
- A RunPod engine's pod serves on one GPU, whatever its `gpu_count`.
- Weights move only as files, and full weights in FP32 every step.

### What each parallelism is for

Inference:

| Parallelism | Splits | Needs | Use it when | vLLM |
|---|---|---|---|---|
| Tensor (TP) | Every layer's matrices across GPUs; an all-reduce per layer | NVLink (inside a node); PCIe works for 2 GPUs at a cost | Weights and KV do not fit one GPU, or each token must come faster | `tensor_parallel_size` |
| Pipeline (PP) | Layers into stages, one per GPU or node | Modest bandwidth between stages | A model does not fit one node; across nodes | `pipeline_parallel_size` (multi-node through Ray) |
| Data (DP) | Requests across copies | Nothing | More throughput than one copy gives | Our replicas; vLLM's own `data_parallel_size` mainly pairs with expert parallel |
| Expert (EP) | A MoE layer's experts across GPUs; tokens sent to their experts and back | Fast all-to-all | Large MoE at high batch; attention data-parallel beside it | `enable_expert_parallel` |
| Context (CP) | One long sequence across GPUs | Fast links | Contexts beyond about 128K tokens | decode context parallel (`-dcp`, for MLA and grouped-query attention); prefill context parallel still in development |

Training:

| Parallelism | Splits | Use it when | In the field |
|---|---|---|---|
| FSDP2 / ZeRO-3 | Weights, gradients and optimizer state across GPUs; gathers each layer for its compute | Dense models and small MoE, up to a few nodes | PyTorch's `fully_shard`; verl, OpenRLHF (DeepSpeed), prime-rl, TRL |
| Tensor (Megatron TP) | Each layer's matrices | Large dense models, inside a node | Megatron-Core; verl, NeMo-RL, slime |
| Pipeline (PP) | Layers into stages | Models beyond a node | Megatron-Core |
| Expert (EP) | Experts across GPUs | Large MoE trained efficiently | Megatron-Core; FSDP2 with expert parallel in prime-rl and AReaL's Archon |
| Context / sequence (CP, SP) | Sequence across GPUs | Long sequences | Megatron-Core, Ulysses (DeepSpeed) |

How the RL frameworks place their trainers and engines and move weights between them:

| Framework | Trainer | Engines | Placement | Weights move by | LoRA | Largest reported |
|---|---|---|---|---|---|---|
| verl | FSDP, FSDP2, Megatron (TP, PP, VPP, CP, EP) | vLLM, SGLang | Colocated (its 3D-HybridEngine reshards between the training and the serving layout), one step off-policy, or fully async on separate node pools | CUDA IPC colocated; NCCL through the checkpoint-engine library, in buckets; a delta backend that sends only changed parameters | FSDP (PEFT) and Megatron (Megatron-Bridge); adapter-only or merged sync | DeepSeek-V3 671B on 96 to 512 GPUs (TP 8, PP 16, EP 8 on 256), Qwen3-235B; LoRA on a 1T MoE (Mind Lab) |
| OpenRLHF | DeepSpeed ZeRO-3 | vLLM (TP, PP) | Separate Ray roles, or all colocated with sleep; async optional | NCCL from rank 0 to every vLLM rank; CUDA IPC colocated | Yes | "70B+" |
| NeMo-RL | DTensor (FSDP2, TP, SP, CP, PP) or Megatron-Core through Megatron-Bridge | vLLM, SGLang, Megatron | Colocated or not; async GRPO | Colocated: CUDA IPC over ZMQ. Otherwise: NCCL broadcast, resharding NCCL ("best for >100s B"), sparse deltas through ZMQ or S3, NIXL (RDMA) | Both backends | DeepSeek-V3 on 256 and 512 H100, Qwen3-235B on 128 and 256 |
| SkyRL | FSDP2, Megatron | vLLM | Colocated or disaggregated | NCCL; CUDA IPC colocated; vLLM's native API | Megatron-Bridge; a Tinker-compatible server with many LoRAs | Qwen3-235B-A22B |
| slime | Megatron | SGLang | Colocated natively, or external engines; fully async | SGLang's tensor update colocated, its distributed update (NCCL) otherwise; deltas over disk or NCCL | Not verified | GLM-4.5 to 5.3, DeepSeek-V3, Qwen3 MoE |
| prime-rl | FSDP2 with TP, CP, EP (no PP) | vLLM | Disaggregated, fully async | NCCL by default, the filesystem when LoRA is on, NIXL; adapters loaded with `/load_lora_adapter` without pausing | Many LoRAs | GLM-5.1 in FP8 across 16 nodes of 8 H200 |
| AReaL | FSDP2, Megatron, Archon (FSDP2 with PP, EP) | SGLang, vLLM | Disaggregated, fully async | Chunked NCCL, or safetensors files | Adapter-only sync | Qwen3-235B agents |
| ROLL | Megatron-Core, DeepSpeed, FSDP2 | vLLM, SGLang | Colocated or disaggregated | NCCL in an update group | Megatron and DeepSpeed | "200B+ MoE", Qwen3-235B |
| TRL | Accelerate (FSDP, DeepSpeed) | vLLM | Colocated by default, or a separate `vllm serve` | NCCL (IPC on a shared GPU) through vLLM's weight-transfer endpoints | Yes | — |
| torchforge | TorchTitan (FSDP2, TP) | vLLM | Monarch actors | TorchStore (RDMA) | No | Development paused: PyTorch consolidates on torchtitan |
| Tinker | Hosted | Hosted | Hosted | Hosted; sampler checkpoints by path | LoRA only (rank 32 by default; experts each `r/k`) | Kimi-K2.6 (1T), Inkling (975B), GLM-5.3, DeepSeek-V3.1 |

Two lessons for the platform. First, nobody trains large MoE efficiently without expert parallel, and the frameworks
that do it at scale use Megatron-Core, now reached through Megatron-Bridge, which also converts names to and from
Hugging Face's. Second, every framework keeps a slow durable path (files) and adds a fast one (IPC or NCCL) inside a
placement; the fast paths have converged on what vLLM now offers natively.

### Which parallelism each size needs

What each size needs, on 80 GB (H100), 141 GB (H200) and 180 GB (B200) GPUs, for a gridworld-like workload (96
sequences of about 2,000 tokens in flight per engine replica). Costs and step times for chosen shapes are in
[the sizing table](#sizing-table).

| Model | Engine replica | LoRA trainer | Full-weight trainer |
|---|---|---|---|
| Qwen3.5-9B | 1 GPU of 24 GB or more (BF16; a 4-bit copy on 16 GB, as now) | 1 GPU (base 18 GB); FSDP2 over 2 to 4 for speed | 3 to 4 H100 or 2 H200, FSDP2 (144 GB of state) |
| Qwen3.5-35B-A3B | 1 H100 at FP8, 1 H200 at BF16; TP 2 halves the time a token takes | 1 H100 (base 70 GB, tight) or 2, FSDP2, expert LoRA | 6 H200 (560 GB of state), FSDP2 with expert parallel |
| Llama 3.3 70B | TP 2 on H100 at FP8; TP 4 at BF16 | 2 H100 at the edge, 4 to 8 for speed, FSDP2 | 8 B200 on one node (1.1 TB of state), or 16 H200 across 2 nodes |
| Qwen3-235B-A22B | TP 4 and EP on H100 or H200 at FP8 (239 GB) | 8 H100 at the edge, or 8 H200; FSDP2 with expert parallel, or Megatron | 32 H200 across 4 nodes (3.8 TB of state), Megatron TP, PP, EP |
| Qwen3.5-397B-A17B | 8 H100 or 4 H200 at FP8 | 8 H200 at the edge, or 8 B200 | 48 or more H200, Megatron, several nodes |
| DeepSeek-V3.1, Kimi K2 (671B to 1T) | 8 H200 or 8 B200 at FP8 | 16 B200 across 2 nodes (2 TB of base at BF16), Megatron EP; or Tinker | 128 or more H200: not planned |

### Gangs: one node, several nodes

**One node, several GPUs** needs almost nothing new from placement: a part's bundle asks for that many whole GPUs,
and `PACK` puts it on a node that has them. What changes is inside the parts:

- the engine host passes `tensor_parallel_size` (and `enable_expert_parallel` for MoE) to vLLM, which starts a
  worker process per GPU inside the actor's GPUs;
- the trainer actor starts its step as one process per GPU (PyTorch's `torch.multiprocessing` with a NCCL process
  group on the node), each holding its shard; rank 0 answers for the step;
- on Kubernetes the part's pod asks for that many GPUs (the RayJob already sizes pods from the demand);
- on RunPod a provider says `gpu_count` (1, 2, 4 or 8, as RunPod offers each type) and the pod is rented with it.

**Several nodes** is new:

- **A placement group per spanning part.** Ray creates a placement group atomically, but a run's single `PACK` group
  cannot say "these four bundles on four different nodes of one fabric, and these others anywhere". So a part that
  spans nodes (a trainer of 16 GPUs, an engine replica of two nodes) reserves a group of its own, `STRICT_SPREAD`,
  one bundle per node of that node's GPUs, beside the run's group for the rest. Several groups are not reserved
  atomically together, but on Kubernetes they need not be: the run's Ray cluster is its own, and Kueue admitted all
  its pods at once, so nothing else competes for them. On one machine nothing spans nodes.
- **Kubernetes.** A worker group per spanning part, as many pods as nodes, each pod the node's GPUs. To keep a group's
  pods on one NVLink domain or InfiniBand block, Kueue's topology-aware scheduling takes an annotation on the group's
  pod template (`kueue.x-k8s.io/podset-required-topology: <level>`), against a `Topology` the cluster defines from
  node labels. Kueue admits a RayJob as one Workload with a PodSet per group (at most 18 PodSets: 17 worker groups).
  LeaderWorkerSet is the way to serve multi-node vLLM without Ray; the platform already has Ray inside each run, so it
  is not needed.
- **The trainer leaves the driver's node.** A spanning trainer's files reach it through the blob store or a volume
  every node mounts, not by path on the driver's node.
- **RunPod.** Multi-GPU pods cover one node. RunPod's Instant Clusters are 2 to 8 nodes of 8 GPUs, joined by
  InfiniBand or RoCE at 3,200 Gb/s for B200, H200 and H100 (1,600 for A100), created through one API call
  (`POST /v2/clusters`, `podCount`, `gpuCountPerPod`, a type `TRAINING` or `RAY`). On demand they go to 16 GPUs; more
  through sales. A spanning trainer on RunPod is one `runpod-trainer` whose lease is an Instant Cluster: the training
  service inside runs its ranks across the cluster's nodes (`NODE_RANK`, `MASTER_ADDR` and `NCCL_SOCKET_IFNAME=ens1`
  are set by RunPod), and the platform still sees one `RemoteTrainer` endpoint. Engines spanning nodes on RunPod are
  deferred: 8 H200 or B200 in one pod hold any model in this note at FP8.

### Colocated and disaggregated

| | Colocated | Disaggregated |
|---|---|---|
| What | Trainer and engines share the same GPUs and take turns (the engines sleep while the trainer steps) | Separate GPUs for each |
| Weights move | On the same GPU: CUDA IPC handles, or files from the local disk | Between GPUs or machines: NCCL, or files |
| Overlap | None: generation and training alternate | Generation of the next step overlaps training (`max_lag` of 1 or more) |
| Costs | Sleep, wake and weight reload each step; the larger of trainer and engine memory needs | Idle time on whichever side is faster; two pools to size |
| Good for | Few GPUs; generation and training of similar length; on-policy runs | More than a node; the platform's asynchronous loop; a trainer much larger than an engine replica |
| In the field | verl's HybridEngine, OpenRLHF's hybrid engine, NeMo-RL's colocated mode, TRL's colocate | prime-rl, AReaL, slime, OpenRLHF's default, NeMo-RL non-colocated |

Both stay. The planner chooses between them by cost and step time (section 3); beyond one node the default is
disaggregated, since the loop is asynchronous already.

### How weights move

| Way | Works across | Speed | Survives a crash | Who uses it |
|---|---|---|---|---|
| Files through the blob store (now) | Anything: pods on RunPod, other clusters, restarts | The store's and the network's throughput: on the order of 1 GB/s a node (RunPod's network volumes 0.2 to 0.4 GB/s, up to 10 at peak) | Yes | The platform; prime-rl (with LoRA); AReaL; NeMo-RL's S3 deltas |
| Files on a node's disk or a shared volume | One node, or one volume | Disk speed: several GB/s on NVMe | Yes, on that disk | The colocated engine's reload today |
| CUDA IPC | Processes on the same GPU | Near zero copy | No | verl, OpenRLHF hybrid engine, vLLM's colocated RLHF example |
| NCCL broadcast | GPUs in one process group | NVLink about 450 GB/s, InfiniBand about 50 GB/s a GPU | No | OpenRLHF, NeMo-RL, verl (disaggregated), vLLM's RLHF example |
| RDMA, pulled point to point | GPUs on one fabric | Near line rate | No | NeMo-RL (NIXL), vLLM's `sharded_rdt` backend, torchforge's TorchStore |
| Deltas | Any of the above | Only what changed: verl reports more than 99% of BF16 bytes unchanged step over step | As the way it rides on | verl, NeMo-RL, slime |

**Resharding** is the same for every way: the trainer gathers each parameter whole, a bucket at a time, in Hugging
Face names, and vLLM's `load_weights` splits each tensor for its own tensor-parallel rank. So a trainer sharded one way
(FSDP over 8 GPUs) feeds engines sharded another (TP 2) with no plan of shards on either side. A Megatron trainer needs
its names converted first (Megatron-Bridge does this as it exports).

**vLLM's native weight transfer.** The pinned vLLM (0.30) has an API for exactly this:
`init_weight_transfer_engine`, then `start_weight_update`, `update_weights` (tensors streamed in packed buffers, 1 GiB
twice over by default) and `finish_weight_update`, with `pause` and `resume` around it (abort, wait for, or keep the
requests in flight), in process or as HTTP endpoints (`--weight-transfer-config`). Its backends are `nccl`, `ipc`
(colocated), `sparse_nccl`, `sharded_rdt` (pulled over NIXL) and `nccl_m2n` (a trainer and engines sharded
differently). TRL's server mode and prime-rl use it. Adapters have their own path: `/v1/load_lora_adapter`, with
`load_inplace` to replace one under the same name for asynchronous RL.

**Recommendation.** Files stay the contract: a serving record names files in the blob store, and every follower,
whatever it runs on, can load them. On top, inside one gang, a full-weight trainer may **push** through vLLM's API:

1. After a step, the trainer writes the serving copy (BF16) to the blob store as now, and streams the same tensors to
   the run's engine hosts: `ipc` where they share the GPU (in place of today's reading the weights back from the file
   on waking), `nccl` where they are in one process group, set up when the gang starts.
2. Each engine host takes the update under the checkpoint's name (with `wait`, so a turn never spans two weights)
   and beats that it holds it; its follower sees the checkpoint already held when the serving record arrives, and
   fetches nothing.
3. A follower that missed the push (a pod outside the gang, an engine host started again) loads the files, as now.

Deltas (only the tensors that changed, or only their changed bytes) are a later saving on either path.

LoRA adapters always move as files: from tens of megabytes to about ten gigabytes (a 1T MoE's), small next to a
step.

### How each topology fits the roles

| Topology | Provider declares | Demand and placement | Validation | Bridges | Serving, following | Leases |
|---|---|---|---|---|---|---|
| Engine replica, TP or EP on one node | `tensor_parallel`, `expert_parallel` on the model offer; `gpus` per replica equal to TP times PP | One bundle of that many GPUs per replica | `memory`: weights and KV fit; TP divides the model's heads | None | Unchanged | A RunPod pod of `gpu_count` GPUs |
| Engine replica across nodes (PP) | `pipeline_parallel`, `nodes` | A `STRICT_SPREAD` group per replica; a worker group per replica on Kubernetes | Nodes available of one fabric | None | Unchanged: the replica is one server | Deferred |
| Trainer on one node (FSDP2) | `gpus` up to a node's | One bundle of that many GPUs | `memory`: shards fit | Gathered serving copy | Unchanged | A RunPod pod of `gpu_count` GPUs |
| Trainer across nodes | `nodes`, `gpus_per_node`, `parallel` (FSDP, or Megatron's TP, PP, EP) | A `STRICT_SPREAD` group; a worker group with a topology annotation | Nodes of one fabric; the parallel sizes divide the model | `hf-from-megatron` where needed | Unchanged | An Instant Cluster as one lease |
| Colocated, several GPUs | `colocate_with`, as now | One bundle shared by trainer ranks and engine workers | Both memory needs fit, the larger at a time | None | Push by CUDA IPC | One pod (`runpod-host`, in progress) |
| Disaggregated | Separate providers | Separate bundles or groups | Each fits | None | Push by NCCL inside the gang; files outside it | Separate leases |

### Interface sketches

A provider in the cluster config, with shapes it offers:

```toml
[inference.local-h100]
kind = "vllm"
gpus = 2                                       # per replica
[inference.local-h100.models."meta-llama/Llama-3.3-70B-Instruct"]
context = 16384
precision = "fp8"
lora_groups = ["attention", "mlp"]
options = { tensor_parallel_size = 2, quantization = "fp8", max_lora_rank = 64, max_num_seqs = 128 }

[trainers.local-fsdp]
kind = "lora"
gpus = [1, 2, 4, 8]                            # the counts a run may ask for (trainer.gpus)
parallel = "fsdp"
models = ["meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen3.5-9B"]
segment_tokens = 16384

[trainers.runpod-cluster]
kind = "runpod-trainer"
trainer = "full"
gpu_types = ["NVIDIA H200"]
nodes = [2]                                    # an Instant Cluster of 2 nodes of 8
parallel = "fsdp"
models = ["Qwen/Qwen3.5-35B-A3B"]
```

A part of a run's demand that spans nodes:

```python
@dataclass(frozen=True)
class Part:
    name: str
    asks: Resources          # per node
    nodes: int = 1           # >1: a placement group of its own, STRICT_SPREAD, one bundle per node
    fabric: str | None = None  # the topology level its nodes must share ("block", "rack"), for Kueue
```

## 3. Choosing GPUs

### Now

A run's GPUs are whatever its providers were configured with: a `vllm` provider's `gpus` per replica, a trainer's
`gpus`, a RunPod provider's `gpu_types`. The capacity rule counts GPUs against `[capacity]` and the free GPUs in the
heartbeats; it knows neither the GPU's type nor its memory, so a run that cannot fit in memory is found out when vLLM
or the trainer runs out of it. `spend_of` prices metered parts per token; scheduled parts cost nothing in it (the
pod's hourly price times a recently measured step time is in progress on another branch). Nothing estimates how long
a step will take.

### The memory model

Per GPU, for a part sharded over `n` GPUs. `P` is total parameters, `A` active parameters, `L` the layers, `h` the
hidden width, `s` a segment's tokens, `c` the sequences an engine holds at once, `t` their average length.

| Part | Memory per GPU |
|---|---|
| LoRA trainer | base weights `2P/n` (BF16; `P/n` at FP8, about `0.55P/n` at 4 bits) + adapter and its Adam state `16·r·Σ(in+out)/n` + activations |
| Full-weight trainer | `16P/n` (FP32 weights and gradients, Adam's two moments) + `2P/n` for a frozen reference when one is asked for + activations |
| Activations, with recomputation | about `2·s·h·L` bytes (each layer's input kept) + one layer recomputed (about `34·s·h` bytes plus attention) + one chunk of logits (`LOGIT_ROWS · vocabulary · 4` bytes): a few GB for 2,000-token segments, tens of GB at 32K |
| Engine | weights `b·P/tp` (b bytes a weight at the serving precision) + KV `c·t·kv/tp` + linear-attention state `c·state/tp` + about 2 to 4 GB of activations, CUDA graphs and fragmentation; vLLM takes `gpu_memory_utilization` of the card and gives the rest after the weights to KV |
| KV bytes per token | `2 · full-attention layers · KV heads · head width · bytes` (MLA: `layers · (latent + rope width) · bytes`) |

KV per token for the models of this note, at BF16 (half at FP8 KV), from each `config.json`:

| Model | Full-attention layers | KV heads × width | KV a token | Also per sequence |
|---|---|---|---|---|
| Qwen3.5-4B | 8 of 32 | 4 × 256 | 32 KiB | linear-attention state |
| Qwen3.5-9B | 8 of 32 | 4 × 256 | 32 KiB | about 25 MB of linear-attention state |
| Qwen3.5-27B | 16 of 64 | 4 × 256 | 64 KiB | linear-attention state |
| Qwen3.5-35B-A3B | 10 of 40 | 2 × 256 | 20 KiB | linear-attention state |
| Qwen3.5-397B-A17B | 15 of 60 | 2 × 256 | 30 KiB | about 94 MB |
| Qwen3-8B | 36 | 8 × 128 | 144 KiB | |
| Qwen3-30B-A3B | 48 | 4 × 128 | 96 KiB | |
| Qwen3-235B-A22B | 94 | 4 × 128 | 188 KiB | |
| Llama 3.3 70B | 80 | 8 × 128 | 320 KiB | |
| gpt-oss-120b | 18 of 36 (the rest a 128-token window) | 8 × 64 | 36 KiB | the window |
| GLM-4.6 | 92 | 8 × 128 | 368 KiB | |
| DeepSeek-V3, Kimi K2 | 61 (MLA) | latent 512 + 64 | 68.6 KiB | |

At 96 sequences of 2,000 tokens, Llama 3.3 70B needs 63 GB of KV and Qwen3.5-9B 6 GB (and 2.4 GB of state): the
attention layout matters as much as the parameter count.

### The throughput model

Four estimates, each with an efficiency factor the platform's own measurements replace once it has them:

- **Decode**, per engine iteration (one token for every sequence in flight): the larger of reading the weights and the
  KV once, `(b·P_read + c·t·kv) / (bandwidth·η_bw·tp)`, and computing, `c·2A / (flops·η_compute·tp)`. `P_read` is all
  the weights of a dense model and of a MoE at RL batch sizes (nearly every expert is chosen by some token);
  `η_bw` about 0.5: Databricks measured 55 to 60% of an H100's bandwidth at batch 1 with TensorRT-LLM and less at
  larger batches, small models on fast GPUs reach less (CPU overheads), and higher tensor parallelism lowers it; plus
  a couple of milliseconds an iteration for scheduling and sampling.
- **Prefill**: `prompt tokens not cached · 2A / (flops·η)`, with η about 0.5.
- **Training**: `k·A·trained tokens / (flops·MFU·n)`, with `k` = 6 for LoRA with recomputation (forward, the forward
  again, the backward through activations; no gradients of the frozen weights) and 8 for full weights; MFU about 0.3
  to 0.4 for dense models (torchtitan's FSDP2 reaches 33 to 42% on Llama 3.1 8B, Megatron 41 to 48%, Llama 3 405B
  was trained at 38 to 43%), and 0.25 for MoE outside Megatron (Megatron-Core reaches 39 to 49% on Qwen2-57B-A14B and
  Mixtral 8x22B; FSDP2 without grouped kernels much less).
- **Publishing**: the serving copy's bytes over the blob store's throughput, or over NCCL when pushed; plus loading.

**An agentic episode is a chain.** Each turn waits for the last turn's reply and the environment's answer, so an
episode needs `turns · tokens sampled a turn` decode iterations one after another, however many GPUs serve it. With
49 turns of 500 tokens that is about 25,000 iterations: at 15 ms each, six minutes for the step's generation whatever
the replicas. More replicas only help when more episodes are in flight than one replica holds well; lower per-token
latency (tensor parallel, FP8, smaller models, speculative decoding, smaller budgets) shortens the chain; and the
asynchronous loop (`max_lag` above 0) overlaps the next step's chains with this step's training. The planner models
the chain, not just tokens per second.

### The GPU catalog

Dense figures in TFLOPS (NVIDIA's sparse figures halved); RunPod's prices per GPU-hour from its pricing page of
2026-09-27, read on 2026-10-05, community and secure cloud. RunPod no longer takes new community hosts, so secure
prices are the ones to plan on.

| GPU | Memory | Bandwidth | BF16 | FP8 | Links | RunPod, community / secure |
|---|---|---|---|---|---|---|
| B200 | 180 GB | 8.0 TB/s | 2,250 | 4,500 (FP4 9,000) | NVLink 1.8 TB/s | $5.98 / $6.79 |
| H200 SXM | 141 GB | 4.8 TB/s | 989 | 1,979 | NVLink 900 GB/s | $3.59 / $4.59; $4.31 in an Instant Cluster |
| H100 SXM | 80 GB | 3.35 TB/s | 989 | 1,979 | NVLink 900 GB/s | $2.69 / $3.49 |
| H100 NVL | 94 GB | 3.9 TB/s | 835 | 1,671 | NVLink bridge 600 GB/s | $2.59 / $3.19 |
| H100 PCIe | 80 GB | 2.0 TB/s | 756 | 1,513 | PCIe 5 | $1.99 / $2.89 |
| A100 SXM 80 GB | 80 GB | 2.04 TB/s | 312 | — | NVLink 600 GB/s | $1.39 / $1.59; $1.79 in an Instant Cluster |
| RTX PRO 6000 Blackwell | 96 GB | 1.6 to 1.8 TB/s | about 500 | about 1,000 | PCIe 5 | $1.69 / $2.09 |
| L40S | 48 GB | 0.86 TB/s | 362 | 733 | PCIe 4 | $0.79 / $1.09 |
| RTX 6000 Ada | 48 GB | 0.96 TB/s | about 364 | about 728 | PCIe 4 | $0.74 / $0.84 |
| RTX A6000 | 48 GB | 0.77 TB/s | 155 | — | PCIe 4 | $0.33 / $0.53 |
| RTX 5090 | 32 GB | 1.79 TB/s | 210 (419 with FP16 accumulation) | 419 | PCIe 5 | $0.69 / $0.99 |
| RTX 4090 | 24 GB | 1.0 TB/s | 165 (330 with FP16 accumulation) | 330 | PCIe 4 | $0.34 / $0.74 |
| MI300X | 192 GB | 5.3 TB/s | 1,307 | 2,615 | Infinity Fabric | not priced |

GeForce cards compute BF16 with FP32 accumulation, which training needs, at half rate, so the lower figure is theirs.
For comparison, an H100 SXM costs about $1.8 an hour on Vast.ai's marketplace, $3.99 to $4.29 at Lambda, $4.50 at
Nebius and $6.16 at CoreWeave (read 2026-10-05). An 8-GPU node is the most one RunPod pod holds; Instant Clusters join
2 to 8 such nodes (16 GPUs on demand, more through sales).

### Prices and availability

- **Hardware facts** (memory, bandwidth, dense TFLOPS at BF16, FP8 and FP4, interconnect, the counts a node or pod
  holds) change only with new hardware: a file in the repository (`rollout_train/gpus.toml`), reviewed like code.
- **RunPod** publishes prices and stock through its API: the REST catalog (`GET /v2/catalog/gpus`, with
  `include=AVAILABILITY` and a product of `POD` or `CLUSTER`: secure and community prices, the most GPUs a pod takes,
  availability NONE to HIGH per data centre) and the GraphQL `gpuTypes` query (`lowestPrice` with `stockStatus`,
  `maxUnreservedGpuCount`, spot and cluster prices). A periodic task (a Ray task on the cluster's Ray, hourly, and
  when the monitor starts) reads them with the account's key and appends a dated record per GPU type, pod size and
  cloud beside the ledger (`prices` table). The planner reads the newest; a price older than a day is shown with its
  age, older than a week is a note in the check.
- **Tinker** publishes `models.json`, its machine-readable catalog. The same task reads it into the same table, and
  the cluster config's `cost` tables become fallbacks that the check notes when they differ from the newest record.
- **Local GPUs** have no price. The cluster config may say an hourly figure for them (power, or what the hardware
  cost spread over its life) so plans that use them compare with rented ones; by default they cost nothing and the
  plan says so.

### The planner

A pure function, beside `spend_of` and `demand`:

```python
@dataclass(frozen=True)
class Target:
    step_minutes: float | None = None          # at most this long a step
    dollars_per_step: float | None = None      # at most this much a step
    dollars_per_hour: float | None = None
    prefer: Literal["cost", "time"] = "cost"   # what ranks plans that meet the rest

@dataclass(frozen=True)
class Shape:
    provider: str                              # an inference provider or trainer of the cluster config
    gpu: str                                   # a catalog GPU type
    gpus: int                                  # per replica (engines) or in all (trainer)
    replicas: int = 1
    nodes: int = 1
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    expert_parallel: bool = False
    precision: str = "bfloat16"
    memory_gib: float = 0                      # estimated per GPU, of the GPU's
    settings: Mapping[str, JsonValue] = {}     # what it sets: channels.policy.provider, .replicas, trainer.gpus, ...

@dataclass(frozen=True)
class Plan:
    trainer: Shape | Literal["metered"]
    engines: Shape | Literal["metered"]
    colocated: bool
    step_seconds: float
    parts: Mapping[str, float]                 # generation, prefill, training, publishing, overlap
    dollars_per_step: float
    dollars_per_hour: float
    prices_at: datetime                        # the oldest price it used
    assumptions: tuple[str, ...]               # every efficiency factor and estimate it used, in words
    measured: bool                             # step times from this platform's own recent steps, not estimates

def plans(settings: RunSettings, cluster: Cluster, environment: EnvironmentFacts, card: ModelCard,
          catalog: Catalog, prices: Prices, target: Target, measured: Measurements) -> list[Plan]: ...
```

It enumerates candidates from what the cluster offers (each provider's GPU type, the counts and shapes it allows, the
metered providers), keeps those whose memory fits with headroom, estimates each one's step and cost, keeps those that
meet the target, and ranks them; the first few go to the form, each with the reason it was chosen or what it would
take to fit. Its inputs are those `spend_of` reads already (the environment's turns, samples a turn and prompt
tokens; the run's group size, groups a step, budgets) plus the model card, `weights`, `trainer.rank`,
`segment_tokens`, the objective (a frozen reference or a teacher adds memory and passes), `episodes_at_once` and
`max_lag`.

**Measured beats estimated.** Every step already records its durations and the trainer's peak memory. Once a
(model, GPU type, shape) has steps recorded, the planner uses their median per-token figures, scaled to the run's
workload, and says so (`measured`); the roofline estimates are for what has not run yet. The pod spend estimate in
progress elsewhere uses the same recorded step times.

### In the New run form, the check and the command line

- **The form** gets a **Plan** step after Budgets: a time target or a spend target, then the planner's first three
  plans as rows (GPU type and count for the trainer and the engines, colocated or not, minutes a step, dollars a step
  and an hour, how old the prices are). Choosing one fills the settings it names, marked as the plan's as a preset's
  fields are; changing any of them afterwards keeps the change and the plan is shown as edited. A run need not take a
  plan.
- **The check** gains a `memory` rule from the same model: a refusal where a part cannot fit on its GPUs at all (the
  weights alone exceed them), a note where it fits with less than 10% headroom or where the estimate rests on a model
  card overridden by hand. A `plan` note says when the chosen settings are more than twice as dear or as slow as the
  best plan for the same target. `spend_of` includes scheduled parts priced by the hour from the price records.
- **The command line**: `rollout plan --preset NAME --minutes 10 --dollars 5` prints the plans; `rollout train
  --plan 1` applies the first.

## Phases

Most value first. Each later phase waits for its trigger: a run someone wants that the earlier phases cannot give.

| Phase | Build | Why first, or the trigger that starts it |
|---|---|---|
| 1. Know the model and the GPU | `ModelCard` from `config.json`; the GPU catalog file; the memory model as a `memory` rule of the check; scheduled parts priced by the hour in `spend_of` (joining the pod spend work in progress); RunPod's and Tinker's prices read into dated records; the planner over what the cluster offers today (provider, replicas, colocated or not), shown in the New run form and `rollout plan` | Now: it costs little, fails runs before they start instead of after, and answers "RunPod or Tinker for this run" with numbers, which is the question the RunPod work raises |
| 2. Engines on several GPUs of one node | `VllmEngine` options passed through from an allow-list; tensor and expert parallel on a model offer; a replica's GPUs from its shape; RunPod providers with `gpu_count`; FP8 serving offers with `precision` | Now, with phase 1: it is a small change, and it lets a 27B to 35B model serve on two GPUs, a 70B on two to four, and any model on 8 H200 at FP8, which is most of the model range a single-node trainer can train |
| 3. Families as packages | The `Architecture` declaration and its entry point; `TARGETS` and the Tinker remap read from it; the renderer conformance kit; `lora_targets` and `architecture` rules; one new family end to end (gpt-oss, which brings the harmony format and MoE LoRA, or a dense family such as Llama 3.3) | When a run wants a model outside Qwen3, Qwen3.5 and Gemma, or a MoE (Qwen3.5-35B-A3B, Qwen3.6-35B-A3B) trained locally with expert LoRA |
| 4. A trainer on several GPUs of one node | Built: FSDP2 in `rollout_lora` for `lora` and `full` (one process per GPU, sharded state, serving copy gathered); a resident trainer that keeps its state between steps and saves it every `trainer.state_every` steps; a provider's `gpus` and `gpu_count`; the `memory` rule. Not built: MoE expert LoRA with grouped kernels | When a run wants full weights above about 4B, or LoRA above about 30B, or a step on one GPU takes longer than the step time the run is willing to wait (the planner shows it) |
| 5. Weights pushed inside a gang | vLLM's weight transfer (`ipc` colocated, `nccl` disaggregated) beside the files; the engine host's "held by push"; deltas later | When publishing a full-weight checkpoint takes more than about a tenth of a step (full weights above about 8B through the blob store) |
| 6. Several nodes | A placement group per spanning part (`STRICT_SPREAD`), worker groups per part, Kueue topology annotations; a trainer whose files do not go by path; RunPod Instant Clusters as one lease for a `runpod-trainer`; pipeline parallel engines across nodes | When a run needs more than 8 GPUs for its trainer: full weights above about 30B on H100 (60B on B200), LoRA on models above about 400B at BF16, or a step target one node cannot meet |
| 7. Megatron for large MoE | A trainer kind over Megatron-Core through Megatron-Bridge (TP, PP, EP, CP), objectives ported from `rollout_objectives` or wrapped, `hf-from-megatron` for serving copies | When a run trains a MoE above about 100B in full, or LoRA on one above about 400B where Tinker lacks the model or costs more than the planner's best scheduled plan for a sustained run |
| Deferred | Context parallel (contexts beyond 128K); images in turns; an SGLang engine; pre-quantized FP8 serving copies; automatic re-planning in the middle of a run; a shared read-only service for untrained channels | Each when a run needs it: an environment with images or 128K contexts, a model SGLang serves markedly better, quantizing at load measured as a real share of a step |

## Open decisions

1. **A resident trainer for large models.** Today a step is a fresh process that loads its parent's files, which keeps
   the trainer stateless and every step recoverable. For full weights above a few billion parameters (and any trainer
   on several GPUs) that costs more than the step. *Recommendation:* keep the fresh process for one-GPU trainers;
   make multi-GPU and full-weight trainers resident, writing the serving copy every step and the full state (FP32
   weights, optimizer) every `trainer.state_every` steps (default 1 for LoRA, 10 for full weights), so a crash costs
   at most that many steps. *Built:* trainers on several GPUs are resident, with `trainer.state_every`; a trainer on
   one GPU, full weights included, keeps the fresh process. A step whose parent's state lacks the full state starts
   from the parent's serving copy with the optimizer afresh: the loop does not yet go back to the newest checkpoint
   that holds it.
2. **Our own FSDP2 trainer, or an existing framework's.** verl, NeMo-RL and SkyRL bring sharded trainers, but each
   brings its own loop, objectives and engine management, which overlap the platform's. *Recommendation:* FSDP2 in
   `rollout_lora` for one node (PyTorch's `fully_shard` over the same model code), so objectives stay one
   implementation; Megatron-Core through Megatron-Bridge, not a whole framework, when phase 7 triggers. *Built:*
   FSDP2 in `rollout_lora`, the step shared in `rollout_objectives`.
3. **Very large models: Tinker or our own gang.** *Recommendation:* LoRA on models above about 200B total parameters
   on Tinker by default (it has Kimi-K2.6, Qwen3.5-397B, DeepSeek-V3.1, GLM-5.3); our own GPUs only where the planner
   shows a sustained run cheaper on a scheduled gang, or for full weights, which Tinker does not train.
4. **MoE LoRA scope.** *Recommendation:* every expert at rank `r / k`, Tinker's choice, where the engine serves it;
   attention and shared experts otherwise; the run's start records which.
5. **Where topology knobs live.** In the cluster config (a provider per shape: `h100-tp2`, `h100-tp4`) or in a run's
   settings within what a provider allows. *Recommendation:* a provider declares a GPU type and the counts and shapes
   it allows; a run chooses `trainer.gpus` and `channels.NAME.replicas` among them; tensor parallel stays on the model
   offer, since it follows from the model and the GPU. *Built:* the counts are the provider's alone (`gpus` for a
   local trainer, `gpu_count` for a RunPod one), a provider per shape; a run says none.
6. **What a local GPU costs in a plan.** *Recommendation:* an optional `cost = { hour = … }` on local providers;
   nothing by default, and the plan says the local GPU was counted as free.
7. **Reading prices automatically.** It needs the RunPod key wherever the task runs (read-only use).
   *Recommendation:* an hourly task with dated records, the cluster config's prices as fallbacks, and the check noting
   prices older than a week.
8. **Serving precision for large models.** *Recommendation:* FP8 on Hopper and Blackwell for models of 70B and up,
   recorded per provider offer; BF16 below; `kl_floor` watched on the first steps of a new pairing, with BF16 as the
   fallback when it is high.
9. **How much the planner decides.** *Recommendation:* advisory. It ranks plans and fills settings only when the user
   picks one (in the form, or `--plan` on the command line); the check judges the settings asked for, never the plan.

## Sizing table

Rough cost of one training step of a gridworld-like workload, by model, weights and shape, from the memory and
throughput models above. They are estimates to choose between shapes, good to about a factor of two, not
measurements.

**The workload.** Four groups of eight episodes a step (`groups_per_step = 4`, `group_size = 8`), all 32 episodes in
flight at once (`episodes_at_once = 32`); 148 samples an episode (49.3 turns, 3 agents sampled each turn: the
gridworld's averages over its rows); prompts of 1,500 tokens, half of them served from the prefix cache; 500 tokens
sampled a turn on average (budgets of 512 thinking and 128 answer tokens). That is 4,736 samples a step: 3.55M prompt
tokens computed, 2.37M sampled, and 9.47M trained, every sample being one segment of about 2,000 tokens (each turn the
agent sees the system prompt and its newest observation). An episode is a chain of about 24,650 decode iterations.

**The assumptions.**

- Engines: one replica per shape listed, 96 sequences in flight; a decode iteration takes the larger of reading the
  weights and KV at 50% of the GPUs' bandwidth and computing at 50% of their FLOPs, plus 2 ms; prefill at 50% of their
  FLOPs. MoE engines read all their weights each iteration. FP8 where the table says so.
- Trainers: FSDP2 (Megatron where the table says so); 6 × active parameters × trained tokens FLOPs for LoRA and 8 ×
  for full weights (activations recomputed); 35% of BF16 peak for dense LoRA, 40% for dense full weights, 25% for MoE
  (with grouped expert kernels and expert parallel). Each step publishes its serving copy: an adapter in 5 s, full
  weights in BF16 at 1 GB/s.
- Disaggregated, with `max_lag` of 1 or more, so the next step's generation overlaps this step's training: a step
  takes the longer of the two. Colocated: the sum.
- Prices: RunPod's secure cloud per GPU-hour on 2026-10-05 ($3.49 H100 SXM, $4.59 H200, $6.79 B200), $4.31 for H200
  in an Instant Cluster, and B200 across nodes assumed at its pod price (RunPod quotes those clusters by sales). Every
  GPU of the shape is paid for the whole step. Pod start, idle time, storage and evals are left out.
- Tinker, for comparison: the same tokens at its prices (cached prefill 80% off) for the model it offers of that
  size; it trains LoRA only.

| Model | Weights | GPUs | Topology | Step | Per step | Tinker per step |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | LoRA | 4 H100 (one pod) | Engine 1 GPU BF16; trainer FSDP2 over 3; disaggregated | 9.5 min | $2.20 | $21.40 |
| Qwen3.5-9B | LoRA | 1 H100 | Colocated, as the platform runs on one GPU today | 34 min | $2.00 | |
| Qwen3.5-9B | full | 4 H100 (one pod) | Engine 1 GPU; trainer FSDP2 over 3 (52 GB a GPU); disaggregated | 10 min | $2.30 | LoRA only |
| Qwen3.5-35B-A3B | LoRA | 4 H100 (one pod) | Engine TP 2 at FP8; trainer FSDP2 over 2 with expert LoRA | 6 min | $1.40 | $16.60 (Qwen3.6-35B-A3B) |
| Qwen3.5-35B-A3B | full | 8 H200 (one pod) | Engine TP 2 BF16; trainer FSDP2 with expert parallel over 6 (97 GB a GPU) | 8 min | $4.75 | LoRA only |
| Llama 3.3 70B | LoRA | 8 H100 (one pod) | Engine TP 2 at FP8; trainer FSDP2 over 6 | 32 min | $15 | not offered (Qwen3.8-27B: $60) |
| Llama 3.3 70B | full | 10 B200 (a pod of 8, a pod of 2) | Engine TP 2 at FP8; trainer FSDP2 over 8 (145 GB a GPU) | 15 min | $17 | LoRA only |
| Qwen3-235B-A22B | LoRA | 12 H200 (a pod of 8, a pod of 4) | Engine TP 4 with EP at FP8; trainer FSDP2 with EP over 8 | 13 min | $12 | not offered (Qwen3.5-397B-A17B: $93) |
| Qwen3-235B-A22B | full | 36 H200 (an Instant Cluster of 4 nodes, a pod of 4) | Engine TP 4 with EP at FP8; trainer Megatron TP, PP, EP over 32 | 13 min | $34 | LoRA only |
| Kimi K2 (1T) | LoRA | 24 B200 (2 nodes, a pod of 8) | Engine TP 8 with EP at FP8; trainer Megatron EP over 16 (BF16 base, 133 GB a GPU) | 14 min | $39 | $68 (Kimi-K2.6) |
| Kimi K2 (1T) | full | 128 or more H200 | 12.4 TB of trainer state | — | not planned | LoRA only |

What the table says:

- **Scheduled GPUs are several times cheaper than Tinker for sustained runs:** about ten times for the 9B and the
  35B-A3B, about twice for 1T LoRA, before idle time, which a run with scale to zero keeps small.
- **Training dominates the cost of a step**, because every sample is trained: Tinker's own bill for the 9B is two
  thirds training. Training fewer segments a step (`trainer.segments_per_step`, as the one-GPU preset does) cuts
  training time in proportion.
- **Generation is a chain:** the 9B's generation takes 9.5 minutes whatever the replicas; TP 2 brings it to about 5,
  and it is why the 35B-A3B shapes give their engines two GPUs.
- **The shape matters as much as the GPU type:** the 35B-A3B in full weights costs $9 a step with a one-GPU engine and
  $4.75 with a TP 2 engine on the same pod, because the trainer stops waiting for the chain. That is the planner's job.

## Sources

All read on 2026-10-05 unless a date is given.

**RL frameworks and engines**

- [Keep the Tokens Flowing: lessons from 16 open-source RL libraries](https://huggingface.co/blog/async-rl-training-landscape)
  (Hugging Face, 2026-03-10)
- verl: [repository](https://github.com/volcengine/verl),
  [fully async](https://verl.readthedocs.io/en/latest/advance/fully_async.html),
  [delta weight sync](https://verl.readthedocs.io/en/latest/advance/delta_weight_sync.html),
  [LoRA](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html),
  [DeepSeek-V3 and Qwen3-235B](https://verl.readthedocs.io/en/latest/perf/dpsk.html);
  [Mind Lab's trillion-parameter LoRA RL](https://macaron.im/mindlab/research/building-trillion-parameter-reasoning-rl-with-10-gpus)
- OpenRLHF: [repository](https://github.com/OpenRLHF/OpenRLHF),
  [architecture](https://openrlhf.readthedocs.io/en/latest/architecture.html),
  [with vLLM](https://vllm.ai/blog/2025-04-23-openrlhf-vllm)
- NeMo-RL: [repository](https://github.com/NVIDIA-NeMo/RL), [refit](https://docs.nvidia.com/nemo/rl/nightly/guides/refit.html),
  [checkpoint engines](https://docs.nvidia.com/nemo/rl/nightly/design-docs/checkpoint-engines.html),
  [performance](https://docs.nvidia.com/nemo/rl/nightly/about/performance-summary.html)
- SkyRL: [repository](https://github.com/NovaSky-AI/SkyRL), [Megatron](https://docs.skyrl.ai/docs/examples/megatron),
  [Tinker-compatible server](https://docs.skyrl.ai/docs/tinker/architecture)
- [slime](https://github.com/THUDM/slime) and its [releases](https://github.com/THUDM/slime/releases)
- prime-rl: [repository](https://github.com/PrimeIntellect-ai/prime-rl),
  [weight broadcast](https://primeintellect.ai/rl-glossary/weight-broadcast), [INTELLECT-2](https://arxiv.org/abs/2505.07291)
- [AReaL](https://github.com/inclusionAI/AReaL), [ROLL](https://github.com/alibaba/ROLL)
- TRL: [vLLM integration](https://huggingface.co/docs/trl/main/en/vllm_integration)
- [torchtune](https://github.com/pytorch/torchtune), [torchforge](https://github.com/meta-pytorch/torchforge),
  [introducing torchforge](https://pytorch.org/blog/introducing-torchforge/)
- vLLM: [parallelism and scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html),
  [data parallel](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html),
  [expert parallel](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html),
  [context parallel](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment.html),
  [weight transfer](https://docs.vllm.ai/en/latest/training/weight_transfer/),
  [native RL APIs](https://vllm.ai/blog/2026-05-28-native-rl-apis) (2026-05-28),
  [sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode.html), [LoRA](https://docs.vllm.ai/en/latest/features/lora.html),
  [multi-LoRA serving](https://vllm.ai/blog/multi-lora) (2026-02-26),
  [MoE LoRA and FlashInfer experts](https://discuss.vllm.ai/t/lora-integration-for-qwen3-5-122b-fails-during-deployment-on-vllm-0-17-0/2466),
  [releases](https://github.com/vllm-project/vllm/releases), [multi-node serving with LWS](https://docs.vllm.ai/en/latest/deployment/frameworks/lws.html)
- SGLang: [for RL](https://docs.sglang.io/advanced_features/sglang_for_rl.html),
  [LoRA](https://docs.sglang.io/advanced_features/lora.html), [MoE LoRA on quantized bases](https://github.com/sgl-project/sglang/pull/22323),
  [expert parallel](https://docs.sglang.io/advanced_features/expert_parallelism.html)
- MoE LoRA in trainers: [PEFT's `target_parameters`](https://huggingface.co/docs/peft/developer_guides/lora),
  [Megatron-Bridge PEFT](https://docs.nvidia.com/nemo/megatron-bridge/nightly/training/peft.html),
  [Unsloth's MoE kernels](https://unsloth.ai/docs/basics/faster-moe)
- [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (Thinking Machines, 2025-09-29);
  [GSPO](https://arxiv.org/abs/2507.18071)

**Tinker**

- [Models and pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/index.md) and
  [`models.json`](https://tinker-docs.thinkingmachines.ai/tinker/models.json)
- [Exporting a LoRA adapter](https://tinker-docs.thinkingmachines.ai/cookbook/deployment/lora-adapter/index.md),
  [Tinker](https://thinkingmachines.ai/tinker/), [general availability](https://thinkingmachines.ai/blog/tinker-general-availability/)

**Kubernetes and Ray**

- [Placement groups](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html),
  [KubeRay with Kueue](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kueue.html),
  [gang scheduling with Kueue](https://docs.ray.io/en/latest/cluster/kubernetes/examples/rayjob-kueue-gang-scheduling.html),
  [multi-host worker groups](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/tpu.html),
  [KubeRay releases](https://github.com/ray-project/kuberay/releases)
- Kueue: [RayJobs](https://kueue.sigs.k8s.io/docs/tasks/run/rayjobs/),
  [topology-aware scheduling](https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling/) and
  [how to use it](https://kueue.sigs.k8s.io/docs/tasks/run/topology_aware_scheduling/),
  [ProvisioningRequest](https://kueue.sigs.k8s.io/docs/concepts/admission_check/provisioning_request/),
  [MultiKueue](https://kueue.sigs.k8s.io/docs/concepts/multikueue/),
  [waiting for pods](https://kueue.sigs.k8s.io/docs/tasks/manage/setup_wait_for_pods_ready/),
  [LeaderWorkerSet](https://kueue.sigs.k8s.io/docs/tasks/run/leaderworkerset/)
- [LeaderWorkerSet](https://lws.sigs.k8s.io/docs/overview/)

**RunPod and other prices**

- [Pricing](https://www.runpod.io/pricing) (updated 2026-09-27), [pod pricing](https://docs.runpod.io/pods/pricing),
  [GPU types](https://docs.runpod.io/references/gpu-types), [network volumes](https://docs.runpod.io/storage/network-volumes),
  [choosing a pod](https://docs.runpod.io/pods/choose-a-pod)
- Instant Clusters: [overview](https://docs.runpod.io/instant-clusters/overview),
  [product page](https://www.runpod.io/product/instant-clusters),
  [configuration](https://docs.runpod.io/instant-clusters/configuration),
  [create a cluster](https://docs.runpod.io/api-reference-v2/clusters/create-a-cluster.md)
- Price APIs: [GPU catalog](https://docs.runpod.io/api-reference-v2/catalog/list-gpu-types.md),
  [GraphQL](https://docs.runpod.io/sdks/graphql/manage-pods.md) and its [schema](https://graphql-spec.runpod.io/)
- [Lambda](https://lambda.ai/pricing), [CoreWeave](https://www.coreweave.com/pricing), [Nebius](https://nebius.com/prices),
  [Vast.ai via computeprices.com](https://computeprices.com/providers/vast)

**GPUs and efficiency**

- NVIDIA: [H100](https://www.nvidia.com/en-us/data-center/h100/), [H200](https://www.nvidia.com/en-us/data-center/h200/),
  [HGX B200 and B300](https://www.nvidia.com/en-us/data-center/hgx/), [A100](https://www.nvidia.com/en-us/data-center/a100/),
  [L40S](https://www.nvidia.com/en-us/data-center/l40s/),
  [RTX PRO 6000](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/),
  [B200 at 180 GB](https://lenovopress.lenovo.com/lp2226-thinksystem-nvidia-b200-180gb-1000w-gpu);
  dense figures for the RTX cards from [flopper.io](https://flopper.io/gpu/nvidia-rtx-pro-6000-blackwell-server-edition)
- [MI300X](https://glennklockwood.com/garden/processors/mi300x)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [torchtitan](https://arxiv.org/html/2410.06511),
  [Llama 3](https://arxiv.org/abs/2407.21783), [Megatron-Core MoE](https://arxiv.org/abs/2504.14960)
- [LLM inference performance engineering](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
  (Databricks), [vLLM v0.6 performance](https://vllm.ai/blog/2024-09-05-perf-update),
  [SemiAnalysis InferenceX](https://inferencex.semianalysis.com/)

**Models and formats**

- `config.json` of [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json),
  [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/config.json),
  [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/raw/main/config.json),
  [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/main/config.json),
  [Llama 3.3 70B](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/raw/main/config.json),
  [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/raw/main/config.json),
  [Kimi K2](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905/raw/main/config.json),
  [GLM-4.6](https://huggingface.co/zai-org/GLM-4.6/raw/main/config.json),
  [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json),
  [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it/raw/main/config.json)
- Formats: [Qwen3.5's chat template](https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/chat_template.jinja),
  [harmony](https://developers.openai.com/cookbook/articles/openai-harmony),
  [DeepSeek-V3.1](https://huggingface.co/deepseek-ai/DeepSeek-V3.1),
  [Kimi K2 tool calls](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/docs/tool_call_guidance.md),
  [Llama 3.1](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md),
  [Gemma 4](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4),
  [vLLM's tool calling](https://docs.vllm.ai/en/latest/features/tool_calling.html)
- Transfer: [S3 performance](https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance.html),
  [hf_transfer](https://github.com/huggingface/hf_transfer)
