# Pretraining and continued pretraining

**Status: proposed.** Nothing here is built. It is read against main `fa49b1b`. Sharded data parallel training
(FSDP2) on one multi-GPU machine is being built on another branch (`gpu_count` on `runpod-trainer`, a resident
trainer, checkpoints in PyTorch's distributed checkpoint format) and is named as in progress where it matters. A
design note: see [Design notes](README.md) for the others.

Can the platform also pretrain a model from nothing, and continue the pretraining of an existing one on text of a
domain, once its trainer shards across the GPUs of a machine? This note says what pretraining needs, how open
pretraining systems provide it, what the platform has that fits and what does not apply, three ways to do it, the one
recommended and its phases, what it costs at the sizes within reach, and the decisions left open. The roles of
[the architecture](../architecture/overview.md) do not change: a pretraining run is a run with a trainer and a data
source and no engines, runners or gateway. Where its trainer runs and on how many GPUs is placement.

Code read for this note: `rollout_train.imitation`, `.datasets`, `.trainer`, `.jobs` (`_imitate`, `imitated`),
`.run_settings`, `.providers`, `.pods.training`; `rollout_lora` (`trainer`, `worker`, `full`, `policy`);
`rollout_objectives.step`. Figures from outside are dated and their sources listed at the end.

## The answer in brief

| Question | Answer |
|---|---|
| Can it? | Yes, at the sizes a budget reaches: models up to about 1B parameters from nothing and continued pretraining of models up to about 9B, each on one node of 8 GPUs. Nothing in the ledger, the blob store, the leases or the monitor stands in the way. What is missing is a data path for raw text at hundreds of thousands of tokens a second, a training loop built for throughput rather than for RL's batches, and a contract for a trainer that runs for days |
| What is missing | Tokenized corpora in the blob store and mixtures of them; a data source that reads them in a deterministic, resumable order; packing; learning-rate schedules over a whole run; FP8 and FP4 recipes; loss-spike handling; checkpoints every so many minutes rather than every step; held-out loss and small scoring evals in the loop; a run that outlives a pod |
| What fits | The likelihood family (pretraining is imitation of raw text, every token trained); the ledger's decided-then-made steps, checkpoints with lineage, retention and the registry; leases, scale to zero, `limits.spend` and `limits.hours`; Kueue's gangs; the monitor's per-step charts and the checkpoint graph |
| What does not apply | Episodes, groups, claims, runners, the gateway, channels, engines, renderers, sandboxes, curricula, `max_lag` and importance weights. They are left out cleanly: a run's kind already decides which parts its driver builds, and a pretraining run builds none of them |
| What does not fit | The `Trainer` protocol: a step is a batch the driver hands over and a fresh process that loads its parent. Pretraining needs a trainer that reads its own data, keeps its state for days, and checkpoints on a clock |
| Low precision | FP8 on H100 and B200 is mature (torchao's float8 and MXFP8, Transformer Engine) and adds 10% to 30% throughput at 1B to 9B, less on smaller models. FP4 (NVFP4) trains only on B200 and B300, through Transformer Engine (Megatron-Core); torchao's NVFP4 training is a prototype. It is shown matching FP8 at 12B on 10T tokens, with 16% of the layers kept in bf16 and the last 18% of training in higher precision; below a few billion parameters its gain is unshown and likely small, because small matrix multiplies leave quantizing as large a cost as the multiply ([precision](#precision-bf16-fp8-and-fp4)) |
| Options | (a) the platform's own trainer, extended; (b) an existing framework (torchtitan, or Megatron-Core through Megatron-Bridge) wrapped as a trainer kind; (c) pretraining kept outside, its checkpoints imported as base models |
| Recommendation | (a): a run kind `pretrain`, corpora and mixtures as records, a data source library, and a **stretch** contract for long-running trainers, implemented first by the resident FSDP2 full-weight trainer in bf16 and FP8 (torchao, on Hugging Face models' linear layers). The platform side is the same for (a) and (b): only the inner loop differs. Megatron-Bridge becomes a second implementation of the stretch contract for the first run that wants NVFP4, or that needs tensor, pipeline or context parallelism, or whose size makes a faster loop worth the work of wrapping it; it is the framework the [scaling note](scaling-models-and-topologies.md) already chooses for large MoE |
| Cost | About $25 to $350 for a 125M to 350M model on 10B to 50B tokens; about $1,620 for 1B on 100B tokens; about $320 to $2,900 for continued pretraining of 4B to 9B on 5B to 20B tokens; all in bf16 on one 8-GPU H100 node at RunPod's secure price of 2026-10-05. FP8 takes a sixth to a fifth off the 1B and 9B runs; NVFP4 on B200 might bring the 9B run to about 60% of its bf16 cost, a guess until measured ([sizing](#sizing)) |
| Several nodes | Needed by none of these; useful only to shorten the wall clock. A run that wants it is an Instant Cluster leased as one trainer, as the [scaling note](scaling-models-and-topologies.md) proposes |

## What pretraining needs

Pretraining is next-token prediction over a corpus of raw text, at a scale where throughput sets the cost and a run
lasts days. Continued pretraining is the same loop started from a trained model, on fewer tokens of a domain's text,
mixed with some of the general text it was trained on so it does not forget. Both need the parts below; the open
systems that provide them are compared after.

### Data

- **Tokenized shards.** A corpus is tokenized once, ahead of training, into flat arrays of token ids (16-bit for
  vocabularies up to 65,536, 32-bit above), documents joined with an end-of-text token, in shards of a few hundred
  megabytes to a few gigabytes. Megatron, OLMo and GPT-NeoX read memory-mapped arrays of this kind; torchtitan and
  Levanter can also tokenize on the fly. Tokenizing ahead keeps the training GPUs from waiting on CPUs, and makes a
  corpus a fixed, named thing.
- **Streaming from object storage.** Shards live in a bucket and are read as training goes, with a local cache and
  prefetch, so a pod needs no copy of the whole corpus on its disk. The bandwidth asked is small: 8 GPUs at 500,000
  tokens a second each read 16 MB/s of 32-bit tokens. MosaicML's Streaming is built for this; OLMo-core and
  Megatron read local or mounted files.
- **A deterministic, resumable order.** The sample at position `i` of the run must be a pure function of the corpus
  (or mixture), the seed and `i`, so a run resumed after a crash, on any number of GPUs, sees the same sequence of
  batches. OLMo-core and Megatron compute a global permutation of sample indices; Streaming's elastic determinism
  does the same across changes of world size. A loader that keeps per-rank iterator state resumes only at the same
  world size.
- **Mixtures and their schedules.** A run draws from several corpora by weight (web text, code, books, the domain's
  text), and the weights may change over the run: a final stretch of higher-quality data is now common (OLMo 2's
  mid-training, the decay phase of WSD schedules). Continued pretraining replays a share of general text, with the
  learning rate warmed up and decayed again: 5% of the old data where the new text is close to it, 25% where it is far
  (English to German), matched retraining on both at 405M and 10B.
- **Packing and sequence length.** Documents are joined and cut into sequences of fixed length (2,048 to 8,192
  tokens), with no padding. Attention may be masked so that a token sees only its own document (Llama 3 did this;
  it matters most at long sequences). Sequence length is a training choice, raised late in a run to extend context.
- **Deduplication and quality filtering** are upstream: done once over the raw text by a pipeline (datatrove, Dolma's
  tools, DCLM's classifiers), not by the trainer. Public corpora come already filtered: FineWeb, FineWeb-Edu, DCLM,
  Dolma. The platform needs only to record what a corpus was made from and how.

### The training loop

- **Throughput first.** The loop's cost is GPU-hours, so its quality is measured as tokens a second per GPU and as
  model FLOPs utilization (MFU): the share of the GPUs' peak arithmetic spent on the model's own forward and backward
  (about `6N` FLOPs a token for `N` parameters, plus attention). Good systems reach 35% to 50% on H100 at the sizes of
  this note. What gets there: large packed batches, bf16 compute with fp32 master weights, fused attention
  (FlashAttention), `torch.compile` or hand-fused kernels, a fused or chunked cross-entropy over the vocabulary, and
  activation checkpointing only as much as memory needs.
- **MFU tracked** every few updates, beside tokens a second, so a regression or a slow node is seen at once.
- **Learning-rate schedules over the whole run:** a linear warmup over the first few hundred to few thousand updates;
  then cosine decay to about a tenth of the peak, or **WSD** (warmup, stable, decay): a constant rate and a short
  decay at the end (10% to 20% of the run). WSD lets a run be extended or branched from any stable-phase checkpoint,
  and matches cosine's loss at the end.
- **Gradient clipping** at a global norm of 1.0, and the norm recorded each update.
- **Loss spikes.** Sudden rises of the loss, mostly at larger sizes and high learning rates. Prevented by
  architecture and settings (QK-norm, z-loss, lower Adam epsilon and beta2 adjustments, no weight decay on
  embeddings); handled, when they happen, by skipping the updates whose gradient norm is far above its recent mean, or
  by going back to a checkpoint before the spike and skipping the batches that caused it (PaLM's practice).
- **z-loss**: a small penalty on the log of the softmax's normalizer (PaLM: `1e-4 · log²Z`), which keeps the output
  logits from drifting. Cheap; used by PaLM, OLMo 2 and others.
- **Mixed precision:** bf16 compute with fp32 master weights and optimizer state is the default everywhere; FP8 and
  FP4 matrix multiplies are the next section.

### Precision: bf16, FP8 and FP4

Low precision applies to the matrix multiplies of the linear layers (forward, input gradient, weight gradient), with
scales kept beside the values; master weights and optimizer state stay in fp32 (torchtitan 0.3 adds bf16 optimizer
states), and attention, norms, the embedding and usually the output layer stay in bf16. What each format needs and
gives:

| Format | Hardware | Where it is implemented | Published gain | What it costs |
|---|---|---|---|---|
| bf16 | Any GPU since Ampere | Everywhere | The baseline: 989 dense TFLOPS on H100, 2,250 on B200 | — |
| FP8, one scale a tensor or a row | H100, H200, B200 (and Ada and consumer Blackwell cards' FP8 units) | torchao float8 (used by torchtitan, OLMo-core), Transformer Engine (Megatron) | Llama 3 8B on 8 H100: +25% tensorwise, +10% rowwise (torchao); with FSDP2: +18% at 1.8B, +28% at 8B, +50% at 70B; OLMo-core 1B 55K to 65K tokens a second a GPU (+18%), 7B 10K to 13K (+30%) | Little: loss within noise of bf16 in these reports; tensorwise scaling can lose range on outliers, rowwise is safer and slower |
| MXFP8, a scale every 32 values | B200, B300 (native) | torchao (torchtitan), Transformer Engine | B200: +25% on Llama 3 8B (torchao), 1.22 to 1.28 times bf16 on Llama 3 70B at up to 1,856 GPUs, convergence equal to bf16 | Quantizing costs as much as it saves where matrices are small (small models, small MoE experts) |
| NVFP4, a scale every 16 values plus one a tensor | B200 (SM 10.0), B300 (SM 10.3) for training | Transformer Engine (`NVFP4BlockScaling`), Megatron-Bridge (`bf16_with_nvfp4_mixed`); torchao: prototype, its performance marked not ready; torchtitan 0.3 lists it | FP4 multiplies have twice FP8's peak on B200 (9 PFLOPS dense) and three times on B300. A 12B model on 10T tokens matched its FP8 twin (MMLU-Pro 62.58 against 62.62), with loss within 1% to 1.5% | A recipe, not a switch: random Hadamard transforms on the weight-gradient inputs, 2-D scaling of weights, stochastic rounding of gradients, 16% of the linear layers (the first two and last eight blocks) in bf16, and the last 18% of training in higher precision. Shown at 12B; "smaller models trained on shorter token horizons may not require all of these techniques", and their gain is not published |

Three consequences for this note:

- **The gain grows with the model.** FP8 adds a few percent at 125M to 350M (overheads dominate small matrix
  multiplies), 10% to 20% at 1B, 25% to 30% at 7B to 9B. FP4 on top of FP8 is unmeasured below billions of
  parameters, and the bf16 layers and transforms take back part of it.
- **FP4 means B200.** NVFP4 training runs on B200 and B300 only, not on H100, H200 or the RTX 5080 (SM 12.0), and its
  production recipe is Transformer Engine's, reached through Megatron-Core. Continued pretraining in FP4 from a bf16
  model is not what the published recipe tested.
- **Low precision is a run's setting and a recorded fact,** like the objective: the format, the recipe and which
  layers stayed in bf16 go into the run's start, so a run and its bf16 or FP8 twin compare in one ledger.

### Parallelism beyond FSDP

| Parallelism | What it splits | Needed when |
|---|---|---|
| Data parallel (DDP) | Nothing: every GPU holds the whole model and its optimizer | The model's state (16 bytes a parameter) and activations fit one GPU: up to about 1B on an 80 GB GPU |
| FSDP2 (ZeRO-3) | Weights, gradients and optimizer state | Above that, up to tens of billions on one or a few nodes. HSDP shards within a node and replicates across nodes |
| Tensor parallel (TP) | Each layer's matrices, inside a node | Large models where FSDP's per-GPU batch becomes too small or its gathers too slow: from about 30B to 70B, or at hundreds of GPUs where the global batch limits data parallelism |
| Pipeline parallel (PP) | Layers into stages, across nodes | Models of about 70B and up spread over many nodes |
| Context parallel (CP) | One long sequence across GPUs | Sequences of about 64K to 128K tokens and up, where activations no longer fit (torchtitan trains Llama 3 8B at 32K on 8 GPUs without it) |

For every run in this note's sizing, FSDP2 (or plain data parallel below 1B) on one node is enough: the models are
small, and sequences of 2,048 to 8,192 tokens fit with activation checkpointing.

### Checkpoints

- **Asynchronous distributed checkpoints.** Each rank writes its shard (PyTorch's DCP); asynchronous saving copies
  the state to CPU memory and writes it while training goes on, so a save stops the GPUs for seconds, not minutes.
- **How often:** by the clock, not by the step: often enough that a failure loses little, rarely enough that saving
  costs little. With saves that stop training for `C` seconds and a mean time between failures `M`, about every
  `√(2CM)`. In practice every 30 to 60 minutes for a node, plus a few kept for good.
- **Resharding on resume.** DCP loads a checkpoint into a different number of GPUs or a different parallel layout, so
  a run can resume on whatever pod it gets next.
- **Keeping a few:** the newest two or three full states (for resuming), a thinned series of the model's weights
  (for evals and branching), and the end of each schedule phase.
- **Publishing a checkpoint as a model:** the weights gathered into Hugging Face safetensors with the configuration
  and tokenizer, so vLLM, the trainers and other tools load it. torchtitan, OLMo-core and Megatron-Bridge each convert
  their own format; a trainer over Hugging Face's model code writes it directly.

### Evals during training

- **Held-out loss** on a fixed set of sequences from each corpus (and from corpora not trained on), every few hundred
  updates. It is the main curve; it is cheap, computed by the trainer on its own GPUs.
- **Small downstream evals** that a model of this size can show progress on: multiple choice scored by likelihood
  (HellaSwag, ARC, PIQA, MMLU's cloze form; OLMo's in-loop evals, DCLM's CORE set). They need the logprobs of given
  continuations, not sampling. Generative evals (GSM8K, code) mean little before instruction tuning at these sizes.

### Long runs

- **Failures.** At the scale of thousands of GPUs, something fails every few hours (Llama 3's 54 days on up to 16K H100
  saw 466 job interruptions, 419 of them unexpected, about 78% of those hardware; GPUs alone 58.7%). At one node over a few days, a failure is rarer but still to plan for: a GPU
  that drops off the bus, a pod whose host is lost, a NCCL timeout.
- **Preemption.** RunPod's on-demand (secure cloud) pods are not preempted; spot pods are, and suit pretraining only
  with frequent checkpoints.
- **Restarts** from the newest full state are what every system does. Elastic training (changing the GPU count
  without a restart, torchft's fault-tolerant data parallelism) pays off only with many replicas; at one or two nodes
  a restart from a checkpoint on a new pod is enough.

### Several nodes

Inside a node, GPUs talk over NVLink (900 GB/s a GPU on H100). Across nodes, data parallel training needs a fabric:
RunPod's Instant Clusters join 2 to 8 nodes at 3,200 Gb/s (400 GB/s a node). With HSDP only the gradients cross
nodes: a 1B model's bf16 gradients (2 GB) take about 10 ms at that rate, against an update of about a second; a 9B
model's (18 GB) take about 0.1 s against an update of several seconds. Over the internet between separate pods
(1 to 10 Gb/s) the same 2 GB takes seconds an update, which is why several nodes means an Instant Cluster and not
several pods. Low-bandwidth methods (DiLoCo) exist but are not needed at these sizes.

### Tokenizer and initial weights

- **The tokenizer** is an input fixed before tokenizing. Training one (BPE on a sample of the corpus) matters for a
  new language or domain; otherwise an existing family's tokenizer is reused. The vocabulary's size is a large share of
  a small model: Qwen3's 151,936 tokens at a width of 768 are 117M parameters, nearly all of a "125M" model; GPT-2's
  50,257 are 39M. And a model meant for SFT and RL here needs a renderer, which exists for Qwen and Gemma tokenizers.
- **Initial weights** for a model from nothing: a configuration (layers, width, heads, vocabulary) and an
  initialization rule (normal at 0.02, scaled for residual layers, or the family's own). Hugging Face's
  `from_config` initializes any family's architecture. For continued pretraining, the base model's weights.

### How open systems provide it

| | Data | Stability | Low precision | Parallelism | Checkpoints | Evals in the loop | State |
|---|---|---|---|---|---|---|---|
| torchtitan (PyTorch) | A Grain loader over JSONL, Hugging Face datasets (random access or streaming) or custom sources; weighted mixtures; concatenate-then-split packing; resumes at the same data-parallel degree only | Clipping, warmup and decay schedules | Float8, MXFP8; NVFP4 listed | FSDP2, TP (async TP), PP, CP, EP | Async DCP, `keep_latest_k`, resharding, Hugging Face safetensors written or converted | Validation loss; downstream offline (lm-eval over vLLM) | 0.3.0 (2026-09-03); releases every two months; extension points "subject to change"; needs PyTorch nightly |
| Megatron-Core, Megatron-Bridge (NVIDIA) | Memory-mapped token shards (`.bin` and `.idx`), blended by weight, global sample indices | Spike and NaN checks | FP8 (several scalings), MXFP8, NVFP4 through Transformer Engine | TP, PP (virtual stages), CP, EP, sequence parallel | Distributed, async; Hugging Face both ways through Megatron-Bridge's `AutoBridge` (Llama, Qwen3, Qwen3.5, Gemma, DeepSeek, GLM, Kimi) | Validation loss | Megatron-Bridge 0.6.0 (2026-08-19) |
| OLMo-core (AI2) | NumPy token shards, deterministic global indices, source mixtures, packing and document masking | z-loss, QK-norm (OLMo 2); updates skipped past 6 standard deviations of loss or gradient norm | FP8 | FSDP2/HSDP, TP, CP, EP | DCP, async; Hugging Face conversion | Held-out perplexity, downstream multiple choice | 3.0.0 (2026-10-01) |
| nanotron (Hugging Face) | Tokenized shards, mixtures by stage | Schedules by stage (WSD for SmolLM2 and 3) | — | DP, TP, PP, ZeRO | Its own; Hugging Face conversion | Loss | Trains SmolLM |
| Levanter (JAX) | Cached tokenized shards from object storage, mixtures, resumable | Schedules | — | FSDP and TP through JAX sharding | Tensorstore, resharding | Held-out loss, harness evals | Merged into Marin (November 2025) |
| Composer, LLM Foundry, Streaming (MosaicML) | Streaming from object storage: the same order at any number of GPUs, resumable mid-epoch | Clipping, spike monitors | FP8 | FSDP, HSDP | Sharded, async, to object storage | In-context learning evals | No release since July 2025 |
| GPT-NeoX (EleutherAI) | Megatron's token shards | Schedules | — | TP, PP, ZeRO (DeepSpeed) | DeepSpeed's | Harness evals | Trained Pythia |
| DeepSpeed | The caller's | — | — | ZeRO 1 to 3, PP, Ulysses sequence parallel | Universal checkpoints for resharding | — | A library |

All of them run a single long process (or one process per GPU) for the whole run, read their own data, checkpoint on a
clock or an update count, and log metrics every few updates.

## What the platform has

### What fits

| Piece | What it gives pretraining | What is missing |
|---|---|---|
| The likelihood family (`sft`) | The objective: cross-entropy over the trained tokens. Pretraining is imitation with every token of a sequence trained | z-loss as a component; a loss over packed sequences rather than sampled spans |
| Imitate runs (`rollout imitate`) | A run kind with a trainer and a dataset and no environment; a start record of its kind; a checkpoint that names its dataset | One step per run, over a dataset of episodes' segments rendered by a renderer: not millions of updates over raw text |
| Datasets in the blob store (`rollout_train.datasets`) | The model of a corpus: a record appended once, a manifest blob, a name in the registry, made by a Ray task | Datasets are chosen from episodes; a corpus is token shards made from outside text |
| The ledger | Steps decided before they are taken and made by their checkpoint (`runs/RUN/steps`, `checkpoints`); fences, so one driver at a time; retention that thins saves; lineage from parents | Nothing that records a data cursor or a schedule's position, and nothing for per-update metrics |
| The checkpoint graph and registry | Lineage from a pretraining run to the SFT and RL runs that start from its checkpoints; bookmarks; a base model root | A root for a model made from nothing |
| Evals and suites | Evals of any checkpoint, on a schedule or by hand, with each checkpoint's history | Scoring evals (likelihood of given continuations) and held-out loss: suites play environments, whose tasks see text, not logprobs |
| The monitor | Runs, their settings, per-step trainer charts (`steps` statistics), the checkpoint graph, spend | Curves by tokens trained, throughput and MFU, the data consumed, a time and a cost to finish |
| Leases, scale to zero, `limits` | A pod leased for the run, released when it ends however it ends, reaped if the run goes away; spend and hours bounded | A run that outlives one pod: a new lease after a lost one, resumed from the newest state |
| Kueue gangs, placement groups | Admission of a run's GPUs whole | A trainer of several nodes (the [scaling note](scaling-models-and-topologies.md)'s phase 6) |
| `runpod-trainer` and its training service | A trainer on a pod of `gpu_count` GPUs, asked for a step by the checkpoint it makes, idempotent by it, polled until made | Steps that last an hour |
| FSDP2 trainer (in progress) | Sharded full weights over a node's GPUs, a resident process, DCP state | A packed, compiled, throughput-first path for text |

### What does not apply

Episodes, groups and their claims, episode runners, the gateway and its signed keys, channels, engines and engine
hosts, followers and serving records, renderers, sandbox pools, environments and curricula, `max_lag`, behaviour
logprobs and importance weights. None of them has anything to do in a pretraining run.

This is clean already. A run's kind decides which settings apply (each `Key` in `rollout_train.run_settings` names
its kinds) and which loop its driver runs (`rollout_train.jobs`): an imitate run has no environment and plays no
episodes. A `pretrain` kind is one more entry in `KINDS`, in `TRAINING` and in neither `SAMPLING` nor `RENDERING`, so
it names no channel, and its demand (`rollout_train.demand`, which counts an engine host for each channel that names
a model and a renderer) is its trainer alone. The only RL machinery it touches is serving, and only to evaluate a
published checkpoint by an ordinary eval run, which builds its own engines.

### What does not fit: the trainer's contract

The `Trainer` protocol says a step is a batch in, a parent's files in, a checkpoint's files out, and that a trainer
"keeps nothing between steps that it cannot be given again". `rollout_lora`'s trainers take each step in a fresh
process that loads the model, train one segment at a time with gradient accumulation, and save the whole model and
optimizer every step. That is right for RL, where a step is minutes of generation and seconds to minutes of training
on a few thousand segments. For pretraining:

- **The driver cannot hand over the batch.** Pretraining reads hundreds of thousands to millions of tokens a second.
  The data must be read by the trainer's ranks from the blob store, by a plan the driver gives.
- **A step cannot be an optimizer update.** At one update a second, a checkpoint each update is impossible and a
  ledger step each update is noise. The unit the ledger should see is a **stretch**: thousands of updates between two
  checkpoints, decided before it is taken and made by its checkpoint, like a step now.
- **The process must stay.** Loading 16 bytes a parameter at every stretch is minutes for a 9B model; a resident
  trainer keeps its state and loads a parent only when it does not already hold it (the scaling note's open decision
  1, being built with FSDP2).
- **The inner loop is per segment.** One segment at a time, positions picked out for sampled spans, logits in chunks
  of 128 rows: built for correctness on RL's ragged segments, not for throughput on packed batches.

So pretraining needs a second trainer contract beside `Trainer`, not a change to it.

## Options

### (a) Extend the platform's own trainer

The resident FSDP2 full-weight trainer in `rollout_lora` gains a packed text path: batches of fixed-length sequences
from the data source, one fused forward and backward per micro-batch, `torch.compile`, FlashAttention through the
model's attention implementation, a chunked fused cross-entropy, schedules, clipping, z-loss, spike handling, async
DCP saves and a Hugging Face serving copy. It runs over Hugging Face's model code, as the rest of the platform does.
FP8 comes from torchao, which converts a model's `nn.Linear` layers in place (`convert_to_float8_training`, and MXFP8
on B200) and gathers FP8 weights under FSDP2: the same library torchtitan and OLMo-core use, applied to the same kind
of layers.

### (b) Wrap an existing framework as a trainer kind

A pod image with the framework pinned; the trainer kind starts its processes over the pod's GPUs with a configuration
the platform writes (model, parallelism, precision recipe, schedule, checkpoint interval), plugs the platform's data
source in or reads the corpora in the framework's own format, reports progress to the platform, and turns each
checkpoint it writes into a stretch made: state uploaded, a Hugging Face copy converted. Two candidates:

- **torchtitan**: PyTorch-native, FSDP2, TP, PP, CP, Float8 and MXFP8 through torchao, Qwen3 and Qwen3.5 among its
  models, Hugging Face safetensors written directly. It moves fast: 0.3.0 replaced its configuration system, its
  extension points are "subject to change", it needs PyTorch nightly, and its data loader resumes only at the same
  data-parallel degree (the platform's data source would replace it). Its NVFP4 rests on torchao's, which is a
  prototype.
- **Megatron-Core through Megatron-Bridge**: Transformer Engine's FP8, MXFP8 and NVFP4 recipes (NVFP4's production
  path, `bf16_with_nvfp4_mixed`), TP, PP, CP, EP, its blended token-shard datasets with global sample indices, and
  `AutoBridge` converting to and from Hugging Face's names for the families the platform uses (Qwen3, Qwen3.5,
  Gemma, Llama). Heavier: NVIDIA's container, Transformer Engine's build, Megatron's own model code. It is the
  framework the scaling note already chooses for large MoE (its phase 7), so one wrapper serves both.

### (c) Keep pretraining out

Pretrain with a framework by hand on rented machines, upload the result to Hugging Face or the blob store, and train
from it here as from any base model. The platform already trains from any model by name.

### Weighed

The platform side is the same for (a) and (b): corpora, mixtures, the data source, the run kind, stretch records,
leases across a long run, the monitor's view, the planner's estimate. The choice between them is only whose inner loop
takes the updates.

| | (a) Own trainer | (b) A framework wrapped | (c) Outside |
|---|---|---|---|
| Effort | The platform side, plus the packed path in a trainer being built anyway. The throughput work (compile, fused loss, attention kernels) is where the time goes; FP8 is a library call and a comparison run | The platform side, plus the image, the launcher, the data plug or a corpus format, progress reports and checkpoint conversion. Less tuning, more integration; every upgrade of the framework is checked again | None here; every run's pods, data and checkpoints by hand |
| Correctness risk | Medium: data order on resume, schedule position on resume, gradient accumulation and loss scaling across ranks, precision. Each is testable on CPU or one GPU against a known curve | Lower for the loop and its precision recipes (used at scale by their authors), higher at the seams: resume, the data plug, conversion of names to Hugging Face's | The user's, every time |
| Models | Any family Hugging Face's code runs, including Qwen3.5's hybrid attention; the checkpoints are what vLLM and the trainers already load | The families the framework implements, with its own model code and a conversion; a family it lacks is a port | Any |
| Precision | bf16, FP8 and MXFP8 (torchao) | Also NVFP4 (Megatron-Bridge, through Transformer Engine) | As the framework |
| Throughput | Below the frameworks' at first; how far is measured, not assumed | Published | As the framework |
| Scale | One node with FSDP2; several with HSDP once the scaling note's phase 6 exists | Up to hundreds of GPUs with TP, PP and CP | Any |
| What it gives | One ledger, one monitor, one spend limit; lineage from pretraining to SFT to RL; scale to zero | The same, with a faster loop and FP4 | A base model and nothing else: no lineage, curves elsewhere, pods left running if forgotten |

What the throughput difference is worth, at the sizes of this note: a run's compute bill scales with 1/MFU. At 30%
against 45% MFU, the 125M to 350M runs cost $10 to $140 more; the 1B on 100B tokens about $700 more; continued
pretraining of a 9B on 20B tokens about $1,300 more. NVFP4 on B200 might take another tenth (1B) to a quarter (9B) off
FP8's cost (a guess: [sizing](#sizing)). Below a thousand dollars a run, the work of wrapping a framework costs more
than it saves; above a few thousand, or for FP4 at all, it is the way.

### Recommendation

**(a), behind a contract that (b) can implement later, with Megatron-Bridge as that (b).**

1. A run kind `pretrain` (from nothing, or continued from any checkpoint or base model), corpora and mixtures as
   records, a data source library, and the stretch contract below.
2. Its first implementation: the resident FSDP2 full-weight trainer being built, with a packed text path, in bf16 and
   FP8 (torchao). Continued pretraining of the models the platform already trains and serves comes first, and those
   are Hugging Face models; the RTX 5080 and one H100 test the whole path before a node is rented.
3. Megatron-Bridge as a second implementation of the same contract, for NVFP4, for TP, PP or CP, or when a run's size
   makes a faster loop pay ([phases](#phases)). torchtitan instead if, by then, torchao's NVFP4 training is no longer
   a prototype and Megatron's weight is the larger cost.

Not (c): importing a checkpoint as a base model works today, and stays possible for models pretrained elsewhere, but
it gives up what the platform is for here (one record from pretraining through SFT to RL, bounded spend, pods that go
away).

## How it maps onto the roles

### The run kind

A **`pretrain`** run trains a model on a mixture of corpora for a number of tokens, from nothing (`trainer.model`
names a configuration and `start` is empty and `pretrain.init` is `scratch`) or from a checkpoint or base model
(`start`, or the base model of `trainer.model`). Continued pretraining is the same kind with a start; there is no
separate kind for it. Its driver builds the run's trainer and nothing else, decides stretches, and records them.

Not `imitate` with a streaming dataset: an imitate run is one step over a dataset of episodes' segments, rendered by a
renderer, with guidance cut and supervision kinds (`importance`, `supervised`, `teacher`). None of that is
pretraining's, and pretraining's schedules, cursor, stretches and spike handling are none of imitation's. They share
the objective's family, not the run.

### Corpora and mixtures

- **A corpus** is a record in the ledger's `corpora` table, appended once under a random id, and a manifest blob: the
  tokenizer (name, revision, a digest of its files), the token width (16 or 32 bits), the end-of-text token, each
  shard's blob reference, tokens and documents, and what it was made from (a Hugging Face dataset and revision, or
  files; the filtering and deduplication done upstream, in words; the licence). A name in the registry, as datasets
  have. `rollout corpus make` tokenizes text into shards as Ray tasks on CPUs and records the corpus; a corpus
  tokenized elsewhere (in Megatron's or NumPy's flat format) is recorded by `rollout corpus add` without being read
  again.
- **A held-out corpus** is a corpus too, or a split of one, named for evaluation.
- **A mixture** is a named, versioned bundle, as a suite is: corpora with weights, and optionally phases (from token
  `t`, these weights), for a final stretch of better data or for continued pretraining's replay. A run names a
  mixture version; its start record says the version, so a mixture edited later does not change a run.
- **Scoring sets** (for multiple-choice evals) are records of the same kind: prompts and their choices, tokenized
  with the run's tokenizer, and the right answer.

### The data source

Not a role of its own: a library every rank of the trainer runs, reading its slice. The driver gives a plan (the
mixture version, the seed, the sequence length, the global batch in sequences, the first sample and how many), and
the source turns sample numbers into tokens:

- sample `i` of the run is a pure function of the plan and `i`: which corpus (by the mixture's weights at that point,
  drawn so that every window of the run holds them), which sequence of it (a seeded permutation of that corpus's
  sequences), and so which shard and offset;
- rank `r` of `w` reads samples `i` with `i mod w = r` of each global batch, so a run resumed on another number of GPUs
  reads the same batches;
- shards are fetched from the blob store ahead of need into the pod's disk, a bounded cache, and memory-mapped;
- updates listed as skipped (after a spike) are passed over by number, so the record says what was not trained on.

The cursor is the number of updates taken. It is in each stretch's record and each checkpoint's metrics, not hidden
in a loader's state.

### The stretch

```py
@dataclass(frozen=True)
class Stretch:
    """Updates `first` to `last` of a run, from a checkpoint's full state (or the initial weights), into a new one."""

    parent: Files | None      # full state: weights, optimizer, the update it was taken at; None: the initial weights
    into: Path
    first: int                # the first update's number
    last: int                 # the last's: the stretch ends with a checkpoint after it
    data: DataPlan            # mixture version, seed, sequence length, global batch, skipped updates
    schedule: Schedule        # the learning rate at every update number: warmup, stable, decay (pure)
    precision: Precision      # bf16, fp8 (tensorwise, rowwise), mxfp8, nvfp4; the layers kept in bf16; when to leave it
    serving_copy: bool        # also write Hugging Face safetensors in bf16, for evals and later runs
    scoring: Sequence[str]    # held-out corpora and scoring sets to score at its end


class Pretrainer(Protocol):
    async def stretch(self, asked: Stretch, progress: Callable[[Progress], None]) -> Stretched: ...
```

- **One stretch is one ledger step.** The driver appends the step (`runs/RUN/steps`: its parent, the checkpoint it
  makes, its first and last update, the plan) before the trainer is called, and the checkpoint is the commit, as for
  a training step. A driver started again finds the step decided and not made and asks for it again, from the same
  parent. Retention thins the saves as it does now.
- **A stretch lasts as long as a checkpoint interval:** updates enough for 30 to 60 minutes (the driver sizes it from
  the measured update time), so a failure loses at most that.
- **The trainer is resident.** It keeps its state between stretches and loads `parent` only when it does not hold it
  (a new pod, a restart). The pod's training service asks for a stretch as it asks for a step now: by the checkpoint
  it makes, idempotent by it, polled until made.
- **Progress** (the newest update, loss, gradient norm, learning rate, tokens a second, MFU) goes into the trainer's
  heartbeat every few seconds, for the monitor's live view. At the stretch's end the trainer writes the stretch's
  per-update series as a blob and returns a summary, which the checkpoint's metrics hold. No ledger record per update.
- **Spikes.** The trainer skips an update whose gradient norm is far above its recent values (the threshold a
  setting; OLMo-core skips past six standard deviations over the last 128 updates) and counts it. If the loss stays high, the stretch fails saying so; the driver records the failure, and
  asks for the stretch again from the last good checkpoint with the offending updates' batches skipped, up to a
  limit. Every skip is in the record.

The FSDP2 trainer implements both `Trainer` (for RL and imitation) and `Pretrainer`. A Megatron-Bridge trainer kind
would implement `Pretrainer` alone for pretraining, and `Trainer` too once the scaling note's phase 7 needs it.

### Settings

What a `pretrain` run's settings would say (none of these exist):

```toml
kind = "pretrain"
"trainer.provider" = "runpod-h100x8"         # a runpod-trainer with gpu_count = 8
"trainer.model" = "Qwen/Qwen3-4B"            # a base model, or a configuration with pretrain.init = "scratch"
start = "qwen3-4b-sft-3"                     # continued pretraining from a checkpoint (else from trainer.model)
"pretrain.mixture" = "chemistry-cpt@2"       # 95% domain text, 5% replay of general text
"pretrain.tokens" = 10_000_000_000
"pretrain.sequence" = 4096
"pretrain.batch_tokens" = 2_097_152          # 512 sequences an update
"pretrain.schedule" = "wsd"                  # or "cosine", "constant"
"pretrain.learning_rate" = 3e-5
"pretrain.warmup_tokens" = 200_000_000
"pretrain.decay_fraction" = 0.15
"pretrain.z_loss" = 1e-4
"pretrain.precision" = "fp8-rowwise"         # or "bf16", "fp8-tensorwise", "mxfp8" (B200), "nvfp4" (B200, B300)
"pretrain.high_precision_from" = 0.82        # nvfp4: the share of the run after which it trains in bf16
"pretrain.checkpoint_minutes" = 45
"pretrain.held_out" = ["chemistry-held-out", "fineweb-edu-held-out"]
"pretrain.scoring" = ["hellaswag-1k", "arc-easy"]
"trainer.max_gradient_norm" = 1.0
"limits.spend" = 600
```

The check refuses a mixture whose corpora's tokenizer differs from the model's, a batch the GPUs cannot divide, a
precision the provider's GPUs or the trainer kind cannot train in (NVFP4 on anything but B200 and B300, or on the
own trainer), and a run whose estimated cost (tokens left over the measured or estimated tokens a second, times the price) is above
`limits.spend`.

### Demand and placement

A `pretrain` run's gang is its trainer: one bundle of the provider's GPUs, on one node. At home that is the local
GPU (small models and tests); on RunPod a `runpod-trainer` of `gpu_count` GPUs, leased for the run and released at its
end. A trainer across nodes is the scaling note's spanning part: a placement group of its own, `STRICT_SPREAD`, and on
RunPod an Instant Cluster as one lease, its ranks set up by the training service inside it. Nothing in the run kind
names a topology: the provider says the GPUs, the trainer says how it shards over them.

A run outlives a pod. When the lease is lost (the pod's host fails, the pod stops beating), the driver releases what is
left, leases a new pod of the same provider, and asks again for the stretch that was not made, from the newest
checkpoint with full state. A run on a pod of another GPU count resumes too, because the checkpoint reshards and the
data source's order does not depend on the number of ranks.

Corpora that pods read live in the pods' store (R2, which charges nothing for reads leaving it); a pod caches the
shards it reads on its own disk.

### The monitor

A `pretrain` run's page shows, in place of groups and episodes:

- tokens trained of the run's total, the time and the dollars to the end at the current rate, and the spend so far;
- the loss against tokens trained (each update's, smoothed), and each held-out corpus's loss at each checkpoint;
- the learning rate, the gradient norm, skipped updates and spikes, marked on the curve;
- tokens a second per GPU and MFU, and the precision it trains in (and when it changes);
- the tokens drawn from each corpus of the mixture, against what its weights asked;
- the checkpoints along the line, which hold full state and which only weights, and each one's scores.

Its checkpoints are a lane in the checkpoint graph whose root is the base model, or, from nothing, the configuration
(`scratch:CONFIG-DIGEST`). SFT, imitation and RL runs that start from one of them hang under it, so a model's line
reads from its pretraining to its last RL step.

## Phases

Each phase waits for its trigger.

| Phase | Build | Trigger |
|---|---|---|
| 1. Corpora and the data source | The `corpora` table and manifests; `rollout corpus make` and `add`; mixtures as named versioned bundles; the data source library, tested on CPU: the same batches at any world size, after any resume, with skips | The first continued pretraining someone wants to run. It needs no GPU and no change to a trainer |
| 2. Pretraining on one node, bf16 and FP8 | The `pretrain` kind and its settings; the stretch contract and its records; the packed path in the resident FSDP2 trainer (bf16, compile, FlashAttention, chunked cross-entropy, schedules, clipping, z-loss, skipping); FP8 and MXFP8 through torchao; async DCP saves and the serving copy; held-out loss per stretch; a new lease after a lost one; the monitor's page | Phase 1, and the FSDP2 trainer merged. Accepted when a small model from nothing reproduces a published curve on one node (GPT-2 124M on FineWeb 10B tokens to validation loss 3.28, the target llm.c and modded-nanogpt use), its FP8 twin lands within noise of it, and a continued pretraining of a Qwen3 model on a domain corpus lowers its domain loss while its general held-out loss stays within a set margin |
| 3. Scoring evals and the estimate | Scoring sets and their multiple-choice evals at each stretch's end; the planner's estimate of a pretraining run (tokens a second measured per model size and GPU type) in the New run form and the check | A run longer than a day, where held-out loss alone does not say whether it is worth going on |
| 4. Megatron-Bridge as a second trainer | A trainer kind over Megatron-Core through Megatron-Bridge behind the stretch contract: the image, the launcher, corpora read as Megatron's token shards or through the data source, progress reports, checkpoints into stretches, `AutoBridge` to Hugging Face; Transformer Engine's NVFP4 and MXFP8 recipes, with the layers kept in bf16 and the switch to bf16 late in the run recorded | The first run that wants NVFP4 (B200 or B300); or one that needs TP, PP or CP (sequences of 64K and up); or a planned run whose cost at the own trainer's measured throughput is more than about $1,000 above its cost at Megatron's or torchtitan's published throughput for that size. Accepted when an NVFP4 run and its FP8 twin of the same model, data and seed differ in loss by no more than the 1% to 1.5% the NVFP4 paper reports |
| 5. Several nodes | An Instant Cluster as one trainer lease, HSDP across its nodes (the scaling note's phase 6) | A run that would take more than about a week on one node, or a model whose state does not fit one node |
| Deferred | Tokenizer training; deduplication and filtering inside the platform (they stay upstream); elastic training; mixture-of-experts pretraining; training across pods without a fabric | Each when a run needs it |

## Sizing

What the runs of the brief cost and how long they take. Estimates to choose between shapes, good to perhaps a factor
of 1.5, not measurements.

**The assumptions.**

- **FLOPs:** `6 · N · tokens` for `N` parameters including embeddings, plus attention: 20% more for the 125M and 350M
  models at 2,048 tokens a sequence (`7.2 · N` a token), 10% more for 1B and up at 4,096 (`6.6 · N`). MFU is counted
  against these.
- **MFU:** 40% on H100 for dense models of 1B and up, 35% for 125M and 350M, in bf16 with a tuned loop. Published
  figures, all on H100 at bf16: LLM Foundry 41% at 1B (44K tokens a second a GPU) and 46% at 7B (10.6K) on 8 GPUs;
  OLMo-core 55K tokens a second a GPU at 1B and 10K at 7B; torchtitan 6.7K at Llama 3.1 8B with `torch.compile`
  (33% to 42% MFU, on H100s held to 500 W); nanochat's 561M model at about 48% (1.1M tokens a second on 8); llm.c's
  1.6B at about 50% (33.6B tokens in 24 hours on 8). The small models' 35% is below what llm.c's 124M reached (10B
  tokens in 45 minutes on 8 H100, about 460K tokens a second a GPU). The own trainer of phase 2 may reach less at
  first; [weighed](#weighed) prices the difference.
- **GPUs:** H100 SXM, 989 dense bf16 TFLOPS; one pod of 8 with NVLink. H200 has the same compute at a third more a
  GPU-hour ($4.59), so it is dearer per token on models this small. B200 has 2.25 dense bf16 PFLOPS and trains about
  2.1 times as many tokens a second as H100 (NVIDIA's Megatron-Bridge figures for Llama 3 8B at FP8) at 1.95 times
  the price ($6.79): the same cost per token, in less time. H100 is what the table prices. RTX 5080: about 110 dense
  bf16 TFLOPS with fp32 accumulation (an estimate scaled from the RTX 5090's 209 by tensor cores and clock; NVIDIA's
  whitepaper is the source to check), at 35% MFU.
- **Prices:** RunPod secure cloud, $3.49 an H100 SXM GPU-hour (pricing page updated 2026-09-27, read 2026-10-05), so
  $27.92 an hour for 8. The local GPU is counted as free. Storage (R2: about $0.015 a GB-month), egress (none from R2),
  pod start, failed stretches and evals are left out; they add a few percent.
- **Memory:** full weights with Adam take 16 bytes a parameter plus activations: a 1B model's 16 GB fits one H100
  (plain data parallel); a 9B model's 144 GB needs FSDP2 over at least 4 H100 (over 8: 18 GB a GPU).

| Run | FLOPs | Tokens a second | Time on 8 H100 | Cost | Fits |
|---|---|---|---|---|---|
| 125M on 10B tokens | 9.0e18 | about 3.1M | 54 min | $25 | One GPU: 7 h on one H100, the same $25; 2.7 days on the RTX 5080 |
| 125M on 50B tokens | 4.5e19 | about 3.1M | 4.5 h | $126 | One GPU (1.5 days on one H100) or one node |
| 350M on 10B tokens | 2.5e19 | about 1.1M | 2.5 h | $71 | One GPU: 20 h on one H100; 7.6 days on the RTX 5080 |
| 350M on 50B tokens | 1.3e20 | about 1.1M | 13 h | $353 | One node (4.2 days on one H100) |
| 1B on 100B tokens | 6.6e20 | about 480K | 2.4 days | $1,620 | One node; one H100 would take 19 days; not the RTX 5080 (16 GB of state on a 16 GB card) |
| 4B, continued, 5B tokens | 1.3e20 | about 120K | 12 h | $323 | One node, FSDP2 (64 GB of state) |
| 4B, continued, 20B tokens | 5.3e20 | about 120K | 1.9 days | $1,290 | One node |
| 9B, continued, 5B tokens | 3.0e20 | about 53K | 1.1 days | $728 | One node, FSDP2 (144 GB of state) |
| 9B, continued, 20B tokens | 1.2e21 | about 53K | 4.3 days | $2,910 | One node; an Instant Cluster of two nodes halves the time at about the same cost |

How the rows are computed: tokens a second = 8 × 989e12 × MFU / FLOPs a token; time = tokens / that; cost = hours ×
$27.92. For example, 1B on 100B tokens: 6.6e9 FLOPs a token, 8 × 989e12 × 0.40 / 6.6e9 = 480K tokens a second (60K a
GPU, between LLM Foundry's 44K and OLMo-core's 55K and a little above both), 100e9 / 480e3 s = 58 hours, × $27.92 =
$1,620.

Checked against published runs: llm.c trained GPT-2 1.6B on 33.6B tokens in 24 hours on one 8 H100 node for $672;
nanochat trains a 561M model on 11.2B tokens in 3 hours 51 minutes on 8 H100 for $92 (at $24 an hour). The table
gives 1.6B on 33.6B tokens 31 hours and 561M on 11.2B tokens 4.5 hours: slower than those tuned loops, as intended.

**In FP8 and FP4.** The two larger runs again, by precision and GPU. The FP8 gains are the published ones for the
nearest size (OLMo-core's +18% at 1B; +28% at 8B with FSDP2); B200 is taken at 2.1 times H100's tokens a second at the
same precision (NVIDIA's Llama 3 8B figures) and $6.79 a GPU-hour; MXFP8 at +15% (1B) and +25% (9B, torchao's 8B
figure) over B200's bf16. **The NVFP4 rows are guesses**, not measurements: +10% at 1B and +30% at 9B over MXFP8, from
FP4's doubled peak less the bf16 layers, the transforms and the quantizing; nothing is published at these sizes.

| Run | Precision | Time on 8 GPUs | Cost |
|---|---|---|---|
| 1B on 100B tokens | H100, bf16 | 2.4 days | $1,620 |
| | H100, FP8 | 2.0 days | $1,370 |
| | B200, bf16 | 1.2 days | $1,500 |
| | B200, MXFP8 | 1.0 day | $1,300 |
| | B200, NVFP4 (a guess) | 0.9 days | about $1,200 |
| 9B, continued, 20B tokens | H100, bf16 | 4.3 days | $2,910 |
| | H100, FP8 | 3.4 days | $2,270 |
| | B200, bf16 | 2.1 days | $2,700 |
| | B200, MXFP8 | 1.7 days | $2,160 |
| | B200, NVFP4 (a guess) | 1.3 days | about $1,660 |

For the 125M and 350M runs, low precision is not worth pricing: their matrix multiplies are too small for FP8 to
save more than a few percent, and FP4's recipe is unshown there.

What the table says:

- **Every run here fits one 8-GPU node**, and the small ones one GPU. Several nodes only shorten the wall clock, at
  the same cost or a little more.
- **The RTX 5080 is for tests and the smallest experiments:** a 125M model on a few billion tokens in a day or two;
  the loop, the data source and resumes checked end to end before paying for a pod.
- **Compute dominates;** data movement does not. 8 H100 at 3M tokens a second read 12 MB/s of 32-bit tokens; a 100B
  token corpus is 400 GB in R2, about $6 a month. Tokenizing it is CPU work done once (a CPU pod or the home cluster's
  Ray workers), not on the GPU pod.
- **Checkpoints are affordable to save often.** A 9B model's full state is about 110 GB; saved asynchronously every
  45 minutes and kept as the newest two plus one in twenty, it is a few hundred gigabytes in the store at a time.
- **Low precision pays at the large end.** FP8 takes a sixth to a fifth off the 1B and 9B runs; B200 costs the same
  per token as H100 at the same precision and halves the time; NVFP4 on B200 may take the 9B run to about 60% of its
  bf16 H100 cost, which is what phase 4 would measure.
- **Out of reach:** a 7B model from nothing on a compute-optimal 1T to 2T tokens is 4.6e22 to 9.2e22 FLOPs, about
  32,000 to 65,000 H100-hours in bf16 ($110,000 to $230,000; perhaps half that in NVFP4 on B200). Pretraining from nothing here means small models; larger ones are
  continued from open weights.

## Open decisions

1. **A run kind of its own, or `imitate` with a streaming dataset.** *Recommendation:* `pretrain`, for pretraining
   from nothing and continued pretraining alike. Imitation's renderer, guidance cuts and supervision kinds are not
   pretraining's, and pretraining's schedule, cursor and stretches are not imitation's; they share the likelihood
   family only.
2. **A new trainer contract, or `Trainer.step` stretched.** *Recommendation:* `Pretrainer.stretch` beside `Trainer`,
   implemented by the same resident FSDP2 trainer; a stretch is a ledger step, so restarts, retention and lineage
   work as they do.
3. **Who reads the data.** *Recommendation:* the trainer's ranks, from the blob store, by a plan the driver gives; the
   cursor is the update count in the record, so resuming never depends on a loader's hidden state.
4. **Per-update metrics.** *Recommendation:* live in the trainer's heartbeat, and a series blob per stretch with a
   summary in the checkpoint's metrics; no ledger record per update.
5. **Mixtures.** *Recommendation:* named, versioned bundles, as suites are; a run's start names the version.
6. **Own loop or a framework first.** *Recommendation:* the own resident FSDP2 trainer first, in bf16 and FP8,
   because the first runs are continued pretraining of Hugging Face models the platform already serves, the
   difference in cost is under a thousand dollars a run at those sizes, and the platform side gets tested on the
   RTX 5080 and one H100 before a node is rented; a framework behind the same contract when phase 4's trigger fires.
7. **Which framework for (b).** *Recommendation:* Megatron-Core through Megatron-Bridge: Transformer Engine is where
   NVFP4 training is supported today, `AutoBridge` converts the platform's families both ways, and the scaling note
   wants it for large MoE anyway, so there is one wrapped framework rather than two. torchtitan if torchao's NVFP4
   training leaves prototype first and Megatron's container and build are the larger cost.
8. **FP4 from the first run.** If the first pretraining run is to be NVFP4, phase 4 comes before phase 2's own
   text path. *Recommendation:* no. Build phase 2 in bf16 and FP8 first, since its data source, stretches, monitor
   and resumes are what phase 4 rests on and they are cheapest to prove on H100 and at home; then run NVFP4 on B200
   beside an FP8 twin of the same model, data and seed, so the ledger holds the comparison the NVFP4 paper made.
   For models under a few billion parameters, expect the FP4 run to be no cheaper than the FP8 one.
9. **The tokenizer of a model from nothing.** *Recommendation:* an existing family's, never trained here until a run
   needs one. Qwen3's for a model meant to go on to SFT and RL, so its renderer applies; at that vocabulary a model
   should be 350M or more, or its embeddings are most of it. GPT-2's only for experiments that reproduce a published
   curve.
10. **Evals in the loop.** *Recommendation:* held-out loss and multiple-choice scoring sets computed by the trainer at
    each stretch's end, on the GPUs it holds; generative suites as ordinary eval runs of chosen serving copies, after
    SFT. An environment cannot score by likelihood (task code sees no logprobs), so scoring sets are not suites.
11. **Precision by default.** *Recommendation:* bf16 for the smallest models, FP8 rowwise on H100 and MXFP8 on B200
    from 1B up, with fp32 master weights and optimizer state; NVFP4 only through the Megatron-Bridge trainer, on B200
    or B300, with the published recipe (16% of linear layers in bf16, the last 18% of the run in higher precision)
    until measurements here say otherwise. The run's start records the format, the recipe and the layers left out.
12. **How often to checkpoint.** *Recommendation:* by the clock (`pretrain.checkpoint_minutes`, 45 by default),
    asynchronous, full state each time, thinned by the run's retention (the newest two and one in twenty kept), with
    a serving copy every few checkpoints and at the end of each phase of the schedule.
13. **Full weights or LoRA for continued pretraining.** *Recommendation:* full weights. In continued pretraining on
    20B tokens of code and of math, LoRA fell short of full fine-tuning at every rank tried (up to 256), though it
    forgot less; it suits fine-tuning, not adding knowledge.
14. **Where corpora live.** *Recommendation:* in the pods' store (R2), cached on the pod's disk; network volumes only
    if the same corpus is read by many runs in one region.

## Sources

All read on 2026-10-05 unless a date is given.

**Frameworks**

- torchtitan: [repository](https://github.com/pytorch/torchtitan), [releases](https://github.com/pytorch/torchtitan/releases)
  and [PyPI](https://pypi.org/project/torchtitan/) (0.3.0, 2026-09-03), [release policy](https://github.com/pytorch/torchtitan/blob/main/docs/release.md),
  [extension](https://github.com/pytorch/torchtitan/blob/main/docs/extension.md), [models](https://github.com/pytorch/torchtitan/tree/main/torchtitan/models)
  and [Qwen3.5](https://github.com/pytorch/torchtitan/tree/main/torchtitan/models/qwen3_5),
  [data loading](https://github.com/pytorch/torchtitan/blob/main/torchtitan/components/data/README.md),
  [checkpoints](https://github.com/pytorch/torchtitan/blob/main/docs/checkpoint.md),
  [evaluation](https://github.com/pytorch/torchtitan/blob/main/docs/evaluation.md),
  [metrics](https://github.com/pytorch/torchtitan/blob/main/docs/metrics.md),
  [experiments (torchft)](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments);
  [the paper](https://arxiv.org/abs/2410.06511) (ICLR 2025);
  benchmarks: [Llama 3 on H100, 2024-12](https://github.com/pytorch/torchtitan/blob/main/benchmarks/llama3_h100_202412_torchtitan.md),
  [async TP, 2025-06](https://github.com/pytorch/torchtitan/blob/main/benchmarks/asyncTP_llama3_h100_2025-06_torchtitan.md),
  [Llama 3 8B on H200, 2025-06](https://github.com/pytorch/torchtitan/blob/main/benchmarks/llama3-8b_h200_202506_trainy-whitefiber.md)
- [torchforge](https://github.com/meta-pytorch/torchforge) (development paused for torchtitan)
- Megatron: [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) ("up to 47% MFU on H100"),
  [datasets](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/datasets_readme.html),
  [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) (0.6.0, 2026-08-19),
  [its performance archive](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary-archive.html) (Llama 3 8B, H100 and B200, FP8)
- OLMo-core: [repository](https://github.com/allenai/OLMo-core) and [PyPI](https://pypi.org/project/ai2-olmo-core/) (3.0.0, 2026-10-01),
  throughput in the READMEs of [v1.0.6](https://raw.githubusercontent.com/allenai/OLMo-core/v1.0.6/README.md) and
  [v1.6.2](https://raw.githubusercontent.com/allenai/OLMo-core/v1.6.2/README.md),
  [data](https://olmo-core.readthedocs.io/en/latest/data/index.html),
  [data loading](https://olmo-core.readthedocs.io/en/latest/guides/data_loading.html),
  [callbacks](https://olmo-core.readthedocs.io/en/latest/train/callbacks.html),
  [evaluator](https://olmo-core.readthedocs.io/en/latest/eval/evaluator.html),
  [skip-step optimizers](https://olmo-core.readthedocs.io/en/latest/optim.html)
- [nanotron's Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (2025-02-19)
- [Levanter](https://github.com/stanford-crfm/levanter) (merged into Marin, November 2025)
- MosaicML: [LLM Foundry's benchmarks](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/benchmarking/README.md),
  [LLM Foundry releases](https://github.com/mosaicml/llm-foundry/releases), [Composer releases](https://github.com/mosaicml/composer/releases),
  [Streaming](https://github.com/mosaicml/streaming)
- [GPT-NeoX](https://github.com/EleutherAI/gpt-neox), [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- PyTorch's FSDP results: [maximizing training throughput](https://pytorch.org/blog/maximizing-training/),
  [with torch.compile](https://pytorch.org/blog/maximizing-training-throughput/)

**Precision**

- [Float8 and MXFP8 training in torchao](https://docs.pytorch.org/ao/main/_sources/workflows/training.md.txt) (gains on
  H100 and B200; NVFP4 training marked prototype), [NVFP4 training tracker](https://github.com/pytorch/ao/issues/3293) (opened 2025-11-05, open)
- [Training with float8 and FSDP2](https://pytorch.org/blog/training-using-float8-fsdp2/) (2024-11-25),
  [float8 rowwise on 2K H200](https://pytorch.org/blog/accelerating-large-scale-training-and-convergence-with-pytorch-float8-rowwise-on-crusoe-2k-h200s/) (2025-04),
  [MXFP8 on 1,856 B200, 1.22x to 1.28x](https://pytorch.org/blog/accelerating-2k-scale-pre-training-up-to-1-28x-with-torchao-mxfp8-and-torchtitan-on-crusoe-b200-cluster/) (2025-09-03),
  [MXFP8 and DeepEP for DeepSeek-V3 on B200](https://pytorch.org/blog/enabling-up-to-41-faster-pre-training-mxfp8-and-deepep-for-deepseek-v3-on-b200-with-torchtitan/) (2026-05-04; small matrix multiplies)
- [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149) (NVIDIA; v2 2026-03-04)
- [Transformer Engine's NVFP4](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html) (training on SM 10.0 and 10.3)
- [Megatron-Bridge's mixed precision recipes](https://docs.nvidia.com/nemo/megatron-bridge/0.6.0/training/mixed-precision.html) (`bf16_with_nvfp4_mixed`, layers pinned in bf16)
- NVIDIA on Blackwell's training speed: [MLPerf Training v5.0](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-up-to-2-6x-higher-performance-in-mlperf-training-v5-0/) (2025-06),
  [v5.1](https://developer.nvidia.com/blog/nvidia-blackwell-enables-3x-faster-training-and-nearly-2x-training-performance-per-dollar-than-previous-gen-architecture/) (2025-12-11)

**Recipes, stability and runs**

- [PaLM](https://arxiv.org/abs/2204.02311) (z-loss; restarting before a spike and skipping 200 to 500 batches),
  [OPT-175B](https://arxiv.org/abs/2205.01068), [OLMo 2](https://arxiv.org/abs/2501.00656) and
  [OLMo 2 32B](https://allenai.org/blog/olmo2-32B) (2025-03-13), [Llama 3](https://arxiv.org/abs/2407.21783)
  (document masking, interruptions, 38% to 43% MFU),
  [small-scale proxies for instability](https://arxiv.org/abs/2309.14322)
- Schedules: [MiniCPM (WSD)](https://arxiv.org/abs/2404.06395),
  [scaling laws beyond fixed training durations](https://arxiv.org/abs/2405.18392), [Chinchilla](https://arxiv.org/abs/2203.15556)
- Continued pretraining: [simple and scalable strategies](https://arxiv.org/abs/2403.08763),
  [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673)
- Small models: [llm.c GPT-2 124M](https://github.com/karpathy/llm.c/discussions/481) (2024-05),
  [llm.c GPT-2 1.6B](https://github.com/karpathy/llm.c/discussions/677) (2024-07-11),
  [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt),
  [nanochat](https://github.com/karpathy/nanochat) and its [first speedrun](https://github.com/karpathy/nanochat/discussions/1) (2025-10-13),
  [SmolLM2](https://arxiv.org/abs/2502.02737), [SmolLM3](https://huggingface.co/blog/smollm3),
  [TinyLlama](https://github.com/jzhang38/TinyLlama), [Pythia](https://arxiv.org/abs/2304.01373)

**Data**

- [datatrove](https://github.com/huggingface/datatrove), [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb),
  [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
  [DCLM-baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0), [Dolma](https://huggingface.co/datasets/allenai/dolma)
- Hybrid attention kernels: [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)

**Checkpoints**

- [Reducing checkpointing times](https://pytorch.org/blog/reducing-checkpointing-times/) (2024-06),
  [6x faster async checkpointing](https://pytorch.org/blog/6x-faster-async-checkpointing/) (2025-04)

**Hardware and prices**

- NVIDIA: [H100](https://www.nvidia.com/en-us/data-center/h100/), [H200](https://www.nvidia.com/en-us/data-center/h200/),
  [HGX B200](https://www.nvidia.com/en-us/data-center/hgx/),
  [the RTX Blackwell whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf),
  [the RTX 5090's bf16 figure](https://forums.developer.nvidia.com/t/rtx-5090-peak-bf16-tensor-tflops/350543)
- RunPod: [pricing](https://www.runpod.io/pricing) (updated 2026-09-27),
  [Instant Clusters](https://docs.runpod.io/instant-clusters) and [their product page](https://www.runpod.io/product/instant-clusters)
  (updated 2026-08-27)
- [Cloudflare R2 pricing](https://developers.cloudflare.com/r2/pricing/)
