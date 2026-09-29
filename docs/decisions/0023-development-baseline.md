# 0023 — Development baseline

Status: **Accepted** · Date: 2026-09-27 · Supersedes [0011](0011-stack.md)

## Context

Development starts with the core library in the local profile ([0016](0016-layers-and-profiles.md)). A few choices
block writing the first line of code; everything else stays deferred (P14).

## Decision

**Language and packaging**
- **Python 3.13.** vLLM, SGLang, DBOS and `renderers` all resolve for 3.13 (checked 2026-09-27).
- One distribution package, **`rollout`**, in a `src/` layout, managed with **uv**. Subpackages follow the layers:
  `rollout.core` first; `rollout.recorder`, `rollout.rollouts`, `rollout.trajectories`, `rollout.inference`,
  `rollout.durable` are added by the milestone that needs them.
- Heavy dependencies are **extras**: `vllm`, `recorder` (`renderers`), `durable` (DBOS), `mcp`, `anthropic`,
  `openai`. The base install needs no GPU. The `sglang` extra arrives with its adapter (M3); it conflicts with
  `vllm` (different torch pins) and, at 0.5.9, with `durable` (it pins a `pyyaml` older than DBOS accepts), so it
  is declared with uv `conflicts` entries or installed in its own environment.
- Ruff lints and formats `src/` and `tests/`; code in `docs/` is illustrative and hand-aligned.
- The durable pump is written in **Python** (answers the open question in [durability](../durability/README.md)).
  Languages for the platform layer and the environment system are chosen when those are designed.

**Code conventions**
- Contract types (canonical content, events, effects, `Sample`, specifications) are **frozen pydantic v2 models**:
  validation, JSON serialization and JSON Schema for tool arguments come from one library. Protocols are
  `typing.Protocol`.
- Digests use RFC 8785 canonical JSON (the `rfc8785` package).
- Full type names, never abbreviations.
- Tooling: `ruff` (lint and format), `pyright` in strict mode for `src/`, `pytest` with `pytest-asyncio`.

**Models**
- Target family: **Qwen3 / Qwen3.5**. The recorder's first renderers are for this family.
- Local development: Qwen3-0.6B for tests; **Qwen3-1.7B trained with LoRA** for the local RL loop (rank 32 on every
  attention and MLP projection). Full fine-tuning of 1.7B does not fit a 16 GB GPU; LoRA fits with the trainer
  resident next to the engine ([LoRA spike](#lora-spike)). After each step the adapter is merged into the base
  weights, pushed into the engine in place, and unmerged, so the engine serves plain weights. Mixture-of-experts plumbing (routing replay) is developed against a small, randomly initialized
  Qwen3-MoE configuration; real MoE models (e.g. Qwen3-30B-A3B) run in the cluster profile.

**Local engine**
- **vLLM first** for the local profile (colocated sleep / wake and in-process weight updates are its well-trodden
  path), behind the engine-adapter protocol; the SGLang adapter follows for the cluster profile. Confirmed by the
  local engine spike (below).
- The colocated local profile runs vLLM's engine core **in the trainer's process**
  (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) and updates weights with `collective_rpc` calling
  `model_runner.model.load_weights`. vLLM starts workers with `spawn` on WSL, so every entry point guards
  `if __name__ == "__main__":`.

**Reference trainer**
- A **minimal in-repository GRPO trainer** (plain PyTorch with `peft` LoRA, importance-weighted, routing-replay-aware
  later) exists
  to validate the `Sample` contract and colocated weight updates end to end. It is test and example code, not the
  production trainer, whose choice stays deferred.

**Small open questions settled so they do not block**
- `score` does not run after a hook raised; the run fails with `TASK_ERROR` and the sample's `outcome` records it.
- Rows of one rollout job are admitted by strict priority, then arrival order. One cursor per job; per-consumer
  cursors are added when several trainers share a job.

## Local engine spike

2026-09-27 · RTX 5080 16 GB (sm_120), WSL2 · vLLM 0.30.0, torch 2.13.0+cu130, transformers 5.17.0 · Qwen3-0.6B,
bf16, eager mode, 25% of GPU memory, `logprobs_mode="processed_logprobs"`.

| Check | Result |
|---|---|
| Load with sleep mode enabled | Pass (16 s warm) |
| Tokens in, per-token logprobs out | Pass |
| Sampled logprobs vs. a prefill re-score at the same weights | Pass: max \|Δ\| 0.146, mean 0.016 |
| Sleep (level 1) and wake | Pass: free memory 10.4 → 14.4 GiB asleep |
| In-place weight update | Pass: a perturbed weight changes logprobs (max Δ 4.98); restoring it restores them exactly |

Decode and prefill kernels disagree slightly in bf16, so the recorder keeps the logprobs observed while sampling
(they are the behavior distribution); `score_tokens` is not a substitute for them. Parity checks compare means,
not maxima.

Rerun at the planned settings (2026-09-28, 45% of GPU memory, 4,096-token context): all five checks pass for
Qwen3-0.6B and Qwen3-1.7B. Sleeping frees 7.1 GiB with either model (7.2–7.3 → 14.4 GiB free). Sampled vs. re-scored
logprobs for 1.7B: mean |Δ| 0.011, max 0.257.

## Colocation spike

2026-09-28 · same machine and versions · one local RL cycle with Qwen3-1.7B: generate a group of 8 × 192 tokens →
engine sleeps (level 1) → AdamW step (bf16 states, gradient checkpointing) → weights pushed with
`collective_rpc(load_weights)` while only the engine's weights are awake → trainer offloaded → KV cache wakes →
generate at the new weights → trainer recomputes that sample's logprobs.

| Measurement | Result |
|---|---|
| The cycle runs end to end | yes; waking weights and KV cache separately (`wake_up(tags=...)`) works |
| Behavior logprobs vs. trainer recomputation, after a real update | mean \|Δ\| 0.013, max 0.365; per-token ratio 0.69–1.18 |
| Generation throughput, warm, group of 8 | 402 tokens/s eager; **1,284 tokens/s with CUDA graphs** |

Trainer memory for one step on 8 sequences of 224 tokens (device: 15.9 GiB; about 1.5 GiB is taken by the desktop
and the sleeping engine):

| Configuration | Trainable parameters | Peak allocated | Fits | Seconds per step |
|---|---|---|---|---|
| Qwen3-1.7B, full, micro-batch 8 | 1,721 M | 19.2 GiB | no | 12.7 (spilled) |
| Qwen3-1.7B, full, micro-batch 1 | 1,721 M | 16.1 GiB | no | 4.5 (spilled) |
| Qwen3-1.7B, embeddings and head frozen, micro-batch 1 | 1,409 M | 13.8 GiB | yes, narrowly | 1.3 |
| Qwen3-0.6B, full, micro-batch 8 | 596 M | 10.8 GiB | yes | 0.6 |

Findings:

- **WSL hides out-of-memory.** The Windows driver spills GPU allocations into system memory instead of failing, so
  an oversized step runs slowly rather than crashing. Local runs must check peak allocation against device memory;
  on native Linux the same step fails.
- **Per-token ratios reach ±30% at identical weights** (bf16 kernels), so the reference trainer uses truncated
  importance weights from the start.
- **Use CUDA graphs locally** (3.2× throughput). Their interaction with sleep, wake and weight updates is not yet
  measured.

## LoRA spike

2026-09-28 · same machine and versions · Qwen3-1.7B, CUDA graphs on, LoRA rank 32 (34.9 M trainable parameters) on
q/k/v/o and gate/up/down projections, `peft` 0.21. Two cycles of: two LoRA steps on a group of 8 × 192 tokens (engine
asleep) → `merge_adapter()` → wake the engine's weights → `collective_rpc(load_weights)` with the merged weights →
`unmerge_adapter()` → wake the KV cache → sample at the new weights → recompute its logprobs with the adapter active.

| Measurement | Result |
|---|---|
| Train step, micro-batch 8 | peak 13.4 GiB allocated (most of it full-vocabulary logits in fp32); 0.4–0.7 s |
| Engine and trainer resident together | 11.0 GiB used with the engine awake; no offload needed |
| Merge, push and wake | 0.5–0.7 s per cycle |
| Behavior vs. trainer logprobs | cycle 1: mean \|Δ\| 0.006, max 0.104 · cycle 2: mean 0.004, max 0.042 (17-token samples) |
| Update reaches the engine | engine vs. base model: mean \|Δ\| 0.387 |
| CUDA graphs across sleep, wake and weight updates | work; 1,294 tokens/s for a group of 8 |

The samples were short (17 tokens), so the agreement figures rest on few tokens; M1's exit criterion measures them
over full episodes. Computing logprobs in chunks instead of materializing fp32 logits for the whole batch would cut
the step's peak memory substantially.

## Consequences

- `docs/development/plan.md` orders the work into milestones with exit criteria.
- ADR-0011's Go-based stack is superseded.
