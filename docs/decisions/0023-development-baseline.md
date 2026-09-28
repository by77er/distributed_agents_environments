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
- Local development: Qwen3-0.6B for tests, Qwen3-1.7B for local training (colocated train + inference must fit a
  16 GB GPU). Mixture-of-experts plumbing (routing replay) is developed against a small, randomly initialized
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
- A **minimal in-repository GRPO trainer** (plain PyTorch, importance-weighted, routing-replay-aware later) exists
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

## Consequences

- `docs/development/plan.md` orders the work into milestones with exit criteria.
- ADR-0011's Go-based stack is superseded.
