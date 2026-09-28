# Our interface vs RL environment specifications and RL frameworks

Status: **Draft** · 2026-09-27 · Question: how do our Task / Agent / Template / rollout / `Sample` / recorder
interfaces compare with RL environment specifications and RL training frameworks, and how should the system
operate for synchronous RL, asynchronous RL and multi-agent training?

Scope of the design under evaluation: [harness](../core/harness/README.md) ([task](../core/harness/task.md),
[agent](../core/harness/agent.md)), [rollouts](../core/rollouts/README.md),
[trajectories](../core/trajectories/README.md), [recorder](../core/recorder/README.md),
[inference](../inference/README.md), ADRs [0007](../decisions/0007-active-recorder.md),
[0008](../decisions/0008-mid-rollout-policy-change.md), [0009](../decisions/0009-stale-kv-importance-sampling.md),
[0012](../decisions/0012-task-agent-loop.md), [0014](../decisions/0014-no-forks-template-recipes.md),
[0015](../decisions/0015-rollout-interface.md).

**Method.** Repositories were shallow-cloned at `main` between 2026-09-24 and 2026-09-28 and their source read
directly; versions and licenses come from PyPI metadata. Statements taken from papers, blogs or READMEs rather than
code say so. Anything not verified is marked **UNVERIFIED**. Sections 2–3 are facts with sources; sections 4–8
are analysis and recommendations.

---

## 1. Summary of recommendations

1. **The core shape holds up.** The field has converged on what we already designed: token-in/token-out via
   renderers with a `bridge_to_next_turn` step, observations masked out, per-token (or per-span) weights versions,
   in-flight weight updates, and stale-KV reuse corrected by importance sampling. Keep Task/Agent, the recorder and
   the five-operation rollout interface.
2. **Extend `Sample`** with the following. All are data, not algorithm:
   - `turns` spans (per-turn boundaries, finish reasons, reply identifiers);
   - optional `routed_experts` (MoE routing replay, R3);
   - `truncation_reason`, and an `outcome` that separates failed, cancelled and excluded;
   - a `mask_reason` set by the task (`run.exclude_from_training`);
   - multimodal references;
   - an optional `transcript_reference` for group judges.
3. **Extend `RolloutJobs`** with:
   - `Cancel(run_ids | label selector)`, for oversampling, dynamic sampling and staleness sweeps;
   - server-side label filtering on `Samples` plus per-ticket awaitables;
   - optional gang admission of a ticket;
   - **multiple trainable channels per job** with `Publish(channel, …)`.
4. **Replace `weights_reference` with a `WeightsSource`** covering checkpoint URI, delta, LoRA adapter and a
   trainer-participating distributed transfer (NCCL / RDMA / checkpoint-engine). Split publish into stage and commit.
5. **Allow cross-run rewards within a job.** A parent may reward its descendants' slots, and descendant samples
   are assembled when the root run is terminal. This is what makes swarms trainable.
6. **Add a prefill-only `ScoreTokens` operation** to the engine adapter and an optional per-job *enrichment*
   (teacher or reference logprobs from a named channel). This covers on-policy distillation and reference-model KL
   without shipping extra models to the trainer.
7. **Adapters in, in priority order:**
   1. Harbor task directories (the de facto interchange);
   2. verifiers v1;
   3. OpenEnv;
   4. Inspect;
   5. Gymnasium/TextArena (in-process).

   Adapters out, in priority order:
   1. **miles/slime** (custom rollout function);
   2. **SkyRL** `GeneratorInterface`;
   3. a Tinker-`Datum` exporter;
   4. AReaL and verl after that.
8. **Staleness control stays in the caller**, as a ten-line AReaL-style admission rule in the adapter. The rollout
   side still knows no algorithm.
9. The `renderers` library our recorder relies on is Prime Intellect's `renderers` (Apache-2.0, 0.1.11). That
   answers the recorder's license open question; model coverage still needs checking per target family.

---

## 2. Environment specifications survey (facts)

### 2.1 Gymnasium (Farama; PyPI 1.3.0, MIT)

- **Definition.** `reset(*, seed, options) -> (observation, info)` and
  `step(action) -> (observation, reward, terminated, truncated, info)`, with typed `observation_space` and
  `action_space` ([core.py](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/core.py)).
- **Termination semantics.**
  - `terminated` means the MDP reached a terminal state.
  - `truncated` means a condition outside the MDP stopped the episode (for example a time limit).
  - The single `done` flag was removed in v0.26 because the distinction matters for bootstrapping.
- **Vector environments.** They declare an autoreset mode (`NEXT_STEP` / `SAME_STEP` / `DISABLED`)
  ([vector_env.py](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/vector/vector_env.py)).
- **Scope.** No messages, tokens, tools, groups or sandboxing. Reward is a per-step scalar. Multi-agent lives
  separately in PettingZoo.

### 2.2 OpenEnv (Meta PyTorch + Hugging Face; PyPI `openenv` 0.6.0, BSD-3-Clause)

- **Repository.** It moved to [huggingface/OpenEnv](https://github.com/huggingface/OpenEnv).
- **Environment interface.** `Environment[ActT, ObsT, StateT]` has these members
  ([interfaces.py](https://github.com/huggingface/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py)):
  - `reset(seed, episode_id, **kwargs) -> Observation`;
  - `step(action, timeout_s) -> Observation`;
  - a `state` property;
  - async variants;
  - an optional `rubric`.
- **Types.** They are Pydantic models:
  - `Observation(done, reward, metadata)`;
  - `Action(metadata)`;
  - `State(episode_id, step_count)`.
- **Serving.** Each environment is a FastAPI server in its own Docker image, exposing `/reset`, `/step`, `/state`,
  `/ws` (WebSocket) and `/mcp`.
- **Distribution.** Environments are distributed as Hugging Face Spaces with an `openenv.yaml` manifest.
- **Container providers.** Local Docker, Kubernetes, Daytona, Modal, HF Sandbox and others.
- **RFCs.**
  - RFC 003 proposes that *all* agent actions be MCP tool calls, via `ListToolsAction` / `CallToolAction`.
  - RFC 004 defines composable `Rubric`s, including trajectory rubrics with discounting and LLM judges.
  - RFC 005 runs agentic harnesses (Claude Code and similar) inside the environment container.
  - RFC 012 (2026-09) states that "training export still requires exact engine tokens and processed log
    probabilities".
- **Episode end and multi-agent.** Episode end is a single `done`, with no terminated/truncated split. There is no
  first-class multi-agent support.
- **Trainer integrations.** TRL `environment_factory`, SkyRL, ART, Unsloth and torchforge
  ([README](https://github.com/huggingface/OpenEnv)).

### 2.3 Prime Intellect verifiers and the Environments Hub (verifiers 0.3.1, MIT)

**Legacy v0 stack**
- It is now removed from `main` ([docs/overview.md](https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/overview.md)).
- Its classes were `SingleTurnEnv`, `MultiTurnEnv`, `ToolEnv`, `StatefulToolEnv` and `SandboxEnv`.
- `MultiTurnEnv` had `env_response(messages, state) -> Messages` and `is_completed(state)`.
- `Rubric` held weighted reward functions. Parameter names in the plural marked *group* reward functions.

**v1** ([docs/v1](https://github.com/PrimeIntellect-ai/verifiers/tree/main/docs/v1))
- `TaskData` is an immutable row.
- `Task` has hooks `setup`, `score(trace, runtime)` and `finalize`, plus `@vf.reward(weight=…)`, `@vf.metric` and
  `@vf.stop`.
- `Taskset` loads tasks. A `Toolset` holds `@vf.tool` methods, served to harnesses **as MCP servers**.
- A `Harness` is the program the model runs in: Claude Code, Codex, mini-swe-agent, terminus-2, and others.
- `Agent` = harness × model × runtime.
- `Env.run(task, agents)` holds multi-agent control flow. Variants include `UserSimEnv`, `AgenticJudgeEnv`,
  `IsolatedVerifierEnv` (which scores in a fresh runtime) and `BestOfNEnv`.

**Token capture**
- A harness never calls the provider directly. An **interception server** speaks the harness's native API.
- For training, `TrainClient` renders prompts to token ids with the `renderers` library and calls a vLLM token-in
  `/generate` route.
- Multi-turn prompts are extended with `renderer.bridge_to_next_turn(prev_prompt_ids, prev_completion_ids,
  new_messages)`, falling back to a full re-render.
- A trace is a **message graph**: each `MessageNode` holds delta `token_ids`, a `mask`, `logprobs`,
  `routed_experts` and so on. Concatenating along a path reproduces the exact tokens
  ([graph.py](https://github.com/PrimeIntellect-ai/verifiers/blob/main/verifiers/v1/graph.py)).
- This is structurally the same as our session tree.

**Runtimes and tasksets**
- Runtimes: subprocess, docker, podman, apptainer, Prime Sandboxes, Modal.
- Built-in tasksets wrap Harbor, OpenEnv, NeMo Gym and TextArena.

**The `renderers` package** (PyPI 0.1.11, Apache-2.0, [repo](https://github.com/PrimeIntellect-ai/renderers))
- It is Prime Intellect's.
- API: `render_ids`, `parse_response`, `bridge_to_next_turn`.
- Hand-written renderers cover DeepSeek V3/V4/R1, GLM-4.5/5, gpt-oss, Kimi K2/K2.5, Llama 3, MiniMax-M2,
  Nemotron 3, Qwen3/3.5/3.6/3.8 and others.
- Other models get a Jinja fallback whose bridge returns `None`, forcing a re-render.

### 2.4 Inspect AI (UK AISI; 0.3.271, MIT)

- **Task.** `Task(dataset, setup, solver, cleanup, scorer, sandbox, epochs, message_limit, token_limit,
  turn_limit, time_limit, …)`.
- **Sample.** `Sample(input, target, id, metadata, sandbox, files, setup)`.
- **Solvers and agents.** A solver is `(TaskState, Generate) -> TaskState`. Agents such as `react(...)` operate on
  `AgentState`.
- **Scoring.** A scorer returns `Score(value, answer, explanation, metadata)`, where `value` may be a mapping
  (multi-dimensional). `Epochs(n, reducer)` repeats samples with reducers such as `pass_at` or `mean`
  ([task.py](https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/_eval/task/task.py)).
- **Sandboxes.** They are a pluggable provider interface (`@sandboxenv`: `exec`, `read_file`, `write_file`, …). The
  built-ins are Docker and local; externals include k8s, Daytona, Modal and EC2.
- **Model interface.** Message-level only: logprobs can be requested, but token ids cannot be captured for
  training.
- **RL use.** No first-party trainer integration was found. The Tinker cookbook uses Inspect for evaluation only.
  Use of Inspect tasks directly as RL environments is **UNVERIFIED**.

### 2.5 TextArena (0.7.4, MIT) and SPIRAL

- **TextArena.**
  - Interface: `reset(num_players, seed)`, `step(action: str) -> (done, info)`,
    `get_observation() -> (player_id, observation)`, `close() -> (rewards per player, game info)`
    ([core.py](https://github.com/LeonGuertler/TextArena/blob/main/textarena/core.py)).
  - Multi-player is native: the environment says whose turn it is.
  - Rewards are terminal and per player.
- **SPIRAL** ([arXiv 2506.24119](https://arxiv.org/abs/2506.24119)).
  - One shared policy is trained by self-play on zero-sum games.
  - It uses **role-conditioned advantage estimation**: a per-(game, role) exponential-moving-average baseline,
    because the group mean in a zero-sum game is ≈ 0 whatever the policy does.
  - prime-rl implements this as `RAEAlgorithm`.

### 2.6 SkyRL-gym (0.4.0, MIT)

- **Interface.** `BaseTextEnv.init(prompt)` and
  `step(action: str) -> BaseTextEnvStepOutput(observations, reward, done, metadata, postprocessed_action)`, with
  tool groups ([base_text_env.py](https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-gym/skyrl_gym/envs/base_text_env.py)).
- **Level.** Environments are message-level. Tokens are handled by the generator (section 3.6).

### 2.7 Terminal-Bench and Harbor (harbor 0.23.0, Apache-2.0)

- **Repository.** Harbor moved to [harbor-framework/harbor](https://github.com/harbor-framework/harbor).
- **Task directory.**
  - It contains `instruction.md`, `task.toml`, `environment/` (Dockerfile or compose), `solution/solve.sh` and
    `tests/test.sh`.
  - `task.toml` declares resources, network mode, allowed hosts, MCP servers and timeouts.
  - The verifier writes `/logs/verifier/reward.txt` (a scalar) or `reward.json` (named rewards).
- **Agents.** `BaseAgent.run(instruction, environment, context)` covers installed agents (claude_code, codex,
  openhands, mini_swe_agent, …) and built-ins (terminus_2, oracle).
- **Environment providers.** About 30, including Docker, Daytona, E2B, Modal, GKE, EC2 and Kata. They implement
  `BaseEnvironment` with `start`, `stop`, `upload`, `download` and `exec`.
- **Trajectories.**
  - They use ATIF (up to v1.8).
  - `AgentContext.rollout_details` carries per-turn `prompt_token_ids`, `completion_token_ids` and `logprobs`, plus
    extras such as routed experts.
- **Adoption.** Harbor itself is not a trainer, but SkyRL, the Tinker cookbook, verifiers, NeMo Gym, OpenEnv and
  rLLM all consume Harbor tasks. There is a registry at [hub.harborframework.com](https://hub.harborframework.com).

### 2.8 SWE-bench-family task images

- **SWE-bench** (`swebench` 5.0.2, MIT).
  - It uses per-instance images `sweb.eval.{arch}.{instance_id}`.
  - `FAIL_TO_PASS` / `PASS_TO_PASS` test lists give FULL / PARTIAL / NO resolution
    ([grading.py](https://github.com/SWE-bench/SWE-bench/blob/main/swebench/harness/grading.py)).
  - The RL reward is usually binary "resolved".
- **R2E-Gym.**
  - Rows carry `docker_image`, with the repository at `/testbed`.
  - Grading compares test output to an expected JSON (per rLLM's builder).
- **SWE-smith** builds per-repository images. **SWE-rebench** ships about 21k pre-built images. Instance counts
  for SWE-Gym and SWE-smith are **UNVERIFIED**.
- **DeepSWE** (rLLM; Qwen3-32B, about 4.5k R2E-Gym environments, 42.2% pass@1 on SWE-bench Verified) masked
  timed-out or max-context trajectories ("compact filtering") ([blog](https://www.together.ai/blog/deepswe)).

### 2.9 Tinker cookbook (tinker-cookbook 0.5.7, Apache-2.0)

- **Environment interface.**
  - `Env.initial_observation() -> (Observation, StopCondition)`.
  - `Env.step(action: list[int]) -> StepResult(reward, episode_done, next_observation, next_stop_condition)`.
  - Observations are `tinker.ModelInput`, so **the environment speaks tokens**. A `MessageEnv` layer adds messages.
- **Groups.** `EnvGroupBuilder.make_envs()` and `compute_group_rewards(trajectory_group, env_group)` make groups
  first-class, including pairwise and zero-sum games.
  - For example, `TwoPlayerCoordinator` shares one TextArena game between two `Env`s so that both players train
    ([rl/types.py](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/rl/types.py)).
- **Renderers.** They advertise `has_extension_property`. When a renderer has it, successive turns merge into one
  datum. Qwen3 with thinking stripped lacks it.
- **Relationship to `renderers`.** The cookbook depends on `tml-renderers` (Apache-2.0). This is distinct from
  Prime Intellect's `renderers`.

### 2.10 NeMo Gym (0.6.0, Apache-2.0) and others

- **NeMo Gym.**
  - It runs as three kinds of FastAPI servers: resources servers (`/seed_session`, `/verify` →
    `{reward, mask_sample}`), agent servers and model servers.
  - The model server returns `prompt_token_ids`, `generation_token_ids` and `generation_log_probs`.
  - v0.6.0 advertises external agent harnesses during RL "while preserving exact token IDs"
    ([repo](https://github.com/NVIDIA-NeMo/Gym)).
  - `mask_sample` is an environment-side "do not train on this" flag.
- **MCP as the action interface** is now common, confirmed in code in:
  - OpenEnv RFC 003;
  - verifiers v1 Toolsets;
  - Harbor `mcp_servers`;
  - NeMo Gym auto-exposure;
  - ART `art.mcp`;
  - TRL `environment_factory`.

---

## 3. RL frameworks' rollout integration survey (facts)

### 3.1 slime (THUDM, Apache-2.0) and miles (radixark fork, Apache-2.0)

**slime plug-in points**
- **Rollout function.** `--rollout-function-path` replaces all orchestration:
  `generate_rollout(args, rollout_id, data_source, evaluation) -> list[list[Sample]]`, i.e. a list of groups.
- **Generate function.** `--custom-generate-function-path` replaces one sample's generation:
  `async (args, sample, sampling_params) -> Sample | list[Sample]`. A list is the fan-out used for compaction or
  sub-agents; siblings share `rollout_id`.
- **Reward functions.** `--custom-rm-path` (per-sample) and `--group-rm` (batched).
- **Filtering.** `--dynamic-sampling-filter-path`, for example `check_reward_nonzero_std`, with
  `--over-sampling-batch-size`.
- Sources: [arguments.py](https://github.com/THUDM/slime/blob/main/slime/utils/arguments.py),
  [customization.md](https://github.com/THUDM/slime/blob/main/docs/en/get_started/customization.md).

**slime `Sample`** ([types.py](https://github.com/THUDM/slime/blob/main/slime/utils/types.py))
- Token fields: `tokens` (prompt + response), `response_length`, `loss_mask` (over the response only),
  `rollout_log_probs`.
- `reward: float | dict`.
- `weight_versions`, one per generate call.
- `rollout_routed_experts` (R3) and `teacher_log_probs` (on-policy distillation).
- `status ∈ {PENDING, COMPLETED, TRUNCATED, ABORTED, FAILED}`, and `session_id` for consistent-hash routing.

**Multi-turn**
- Each trajectory is one concatenated sequence.
- `append_response_tokens(..., trainable=False)` gives observation tokens loss mask 0 and logprob 0.

**Agent adapters**
- `OpenAIAdapter` and `AnthropicAdapter` render messages, call SGLang with `input_ids` and export the sampled ids
  and logprobs without retokenizing ([agent.md](https://github.com/THUDM/slime/blob/main/docs/en/get_started/agent.md)).
- Routing uses an `X-SMG-Routing-Key` header.

**Asynchrony**
- `--partial-rollout` aborts servers when enough groups exist and re-queues partial samples, which resume next round
  with KV recomputed. `--mask-offpolicy-in-partial-rollout` masks the earlier tokens.
- A fully asynchronous rollout function exists. In it, groups containing aborted samples restart from scratch.

**Corrections and MoE**
- `--use-tis`, custom TIS/MIS functions, and `--use-rollout-routing-replay` (R3).
- Reference logprobs are computed by the Megatron trainer.

**Weight sync**
- `--update-weight-transport nccl|disk` and `--update-weight-mode full|delta`.
- SGLang `update_weights_from_{tensor,distributed,disk}`, each tagged with `weight_version`.
- External engines attach with `--rollout-external-engine-addrs`.

**miles additions** ([repo](https://github.com/radixark/miles); v0.1 shipped 2026-08, tech report arXiv 2609.08368, not read in full)
- **Version spans.** `weight_versions` becomes a list of `WeightVersionSpan(version, abs_start, abs_end)` per
  call, so versions are tracked per token span even across mid-request updates.
- **Three plug-in layers.** Agent function → generate function → rollout function. Connectors exist for Harbor,
  NeMo Gym, OpenEnv, verifiers, HUD, Strands and τ-bench.
- **TITO session server.**
  - Agents call `{base_url}/sessions/<id>/v1/chat/completions`.
  - The server keeps prompt and output ids per turn and tokenizes only appended suffixes.
  - v2 is an append-only tree yielding one `Sample` per leaf, which is our session tree again.
- **Fully asynchronous mode.**
  - `--max-weight-staleness`.
  - `--pause-generation-mode abort|retract|in_place`. `in_place` resumes on stale KV; `retract` recomputes.
- **Other.**
  - `--true-on-policy`: bitwise train/inference logprob match.
  - `--update-weight-transfer-mode broadcast|p2p|disk-delta`, with P2P RDMA.
  - Multi-policy training (`train_multi_policy.py`).
  - A Tinker-compatible server.

### 3.2 AReaL (inclusionAI / areal-project, Apache-2.0; package 2.1.0)

- **Paper** ([arXiv 2505.24298](https://arxiv.org/abs/2505.24298)):
  - Interruptible rollouts: abort, load new weights, *recompute* KV, continue.
  - Staleness bound ⌊(N_r − 1)/B⌋ ≤ i + η.
  - Decoupled PPO against a proximal policy.
  - 2.77× speedup over synchronous training.
- **Plug-in.** `RolloutWorkflow.arun_episode(engine, data) -> dict | None`, or any agent `run(data, base_url=…)`
  that uses an OpenAI client ([workflow_api.py](https://github.com/inclusionAI/AReaL/blob/main/areal/api/workflow_api.py)).
- **Engine API.** `InferenceEngine.agenerate(ModelRequest) -> ModelResponse`, where `ModelResponse` carries
  `output_tokens`, `output_logprobs`, **`output_versions` (per token)**, `stop_reason` and `routed_experts`.
- **Abort and resume.** `RemoteInfEngine.agenerate` loops on abort: it pins the version, extends versions per token
  and resubmits `input_ids + output_tokens`.
- **Multi-turn.** Sequences are concatenated with loss mask 0 on inputs and version −1 on non-sampled tokens.
- **`ArealOpenAI`.**
  - It records every call with `set_reward` and `apply_reward_discount(turn_discount)`.
  - `export_interactions('concat' | 'individual')` returns either prefix-tree leaves or one sample per turn.
- **Groups and filtering.** `submit(..., group_size, drop_incomplete_group, min_usable_group_size,
  should_accept_fn)`.
- **Staleness and corrections.**
  - Staleness is `max_head_offpolicyness` (η); η = 0 means synchronous.
  - Proximal logprobs are recomputed or approximated log-linearly.
  - Token or sequence importance sampling, plus IcePop/KPop presets.
- **Weight sync and versions.**
  - Weight sync: disk, xccl or awex.
  - AReaL 2.0 (2026-07) is split into training, inference, agent and weight-update services. Its tech-report
    arXiv id comes from the README only (**UNVERIFIED**).

### 3.3 verl (volcengine, Apache-2.0; 0.10.0.dev)

- **Agent loop.**
  - `AgentLoopBase.run(sampling_params, **kwargs) -> AgentLoopOutput(prompt_ids, response_ids, response_mask,
    response_logprobs, routed_experts, multi_modal_data, reward_score, num_turns, …)`
    ([agent_loop.py](https://github.com/volcengine/verl/blob/main/verl/experimental/agent_loop/agent_loop.py)).
  - `ToolAgentLoop` is a state machine.
  - `BaseInteraction` (user simulation) has been removed from the tree.
- **Tokens.**
  - "Continuous Token" is now the only tokenization path. A per-family `ContinuousTokenBuilder` merges context and
    assistant tokens, and inserted boundary tokens get no loss and no logprob
    ([agent_loop.rst](https://github.com/volcengine/verl/blob/main/docs/advance/agent_loop.rst)).
  - Sticky sessions bind a request to one server.
- **Groups and filtering.** Group size is `rollout.n`. DAPO filtering is `algorithm.filter_groups`.
- **Asynchrony.**
  - The fully-async and one-step-off recipes have Rollouter, MessageQueue, Trainer and ParameterSynchronizer
    components, with `staleness_threshold` and `partial_rollout`.
  - The newer v1 async trainer (`sync | colocate_async | separate_async`) keeps tokens and logprobs across aborts
    and reconstructs KV.
  - Versions are tracked **per trajectory** (min and max global step), not per token
    ([v1_async_trainer.md](https://github.com/volcengine/verl/blob/main/docs/advance/v1_async_trainer.md)).
- **Rollout correction.** `rollout_is: token | sequence`, rejection sampling, and presets for TIS, MIS, Geo-RS and
  K3-RS ([rollout_corr](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)).
- **Weight sync.** `checkpoint_engine` backends: naive, nccl, nixl, kimi_ckpt_engine, mooncake, delta.
- **MoE.** R3 via `enable_rollout_routing_replay`.

### 3.4 PipelineRL (ServiceNow, Apache-2.0)

- **In-flight weight updates** ([arXiv 2509.19128](https://arxiv.org/abs/2509.19128)). vLLM is paused with
  `mode="keep", clear_cache=False` during an NCCL broadcast, so **stale KV is kept**
  ([vllm1.py](https://github.com/ServiceNow/PipelineRL/blob/main/pipelinerl/vllm1.py)).
- **Plug-in.** `generate_rollout(cfg, llm, problem, session) -> RolloutResult(training_texts, …)`.
- **Multi-turn is segmented.** There is one `TrainingText` per LLM call. The prompt is retokenized from messages;
  the output ids are the engine's.
- **Versions and lag.** Versions are stamped per rollout at its start. Lag is bounded by `max_lag` in samples.
- **Loss.** The default loss clamps the new/old ratio (a truncated importance-sampling-style weight) against the
  actor's logprobs.

### 3.5 SkyRL (NovaSky, Apache-2.0; v0.3.0)

- **Generator interface.** `GeneratorInterface.generate(GeneratorInput) -> GeneratorOutput`
  ([base.py](https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl/train/generators/base.py)).
  - `GeneratorOutput` holds `prompt_token_ids`, `response_ids`, `rewards` (per trajectory **or per token**),
    `loss_masks`, `rollout_logprobs`, `stop_reasons`, `trajectory_ids`, `rollout_expert_indices` (R3),
    `is_last_step` (step-wise mode) and `pixel_values`.
- **Token handling.** It is token-in/token-out by default.
  - Step-wise mode emits one row per LLM turn.
  - Step-wise mode is motivated explicitly by retokenization drift and non-append contexts.
  - Advantages are computed on the last step and broadcast to the other steps.
- **Fully asynchronous trainer.**
  - `max_staleness_steps` (default 4), with capacity gating so that no trajectory is dropped.
  - In-flight updates with **stale KV reused by default**; a per-version cache salt is optional.
- **Off-policy corrections.** Geometric sequence masking, token TIS and IcePop.
- **Integrations.** SkyRL implements the Tinker API. The Harbor generator consumes Harbor's per-turn
  `rollout_details`.

### 3.6 rLLM (Agentica, Apache-2.0)

- **Plug-in.**
  - `AgentFlow.run(task, AgentConfig(base_url, session_uid, …)) -> Episode | Trajectory`, plus an `Evaluator`.
  - Backends: verl, Tinker, Fireworks ([types.py](https://github.com/rllm-org/rllm/blob/main/rllm/types.py)).
  - `Step` carries `prompt_ids`, `response_ids`, `logprobs`, `routing_matrices` and `weight_version`.
- **Token capture.** `rllm-model-gateway` is a session proxy that records vLLM `return_token_ids`. Its
  `TokenAccumulator` uses `renderers.bridge_to_next_turn`
  ([gateway](https://github.com/rllm-org/rllm/tree/main/rllm-model-gateway)).
- **Rows.** Prefix-extending steps merge into one row; a non-cumulative step starts a new row.
- **Groups and advantages.**
  - Groups are keyed `task_id:trajectory_name`, so each **role gets its own group**.
  - `estimator_map` sets the estimator per role.
- **Asynchrony.** `SyncCoordinator` applies the "verl/AReaL" quota: `(1 + staleness) × sync_step × mini_batch`.

### 3.7 Agent Lightning (Microsoft, MIT; v1.0)

- **Architecture.** API Gateway (rollouts and events) → Rollout Controller (subprocess or Kubernetes Job per agent)
  → verl trainer.
- **Plug-in.** Agents change only their base URL:
  `/proxy/rollout/{rollout_id}/attempt/{attempt_id}/mode/train/openai/v1/...`.
- **Rewards.** Agents POST them to `AGL_EVENT_URL`.
- **Token capture.** The proxy forces `return_token_ids`, `logprobs` and optionally `return_routed_experts`, and
  **overrides temperature** ([proxy.py](https://github.com/microsoft/agent-lightning/blob/main/agentlightning/server/proxy.py)).
  This is our recorder's compatibility-endpoint `OVERRIDE` mode.
- **Aggregation.** `trajectory` merges exact prefix extensions and masks observations; `transition` gives one row
  per call.
- **Asynchrony.** A group is used only when all siblings finish. Before a weight update the gateway pauses new
  requests and drains in-flight ones.
- **Retokenization drift.** It is documented in the vLLM blog
  ["No More Retokenization Drift"](https://vllm.ai/blog/2025-10-22-agent-lightning). The three causes are
  non-unique BPE splits, tool-call parse and re-render, and chat-template history rewriting.

### 3.8 OpenRLHF (Apache-2.0)

- **Agent interface.** `AgentExecutorBase.execute(prompt, label, sampling_params, …)` and
  `AgentInstanceBase.reset/step`
  ([agent.py](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/utils/agent.py)).
- **Tokens.** It works in token space. Environment feedback text is tokenized alone, without the chat template,
  and the environment must emit template markers itself.
- **Asynchrony.** `--train.async_enable` and `--train.partial_rollout_enable` pause/resume vLLM around the
  broadcast, so "in-flight samples may contain tokens from both old and new weights".
- **Importance-sampling correction.** Level `token | seq`, mode `mask | clip`, gating `ratio | binary_kl | tv`.
- **Reference model.** A separate Ray actor group computes reference logprobs.

### 3.9 prime-rl (Prime Intellect, Apache-2.0)

- **Architecture.**
  - Inference is vLLM behind a token-in route using `renderers`.
  - A CPU orchestrator owns the verifiers environments and computes advantages.
  - The FSDP2 trainer receives packed batches over ZMQ or the filesystem
    ([overview.md](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/overview.md)).
- **Asynchrony.**
  - The pipeline is always at least one step off-policy.
  - `max_off_policy_steps` (default 8) is measured from the **oldest version that generated any token** of a
    rollout.
  - Stale episodes are **cancelled in flight**.
- **Advantages and filtering.**
  - The default advantage is `reward − mean(group)`, with no std normalization.
  - Zero-advantage filtering refills the batch, and there is a difficulty-pool curriculum.
  - Other algorithms: `rae`, `hierarchical_grpo`, `opd` (on-policy distillation), `echo`.
- **Losses and replay.**
  - The loss uses inference logprobs directly (IPO-style masked IS), with IcePop optional.
  - Reference KL comes from a vLLM "reference scoring" server using `prompt_logprobs`.
  - **Router Replay** is available.
  - **Sampling Replay** lets vLLM return top-p/top-k sampling masks so the trainer can renormalize
    ([inference.md](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/inference.md)).
- **Weight broadcast.** `filesystem | nccl | nixl`.

### 3.10 NeMo RL (NVIDIA, Apache-2.0)

- **Environment interface.**
  `EnvironmentInterface.step(message_log_batch, metadata) -> EnvironmentReturn(observations, metadata,
  next_stop_strings, rewards (tensor or dict), terminateds, answers)`
  ([interfaces.py](https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/environments/interfaces.py)).
- **NeMo Gym path.** The rollout **asserts** token contiguity ("Non-contiguous messages found!").
- **Asynchronous GRPO.**
  - `max_trajectory_age_steps`, `in_flight_weight_updates`, `recompute_kv_cache_after_weight_updates`.
  - Importance-sampling correction is mandatory
    ([async-grpo.md](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/async-grpo.md)).
- **Losses.** TIS, IcePop, seq-mask-TIS, GSPO and CISPO. The trainer computes `prev_logprobs` and reference
  logprobs.
- **Refit.** CUDA IPC, NCCL, sparse deltas (ZMQ or S3) and NIXL.

### 3.11 ART (OpenPipe, Apache-2.0), TRL (Apache-2.0), Tinker (Apache-2.0)

- **ART.**
  - Plug-in is a user `rollout(model, scenario) -> Trajectory`, grouped with
    `gather_trajectory_groups(TrajectoryGroup(...))`.
  - `ruler_score_group` gives **relative LLM-judge rewards within a group**.
  - Raw HTTP exchanges are captured with vLLM `return_token_ids`, merged by prefix.
  - `PipelineTrainer(max_steps_off_policy=4)`.
  - Backends: local (Unsloth/vLLM, LoRA), Megatron, Tinker, serverless
    ([ART](https://github.com/OpenPipe/ART)).
- **TRL.**
  - `GRPOTrainer(reward_funcs, tools, rollout_func, environment_factory)`. The environment's public methods become
    tools; `reset(**row)` and an optional `get_reward()`.
  - `rollout_func` must return `prompt_ids`, `completion_ids` and `logprobs`, plus an optional `env_mask`.
  - Multi-turn needs a prefix-preserving chat template.
  - vLLM importance-sampling correction is on by default (`sequence_mask`).
  - The experimental async GRPO **retokenizes every turn** and forks a row on a rewrite
    ([grpo_trainer.py](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)).
- **Tinker.**
  - `forward_backward(data: list[Datum], loss_fn ∈ {cross_entropy, importance_sampling, ppo, cispo, dro})`.
  - RL losses take **exactly** `target_tokens`, `logprobs` (the sampler's q) and `advantages`.
  - `SamplingClient.compute_logprobs` scores sequences (used for distillation).
  - `AsyncConfig(max_steps_off_policy)` **requeues** stale groups
    ([losses](https://tinker-docs.thinkingmachines.ai/tinker/losses/)).

### 3.12 Weight transfer and correction literature

- **Weight transfer.** Kimi [checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine) (MIT) reports
  about 20 s to update a 1T-parameter model across thousands of GPUs, using pipelined H2D → broadcast → CUDA-IPC or
  Mooncake P2P.
- **Mismatch and correction.**
  - Yao et al. ([blog](https://fengyao.notion.site/off-policy-rl)): token-level TIS.
  - ["When Speed Kills Stability"](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)
    argues for sequence-level MIS.
  - IcePop ([arXiv 2510.18855](https://arxiv.org/abs/2510.18855)).
  - R3 ([arXiv 2510.11370](https://arxiv.org/abs/2510.11370)). SGLang ≥ 0.5.13 returns routed experts natively
    ([SGLang issue #16379](https://github.com/sgl-project/sglang/issues/16379)).
- **Partial rollouts.** APRIL ([arXiv 2509.18521](https://arxiv.org/abs/2509.18521)) over-provisions, stops at the
  target count, and continues partials the next step.
- **Keeping sequences across updates.** Magistral ([arXiv 2506.10910](https://arxiv.org/abs/2506.10910)) keeps
  in-flight sequences across updates without recomputing KV.
- **Distillation.** Thinking Machines'
  [on-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) uses per-token teacher
  logprobs (reverse KL as the advantage).
- **Multi-agent.**
  - MARTI ([repo](https://github.com/TsinghuaC3I/MARTI)): per-agent trainers.
  - M-GRPO ([arXiv 2511.13288](https://arxiv.org/abs/2511.13288)): separate main and sub-agent policies.
  - AgentFlow / Flow-GRPO ([arXiv 2510.05592](https://arxiv.org/abs/2510.05592)): trains only the planner and
    broadcasts the outcome.
  - Self-play SWE-RL ([arXiv 2512.18552](https://arxiv.org/abs/2512.18552)): one policy injects and repairs bugs.

---

## 4. Comparison matrix (analysis)

### 4.1 Environment specifications vs our Task

| | Unit of definition | Action / observation | Tools | Reward | Ending | Groups | Isolation | Multi-agent |
|---|---|---|---|---|---|---|---|---|
| **Ours** | `Task` class + `Template` recipe; row = parameters | canonical messages; agent chooses context | `@tool`, imports, environment-provided MCP | `Observation.reward` per reply, `run.reward(slot, key)`, `score` | `TERMINATED` / `TRUNCATED` | caller labels | microVM per environment; task host sandboxed | slots in one run; `spawn` child runs |
| Gymnasium | `Env` object | arbitrary spaces | — | per step | terminated / truncated | — | in-process | PettingZoo |
| OpenEnv | container + HTTP server | Pydantic action / observation | MCP (RFC 003) | per step + rubrics | `done` | trainer | container | — |
| verifiers v1 | Taskset / Task / Toolset / Harness | messages via interception | MCP toolsets | weighted `@vf.reward` | stop conditions | orchestrator `group_size` | runtimes (docker, Prime, Modal) | `Env.run(task, agents)` |
| Inspect | `Task(dataset, solver, scorer, sandbox)` | messages | `@tool` | `Score` (multi-dim) | limits | `epochs` | sandbox providers | agents / handoff |
| TextArena | `Env` | strings | — | terminal, per player | `done` | — | in-process | native |
| SkyRL-gym | `BaseTextEnv` | messages / strings | tool groups | per step | `done` | trainer | in-process / remote | — |
| Harbor | task directory | agent's own | agent's own + MCP | `reward.json` (named) | timeouts | trainer | ~30 providers | multi-step tasks |
| Tinker | `Env` + `EnvGroupBuilder` | **tokens** | via renderer | per step + group | `episode_done` | **first-class** | user's | via group builder |
| NeMo Gym | resources + agent servers | Responses API | MCP | `/verify` + `mask_sample` | — | trainer | servers | agent servers |

**Where we are ahead.**
- Terminated vs truncated at the LLM level: only Gymnasium keeps it.
- Durable, replayable task code.
- A content-addressed template cache paid once per row.
- Rewards bound to specific replies.
- One interface for tool-driven and environment-driven tasks.
- Environments that outlive runs.
- Security at the environment layer.

**Where we are behind.**
- No group-level reward hook (Tinker `compute_group_rewards`, verifiers group rewards, ART RULER).
- No environment-side "do not train" flag (NeMo Gym `mask_sample`, DeepSWE compact filtering).
- No way to address a service inside an environment except `execute` (OpenEnv needs HTTP/WebSocket to a port).
- Templates do not build Dockerfiles (Harbor, OpenEnv and SWE images are Dockerfile-based); that is left to
  external pipelines.

### 4.2 Trainers vs our rollout interface and `Sample`

| | Token capture | Multi-turn form | Version granularity | In-flight update | Stale KV | Groups | What a rollout source must provide |
|---|---|---|---|---|---|---|---|
| **Ours** | active recorder, renderers, tokens-in | one `Sequence` per renderer epoch + loss mask | **per token** (exact via abort-and-resubmit) | yes (WUC) | **reused**, bounded by `max_kv_age` | caller (labels) | `Sample` stream + publish |
| slime | TITO via SGLang, adapters | concatenated; fan-out list | per call | abort → re-queue (recompute) | no | barrier per group | `list[list[Sample]]` |
| miles | TITO session server (tree) | one per leaf | per token span | abort / retract / in_place | optional | barrier | agent / generate / rollout functions |
| AReaL | `agenerate` token ids | concat or per interaction | per token | abort + resubmit | recompute | `group_size` | `RolloutWorkflow` or OpenAI agent |
| verl | Continuous Token builder | concatenated | per trajectory | abort; KV rebuilt | no | `rollout.n` | `AgentLoopBase` |
| PipelineRL | output ids; prompt retokenized | per call | per rollout | pause keep | **yes** | `attempts` | `generate_rollout` |
| SkyRL | TITO; step-wise | concat or step-wise | per group | pause / resume | **yes** (default) | `n_samples_per_prompt` | `GeneratorInterface` |
| rLLM | gateway + renderers | prefix-merge rows | per step | via backend | backend | per (task, role) | `AgentFlow` + gateway |
| Agent Lightning | proxy, `return_token_ids` | prefix-merge / transitions | per request | drain | — | all siblings | base URL + reward event |
| prime-rl | renderers, token-in | prefix-merge; break → new | oldest version per rollout | yes | — | `group_size` | verifiers environment |
| NeMo RL | message log with ids | concatenated; asserts contiguity | per trajectory | optional | optional recompute | per prompt | `EnvironmentInterface` / Gym |
| TRL | prefix-preserving template | concatenated | per sample | async (experimental) | — | `num_generations` | `rollout_func` dict |
| Tinker | renderer, token env | merge when extension holds | — | server-side | — | `EnvGroupBuilder` | `Datum` list |

**Reading of the matrix.**
1. Our data model is the *most exact* in the table. Per-token versions are exact without engine patches; only
   AReaL and miles match this.
2. We are one of three designs (with PipelineRL and SkyRL) that reuse stale KV deliberately.
3. The session tree matches verifiers v1's message graph and miles' session-server v2 tree.
4. What we lack is the set of fields and controls that trainers added in 2025–2026: routed experts, turn
   boundaries, teacher logprobs, cancellation, multi-policy, and fast weight transport.

---

## 5. Gaps in our interface and proposed changes (analysis)

Each gap lists who needs it, and a concrete change that keeps ADR-0015's rule that the rollout side knows no
algorithm.

### 5.1 `Sample`: proposed schema

```proto
message Sample {
  // existing: sample_id, cursor, job_id, run_id, slot, labels, sequences, ending, complete, policy_id, sampling
  Outcome outcome = 12;                   // COMPLETED | TASK_ERROR | CANCELLED | INFRASTRUCTURE_EXHAUSTED
  TruncationReason truncation_reason = 13;// MAX_TURNS | TIME | BUDGET | CONTEXT_OVERFLOW | OUTPUT_LENGTH | TASK (when ending = TRUNCATED)
  repeated string mask_reasons = 14;      // set by the task via run.exclude_from_training(reason); trainers default to skip
  string root_run_id = 15;                // for descendants of a swarm; equals run_id otherwise
  BlobReference transcript = 16;          // optional: canonical messages of the selected path (for judges, debugging)
  map<string, Enrichment> enrichments = 17; // optional per-job enrichments, e.g. "teacher" logprobs (5.5)
}

message Sequence {
  // existing: tokens, loss_mask, behavior_logprobs, weights_version, rewards
  repeated TurnSpan turns = 6;            // one per sampled turn on the path
  BlobReference routed_experts = 7;       // optional: int32 [positions, layers, top_k], for routing replay (R3)
  repeated MediaReference media = 8;      // optional: images / audio placed at token offsets
  EpochReason epoch_reason = 9;           // why this sequence starts (START | CONTEXT_EDIT | PREFIX_MISMATCH | …)
}

message TurnSpan {
  uint32 context_start = 1;               // first token of the observation this turn responds to
  uint32 sampled_start = 2;               // first sampled token of the reply
  uint32 sampled_end = 3;                 // exclusive
  string reply_effect_id = 4;             // joins to run-log observations and rewards
  FinishReason finish = 5;                // STOP | LENGTH | TOOL_USE | ABORTED_RESUBMITTED
  uint32 interruptions = 6;               // weight transitions inside this turn
}
```

**Why each field:**

- **`turns`.** Needed for step-wise and per-turn advantages:
  - SkyRL step-wise mode;
  - rLLM `broadcast` / `per_step`;
  - Agent Lightning `transition`;
  - AReaL `individual`;
  - the Tinker/PipelineRL per-call rows;
  - DAPO overlong filtering, which needs the finish reason of the last turn;
  - AgentFlow's outcome broadcast.

  Without turn boundaries an adapter would have to re-derive turn structure from `loss_mask` runs. That breaks when
  a turn's sampled span is split by an abort-and-resubmit.
- **`routed_experts`.** Routing replay (R3) is supported by slime, miles, verl, SkyRL, rLLM, prime-rl, NeMo RL and
  ART; MoE policies need it.
  - This requires a new engine-adapter capability: SHOULD return routed experts. Positions include context tokens,
    which the engine produced at prefill.
  - The recorder stores them per span as a blob.
  - An open issue is what happens on a prefix-cache hit: the engine must retain routing for cached positions (SGLang
    `routed_experts_start` suggests it does). That needs a spike (8.2).
- **`outcome`, `truncation_reason`, `mask_reasons`.** Today `ending = NONE` conflates task errors, cancellation and
  exhausted retries. DeepSWE's compact filtering and NeMo Gym's `mask_sample` show that environments must be able to
  say "do not train on this" (for example a sandbox timeout that is not the policy's fault).
  - Proposal: add `run.exclude_from_training(reason: str)` to `RunContext`. It records a `reward.assigned`-like
    event, so no new event type is needed if it is modelled as a reserved key. Trainers skip such samples by default.
- **`media`.** verl `multi_modal_data`, SkyRL `pixel_values` and AReaL `image_data` show VLM training needs the raw
  media, not only placeholder tokens.
- **`transcript`.** Group judges (ART RULER, verifiers group rewards, pairwise reward models) need the canonical
  output of each group member. Fetching it by `run_id` from the Control API works but costs one call per sample at
  30k samples/s. It should be opt-in per job.
- **`root_run_id`.** See 5.4.

**What stays out of `Sample`, deliberately:**
- Reference-model logprobs, advantages, returns, KL: the trainer computes these (verl, NeMo RL, OpenRLHF, AReaL all
  do).
- Top-k logprobs: already opt-in in the recorder. Expose them as a blob only in an enrichment.
- Sampling masks (prime-rl Sampling Replay): not needed because trainable channels are temperature-only. If we ever
  relax that, the sampling mask becomes a `Sequence` field.

**Loss-mask conventions.** Our boolean mask ("sampled by this slot's policy") matches every framework surveyed:
slime `loss_mask`, verl `response_mask`, SkyRL `loss_masks`, TRL `env_mask`, Tinker `mask`.

Two details must be written into the trajectories contract:
1. Renderer-inserted boundary tokens (for example the newline after an end-of-turn token) are `CONTEXT` with
   mask 0. This is verl's Continuous Token rule.
2. The stop token the model sampled is mask 1 and has a logprob.

Sequence-level loss normalization when one run yields several sequences (slime's `rollout_id` siblings, Agent
Lightning `per_rollout_mean`, AReaL `rollout_reward`) is the adapter's job. `sample_id` groups them already.

### 5.2 `RolloutJobs`: proposed changes

```proto
service RolloutJobs {
  // existing: StartJob, Run, Samples, Acknowledge, Publish, GetJob, CancelJob
  rpc Cancel(CancelRunsRequest) returns (CancelRunsResult);
  //   {job_id, oneof {run_ids[], label_selector}, mode: CANCEL | STOP_ADMISSION_ONLY}
  //   Cancelled runs still produce a Sample with outcome = CANCELLED (and complete = false),
  //   so counting stays exact; callers may ignore them.
  rpc Samples(SamplesRequest) returns (stream Sample);
  //   add: label_selector (server-side filter), slots[], include_masked
}

message RunRequest {                      // existing fields + …
  bool admit_together = 7;                // gang admission: all `count` runs of this ticket are admitted at once or not at all
}

message StartJobRequest {                 // existing fields + …
  map<string, string> trainable_channels = 9;   // slot → channel; replaces the single "job's channel"
  repeated EnrichmentSpecification enrichments = 10;
  bool attach_transcripts = 11;
}

message PublishRequest {
  string job_id = 1;
  string channel = 2;                     // which trainable channel (multi-policy jobs)
  WeightsSource weights = 3;
  PublishPhase phase = 4;                 // STAGE (transfer while serving) | COMMIT (abort-before-update + swap) | STAGE_AND_COMMIT
}

message WeightsSource {
  oneof source {
    string checkpoint_uri = 1;            // full checkpoint in object storage
    Delta delta = 2;                      // {parent_version, uri}: sparse/delta update (slime, SkyRL, NeMo RL, verl)
    LoraAdapter lora = 3;                 // {base_version, uri}: adapter-only (ART, Tinker, miles)
    DistributedTransfer distributed = 4;  // {backend: NCCL | NIXL | MOONCAKE | CHECKPOINT_ENGINE, rendezvous}: trainer participates
  }
}
```

**Why each change:**

- **`Cancel`.**
  - Oversampling and dynamic sampling (slime `--over-sampling-batch-size`, APRIL) stop surplus work once enough
    groups have variance.
  - prime-rl cancels in-flight episodes that exceed `max_off_policy_steps`.
  - Rows that are known bad (for example a broken environment image) are dropped.
  - Today the only lever is `CancelJob`.
- **`label_selector` on `Samples`, and per-ticket awaitables in the client.** AReaL workflows, verl agent loops,
  slime generate functions and TRL `rollout_func` are all *pull per episode*: the framework calls "run this row, give
  me the result". A job-wide stream forces the adapter to demultiplex client-side, which is fine for one consumer.
  - When the trainer's rollout workers are many processes, each should read only its own tickets. The client grows
    `ticket = job.run(...)` and `await ticket.samples()`.
- **`admit_together`.** Most trainers need whole groups (slime, Agent Lightning and SkyRL all have group barriers).
  Under buffer pressure, independent admission can admit 5 of 8 runs of a group and leave the rest queued behind
  other rows. That inflates group completion latency and staleness spread within a group.
  - Gang admission is an admission rule, not grouping semantics: the ticket is still just `count` runs.
- **`trainable_channels` and `Publish(channel)`.** Multi-policy training (miles `train_multi_policy.py`, MARTI,
  M-GRPO) and "several trainable identities" (7.3) need more than one trainable channel per job. Today the job has
  exactly one.
- **`WeightsSource` and staged publish.** Every framework surveyed transfers weights trainer → engine over
  NCCL, RDMA or CUDA IPC, or as deltas. A full checkpoint through object storage is the slowest path. The WUC's
  abort-before-update pause should cover only the *swap*, not the transfer:
  - `STAGE` moves bytes into engine-host memory while the old version serves (checkpoint-engine's pipeline);
  - `COMMIT` runs pause → abort → swap → resume.

  `DistributedTransfer` means the trainer joins a collective with the engines. So the WUC must expose a rendezvous
  and the trainer's adapter must speak it. That is a real coupling, contained in the adapter and the WUC.

### 5.3 Harness and Task additions

- **`run.exclude_from_training(reason)`**: see 5.1.
- **`Environment.request(port, method, path, body)` and WebSocket.** Also accept MCP streamable-HTTP (not only
  stdio) for `provided_tools`. This is needed so OpenEnv servers, NeMo Gym resources servers and Harbor `mcp_servers`
  inside an environment are callable without `curl` through `execute`.
- **Group-level rewards.** These stay out of the task, because a task instance sees one run. Two supported patterns:
  1. **Caller-side**: the trainer adapter post-processes a complete group. This fits RULER and pairwise preference.
  2. **Judge job**: the caller submits a row containing the group's `transcript` references to an `EVALUATE` job
     whose task samples a frozen judge slot and returns per-member scores.

  Pattern 2 keeps judges durable, recorded and rate-limited by the same machinery. Document both; add no hook.

### 5.4 Multi-run (swarm) rewards and assembly

Today each run's samples are assembled when that run is terminal, and `run.reward` addresses slots of the calling
run only. In a swarm the outcome is known at the root, after the children finished. Proposal:

- Runs spawned by a run inside a rollout job **inherit the job** (and its labels, plus
  `{"root_run": …, "role": …}`). Their trainable slots produce samples in the same job log.
- `run.reward(value, *, slot, key, run_id=None)`: a run may assign rewards to slots of its **descendants**, even
  after they are terminal. This becomes a `reward.assigned` event in the *parent's* log with a `target_run_id`.
- **Assembly trigger** becomes: the *root* run is terminal and all sessions of the tree are `CLOSED`.
- This costs latency for descendants. For long swarms that is acceptable. For short ones the root ends soon anyway.

### 5.5 Enrichments (teacher and reference logprobs)

- **Engine operation.** `ScoreTokens(tokens, positions) -> logprobs` (prefill-only; vLLM `prompt_logprobs`, SGLang
  `return_logprob` with `logprob_start_len`) on the engine adapter.
- **Job configuration.** A job may declare `EnrichmentSpecification{name: "teacher", channel: "qwen3-235b/frozen",
  positions: SAMPLED}`. The assembler scores each sample's sequences on that channel before emitting it.
- **Uses.**
  - On-policy distillation (Thinking Machines, prime-rl `opd`, slime `teacher_log_probs`, miles `opd_reverse_kl`).
  - Reference-KL without hosting a reference model inside the trainer (prime-rl's "reference scoring" server does
    exactly this).
- **Placement.** It is algorithm-neutral data and optional, but it adds inference load and a pipeline stage. Mark it
  optional (P10).
- **Alternative.** The trainer calls a scoring endpoint itself. Simpler, but the trainer then needs access to our
  inference fleet. **Recommendation:** build `ScoreTokens` first and enrichment second.

### 5.6 Things that look like gaps but are not

- **Partial-rollout recycling** (slime, APRIL, verl) exists to avoid losing work when a synchronous step ends. With
  in-flight weight updates, runs simply keep going across versions. There is nothing to recycle and nothing to
  re-prefill (stale KV).
  - The only related need is *cancellation* for over-provisioned work (5.2).
- **Group barriers** belong in the caller (ADR-0015). Gang admission (5.2) makes them cheap.
- **Adaptive staleness control** belongs in the caller as an admission rule (7.2). The buffer bound remains the
  system's backstop.
- **Reference logprobs in `Sample`**: the trainer computes them, or they come via enrichment (5.5).

---

## 6. Adapters (analysis)

### 6.1 Environments in

Order is by value, where value = task volume in the ecosystem × fit.

**1. Harbor task directory → `Task`.** Highest value: SkyRL, Tinker, verifiers, NeMo Gym, OpenEnv and rLLM all
consume it, and Terminal-Bench 2 ships in it.

```python
class HarborTask(Task):
    """Runs one Harbor task directory; parameters = {"task_reference": ..., "image": ...}."""
    max_turns = 200

    def __init__(self, parameters: dict) -> None:
        self.task_reference = parameters["task_reference"]         # content-addressed task directory
        self.environment_specification = EnvironmentSpecification(
            template=Template(base=parameters["image"]),           # environment/Dockerfile baked by an external pipeline
            security="untrusted-allowlist",                         # from task.toml network_mode / allowed_hosts
        )
        self.instruction: str = parameters["instruction"]

    async def setup(self, run: RunContext) -> None:
        self.workspace = await run.environments.create(self.environment_specification)

    async def start(self, run: RunContext) -> Observation:
        return Observation(self.instruction)

    @tool
    async def bash(self, command: str) -> str:
        """Run a shell command in the task container."""
        return (await self.workspace.execute(command, timeout=600)).output

    async def score(self, run: RunContext) -> float | None:
        await self.workspace.put(asset_for(self.task_reference, "tests"), "/tests", extract=True)
        await self.workspace.execute("bash /tests/test.sh", timeout=900)
        rewards = parse_harbor_rewards(await self.workspace.get("/logs/verifier/reward.json"))  # or reward.txt
        for key, value in rewards.items():
            run.reward(value, key=key)
        return rewards.get("reward")

    async def teardown(self, run: RunContext) -> None:
        await self.workspace.destroy()
```

- **Verification environment.** Harbor runs tests in the agent's container by default. Our trust-boundaries rule
  prefers `scratch` verification; honor Harbor's `[verifier] environment_mode` where it asks for isolation.
- **Harbor's installed agents** (claude_code, codex, openhands) are foreign harnesses. Run them *inside* the
  environment as an unmanaged harness pointed at a recorder session: `base_url` plus a scoped credential, with egress
  allowlisted to the recorder. The task's `respond` then just waits for the harness process.
  - This is the "any harness is trainable" path (R6), and it is safe for external tenants (Q10) because the foreign
    code runs in the microVM, not the task host.
- **Reverse adapter.** Implement Harbor's `BaseEnvironment` (`start`, `stop`, `upload`, `download`, `exec`) over
  our Environment Manager, so Harbor users can evaluate on our microVMs. It is cheap and a good adoption path.

**2. verifiers v1.** Two routes:
- *Foreign program*: run a verifiers `Env` / harness in an environment. Its interception server's upstream becomes
  our recorder session, and its `Trace` rewards become `run.reward(value, key=name)`. This preserves verifiers
  semantics exactly; our task is a thin wrapper.
- *Native*: map `Taskset` → rows, `Task.setup` / `score` → our hooks, `Toolset` → provided tools (MCP) and
  `@vf.reward` → keyed rewards. Better durability, but it re-implements verifiers' harness semantics. Also offer a
  verifiers *runtime* backed by our environments (verifiers already has docker, Prime and Modal runtimes).

**3. OpenEnv.**
- Map the space's image to `Template(base=...)`.
- Calls go through `Environment.request` (5.3). Until that exists, use `execute("curl …")`.
- `Observation.reward` → `Observation.reward`; `done` → `End()`. It is always `TERMINATED` unless metadata says
  otherwise, because OpenEnv cannot express truncation.
- With RFC 003 (MCP) environments, list tools once in `setup` and expose them via `run.tools.add(...)`.

**4. Inspect.**
- Dataset `Sample` → row; `sandbox` → `Template`; `scorer` → `score`, with a multi-dimensional `Score.value` → keyed
  rewards; `message_limit` / `turn_limit` → `max_turns`.
- The solver becomes our default agent. Alternatively, run Inspect itself in an environment with its `openai-api`
  provider pointed at the recorder.
- **Reverse adapter:** an Inspect sandbox provider (`@sandboxenv("dae")`) over our environments.

**5. Gymnasium, TextArena, SkyRL-gym (in-process).**
- These are deterministic Python. They can run *inside the task host* under replay, seeded from `run.random`, with
  environment state pickled in snapshots.
- A `GymnasiumTask` needs an observation→message renderer and an action parser.
- This path is only acceptable for **trusted** task code. For tenants (Q10), run them inside an environment and talk
  over `Environment.request`.

**6. Tinker `Env`.** Token-level environments violate P4 (the task never sees tokens). Adapt only `MessageEnv`.

**7. NeMo Gym resources servers.** Run them as an environment image (or bind as an import through the tool
router). Call `/verify` from `score`, and map `mask_sample` → `run.exclude_from_training`.

### 6.2 Trainers out

**Shared core.** Every adapter builds on one `SampleGroups` helper: stream → deduplicate on `sample_id` → group by a
label key → emit complete groups (`count` known per ticket) → acknowledge.

```python
@dataclass
class GroupTicket:
    group_key: str
    expected: int                                   # count × trainable slots
    samples: list[Sample] = field(default_factory=list)

class SampleGroups:
    def __init__(self, job: RolloutJob, group_label: str = "group") -> None:
        self.job, self.group_label = job, group_label
        self.open: dict[str, GroupTicket] = {}
        self.seen: set[str] = set()

    def expect(self, group_key: str, expected: int) -> None:
        self.open[group_key] = GroupTicket(group_key, expected)

    async def complete_groups(self, cursor: int) -> AsyncIterator[tuple[GroupTicket, int]]:
        async for sample in self.job.samples(cursor):
            if sample.sample_id in self.seen:       # at-least-once delivery
                continue
            self.seen.add(sample.sample_id)
            group = self.open[sample.labels[self.group_label]]
            group.samples.append(sample)
            if len(group.samples) == group.expected:
                del self.open[group.group_key]
                yield group, sample.cursor
```

**1. miles / slime (first).**
- Adapter: a `--rollout-function-path` that returns `list[list[slime.Sample]]` from `SampleGroups`.
- Conversion: `tokens = sequence.tokens`; `response_length = len(tokens) − first sampled index`;
  `loss_mask = sequence.loss_mask[prompt_length:]`; `rollout_log_probs` with 0.0 at masked positions;
  `reward = sum(rewards)` or a dict by key; `status` from `ending` / `outcome`.
- Versions: miles `WeightVersionSpan`s come from runs of `weights_version`. R3 comes from `routed_experts`.
  Multi-sequence samples become siblings sharing `rollout_id` (slime's fan-out convention).
- Weights: replace slime's `update_weights` with `job.publish(channel, WeightsSource(distributed=…))`. slime already
  supports external engines, so the remaining work is making the WUC the target of its NCCL/P2P broadcast.
  **UNVERIFIED** that this can be done without patching slime.
- Why first: miles shares our premises (per-span versions, TITO tree, stale KV `in_place`, multi-policy, R3) and has
  connectors for the same ecosystems.

**2. SkyRL.**
- Adapter: `GeneratorInterface.generate(GeneratorInput) -> GeneratorOutput`.
  - Submit one ticket per prompt with `count = n_samples_per_prompt`, await the tickets, and fill `response_ids`,
    `loss_masks`, `rollout_logprobs`.
  - `rewards` can be **per token** (ours map directly), plus `rollout_expert_indices` and `trajectory_ids`.
  - Step-wise mode maps to `turns`.
- Its fully asynchronous trainer with stale-KV reuse matches ADR-0009.

**3. Tinker `Datum` exporter.**
- `target_tokens`, `logprobs` (= `behavior_logprobs`) and `advantages` are *exactly* the Tinker RL loss inputs.
- A generic exporter serves every Tinker-API trainer: Tinker itself, SkyRL tx, miles `serve_tinker`.
- Weight sync from Tinker is out-of-band: saved weights → `WeightsSource(lora=…)`.

**4. AReaL.**
- A `RolloutWorkflow` whose `arun_episode` submits a ticket and awaits its sample, returning AReaL's tensor dict
  (`versions` from `weights_version`, −1 where masked).
- AReaL's weight-update service must be redirected to our publish. AReaL 2.0's service split may make this clean.
  **UNVERIFIED**.

**5. verl.**
- An `AgentLoopBase` returning `AgentLoopOutput`.
- verl expects its own server manager and checkpoint engine. It is a worse fit, because versions are per trajectory
  and weight sync is assumed internal.

**6. TRL.** `rollout_func` returning `prompt_ids`, `completion_ids`, `logprobs`, `env_mask`. Useful for small
users; not a scale target.

**What every adapter must do**, and what we should document as the "trainer contract":
- deduplicate on `sample_id`;
- skip `mask_reasons` and `complete = false` by default;
- log every normalization (sum rewards, split sequences);
- compute a per-token lag `current_version − weights_version` and apply the framework's IS/MIS/masking with it.

---

## 7. Operating walkthroughs (analysis)

All three walkthroughs share one job setup:

```python
binding = RunBinding(
    models={"policy": ModelBinding(recorded=Recorded(channel="acme/coder-train", mode=Mode.ACTIVE,
                                                     sampling=SamplingParameters(temperature=1.0)))},
    security_classes={"untrusted-allowlist": operator_profile("pypi-only")},
)
job = rollouts.start(task=FixFailingTest, agent=DefaultAgent, binding=binding,
                     buffer_samples=4096, mode=JobMode.TRAIN)
```

### 7.1 Synchronous RL (on-policy GRPO)

**Semantics.** Every token trained at step *k* was sampled by version *k*, so no rollout spans a publish.

**Loop.**
1. Pick *B* rows.
2. `run(row, labels={"group": key, "step": k}, count=G, admit_together=True)` for each row.
3. Collect all *B* groups.
4. Train.
5. `publish` and **wait for the Operation** (channel advanced, readiness met).
6. Only then submit step *k + 1*.

Because nothing is in flight at publish time, abort-before-update has nothing to abort.

```python
async def synchronous_training(job: RolloutJob, trainer: Trainer, dataset: RowDataset, group_size: int) -> None:
    groups = SampleGroups(job)
    cursor = job.acknowledged_cursor()
    version = trainer.version()
    for step, rows in enumerate(dataset.batches()):
        for row in rows:
            key = f"{step}:{row.row_id}"
            groups.expect(key, expected=group_size)            # one trainable slot
            job.run(row.parameters, labels={"group": key, "step": str(step)},
                    count=group_size, admit_together=True)
        batch: list[GroupTicket] = []
        async for group, cursor in groups.complete_groups(cursor):
            batch.append(group)
            if len(batch) == len(rows):
                break
        for group in batch:                                    # assert on-policy; cheap, catches bugs
            for sample in group.samples:
                for sequence in sample.sequences:
                    assert all(v == version for v, m in zip(sequence.weights_version, sequence.loss_mask) if m)
        usable = [g for g in batch if reward_variance(g) > 0]  # dynamic sampling happens in the caller
        trainer.train_step(to_trainer_batch(usable))
        job.acknowledge(cursor)
        operation = job.publish(channel="acme/coder-train", weights=trainer.weights_source(),
                                phase=PublishPhase.STAGE_AND_COMMIT)
        version = await operation.result()                     # blocks until the channel serves the new version
```

**Operational notes.**
- **Stragglers dominate.** A step waits for its slowest episode. Use the task's `max_turns` and time budgets
  (`End(truncated=True)`) to cap the tail. The DAPO pattern (oversample 1.5×, cancel the rest with
  `job.cancel(label_selector={"step": str(step)})`) trades inference for latency.
- **Idle inference.** On a disaggregated fleet, inference sits idle while the trainer trains and the trainer sits
  idle during rollouts. Colocation, which verl, slime and TRL all default to, is not in our design: the inference
  fleet is separate.
  - So synchronous mode is for correctness baselines, small experiments and evaluation-gated training.
  - Production should use one-step-off (prime-rl's default), which is 7.2 with a lag bound of 1.
- **Infrastructure retries** keep labels, so counts stay exact. Task errors produce an `outcome = TASK_ERROR`
  sample, which is counted but not trained on.
- **Numeric mismatch remains** even when synchronous. Recorded behavior logprobs ≠ trainer logprobs (the
  Yao et al. finding), so apply TIS or MIS anyway (P9).

### 7.2 Asynchronous RL with in-flight weight updates

**Semantics.**
- Rows are submitted continuously and runs never wait for training.
- Each `publish` reaches running episodes mid-generation: the WUC aborts, the recorder records the partial at
  v_old and resubmits at v_new on stale KV.
- A sample's tokens carry mixed versions. The trainer corrects with importance sampling against the recorded
  behavior logprobs, and masks tokens whose lag or ratio is extreme.

**Staleness control stays in the caller.** It is AReaL's rule, expressed on our interface: never have more than
`(version + η + 1) × B` groups submitted.

```python
async def asynchronous_training(job: RolloutJob, trainer: Trainer, dataset: RowDataset,
                                group_size: int, groups_per_step: int, maximum_lag: int) -> None:
    groups = SampleGroups(job)
    cursor = job.acknowledged_cursor()
    submitted = 0
    rows = dataset.stream()

    async def feeder() -> None:
        nonlocal submitted
        while True:
            budget = (trainer.version() + maximum_lag + 1) * groups_per_step   # AReaL-style admission rule
            while submitted < budget:
                row = next(rows)
                key = f"{row.row_id}:{submitted}"
                groups.expect(key, expected=group_size)
                job.run(row.parameters, labels={"group": key}, count=group_size, admit_together=True)
                submitted += 1
            await trainer.version_changed()

    feeding = asyncio.create_task(feeder())
    batch: list[GroupTicket] = []
    async for group, cursor in groups.complete_groups(cursor):
        if reward_variance(group) == 0:
            continue                                            # filtered; the feeder keeps the pipe full
        batch.append(group)
        if len(batch) < groups_per_step:
            continue
        current = trainer.version()
        data = to_trainer_batch(batch)
        data.token_lag = [[current - v for v in s.weights_version] for s in data.sequences]
        data.token_mask &= lag_mask(data.token_lag, maximum_lag)   # mask outliers by per-token version (ADR-0015)
        trainer.train_step(data, correction=Correction.SEQUENCE_MASKED_IMPORTANCE_SAMPLING)
        job.acknowledge(cursor)
        job.publish(channel="acme/coder-train", weights=trainer.weights_source(),
                    phase=PublishPhase.STAGE_AND_COMMIT)       # not awaited: runs keep going
        batch = []
    feeding.cancel()
```

**What happens underneath, per publish:**
1. The trainer adapter streams weights to engine hosts (`STAGE`, while serving).
2. The WUC pauses admission per replica, aborts in-flight generations and swaps (`COMMIT`). It also flushes KV on
   replicas whose oldest KV exceeds `max_kv_age`.
3. The recorder appends each partial as a `SAMPLED` span at v_old and resubmits `input ++ partial` at v_new, which is
   a prefix-cache hit.

Task and agent code see nothing.

**Tuning knobs, and what they bound:**

| Knob | Bounds | Where |
|---|---|---|
| `maximum_lag` (η) | how far submissions run ahead of training | caller (above) |
| `buffer_samples` | memory and queueing of finished, unconsumed samples | job |
| `max_turns`, time budgets | episode length, hence the version span within one sample | task |
| `max_kv_age` | age of stale KV conditioning new tokens | channel (operator) |
| per-token lag mask / MIS thresholds | which tokens train | trainer |
| staged publish | pause length per update (swap only) | adapter + WUC |

**Differences from the frameworks.**
- AReaL recomputes KV on interrupt. We keep it, like PipelineRL, SkyRL (default) and miles `in_place`.
- prime-rl measures lag from the *oldest* token version and cancels stale episodes. We can do the same with
  `job.cancel(run_ids)` from the caller if long episodes accumulate lag beyond usefulness. The default is to let them
  finish and mask per token, which wastes less inference.
- **Group spread.** Within one group, members finish at different versions. GRPO's group baseline is still valid
  as a baseline, and importance sampling corrects each member's own tokens. Gang admission keeps members' start
  times together.

**Failure behaviour.**
- A trainer crash stops acknowledgement. The buffer fills, admission stops, and runs in flight finish.
- The trainer restarts from its checkpoint and its last acknowledged cursor. Samples after the cursor are
  redelivered and deduplicated.
- The feeder's `submitted` counter must be recomputed from `GetJob`, as runs submitted minus runs consumed.
  *Proposal: expose these counters in `Job`.*

### 7.3 Multi-agent training

#### (a) Self-play: one policy, several roles, one run

Both roles are model slots of one task, bound to the **same trainable channel**. Each slot is its own recorder
session, so each run yields **two samples**, distinguished by `slot`. The opponent's context is built by task code
(P4 holds: messages only).

```python
class TicTacToeSelfPlay(Task):
    models = {"policy": ModelSlot(), "opponent": ModelSlot(trainable=True)}
    max_turns = 9

    def __init__(self, parameters: dict) -> None:
        self.game = TicTacToe()                             # deterministic, pure Python (or TextArena under replay)

    async def start(self, run: RunContext) -> Observation:
        self.policy_moves_first = run.random.random() < 0.5
        if not self.policy_moves_first:
            await self.opponent_move(run)
        return Observation(self.game.render_for(self.game.current_player))

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        if not self.game.apply(parse_move(reply.text)):
            return self.finish(run, policy_reward=-1.0, opponent_reward=0.0)   # illegal move loses
        if self.game.over:
            return self.settle(run)
        await self.opponent_move(run)
        if self.game.over:
            return self.settle(run)
        return Observation(self.game.render_for(self.game.current_player))

    async def opponent_move(self, run: RunContext) -> None:
        context = [system_message(RULES), user_message(self.game.render_for(self.game.current_player))]
        reply = await run.models["opponent"].sample(context)
        if not self.game.apply(parse_move(reply.text)):
            self.game.forfeit(self.game.current_player)

    def settle(self, run: RunContext) -> Observation:
        policy_reward = self.game.outcome_for(self.policy_player)   # +1 / 0 / -1
        return self.finish(run, policy_reward, -policy_reward)

    def finish(self, run: RunContext, policy_reward: float, opponent_reward: float) -> Observation:
        run.reward(opponent_reward, slot="opponent")
        return End(reward=policy_reward)
```

**Caller side.**
- Group by `(row, slot)` for GRPO, or use SPIRAL-style role-conditioned baselines keyed by `(environment, slot)`.
  In zero-sum games the plain group mean is ≈ 0, so this matters.
- Both slots' samples go to the same trainer, because they share one policy.
- The opponent slot builds its own context each move (no history). If it should see history, the task keeps a
  per-role transcript. An `Agent` object can be reused for the opponent by calling
  `opponent_agent.select_context(...)` with a role-specific `History`. *Proposal: an SDK helper
  `run.act(slot, agent, history)` that runs any `Agent` against another slot, so self-play tasks don't hand-roll
  context.*

**League / past-self opponents.**
- Bind `opponent` to a **frozen channel pinned to an older version** (`acme/coder-frozen-v40`), and mark the slot
  `trainable=False` in the binding.
- Each pinned version needs replica capacity. LoRA-based policies make pools of past selves cheap (multi-LoRA
  serving), which is an argument for LoRA in self-play.
- The Policy Registry must allow channels pinned to historical versions; verify this is permitted.

#### (b) One trainable agent in a swarm of frozen agents

**Setup.**
- A root task (the orchestrator) spawns child runs: planner, several workers, a reviewer.
- Only the `worker` role is trainable. Its binding points at the trainable channel, while the others use frozen
  channels or API models (direct adapters, not recorded).
- The outcome is known only at the root. This needs 5.4: descendants inherit the job, and the root assigns rewards
  to descendants.

```python
class SwarmSoftwareProject(Task):
    models = {"policy": ModelSlot(trainable=False)}          # the orchestrator itself is frozen here

    async def setup(self, run: RunContext) -> None:
        self.repository = await run.environments.create(REPOSITORY_SPECIFICATION)   # shared workspace

    async def start(self, run: RunContext) -> Observation:
        return Observation(self.parameters["goal"])

    @tool
    async def delegate(self, subtask: str) -> str:
        """Give a subtask to a worker agent; returns its report."""
        child = await run.spawn(RunSpecification(
            task=TaskReference(code_reference=WORKER_PACKAGE, class_name="WorkerTask",
                               parameters={"subtask": subtask, "repository": self.repository.environment_id}),
            agent=AgentReference(code_reference=WORKER_PACKAGE, class_name="DefaultAgent"),
            binding=WORKER_BINDING,                           # "policy" → trainable channel acme/worker-train
        ))
        self.children.append(child.run_id)
        return (await child.result()).text

    async def score(self, run: RunContext) -> float | None:
        async with run.environments.scratch(VERIFIER_SPECIFICATION) as clean:
            passed = await run_acceptance_tests(clean, self.repository)
        outcome = 1.0 if passed else 0.0
        for child_run_id in self.children:                   # proposed: rewards to descendants (5.4)
            run.reward(outcome, slot="policy", key="team_outcome", run_id=child_run_id)
        return None                                          # the orchestrator is not trained
```

**Notes.**
- The shared workspace is an attachment (P11: the root owns it, workers borrow it).
- **Credit assignment.** Broadcasting the team outcome to every worker is AgentFlow's choice. Per-worker process
  rewards from the reviewer can be added with further keys (`key="review"`). The caller combines keys.
- **Grouping.** A group is the *G* repetitions of the root row (`labels["group"]`). Each root yields a variable
  number of worker samples, so the caller normalizes per root (for example, averaging each root's worker advantages,
  like slime `rollout_id` siblings) or groups by `(row, role)` like rLLM.
- **Non-stationarity.** Frozen teammates make the environment stationary, which is the main reason to train one
  role at a time.

#### (c) Several trainable identities

**Setup.**
- Two or more roles, each a **separate policy lineage**. For example, `planner` on `acme/planner-train` and `coder`
  on `acme/coder-train`.
- One job lists both in `trainable_channels`.
- The caller routes samples by `policy_id` to two trainer groups (or one multi-policy trainer, as in miles). Each
  publishes to its own channel.

```python
job = rollouts.start(
    task=PlanAndCode, agent=DefaultAgent, binding=two_policy_binding,
    trainable_channels={"planner": "acme/planner-train", "coder": "acme/coder-train"},
    buffer_samples=8192,
)
trainers = {"acme/planner": planner_trainer, "acme/coder": coder_trainer}
batches: dict[str, list[GroupTicket]] = defaultdict(list)

async for group, cursor in SampleGroups(job, group_label="group").complete_groups(cursor):
    for policy_id, samples in split_by_policy(group).items():  # role-specific groups (rLLM: task_id:role)
        batches[policy_id].append(GroupTicket(group.group_key, len(samples), samples))
    for policy_id, pending in batches.items():
        if len(pending) >= GROUPS_PER_STEP[policy_id]:
            trainer = trainers[policy_id]
            trainer.train_step(to_trainer_batch(pending))
            job.publish(channel=CHANNEL_OF[policy_id], weights=trainer.weights_source(),
                        phase=PublishPhase.STAGE_AND_COMMIT)
            pending.clear()
    job.acknowledge(cursor)                               # only when all trainers have consumed through cursor
```

**Notes.**
- **Acknowledgement is joint.** The job's cursor is single, so acknowledge the minimum consumed position across
  trainers. Alternatively, give each policy its own job over the same rows, which is simpler but duplicates runs.
  Recommendation: one job, with a cursor per consumer as a follow-up if needed.
- **Asynchronous per identity.** Each channel advances independently. A sample's tokens carry the version of *their
  own* policy; the other role's version is visible in its own sample. For analyses that need the joint state,
  `root_run_id` joins them.
- **Training schedules.** To reduce non-stationarity, alternate which identity trains: publish only one channel per
  phase, keeping the other channel's version fixed. This is a caller policy.

**Durable agent identities (proposal 6 in the research charter).**
- An identity's *policy binding* is a channel. Training the identity means advancing that channel through a job
  like the one above.
- Two open issues:
  1. Samples from the identity's **production traffic**, meaning runs outside any rollout job, need an "observe" job
     mode that collects samples from recorded sessions labelled with the identity. Rewards would arrive later,
     from humans or outcomes, via `reward.assigned` against finished runs.
  2. Channel contracts must not weaken while long-lived identities use them (existing rule).

  Both are open questions (8.1).

---

## 8. Open questions and spike tests

### 8.1 Open questions

1. **Which trainer first?** This report recommends miles/slime, then SkyRL. Confirm against the team's model
   families (MoE → R3 is mandatory) and parallelism needs (Megatron vs FSDP).
2. **Trainer-participating weight transfer.** Is it acceptable that the trainer joins a collective with our engines
   (`DistributedTransfer`)? It couples network topology: trainer and inference must share an RDMA/NCCL fabric,
   which may cross cells. The alternative is object storage plus deltas, which is slower but portable (N2).
3. **Cross-run rewards (5.4).** Should the target be restricted to descendants, or allowed for any run in the same
   job? Descendants-only keeps the ownership model (P11) simple.
4. **Samples from runs outside jobs** (durable identities, production traffic): new job mode, retention and consent
   rules, and delayed rewards.
5. **`Environment.request` vs MCP-over-HTTP only.** Which transport do we expose for services inside environments?
   Both affect the envd contract and the security profile (port exposure inside the guest only).
6. **Dockerfile builds in `Template`.** Harbor, OpenEnv and SWE ecosystems ship Dockerfiles. Do we add
   `Template(dockerfile=...)` (BuildKit is already in the build pipeline), or require pre-baked images?
7. **Temperature-only trainable channels vs Sampling Replay.** prime-rl shows top-p/top-k can be trained on with
   returned sampling masks. Revisit if agents need top-p for quality.
8. **Opponent pools.** Are channels pinned to historical versions allowed, and what does capacity look like? LoRA vs
   full weights.
9. **Enrichment placement.** Should enrichment happen in the assembler (latency, inference load) or in the trainer
   (access to our fleet)?

### 8.2 Spike tests

| # | Spike | Pass criterion |
|---|---|---|
| S1 | **miles adapter, synchronous**: one small dense model; rollouts through our recorder; train with miles; compare trainer-recomputed logprobs to recorded behavior logprobs at the same version | mean abs diff comparable to miles' own SGLang path; learning curve matches miles-native within noise |
| S2 | **Asynchronous with in-flight updates**: publish every step on a 30k-token agentic task; measure abort-and-resubmit overhead, prefix-cache hit rate after swap, per-token lag distribution, IS ratio / effective sample size vs `max_kv_age` ∈ {0, 2, 8} | cache hit ≥ 90% after swap; ESS degradation at `max_kv_age = 8` small relative to lag 1–2 |
| S3 | **Staged publish**: 30B and 235B MoE; checkpoint URI vs delta vs NCCL broadcast vs checkpoint-engine; measure pause time (commit only) and total transfer | commit pause ≤ a few seconds; transfer overlapped with serving |
| S4 | **R3 capture through the recorder**: SGLang `return_routed_experts` with prefix-cache hits and abort-and-resubmit; blob size per token | routed experts available for every position incl. cached prefixes; storage cost acceptable |
| S5 | **Harbor adapter**: run Terminal-Bench 2 tasks via `HarborTask` with (a) our default agent and (b) Claude Code / Codex installed in the microVM pointed at a recorder session | scores match Harbor's own runner; prefix-mismatch epoch rate for foreign harnesses measured |
| S6 | **Gang admission vs independent admission** under buffer pressure: group completion latency (p50/p99) and intra-group version spread | gang admission cuts p99 group latency materially |
| S7 | **Swarm rewards**: root with 8 spawned workers, rewards assigned to descendants, assembly at root terminal | samples for all workers carry `team_outcome`; no sample assembled before the root ends |
| S8 | **Self-play**: TextArena Kuhn Poker with two trainable slots on one channel, role-conditioned baselines in the caller | reproduces SPIRAL's qualitative result (win rate vs fixed opponents increases) |
| S9 | **Renderer coverage**: target model families against `renderers` `bridge_to_next_turn`, including thinking-stripping templates | epoch splits only where the template rewrites history; logged per family |
| S10 | **Enrichment**: teacher logprobs via `ScoreTokens` at 30k samples/s scale model | throughput and added latency per sample |

---

## 9. Sources

Environment specifications and hubs
- Gymnasium: [repository](https://github.com/Farama-Foundation/Gymnasium), [core.py](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/core.py), [vector_env.py](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/vector/vector_env.py), [autoreset modes](https://farama.org/Vector-Autoreset-Mode)
- OpenEnv: [repository](https://github.com/huggingface/OpenEnv), [interfaces.py](https://github.com/huggingface/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py), [docs](https://huggingface.co/docs/openenv), [hub](https://huggingface.co/openenv), [TRL integration](https://huggingface.co/docs/trl/openenv)
- verifiers: [repository](https://github.com/PrimeIntellect-ai/verifiers), [v1 docs](https://github.com/PrimeIntellect-ai/verifiers/tree/main/docs/v1), [graph.py](https://github.com/PrimeIntellect-ai/verifiers/blob/main/verifiers/v1/graph.py); [renderers](https://github.com/PrimeIntellect-ai/renderers); [Environments Hub](https://app.primeintellect.ai/dashboard/environments), [research-environments](https://github.com/PrimeIntellect-ai/research-environments)
- Inspect AI: [repository](https://github.com/UKGovernmentBEIS/inspect_ai), [task.py](https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/_eval/task/task.py), [docs](https://inspect.aisi.org.uk/)
- TextArena: [repository](https://github.com/LeonGuertler/TextArena), [core.py](https://github.com/LeonGuertler/TextArena/blob/main/textarena/core.py); SPIRAL: [arXiv 2506.24119](https://arxiv.org/abs/2506.24119), [repository](https://github.com/spiral-rl/spiral)
- SkyRL-gym: [base_text_env.py](https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-gym/skyrl_gym/envs/base_text_env.py)
- Harbor: [repository](https://github.com/harbor-framework/harbor), [hub](https://hub.harborframework.com); Terminal-Bench 1: [repository](https://github.com/harbor-framework/terminal-bench-1)
- SWE-bench: [repository](https://github.com/SWE-bench/SWE-bench), [grading.py](https://github.com/SWE-bench/SWE-bench/blob/main/swebench/harness/grading.py); SWE-Gym [arXiv 2412.21139](https://arxiv.org/abs/2412.21139); SWE-smith [arXiv 2504.21798](https://arxiv.org/abs/2504.21798); R2E-Gym [arXiv 2504.07164](https://arxiv.org/abs/2504.07164); SWE-rebench [arXiv 2505.20411](https://arxiv.org/abs/2505.20411); DeepSWE [blog](https://www.together.ai/blog/deepswe)
- Tinker cookbook: [repository](https://github.com/thinking-machines-lab/tinker-cookbook), [rl/types.py](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/rl/types.py), [data_processing.py](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/rl/data_processing.py); Tinker [losses](https://tinker-docs.thinkingmachines.ai/tinker/losses/)
- NeMo Gym: [repository](https://github.com/NVIDIA-NeMo/Gym), [base_resources_server.py](https://github.com/NVIDIA-NeMo/Gym/blob/main/nemo_gym/base_resources_server.py)
- BrowserGym: [repository](https://github.com/ServiceNow/BrowserGym); WebArena: [task configs](https://github.com/web-arena-x/webarena/blob/main/config_files/test.raw.json)

RL frameworks
- slime: [repository](https://github.com/THUDM/slime), [types.py](https://github.com/THUDM/slime/blob/main/slime/utils/types.py), [arguments.py](https://github.com/THUDM/slime/blob/main/slime/utils/arguments.py), [customization](https://github.com/THUDM/slime/blob/main/docs/en/get_started/customization.md), [agent adapters](https://github.com/THUDM/slime/blob/main/docs/en/get_started/agent.md), [fully async](https://github.com/THUDM/slime/blob/main/examples/fully_async/README.md)
- miles: [repository](https://github.com/radixark/miles); Miles v0.1 [arXiv 2609.08368](https://arxiv.org/pdf/2609.08368) (not read in full)
- AReaL: [repository](https://github.com/inclusionAI/AReaL), [workflow_api.py](https://github.com/inclusionAI/AReaL/blob/main/areal/api/workflow_api.py), [paper arXiv 2505.24298](https://arxiv.org/abs/2505.24298)
- verl: [repository](https://github.com/volcengine/verl), [agent_loop.py](https://github.com/volcengine/verl/blob/main/verl/experimental/agent_loop/agent_loop.py), [agent_loop.rst](https://github.com/volcengine/verl/blob/main/docs/advance/agent_loop.rst), [v1 async trainer](https://github.com/volcengine/verl/blob/main/docs/advance/v1_async_trainer.md), [rollout correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)
- PipelineRL: [repository](https://github.com/ServiceNow/PipelineRL), [vllm1.py](https://github.com/ServiceNow/PipelineRL/blob/main/pipelinerl/vllm1.py), [arXiv 2509.19128](https://arxiv.org/abs/2509.19128)
- SkyRL: [repository](https://github.com/NovaSky-AI/SkyRL), [generators/base.py](https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl/train/generators/base.py)
- rLLM: [repository](https://github.com/rllm-org/rllm), [types.py](https://github.com/rllm-org/rllm/blob/main/rllm/types.py), [model gateway](https://github.com/rllm-org/rllm/tree/main/rllm-model-gateway), [sync_coordinator.py](https://github.com/rllm-org/rllm/blob/main/rllm/trainer/sync_coordinator.py)
- Agent Lightning: [repository](https://github.com/microsoft/agent-lightning), [proxy.py](https://github.com/microsoft/agent-lightning/blob/main/agentlightning/server/proxy.py), [asynchronous training](https://github.com/microsoft/agent-lightning/blob/main/docs/35-asynchronous-training.md); vLLM blog [No More Retokenization Drift](https://vllm.ai/blog/2025-10-22-agent-lightning)
- OpenRLHF: [repository](https://github.com/OpenRLHF/OpenRLHF), [agent.py](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/utils/agent.py), [ppo_trainer_async.py](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ppo_trainer_async.py)
- prime-rl: [repository](https://github.com/PrimeIntellect-ai/prime-rl), [overview](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/overview.md), [algorithms](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/algorithms.md), [inference](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/inference.md), [grpo.py](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/orchestrator/algo/grpo.py); INTELLECT-2 [arXiv 2505.07291](https://arxiv.org/abs/2505.07291)
- NeMo RL: [repository](https://github.com/NVIDIA-NeMo/RL), [interfaces.py](https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/environments/interfaces.py), [async GRPO](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/async-grpo.md), [refit](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/refit.md)
- ART: [repository](https://github.com/OpenPipe/ART)
- TRL: [grpo_trainer.py](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py), [grpo_config.py](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py), [async GRPO](https://github.com/huggingface/trl/blob/main/trl/experimental/async_grpo/async_grpo_config.py)
- Tinker: [SDK](https://github.com/thinking-machines-lab/tinker); [on-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- torchforge: [README](https://github.com/meta-pytorch/torchforge/blob/main/README.md); ROLL: [repository](https://github.com/alibaba/ROLL)

Weight transfer, corrections, papers
- Kimi [checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine)
- R3 [arXiv 2510.11370](https://arxiv.org/abs/2510.11370); SGLang R3 roadmap [issue #16379](https://github.com/sgl-project/sglang/issues/16379)
- Yao et al., [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/off-policy-rl); [When Speed Kills Stability](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda); IcePop [arXiv 2510.18855](https://arxiv.org/abs/2510.18855)
- APRIL [arXiv 2509.18521](https://arxiv.org/abs/2509.18521); Magistral [arXiv 2506.10910](https://arxiv.org/abs/2506.10910); LlamaRL [arXiv 2505.24034](https://arxiv.org/abs/2505.24034); AsyncFlow [arXiv 2507.01663](https://arxiv.org/abs/2507.01663); StreamRL [arXiv 2504.15930](https://arxiv.org/abs/2504.15930); Laminar [arXiv 2510.12633](https://arxiv.org/abs/2510.12633)
- Multi-agent: MARTI [repository](https://github.com/TsinghuaC3I/MARTI); MAPoRL [arXiv 2502.18439](https://arxiv.org/abs/2502.18439); Self-play SWE-RL [arXiv 2512.18552](https://arxiv.org/abs/2512.18552); AgentFlow / Flow-GRPO [arXiv 2510.05592](https://arxiv.org/abs/2510.05592); M-GRPO [arXiv 2511.13288](https://arxiv.org/abs/2511.13288)

**Unverified items collected:**
- Instance counts for SWE-Gym and SWE-smith.
- Inspect used directly as an RL environment by any trainer.
- The AReaL 2.0 tech-report id.
- The date the slime router was removed.
- AReaL's reference-logprob placement.
- Token-level reward placement in verl and per-turn reward placement in SkyRL.
- Whether slime or AReaL weight sync can be redirected to an external controller without patches.
- torchforge's OpenEnv integration.
- The LightningRL credit-assignment details.
