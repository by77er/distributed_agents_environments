# Prime Intellect's verifiers environments, here

See [verifiers environments](../implementations/rollout-verifiers.md),
[harnesses over HTTP](../libraries/rollout-train/harness-endpoint.md), [evals](../libraries/rollout-train/evals.md),
[Thinking Machines' API](thinking-machines.md#prime-intellect-for-contrast)

**A record of a spike.** This page asks how far Prime Intellect's `verifiers` environments run on this system, and
what does not carry over. `rollout_verifiers` and the gateway's three APIs exist; everything marked *proposed* is
not built. The facts about verifiers were read from its source (the pinned `0.3.2.dev185`, and `main` at
`484e6de6c`, 2026-10-03) and from docs.primeintellect.ai on 2026-10-04. Anything not confirmed there is marked
*unverified*, and the list at the end collects them.

## The answer in brief

- **A verifiers v1 environment runs here as an environment**, with no change to the loop, the scheduler, evals or the
  trainer. Its harness reaches the model through verifiers' own interception server, which relays each request in
  the harness's own API to the episode's model address. The gateway renders and samples every request, so the
  tokens and logprobs are recorded by this system and are exact. The adapter, `rollout-verifiers`, is a uv project of
  its own, locked apart from the platform: verifiers' pins never reach the platform's lock.
- **The gateway speaks the three APIs verifiers relays:** Chat Completions (verifiers' `null`, `bash` and most
  harnesses), Responses (Codex) and Messages (Claude Code), with streaming, tool calls, reasoning, errors and
  `Idempotency-Key` in each. This is also the front door for testing black-box harnesses against each other.
- **The spike:** `primeintellect/gsm8k` 0.1.4 from the Environments Hub, Qwen/Qwen3-0.6B on vLLM. The base model
  solved 36 of 100 starts of a frozen suite through `rollout eval`; verifiers' own `vf-eval`, against the gateway,
  scored 0.39 on the same tasks, with the same outcome on 83 of 100 (both measured before the gateway closed
  Qwen3's thinking at its budget, so partly limited by finishing within 768 tokens). A LoRA run took 3 steps. The
  recorded logprobs differ from the trainer's, recomputed on the same tokens, by 0.0167 per token on average (median
  0.0007), as vLLM and the trainer always do here: the tokens are the ones sampled.
- **What does not carry over:** an episode is all or nothing (ours can resume mid-episode); every agent of a
  multi-agent verifiers environment would share the one model address; tasksets have no order, so our curriculum
  has nothing to unlock; splits are whatever each taskset's config calls them.
- **The v1 API is young and moves fast:** 189 commits in the six weeks since 0.3.1, a removed v0 stack, renamed
  commands and changed pins. Hub packages written for v0 do not load on the version that installs beside vLLM.
- **Exporting our environments as verifiers packages** is possible for single-agent ones (a taskset of our starts,
  a harness that runs our program against the interception endpoint) and costs our durability and exact recording
  inside Prime's stack ([below](#exporting-our-environments-as-verifiers-packages)).

## How the pieces map

| verifiers v1 | Here | How well |
|---|---|---|
| A taskset's tasks (`TaskData`: prompt, answer, files) | A row's starts: the task's data is the start's parameters | Cleanly. A suite freezes each task's data in the ledger |
| `TasksetConfig` (split, dataset, size) | `train` and `eval` settings: the training row and the eval data | Cleanly, but each taskset names its own settings |
| `@vf.reward` methods over a `Trace` | The episode's reward (their weighted sum) and `result` | Cleanly. `solved` is ours: a threshold on the reward |
| `@vf.metric` | The episode's `result` | Cleanly |
| A harness (`null`, `bash`, Codex, Claude Code) | A harness inside the environment, given a model address | Cleanly: verifiers runs it |
| The interception server | The gateway's APIs for harnesses | Both stand between harness and model; verifiers' relays to ours |
| verifiers' renderers rebuilding tokens from requests | Our renderer rendering each request before sampling it | Ours records; verifiers' eval client records no tokens |
| A runtime (`subprocess`, `docker`, `prime`, `modal`) | Where the episode runner runs the program | `subprocess` here; remote ones need a tunnel ([below](#runtimes-and-sandbox-leases)) |
| `Env.run(task, agents)` with several agents | A program with several model slots | Not mapped: one address for every agent ([below](#multi-agent-environments)) |
| `Episode`, retried whole | An episode, resumable mid-way under the durable runner | A verifiers episode is one opaque step ([below](#durability)) |
| `BestOfNEnv`, `-r n` rollouts of a task | A group: `group_size` episodes of one start | Ours is the unit of a group-relative update |
| `vf-eval -n N` over the first tasks | A suite: frozen starts, one per seed | Different selection; both reproducible |

## The spike

Run on 2026-10-04 on one RTX 5080, from `implementations/rollout-verifiers`, on GSM8K
(`rollout_verifiers.environments:gsm8k`) with the profile `examples/gsm8k_vllm.toml`, with every run in the shared
ledger.

**Installing and running it.** `implementations/rollout-verifiers` is a uv project of its own, locked apart from the
workspace, with the Hub's `gsm8k` 0.1.4 wheel among its dependencies; its `spike` group adds vLLM, the LoRA trainer and
the Qwen renderers (all by path from the workspace). From that directory:

```bash
uv sync --group spike
uv run rollout suite make math --environment rollout_verifiers.environments:gsm8k --data gsm8k-test-100 --ledger sqlite:///$HOME/.cache/rollout/ledger.db
uv run --group spike rollout eval examples/gsm8k_vllm.toml math --directory ~/.cache/rollout/runs/e2e-prime-eval-base --name e2e-prime-eval-base
uv run --group spike rollout train examples/gsm8k_vllm.toml rollout_verifiers.environments:gsm8k --groups 8 --groups-per-step 2 --directory ~/.cache/rollout/runs/e2e-prime-lora --name e2e-prime-lora
uv run --group spike rollout eval examples/gsm8k_vllm.toml math --checkpoint e2e-prime-lora:3 --directory ~/.cache/rollout/runs/e2e-prime-eval-lora --name e2e-prime-eval-lora
```

The environment's eval data is the whole test split (`gsm8k-test`) and its first 100 tasks (`gsm8k-test-100`), which
the suite `math` names as its entry's starts. The spike's own suite, `e2e-prime-gsm8k-test`, was drawn by hand: 100
starts of the test split with seeds 1 to 100 (93 distinct tasks), made before environments had eval data of their own.

verifiers' own eval ran as `vf-eval primeintellect/gsm8k --env.taskset.split test --env.agent.harness.id null
--env.agent.runtime.type subprocess --client.base-url URL --client.api-key-var VARIABLE --select.include.idx …
--no-push`, against a gateway serving the same channel on `127.0.0.1:8801` with a key made for one session. The
logprobs were scored again with `rollout_lora.policy.Policy.logprobs`. Those two scripts are not in the repository.
The runs below were played while the adapter was still resolved in the workspace's lock, except
`e2e-prime-eval-standalone`, which played the same suite from the project's own lock.

**The environment.** `primeintellect/gsm8k` 0.1.4, installed from the Hub's wheel. It is a v1 taskset: a train split
of 7,473 tasks and a test split of 1,319, a prompt asking for the answer after `####`, and one reward, `correct`,
which runs a `math-verify` script inside the rollout's runtime. It was played by the `null` harness (a tool-less chat
loop, run as a uv script) in the `subprocess` runtime.

**The channel.** Qwen/Qwen3-0.6B, renderer `qwen3`, `thinking_tokens = 512`, `answer_tokens = 256`, turns of at
most 2,048 tokens, a LoRA adapter of rank 16. Qwen3 opens its own thinking (the `qwen3` renderer's `ThinkingFormat`
has `prompt_opens=False`). The gateway samples such thinking in two phases: room to open the block, a forced
close at the budget if the model is still thinking, then the answer. The spike ran before it did, when Qwen3 was
sampled in one phase and the two budgets acted as one cap of 768 sampled tokens: 71 of the base model's 100
eval episodes and 26 of the 32 training episodes ran into it, mostly while still thinking, and scored 0. The solve
rates below were measured then, and are limited partly by finishing within that cap.

| Run (ledger name) | What | Result |
|---|---|---|
| suite `e2e-prime-gsm8k-test` | 100 starts of the test split, seeds 1 to 100 (93 distinct tasks) | |
| `e2e-prime-eval-base` | `rollout eval`, the base model, 1 episode a start | solved 36 of 100 (mean reward 0.36) |
| `e2e-prime-lora` | `rollout train`, 8 groups of 4, 2 groups a step | 3 steps (2 groups had equal rewards and taught nothing); checkpoints `mkzlyrlnmwumvzlp` (released), `lmrzxkonrylyznox`, `vmwqpuouttquvwpx` |
| `e2e-prime-eval-lora` | `rollout eval`, checkpoint `e2e-prime-lora:3` | solved 31 of 100 (mean reward 0.31) |
| `e2e-prime-eval-standalone` | `rollout eval`, the base model again, from the adapter's own project and lock (`uv run --group spike`) | solved 39 of 100 (mean reward 0.39) |
| `vf-eval` (not in the ledger) | verifiers' own eval of the suite's 93 tasks, its client pointed at a gateway serving the same channel, `--no-push` | mean reward 0.387 (0.390 over the suite's 100 starts) |

The two evals of the base model played each task once with the same channel and sampling, so they differ only by
the draw: the same outcome on 83 of 100 starts, 7 solved only by ours and 10 only by verifiers'. Three steps of 8
segments each did not change the solve rate beyond the draw either (0.31 against 0.36).

**The training run.** Each step trained 8 segments, about 6,000 sampled tokens, in 3 optimizer steps. Its own
statistics:

| Step | `mean_mismatch` | `kl_floor` | `kl_moved` | `clip_fraction` |
|---|---|---|---|---|
| 1 | 0.0168 | 0.0005 | 0.0081 | 0.025 |
| 2 | 0.0362 | 0.0047 | 0.0056 | 0.012 |
| 3 | 0.0408 | 0.0067 | 0.0059 | 0.008 |

Step 1's segments were sampled by the weights it started from, so its `mean_mismatch` is the engine's and the
trainer's numerical difference alone; later steps add staleness. Each step moved the policy by a KL of 0.006 to
0.008.

**Token exactness.** Every recorded segment of the base model's turns was scored again by `rollout_lora`'s policy
(the base model, its adapter zero), token for token, against the logprobs the gateway kept:

| Segments | Sampled tokens | Mean \|recorded − trainer\| | Mean (recorded − trainer) | Median \|·\| | 99th percentile \|·\| | Largest |
|---|---|---|---|---|---|---|
| `e2e-prime-eval-base`, all 100 | 70,636 | 0.0167 | 0.0009 | 0.0007 | 0.154 | 0.53 |
| `e2e-prime-lora`, the 20 sampled by the base model | 14,811 | 0.0169 | 0.0004 | 0.0006 | 0.169 | 0.44 |
| `vf-eval`'s 93, recorded by the gateway | 67,013 | 0.0167 | 0.0006 | 0.0007 | 0.153 | 0.85 |

This is the 0.016 per token measured between this trainer and vLLM before ([LoRA trainer](../implementations/rollout-lora.md#measurements)).
A token recorded out of place, or a prompt rendered differently from what was sampled, would show differences of
whole nats. verifiers' own trace of the same exchanges has no tokens: its eval client relays the requests and keeps
the replies.

## What does not carry over

### Multi-agent environments

A verifiers `Env` declares each agent (a role) as an `AgentConfig`: its harness, runtime, model and client, each
falling back to the run's. `run(task, agents)` programs the control flow; agents run one at a time unless
`--env.max-concurrent-agents` says otherwise; `SharedAgenticJudgeEnv` gives the judge the solver's runtime, and
`setup()` may mark an agent untrainable (`agents.judge.trainable = False`).

Here a program has named model slots, each bound to a channel (recorded, trained) or a direct model (not trained),
and each with its own address. `rollout_verifiers` hands verifiers one client, the run's `policy` address, so every
agent of a multi-agent environment would be sampled, recorded and trained as the one policy, the judge included.
*Proposed:* give each role whose `AgentConfig.client` is unset the address of a slot of the same name (the program
declaring a slot per role), so that a judge bound to a direct model is neither recorded nor trained. Sharing a
runtime between agents is verifiers' business inside the episode and needs nothing here. Multi-agent environments
were not run (*unverified* beyond reading the code).

### Durability

A verifiers rollout is all or nothing. `run_episode_with_retry` retries the whole episode on failure, and
`vf-eval --resume` plays errored rollouts again from the start. Here a program can resume mid-episode under the
durable runner: its effects are recorded, and a sample repeated under its effect id returns the recorded result.

`VerifiersProgram.main` is one opaque call. If its process dies, the episode is played again from the start: a new
interception server, a new harness process, new samples. The harness's requests carry no `Idempotency-Key` (the
official OpenAI and Anthropic Python clients send none by default; *unverified* for Codex and Claude Code), so a
replayed request is sampled again; the
gateway keeps the newer turn where the prompt repeats exactly ([what a session exports](../libraries/rollout-train/recorder.md#what-a-session-exports)),
and a single-turn episode loses nothing. A long multi-turn harness (Codex on a repository) loses the work done.
Making it resumable would need verifiers to replay a trace into a harness, which it does not offer for subprocess
harnesses (`SUPPORTS_RESUME` is about user turns over ACP, *unverified* for resuming a crashed rollout).

### Curricula

Our curriculum unlocks rows in the environment's order, easiest first, and weights each by how often its groups'
rewards differed. A taskset is an unordered dataset with no difficulty. Making each task a row would unlock the
first three tasks and train on them until solved. So a split is one row and its tasks are its starts, drawn at
random: the curriculum has one row and nothing to unlock. *Proposed:* a flat curriculum over task rows (no
unlocking, weighted by signal) would recover per-task signal weighting for small tasksets.

### Splits

`TasksetConfig` fields are each taskset's own: `gsm8k` has `split: train | test`, `reverse_text` has
`dataset_name` and `dataset_split`. Some tasksets have one split only. So `train` and `eval` take any settings, and
nothing checks that the eval data is held out. A task's `idx` is its position in its split's `load()`; a suite keeps
the task's data whole, so an eval plays the same tasks even if the dataset on the Hub changes. verifiers loads the
dataset again on every eval and selects by position or key.

### Sampling and caps

verifiers' interception server owns sampling: it writes the run's `sampling` into each request it relays. The
gateway ignores a client's sampling parameters, so `--sampling.temperature` and the like have no effect against
it; the channel samples as its binding says. A cap on the output is honoured.

### Ports and processes

Each episode starts an interception server on an ephemeral loopback port (verifiers binds `127.0.0.1:0` when no
remote consumer needs a tunnel) and, under `subprocess`, one harness process. The `null` harness is a uv script
whose dependencies are fetched the first time.

## Runtimes and sandbox leases

verifiers' `Runtime` is an abstract class: `start`, `run(argv, env)`, `read`, `write`, `expose`, `host_url`. Its
config is a closed union of seven types (`subprocess`, `docker`, `podman`, `apptainer`, `prime`, `modal`, `e2b`),
chosen by `type`. A remote runtime reaches the interception server through a tunnel: a Prime tunnel (which needs a
Prime account) or one of your own (`tunnel.type = "custom"`, a URL and a port; it binds every interface).

*Proposed:* a sandbox pool ([sandboxes](../libraries/rollout/sandboxes.md)) could supply the runtime in one of two ways.

1. **A lease as a container host.** The lease hands out a machine with Docker; verifiers' `docker` runtime runs on
   it (`DOCKER_HOST`), and the interception server is reached at the runner's address through a `custom` tunnel.
   No verifiers change; the lease is claimed by the episode's claim before `play` and released after it.
2. **A lease as a verifiers runtime.** A `Runtime` subclass over the lease's exec and file calls. The closed config
   union means this needs a change in verifiers (or a patch of its union) to be selectable by `type`.

Either way the model calls still go harness → interception server → the gateway, so tokens stay exact.

## How stable the v1 API looked

- **Fast-moving.** verifiers 0.3.1 was released on 2026-08-24; there have been 172 development releases of 0.3.2
  since (up to `dev185`), the newest on 2026-10-04. From the 0.3.1 tag to `main` (2026-10-03), 189 commits, 171 of
  them under `verifiers/v1`: 136 files changed, 11,302 lines added and 5,581 removed.
- **Breaking changes in that window.** The v0 stack (`load_environment()` returning `vf.SingleTurnEnv`, rubrics,
  parsers) was removed on 2026-08-31 (`feat!: remove the legacy (v0) stack`). The `default` harness became `bash`.
  `KeptTokens` was replaced by `SamplingMask`. The built-in Lean taskset and the pool harness were removed.
- **The Hub mixes both.** `primeintellect/gsm8k` 0.1.4 is a v1 taskset (the Hub says `runtime: VERIFIERS_V1`), while
  `primeintellect/reverse-text` 0.1.4 is a v0 `load_environment()` package. v0 packages do not load on 0.3.2.
- **Dependency pins move.** 0.3.1 requires `mcp<2`, which vLLM 0.30 (`mcp>=2`) cannot sit beside. The 0.3.2
  development releases require `mcp==2.0.0`, `openai>=2.54,<3` and a pre-release of Prime's `renderers`. So the
  version that installs beside vLLM is a development release, pinned exactly. Resolved together with the platform,
  it would move the workspace's `openai` from 3.19 to 2.54 and `mcp` from 2.2 to 2.0; so `rollout-verifiers` is a
  project of its own with its own lock, and the platform's lock does not know it.
- **Docs and code drift.** The hosted docs say `uv run eval`; the command is `vf-eval` (the repository's own docs say
  so). The hosted docs list `--no-serve`, which the pinned `vf-eval` refuses. Nothing named `verifiers.v1.packages`
  exists in the pinned release or on `main`; the v1 API is `verifiers.v1`. `vf-eval` uploads its results to Prime
  unless given `--no-push`.
- **The core shapes held.** `Taskset.load`, `Task` with `@vf.reward`, `TaskData`, `Env.run_episode`,
  `load_environment(EnvConfig)` and the interception server have the same shape in 0.3.1 and on `main`. The adapter
  uses only those, plus `EvalClientConfig` and `ModelContext`.

## Exporting our environments as verifiers packages

*Proposed*, not built. A single-agent environment of ours could be published as a v1 package:

1. **A taskset** whose `load()` yields one `TaskData` per start (the environment's rows, each with seeds, and its eval data), the start's
   parameters as a field. `INFINITE` with a generator for environments that draw starts forever.
2. **A harness** whose `launch(ctx, trace, runtime, endpoint, secret, …)` runs our program in the runtime: a runner
   process (`rollout.local`) with the program's slot bound to a direct model at `endpoint` with key `secret`. The
   interception server accepts Responses, so `rollout_openai.ResponsesEndpoint(ApiKey(secret, base_url=endpoint))`
   would serve the slot (*unverified*: not tried against verifiers' interception server). The runtime needs our
   packages: a uv script, or an image.
3. **A reward** that reads what the program decided. Our programs compute their reward inside (`run.reward`); the
   harness would write it to a file in the runtime and a `@vf.reward` method would read it with `runtime.read`, as
   gsm8k's reward runs its verifier in the runtime.

What it costs: the program's tools stay inside the program (they need not become MCP toolsets); durability is gone
(a rollout is all or nothing); tokens are rebuilt by verifiers' renderers from the requests, so our renderer's exact
segments are not what Prime trains on, and whether `renderers` has Qwen3.5's format is *unverified*. The Minecraft
team does not fit: it needs a Paper server and a Node harness in the runtime, and its several agents share one
program, which verifiers would see as one agent with one endpoint.

## Unverified

- Multi-agent verifiers environments, `bash`, Codex and Claude Code harnesses were not run against the gateway;
  the Responses and Messages endpoints were tested with the official clients, not with Codex or Claude Code.
- Whether harnesses other than `null` send `Idempotency-Key`.
- Whether verifiers can resume a crashed subprocess rollout.
- Whether our `ResponsesEndpoint` works against verifiers' interception server, and whether Prime's `renderers`
  covers Qwen3.5.
- Remote runtimes (`prime`, `modal`) and tunnels: not tried (they need a Prime account).
