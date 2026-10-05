# verifiers environments

For whoever wants Prime Intellect's verifiers environments here: installing, GSM8K, how an environment and an
[episode](../libraries/rollout-train/episodes.md) map, and checking one.

**Read first:** [Train any harness over HTTP](../libraries/rollout-train/harness-endpoint.md). **Next:**
[Contracts](../libraries/rollout/contracts/README.md).

Code: `rollout_verifiers` · See [harnesses over HTTP](../libraries/rollout-train/harness-endpoint.md),
[Prime Intellect's verifiers](../research/prime-compat.md)

Prime Intellect's [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) v1 describes an environment as a
**taskset** (its tasks, and how each is scored) played by a **harness** (the program the model runs in) in a
**runtime**, one rollout per task. This package wraps any such environment, a package from the Environments Hub or
one built into verifiers, as an [environment](../guide/reference.md#environment) that runs train
on and evals play.

It is a uv project of its own, with its own lock, outside the workspace: verifiers ships development releases daily
and pins what it needs (`openai<3`, `mcp==2.0.0`, a pre-release of Prime's `renderers`), so it is resolved apart from
the platform. It pins `verifiers==0.3.2.dev185` and the Hub's `gsm8k` 0.1.4 wheel, and depends on the workspace's
`rollout` (and, for its tests and a run's driver, `rollout-train`; for training here, in its `spike` group, the vLLM
engine, the LoRA trainer and the Qwen renderers) by path.

```py
from rollout_verifiers import VerifiersEnvironment

gsm8k = VerifiersEnvironment(
    "primeintellect/gsm8k", train={"split": "train"}, eval={"split": "test"}, eval_subsets=[100]
)
```

## Installing and running

```bash
cd implementations/rollout-verifiers
uv sync                  # the adapter, GSM8K, rollout-train (the commands, the database ledger) and the tests
uv run pytest tests
uv sync --group spike    # with vLLM, the LoRA trainer and the Qwen renderers: training on this machine
```

Another environment is installed into the project the same way: its Hub wheel as a dependency, and an object of its
own in `rollout_verifiers.environments`.

## GSM8K

`rollout_verifiers.environments:gsm8k` is GSM8K (grade-school math word problems) from the Environments Hub,
`primeintellect/gsm8k` 0.1.4, played by the `null` harness: one turn, in which the model reasons and gives its final
number after `#### `; the taskset's reward runs `math-verify` on it against the problem's answer, 1 or 0.

| | |
|---|---|
| Training data | the train split: 7,473 problems, one row |
| `gsm8k-test` | eval data: the whole test split, 1,319 problems, in order |
| `gsm8k-test-100` | eval data: its first 100 problems, the same starts |
| Description | rewards in [0, 1], `solved` at 1, duration in turns |

Eval data is the environment's own; a [suite](../libraries/rollout-train/evals.md#suites) names it in an entry, with how
its episodes play. The suite `math`, in the shared [ledger](../libraries/rollout-train/checkpoints.md#the-ledger), has
one entry: `gsm8k-test-100`, one episode of each start, 1,024 tokens of thinking and 512 of answer:

```bash
uv run rollout suite make math --environment rollout_verifiers.environments:gsm8k --data gsm8k-test-100 \
    --episodes 1 --thinking-tokens 1024 --answer-tokens 512 --cluster
```

Its tasks load from Hugging Face's `openai/gsm8k` (the first time, over the network) once its rows, a start, its
description or its eval data are asked for; importing it loads nothing. The root workspace's environment cannot
import it (verifiers is not in the platform's lock): there, the monitor lists it from the cluster config, the runs
and suites that name it, and its page says it does not load, with what the ledger has of it.

A run of it starts in this project's Python: the cluster config's `[environments]` entry names the project, and its
driver runs in the project's `.venv` (or the entry's `interpreter`):

```toml
[environments."rollout_verifiers.environments:gsm8k"]
project = "/opt/rollout/source/implementations/rollout-verifiers"   # the project's directory, on the cluster's nodes
interpreter = "/opt/rollout/verifiers/bin/python"                   # where its Python is built there
```

The preset `gsm8k-tinker` plays it with the base `Qwen/Qwen3.5-9B` sampled at Tinker
([Tinker](rollout-tinker.md#evals-of-a-base-model)); a run's settings that name a `vllm` provider and the LoRA trainer
train an adapter over Qwen3-0.6B on this machine, with the project synced with its `spike` group. From the project's
directory (where `env check` plays an episode in this process):

```bash
uv run rollout eval math --preset gsm8k-tinker --name gsm8k-tinker-base
uv run rollout env check rollout_verifiers.environments:gsm8k
uv run rollout train rollout_verifiers.environments:gsm8k --model Qwen/Qwen3-0.6B --provider local-vllm \
    --renderer rollout_qwen:qwen3 --trainer local-lora --groups 8 --groups-per-step 2
```

## The environment

`VerifiersEnvironment(taskset, *, harness="null", train=None, eval=None, runtime=None, environment=None, limit=None,
eval_size=None, eval_subsets=(), solved_at=1.0)`:

| | |
|---|---|
| `taskset` | The taskset's id: `owner/name` of a Hub package, which must be installed, or a built-in |
| `harness` | The harness's id. `null` is verifiers' tool-less chat loop; `bash`, `codex` and `claude_code` are others |
| `train`, `eval` | Settings of the taskset's config for the training and the eval data: its split, its dataset, its size. Without `eval`, no eval data |
| `runtime` | Where each rollout runs: `{"type": "subprocess"}` by default |
| `environment` | Any other setting of verifiers' environment config (`timeout`, `retries`) |
| `limit`, `eval_size` | Only the first training tasks (a taskset without end needs it), and the first eval tasks |
| `eval_subsets` | Sizes: the first so many eval tasks, each eval data of its own |
| `solved_at` | An episode whose reward reaches it solved its task |

- **Rows and starts.** The training data is one row; its tasks are its starts. A group's start is a task drawn with
  the group's random generator, so every episode of a group plays the same task, and a suite's seed always draws the
  same task. A start carries the task's data whole (`{"environment", "task", "solved_at"}`): whoever plays it needs
  no dataset. The tasks are loaded once, the first time the rows, a start or the description are asked for.
- **Eval data.** `evals()` names every task of the eval data (the first `eval_size`), in order, each a start of
  the row `eval`, which training never plays, after the taskset and its eval settings (`gsm8k-test`), and the first so
  many of them for each eval subset, with their number after (`gsm8k-test-100`): a task is the same start in each. A
  suite's entry names one (`rollout suite make NAME --environment MODULE:NAME --data EVAL_DATA`).
- **Version.** The taskset's package and its version, verifiers' version, and a digest of the settings above, such as
  `gsm8k 0.1.4, verifiers 0.3.2.dev185, settings 1a2b3c4d`.
- **Description.** Results say `solved` and, as `duration`, the turns the harness took. Rewards fall from the sum of
  the task's negative reward weights to that of its positive ones: verifiers scores fall in [0, 1] by convention, not
  by contract.
- **The program** is `VerifiersProgram`: one verifiers episode of the start's task.

## An episode

`VerifiersProgram.main` asks for the run's model address and calls `play(environment, task, address)`, which plays
one verifiers episode in the runner's process:

1. It builds verifiers' environment from the config, and its task from the start's data.
2. verifiers stands its own interception server between the harness and the model, and relays each of the harness's
   requests, in the harness's own API (Chat Completions for `null`, Responses for Codex, Messages for Claude Code), to
   the address. The gateway renders and samples each, so the episode's tokens and logprobs are recorded by this
   system, not rebuilt by verifiers.
3. The task's rewards score the trace. The episode's reward is their weighted sum; its `result` is each reward, each
   metric, the turns the harness took, and whether the episode solved its task.

An episode verifiers could not play (the harness failed, the model could not be reached) fails, with what verifiers
said. The key goes to verifiers' client by a name with `API_KEY` in it, which keeps it out of a subprocess runtime's
environment.

## Runtimes

`subprocess` runs each rollout's harness as a process on the runner's machine, which suits tool-less harnesses
and debugging. `docker` runs it in a local container. verifiers' sandbox runtimes (`prime`, `modal`) are remote: the
interception server is then reached through a tunnel: a Prime tunnel, which needs a Prime account, or one of
your own (`custom`).

## Checking one

`rollout env check MODULE:NAME` passes its rows, description, starts and eval data (for GSM8K: one row, rewards in
[0, 1], 1,319 and 100 eval starts on a row training never plays). Its scripted episode fails: the check's model is not
served over HTTP, and the harness needs an address.

## Tests

`implementations/rollout-verifiers/tests/` plays a two-task taskset defined in the test through the `null` harness,
against a gateway served on `127.0.0.1:8809`, and checks GSM8K's configuration and eval data with its tasks stood in
for (no network). It runs in the project's own environment (`uv run pytest tests` there),
needs `uv` (the harness runs as a uv script) and, the first time, the network. Elsewhere it skips: it needs
`verifiers`.
