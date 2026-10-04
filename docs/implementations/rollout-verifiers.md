# verifiers environments

Code: `rollout_verifiers` · See [the recorder over HTTP](../libraries/rollout-train/harness-endpoint.md),
[Prime Intellect's verifiers](../research/prime-compat.md)

Prime Intellect's [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) v1 describes an environment as a
**taskset** (its tasks, and how each is scored) played by a **harness** (the program the model runs in) in a
**runtime**, one rollout per task. This package wraps any such environment, a package from the Environments Hub or
one built into verifiers, as a catalog that runs train on and evals play. It is installed with
`uv sync --extra verifiers` (or `--all-extras`), and pins `verifiers==0.3.2.dev185`.

```py
from rollout_verifiers import VerifiersEnvironment

train = VerifiersEnvironment("primeintellect/gsm8k", train={"split": "train"}, eval={"split": "test"})
test = train.evaluation()
```

`implementations/rollout-verifiers/examples` has this catalog (`prime_gsm8k.py`, not `gsm8k.py`: a catalog's module
must not take the name of the environment's package) and a profile for it. The profile serves the model endpoint
(`serve`): the harness needs an address.

```bash
uv pip install https://hub.primeintellect.ai/primeintellect/gsm8k/@eb4818f9/gsm8k-0.1.4-py3-none-any.whl
export PYTHONPATH=implementations/rollout-verifiers/examples
uv run rollout suite make gsm8k-test --catalog prime_gsm8k:test --seeds 1,2,3 --ledger sqlite:///$HOME/.cache/rollout/ledger.db
uv run rollout eval implementations/rollout-verifiers/examples/prime_gsm8k.toml gsm8k-test
uv run rollout train implementations/rollout-verifiers/examples/prime_gsm8k.toml prime_gsm8k:train --groups 8
```

## The catalog

`VerifiersEnvironment(taskset, *, harness="null", train=None, eval=None, runtime=None, environment=None, limit=None,
solved_at=1.0)`:

| | |
|---|---|
| `taskset` | The taskset's id: `owner/name` of a Hub package, which must be installed, or a built-in |
| `harness` | The harness's id. `null` is verifiers' tool-less chat loop; `bash`, `codex` and `claude_code` are others |
| `train`, `eval` | Settings of the taskset's config for the training and the eval data: its split, its dataset, its size |
| `runtime` | Where each rollout runs: `{"type": "subprocess"}` by default |
| `environment` | Any other setting of verifiers' environment config (`timeout`, `retries`) |
| `limit` | Only the first tasks. A taskset without end needs it |
| `solved_at` | An episode whose reward reaches it solved its task |

- **Rows and starts.** The training data is one row; its tasks are its starts. A group's start is a task drawn with
  the group's random generator, so every episode of a group plays the same task, and a suite's seed always draws the
  same task. A start carries the task's data whole (`{"environment", "task", "solved_at"}`): whoever plays it needs
  no dataset. The tasks are loaded once, the first time the catalog's rows or a start are asked for.
- **Train and eval data.** `evaluation()` is the same environment over the `eval` settings: a catalog of its own,
  whose one row is the eval data. Make suites of it.
- **The program** is `VerifiersProgram`: one verifiers episode of the start's task.

## An episode

`VerifiersProgram.main` asks for the run's model address and calls `play(environment, task, address)`, which plays
one verifiers episode in the runner's process:

1. It builds verifiers' environment from the config, and its task from the start's data.
2. verifiers stands its own interception server between the harness and the model, and relays each of the harness's
   requests, in the harness's own API (Chat Completions for `null`, Responses for Codex, Messages for Claude Code), to
   the address. The recorder renders and samples each, so the episode's tokens and logprobs are recorded by this
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

## Tests

`tests/rollout_verifiers/` plays a two-task taskset defined in the test through the `null` harness, against a
recorder served on `127.0.0.1:8809`. It needs `uv` (the harness runs as a uv script) and, the first time, the network.
