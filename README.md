# rollout

Durable, scalable agents and environments for agent products and reinforcement learning. Tasks define environments
and tools, agents decide what the model sees, and a loop runs episodes; rollout jobs turn finished runs into episodes
with token trajectories, a training loop steps a trainer on them, and a profile says which engines and machines stand
behind it. It runs on one machine with one GPU; a durable runner, more engines and tool sets on other machines are
changes to the profile.

- [Three ways in](docs/guide/perspectives.md): building an environment, designing training, deploying
- [Developer guide](docs/guide/README.md), and the [API reference](docs/guide/reference.md)
- [All documentation](docs/README.md), with the table of packages

## Layout

The repository is a uv workspace of Python 3.13 packages.

| Directory | What is in it |
|---|---|
| `libraries/` | `rollout`, what environments are written against, and `rollout-train`, reinforcement learning on it |
| `implementations/` | One package per implementation of an interface the libraries define: a durable runner, a vLLM engine, a LoRA trainer, Qwen and Gemma renderers, computers for tasks, an OpenAI model endpoint, an S3 blob store |
| `products/` | Applications: the project assistant and agent sessions |
| `environments/` | Environments to train on: the Minecraft team, which depends on `rollout` only |
| `docs/`, `tests/`, `scripts/`, `deploy/` | The documentation, the tests, the reference generator and a training wrapper, local services |

## Development

```sh
uv sync                     # everything that needs no GPU
uv sync --all-extras        # with the engine, the trainer and the model families' renderers (Linux, NVIDIA GPU)
uv run pytest
uv run ruff check && uv run ruff format --check
uv run pyright
uv run python scripts/generate_reference.py   # after changing public names or docstrings
uv run --group docs mkdocs serve             # the docs as a local site: http://127.0.0.1:8000
uv run --group docs mkdocs build             # or write it to site/ (opens straight from disk)
```
