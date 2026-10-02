# rollout

Durable, scalable agents and environments for agent products and reinforcement learning. The core is a Python library
that runs on one machine (one GPU is enough); durability, fleet-scale execution and environments are optional layers.

- Three ways in (building an environment, designing training, deploying): [docs/guide/perspectives.md](docs/guide/perspectives.md)
- Developer guide: [docs/guide/README.md](docs/guide/README.md) (API reference: [docs/guide/reference.md](docs/guide/reference.md))
- All documentation: [docs/README.md](docs/README.md)

## Development

```sh
uv sync                     # base install (no GPU needed)
uv sync --all-extras        # with the inference engine and the trainer (Linux, NVIDIA GPU)
uv run pytest
uv run ruff check && uv run ruff format --check
uv run pyright
uv run python scripts/generate_reference.py   # after changing public names or docstrings
uv run --group docs mkdocs serve             # the docs as a local site: http://127.0.0.1:8000
uv run --group docs mkdocs build             # or write it to site/ (opens straight from disk)
```
