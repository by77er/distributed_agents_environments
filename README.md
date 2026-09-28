# rollout

Durable, scalable agents and environments for agent products and reinforcement learning. The core is a Python library
that runs on one machine (one GPU is enough); durability, fleet-scale execution and environments are optional layers.

- Design: [docs/README.md](docs/README.md)
- Development plan: [docs/development/plan.md](docs/development/plan.md)

## Development

```sh
uv sync                     # base install (no GPU needed)
uv sync --extra vllm        # local inference engine (Linux, NVIDIA GPU)
uv run pytest
uv run ruff check && uv run ruff format --check
uv run pyright
```
