# The platform: every workspace package with every extra (vLLM, the trainers, Tinker) in /opt/rollout/venv,
# rollout-verifiers (a project locked apart) in /opt/rollout/verifiers, a Java runtime for Paper and Node for the
# Minecraft harness. Ray's pods (KubeRay), the gateway, the monitor and the launchers all run it.
# Built from the repository's root:
#   buildctl build --frontend dockerfile.v0 --local context=. --local dockerfile=deploy/images/platform ...
FROM ghcr.io/astral-sh/uv:0.11.19 AS uv

FROM node:22-trixie-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl wget git openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv

ENV UV_PYTHON_INSTALL_DIR=/opt/rollout/python \
    UV_PROJECT_ENVIRONMENT=/opt/rollout/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/rollout/venv/bin:$PATH
WORKDIR /opt/rollout/source

# Dependencies first, from the lock alone, so a change to the sources reuses this layer.
COPY pyproject.toml uv.lock ./
COPY libraries libraries
COPY implementations implementations
COPY products products
COPY environments environments
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.13 \
    && uv sync --frozen --all-packages --all-extras --no-dev --no-install-workspace
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-packages --all-extras --no-dev \
    && cd environments/minecraft/harness && npm ci --omit=dev \
    && ray --version && java -version 2>&1 | head -1 && node --version
# rollout-verifiers in a Python environment of its own, from its own lock: GSM8K's launcher and the evals it starts;
# rsync, which deploy/k3s/migrate.sh copies run directories with.
RUN apt-get update && apt-get install -y --no-install-recommends rsync && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_PROJECT_ENVIRONMENT=/opt/rollout/verifiers uv sync --frozen --project implementations/rollout-verifiers \
    && /opt/rollout/verifiers/bin/python -c "import ray, rollout_s3, rollout_train, rollout_verifiers"

COPY . .
