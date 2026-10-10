# A host pod: a stock vLLM server and the follower that keeps it serving what the run that holds the pod says, the
# training service over rollout-lora's trainers beside it on the same GPU, Minecraft's sandbox pool on the pod's spare
# CPUs and memory, the certificate's renewal, and Envoy in front with mutual TLS (deploy/images/host/README.md).
# Built from the repository's root, by .github/workflows/images.yml:
#   docker build -f deploy/images/host/Dockerfile --build-arg VLLM_VERSION=0.30.0 .
# VLLM_VERSION is the workspace's (uv.lock), so the server is the vLLM the engines here are tested with. The platform's
# packages run on the vLLM image's Python, with its PyTorch, transformers and CUDA libraries (deploy/images/README.md).
ARG VLLM_VERSION=0.30.0
FROM envoyproxy/envoy:v1.38.5 AS envoy
FROM smallstep/step-cli:0.31.0 AS step
FROM ghcr.io/astral-sh/uv:0.11.19 AS uv
# Java 21 (a runtime: Paper runs on it) and Node 22 (the Minecraft harness's bots), each a self-contained build.
FROM eclipse-temurin:21-jre AS java
FROM node:22-bookworm-slim AS node
FROM vllm/vllm-openai:v${VLLM_VERSION} AS base

# The Minecraft harness's packages, from its lock alone.
FROM node AS harness
WORKDIR /harness
COPY environments/minecraft/harness/package.json environments/minecraft/harness/package-lock.json ./
RUN npm ci --omit=dev

# The locked dependencies, apart: a change to the sources alone leaves this file, and the layer installed from it, as
# they were.
FROM base AS locked
RUN --mount=type=bind,target=/source --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    cd /source && deploy/images/common/install.sh export "rollout-train[http]" rollout-s3 rollout-lora minecraft-team \
        >/locked.txt

FROM base
COPY --from=envoy /usr/local/bin/envoy /usr/local/bin/envoy
COPY --from=step /usr/local/bin/step /usr/local/bin/step
COPY --from=java /opt/java/openjdk /opt/java/openjdk
COPY --from=node /usr/local/bin/node /usr/local/bin/node
ENV JAVA_HOME=/opt/java/openjdk PATH=/opt/java/openjdk/bin:$PATH
RUN envoy --version && step version && java -version && node --version
RUN --mount=from=locked,source=/locked.txt,target=/locked.txt \
    --mount=type=bind,source=deploy/images/common/install.sh,target=/install.sh \
    --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    /install.sh dependencies /locked.txt
COPY --chmod=755 deploy/images/common/pki.sh deploy/images/common/supervise.sh deploy/images/host/entrypoint.sh \
    /opt/rollout/bin/
COPY deploy/images/host/envoy.yaml /etc/envoy/envoy.yaml
COPY deploy/images/common/sds/ /etc/envoy/sds/
RUN --mount=type=bind,target=/source,rw --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    cd /source && deploy/images/common/install.sh packages libraries/rollout libraries/rollout-train \
        implementations/rollout-s3 implementations/rollout-objectives implementations/rollout-lora
# The Minecraft environment where its package finds its plugin's sources, its servers' configuration and its harness
# (beside it, with the harness's packages): installed editable, from /opt/rollout/minecraft.
COPY environments/minecraft /opt/rollout/minecraft
COPY --from=harness /harness/node_modules /opt/rollout/minecraft/harness/node_modules
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    UV_NO_CACHE=1 UV_COMPILE_BYTECODE=1 \
        uv pip install --python /opt/rollout/venv/bin/python --no-deps --editable /opt/rollout/minecraft \
    && /opt/rollout/venv/bin/python -c "import minecraft_team.worlds"
WORKDIR /

EXPOSE 8443
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s \
    CMD /opt/rollout/venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8081/healthz'); urllib.request.urlopen('http://127.0.0.1:8001/healthz')"
ENTRYPOINT ["/opt/rollout/bin/entrypoint.sh"]
CMD []
