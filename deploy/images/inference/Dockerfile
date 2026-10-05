# An inference pod: a stock vLLM server, the follower that keeps it serving what its run's channel should, the
# certificate's renewal, and Envoy in front with mutual TLS (deploy/images/inference/README.md).
# Built from the repository's root, by .github/workflows/images.yml:
#   docker build -f deploy/images/inference/Dockerfile --build-arg VLLM_VERSION=0.30.0 .
# VLLM_VERSION is the workspace's (uv.lock), so the server is the vLLM the engines here are tested with. The follower
# runs on the vLLM image's Python, with the packages it has (deploy/images/README.md).
ARG VLLM_VERSION=0.30.0
FROM envoyproxy/envoy:v1.38.5 AS envoy
FROM smallstep/step-cli:0.31.0 AS step
FROM ghcr.io/astral-sh/uv:0.11.19 AS uv
FROM vllm/vllm-openai:v${VLLM_VERSION} AS base

# The locked dependencies, apart: a change to the sources alone leaves this file, and the layer installed from it, as
# they were.
FROM base AS locked
RUN --mount=type=bind,target=/source --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    cd /source && deploy/images/common/install.sh export "rollout-train[http]" rollout-s3 >/locked.txt

FROM base
COPY --from=envoy /usr/local/bin/envoy /usr/local/bin/envoy
COPY --from=step /usr/local/bin/step /usr/local/bin/step
RUN envoy --version && step version
RUN --mount=from=locked,source=/locked.txt,target=/locked.txt \
    --mount=type=bind,source=deploy/images/common/install.sh,target=/install.sh \
    --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    /install.sh dependencies /locked.txt
COPY --chmod=755 deploy/images/common/pki.sh deploy/images/common/supervise.sh deploy/images/inference/entrypoint.sh \
    /opt/rollout/bin/
COPY deploy/images/inference/envoy.yaml /etc/envoy/envoy.yaml
COPY deploy/images/common/sds/ /etc/envoy/sds/
RUN --mount=type=bind,target=/source,rw --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    cd /source && deploy/images/common/install.sh packages libraries/rollout libraries/rollout-train \
        implementations/rollout-s3
WORKDIR /

EXPOSE 8443
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s \
    CMD /opt/rollout/venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8081/healthz')"
ENTRYPOINT ["/opt/rollout/bin/entrypoint.sh"]
CMD []
