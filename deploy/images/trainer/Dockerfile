# A training pod: the training service over rollout-lora's trainers (a process per GPU under torchrun, kept between
# steps), the
# certificate's renewal, and Envoy in front with mutual TLS (deploy/images/trainer/README.md).
# Built from the repository's root, by .github/workflows/images.yml:
#   docker build -f deploy/images/trainer/Dockerfile .
# PyTorch's image, of uv.lock's PyTorch, and the CUDA libraries its wheels bring; the pod's driver is the host's. The
# platform's packages run on its Python, with its PyTorch (deploy/images/README.md). It has the C compiler Triton
# builds its kernels' launchers with when a step first runs them.
ARG TORCH_IMAGE=pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime
FROM envoyproxy/envoy:v1.38.5 AS envoy
FROM smallstep/step-cli:0.31.0 AS step
FROM ghcr.io/astral-sh/uv:0.11.19 AS uv
FROM ${TORCH_IMAGE} AS base

# The locked dependencies, apart: a change to the sources alone leaves this file, and the layer installed from it, as
# they were.
FROM base AS locked
RUN --mount=type=bind,target=/source --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    cd /source && deploy/images/common/install.sh export "rollout-train[http]" rollout-s3 rollout-lora >/locked.txt

FROM base
COPY --from=envoy /usr/local/bin/envoy /usr/local/bin/envoy
COPY --from=step /usr/local/bin/step /usr/local/bin/step
RUN envoy --version && step version
RUN --mount=from=locked,source=/locked.txt,target=/locked.txt \
    --mount=type=bind,source=deploy/images/common/install.sh,target=/install.sh \
    --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    /install.sh dependencies /locked.txt
COPY --chmod=755 deploy/images/common/pki.sh deploy/images/common/supervise.sh deploy/images/trainer/entrypoint.sh \
    /opt/rollout/bin/
COPY deploy/images/trainer/envoy.yaml /etc/envoy/envoy.yaml
COPY deploy/images/common/sds/ /etc/envoy/sds/
RUN --mount=type=bind,target=/source,rw --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    cd /source && deploy/images/common/install.sh packages libraries/rollout libraries/rollout-train \
        implementations/rollout-s3 implementations/rollout-objectives implementations/rollout-lora
WORKDIR /

EXPOSE 8443
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD /opt/rollout/venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/healthz')"
ENTRYPOINT ["/opt/rollout/bin/entrypoint.sh"]
CMD []
