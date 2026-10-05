# A training pod: the training service over rollout-lora's trainers (each step in a fresh process on the GPU), the
# certificate's renewal, and Envoy in front with mutual TLS (deploy/images/trainer/README.md).
# Built from the repository's root, by .github/workflows/images.yml:
#   docker build -f deploy/images/trainer/Dockerfile .
# PyTorch's wheels bring their CUDA libraries; the pod's driver is the host's.
FROM envoyproxy/envoy:v1.38.5 AS envoy
FROM smallstep/step-cli:0.31.0 AS step
FROM ghcr.io/astral-sh/uv:0.11.19 AS uv

FROM ubuntu:24.04
# (a C compiler: Triton builds its kernels' launchers when a step first runs them)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=envoy /usr/local/bin/envoy /usr/local/bin/envoy
COPY --from=step /usr/local/bin/step /usr/local/bin/step
COPY --from=uv /uv /usr/local/bin/uv
RUN envoy --version && step version

COPY . /opt/rollout/source
WORKDIR /opt/rollout/source
RUN deploy/images/common/install.sh "rollout-train[http]" rollout-s3 rollout-lora -- \
        libraries/rollout libraries/rollout-train implementations/rollout-s3 implementations/rollout-objectives \
        implementations/rollout-lora \
    && mkdir -p /opt/rollout/bin /etc/envoy/sds \
    && cp deploy/images/common/pki.sh deploy/images/common/supervise.sh deploy/images/trainer/entrypoint.sh \
        /opt/rollout/bin/ \
    && cp deploy/images/trainer/envoy.yaml /etc/envoy/envoy.yaml \
    && cp deploy/images/common/sds/*.yaml /etc/envoy/sds/ \
    && chmod +x /opt/rollout/bin/*.sh \
    && cd / && rm -rf /opt/rollout/source /root/.cache/uv
WORKDIR /

EXPOSE 8443
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD /opt/rollout/venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/healthz')"
ENTRYPOINT ["/opt/rollout/bin/entrypoint.sh"]
CMD []
