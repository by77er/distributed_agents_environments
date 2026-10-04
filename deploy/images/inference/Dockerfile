# An inference pod: a stock vLLM server, the follower that keeps it serving what its run's channel should, the
# certificate's renewal, and Envoy in front with mutual TLS (deploy/images/inference/README.md).
# Built from the repository's root, by .github/workflows/images.yml:
#   docker build -f deploy/images/inference/Dockerfile --build-arg VLLM_VERSION=0.30.0 .
# VLLM_VERSION is the workspace's (uv.lock), so the server is the vLLM the engines here are tested with.
ARG VLLM_VERSION=0.30.0
FROM envoyproxy/envoy:v1.38.5 AS envoy
FROM smallstep/step-cli:0.31.0 AS step
FROM ghcr.io/astral-sh/uv:0.11.19 AS uv

FROM vllm/vllm-openai:v${VLLM_VERSION}
COPY --from=envoy /usr/local/bin/envoy /usr/local/bin/envoy
COPY --from=step /usr/local/bin/step /usr/local/bin/step
COPY --from=uv /uv /usr/local/bin/uv
RUN envoy --version && step version

COPY . /opt/rollout/source
WORKDIR /opt/rollout/source
RUN deploy/images/common/install.sh "rollout-train[http]" rollout-s3 -- \
        libraries/rollout libraries/rollout-train implementations/rollout-s3 \
    && mkdir -p /opt/rollout/bin /etc/envoy/sds \
    && cp deploy/images/common/pki.sh deploy/images/common/supervise.sh deploy/images/inference/entrypoint.sh \
        /opt/rollout/bin/ \
    && cp deploy/images/inference/envoy.yaml /etc/envoy/envoy.yaml \
    && cp deploy/images/common/sds/*.yaml /etc/envoy/sds/ \
    && chmod +x /opt/rollout/bin/*.sh \
    && cd / && rm -rf /opt/rollout/source /root/.cache/uv
WORKDIR /

EXPOSE 8443
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s \
    CMD /opt/rollout/venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8081/healthz')"
ENTRYPOINT ["/opt/rollout/bin/entrypoint.sh"]
CMD []
