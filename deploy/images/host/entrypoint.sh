#!/usr/bin/env bash
# A host pod (deploy/images/host/README.md): its certificate first, then the renewal daemon, Envoy, the vLLM server
# (with its share of the GPU's memory), the follower and the training service. When any of them ends, the others are
# ended and the container exits.
set -euo pipefail
source /opt/rollout/bin/supervise.sh

for name in ROLLOUT_POD_NAME ROLLOUT_MODEL ROLLOUT_TRAINER ROLLOUT_TRAINER_MODEL ROLLOUT_LEDGER ROLLOUT_BLOBS \
    STEP_CA_URL STEP_FINGERPRINT; do
    if [ -z "${!name:-}" ]; then
        echo "$name is not set" >&2
        exit 1
    fi
done

/opt/rollout/bin/pki.sh bootstrap
unset STEP_TOKEN # (used, and good for nothing more: no other process sees it)

export HF_HOME=${HF_HOME:-/workspace/huggingface}
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export ROLLOUT_ROLE=host
# (what the training service keeps and the follower loads, on the pod's own disk: no round trip through the bucket)
export ROLLOUT_BLOB_CACHE=${ROLLOUT_BLOB_CACHE:-/workspace/blobs}
read -r -a vllm_options <<<"${VLLM_ARGS:---gpu-memory-utilization 0.42}"
lora_options=(--max-loras "${VLLM_MAX_LORAS:-2}")
if [ -n "${VLLM_MAX_CPU_LORAS:-}" ]; then
    lora_options+=(--max-cpu-loras "$VLLM_MAX_CPU_LORAS")
fi
sleeping=()
if [ "${ROLLOUT_SLEEP_VLLM:-}" = 1 ]; then
    sleeping=(--enable-sleep-mode)
fi

start certificates /opt/rollout/bin/pki.sh renew
start envoy envoy --config-path "${ENVOY_CONFIG:-/etc/envoy/envoy.yaml}" --log-level "${ENVOY_LOG_LEVEL:-warn}"
start vllm vllm serve "$ROLLOUT_MODEL" --host 127.0.0.1 --port 8000 \
    --enable-lora --max-lora-rank "${VLLM_MAX_LORA_RANK:-32}" "${lora_options[@]}" "${sleeping[@]}" \
    --logprobs-mode processed_logprobs "${vllm_options[@]}"
start trainer /opt/rollout/venv/bin/python -m rollout_train.pods.training
# (the follower says the pod is ready once vLLM serves what the run says and the training service holds its trainer)
export ROLLOUT_TRAINER_URL=${ROLLOUT_TRAINER_URL:-http://127.0.0.1:8001}
start follower /opt/rollout/venv/bin/python -m rollout_train.pods.inference

supervise
