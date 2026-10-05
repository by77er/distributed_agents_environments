#!/usr/bin/env bash
# An inference pod (deploy/images/inference/README.md): its certificate first, then the renewal daemon, Envoy, the
# vLLM server and the follower. When any of them ends, the others are ended and the container exits.
set -euo pipefail
source /opt/rollout/bin/supervise.sh

for name in ROLLOUT_POD_NAME ROLLOUT_MODEL ROLLOUT_LEDGER ROLLOUT_BLOBS STEP_CA_URL STEP_FINGERPRINT; do
    if [ -z "${!name:-}" ]; then
        echo "$name is not set" >&2
        exit 1
    fi
done

/opt/rollout/bin/pki.sh bootstrap
unset STEP_TOKEN # (used, and good for nothing more: no other process sees it)

export HF_HOME=${HF_HOME:-/workspace/huggingface}
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
read -r -a vllm_options <<<"${VLLM_ARGS:-}"
# Adapters one batch may mix: every bound run's window (its max_lag + 1) side by side; the CPU cache holds more.
lora_options=(--max-loras "${VLLM_MAX_LORAS:-2}")
if [ -n "${VLLM_MAX_CPU_LORAS:-}" ]; then
    lora_options+=(--max-cpu-loras "$VLLM_MAX_CPU_LORAS")
fi

start certificates /opt/rollout/bin/pki.sh renew
start envoy envoy --config-path "${ENVOY_CONFIG:-/etc/envoy/envoy.yaml}" --log-level "${ENVOY_LOG_LEVEL:-warn}" \
    --concurrency "${ENVOY_CONCURRENCY:-4}" # (by default Envoy runs a worker per hardware thread the machine has, not the pod)
start vllm vllm serve "$ROLLOUT_MODEL" --host 127.0.0.1 --port 8000 \
    --enable-lora --max-lora-rank "${VLLM_MAX_LORA_RANK:-32}" "${lora_options[@]}" \
    --logprobs-mode processed_logprobs "${vllm_options[@]}"
start follower /opt/rollout/venv/bin/python -m rollout_train.pods.inference

supervise
