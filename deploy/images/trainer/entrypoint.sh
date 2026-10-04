#!/usr/bin/env bash
# A training pod (deploy/images/trainer/README.md): its certificate first, then the renewal daemon, Envoy and the
# training service. When any of them ends, the others are ended and the container exits.
set -euo pipefail
source /opt/rollout/bin/supervise.sh

for name in ROLLOUT_POD_NAME ROLLOUT_TRAINER ROLLOUT_TRAINER_MODEL ROLLOUT_LEDGER ROLLOUT_BLOBS STEP_CA_URL \
    STEP_FINGERPRINT; do
    if [ -z "${!name:-}" ]; then
        echo "$name is not set" >&2
        exit 1
    fi
done

/opt/rollout/bin/pki.sh bootstrap
unset STEP_TOKEN # (used, and good for nothing more: no other process sees it)

export HF_HOME=${HF_HOME:-/workspace/huggingface}

start certificates /opt/rollout/bin/pki.sh renew
start envoy envoy --config-path "${ENVOY_CONFIG:-/etc/envoy/envoy.yaml}" --log-level "${ENVOY_LOG_LEVEL:-warn}"
start trainer /opt/rollout/venv/bin/python -m rollout_train.pods.training

supervise
