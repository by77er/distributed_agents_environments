#!/usr/bin/env bash
# Stops the platform's services on the host (deploy/k3s/cutover.md, step 1): the launchers, the gateway, the monitors,
# and the host's Ray head with its workers. Tests' own Ray sessions (under ~/.cache/rollout/ray-tests) and the
# cluster's containers are left running.
set -uo pipefail

pkill -u "$USER" -INT -f 'rollout launcher .*--name gsm8k'          # the GSM8K launcher
pkill -u "$USER" -INT -f 'rollout gateway .*gsm8k_tinker.toml'      # the gateway on 127.0.0.1:8900
pkill -u "$USER" -INT -f '^\S*python3? \S*rollout monitor '         # the monitors on 8765 and 8766
pkill -u "$USER" -INT -f 'rollout_train.cli launcher --ledger sqlite'   # the Minecraft launcher (a Ray job's process)
head=$(pgrep -u "$USER" -f '^\S*gcs_server --log_dir=/home/bit/.cache/ray/session' | head -1)
if [[ -n $head ]]; then
  pkill -TERM -s "$(ps -o sid= -p "$head" | tr -d ' ')"   # the host's Ray head and its workers: one process session
fi
sleep 5
left=$(pgrep -u "$USER" -f 'rollout(_train\.cli)? (launcher|gateway|monitor)|^\S*gcs_server --log_dir=/home/bit/.cache/ray/' \
  | while read -r pid; do grep -q kubepods "/proc/$pid/cgroup" 2>/dev/null || ps -o pid=,args= -p "$pid" | cut -c1-160; done)
if [[ -n $left ]]; then
  echo "still running:"; echo "$left"; exit 1
fi
echo "the host's services are stopped"
