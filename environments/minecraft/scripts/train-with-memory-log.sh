#!/bin/bash
# Run `minecraft-swarm train` with a memory log on disk, so a machine that runs out of memory leaves evidence.
#
#   train-with-memory-log.sh RUN_DIRECTORY [train options...]
#
# Writes RUN_DIRECTORY/train.log and RUN_DIRECTORY/memory.log (available system memory and GPU memory every 2 s), and
# serves the monitor (each episode and each agent, live) on http://localhost:${MONITOR_PORT:-8765} while it runs.
set -u
run="$1"; shift
mkdir -p "$run/feed"
(
  while true; do
    available=$(awk '/MemAvailable/ {printf "%.1f", $2 / 1048576}' /proc/meminfo)
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    echo "$(date +%T) ${available} GiB available, GPU ${gpu} MiB"
    sleep 2
  done
) >> "$run/memory.log" &
logger=$!
cd "$(dirname "$0")/../../.." || exit 1
uv run rollout-monitor "$run/feed" --port "${MONITOR_PORT:-8765}" > "$run/monitor.log" 2>&1 &
monitor=$!
trap 'kill "$logger" "$monitor" 2>/dev/null' EXIT
uv run minecraft-swarm train "$run" "$@" > "$run/train.log" 2>&1
status=$?
echo "exit $status" >> "$run/train.log"
exit "$status"
