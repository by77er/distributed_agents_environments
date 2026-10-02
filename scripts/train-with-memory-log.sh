#!/bin/bash
# Run `rollout train` with a memory log on disk, so a machine that runs out of memory leaves evidence.
#
#   train-with-memory-log.sh RUN_DIRECTORY PROFILE CATALOG [train options...]
#
# Writes RUN_DIRECTORY/train.log, memory.log (available system memory and GPU memory every 2 s) and, under WSL,
# host-memory.log (every 15 s: Windows' free memory, and GPU memory spilled into system memory, which should be 0), and
# serves the monitor (each episode and each agent, live) on http://localhost:${MONITOR_PORT:-8765} while it runs.
set -u
run="$1"; profile="$2"; catalog="$3"; shift 3
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
# Windows' own view of memory (the host, not the WSL VM): the VM's file cache counts against the host.
windows=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
if [ -x "$windows" ]; then
  (
    while true; do
      line=$(timeout 20 "$windows" -NoProfile -Command "\$o = Get-CimInstance Win32_OperatingSystem; \$v = (Get-Process vmmemWSL -ErrorAction SilentlyContinue).WorkingSet64; \$s = ((Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples | Measure-Object CookedValue -Sum).Sum; Write-Output \"host free \$([math]::Round(\$o.FreePhysicalMemory/1MB,1)) GB of \$([math]::Round(\$o.TotalVisibleMemorySize/1MB,1)), WSL holds \$([math]::Round(\$v/1GB,1)) GB, GPU spilled \$([math]::Round(\$s/1GB,1)) GB\"" 2>/dev/null | tr -d '\r')
      echo "$(date +%T) $line"
      sleep 15
    done
  ) >> "$run/host-memory.log" &
  host=$!
else
  host=
fi
cd "$(dirname "$0")/.." || exit 1
uv run rollout monitor "$run/feed" --port "${MONITOR_PORT:-8765}" > "$run/monitor.log" 2>&1 &
monitor=$!
trap 'kill "$logger" "$monitor" $host 2>/dev/null' EXIT
uv run rollout train "$profile" "$catalog" --directory "$run" "$@" >> "$run/train.log" 2>&1 &  # (a run started again goes on in the same log)
trainer=$!
# A signal to this script goes on to the trainer, and the script waits for it to end: otherwise the trainer would
# run on alone, with nothing logging its memory.
trap 'kill -INT "$trainer" 2>/dev/null' INT
trap 'kill -TERM "$trainer" 2>/dev/null' TERM HUP
status=0
while kill -0 "$trainer" 2>/dev/null; do
  wait "$trainer"
  status=$?
done
echo "exit $status" >> "$run/train.log"
exit "$status"
