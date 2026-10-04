#!/usr/bin/env bash
# Stops the platform's services on the host (deploy/k3s/cutover.md, step 1): the launchers, the gateway, the monitors,
# and the host's Ray head with its workers. Tests' own Ray sessions (under ~/.cache/rollout/ray-tests) and the
# cluster's containers are left running. Each is asked to stop (SIGINT), given 30 seconds, then terminated (SIGTERM).
set -uo pipefail

SERVICES='^\S*(python3?|uv run) \S*(rollout|rollout_train\.cli|-m rollout_train\.cli) (launcher|gateway|monitor) '
HEAD='^\S*gcs_server --log_dir=/home/bit/.cache/ray/session'

host() {  # the services' processes and the host's Ray head's session, outside the cluster's containers
  local pid head
  for pid in $(pgrep -u "$USER" -f "$SERVICES"); do
    grep -q kubepods "/proc/$pid/cgroup" 2>/dev/null || echo "$pid"
  done
  head=$(pgrep -u "$USER" -f "$HEAD" | head -1)
  [[ -n $head ]] && pgrep -s "$(ps -o sid= -p "$head" | tr -d ' ')"
}

pids=$(host | sort -u)
[[ -z $pids ]] && { echo "nothing of the platform runs on the host"; exit 0; }
kill -INT $pids 2>/dev/null
for _ in $(seq 30); do
  pids=$(host | sort -u)
  [[ -z $pids ]] && break
  sleep 1
done
if [[ -n $pids ]]; then
  kill -TERM $pids 2>/dev/null
  sleep 5
  pids=$(host | sort -u)
fi
if [[ -n $pids ]]; then
  echo "still running:"; ps -o pid=,args= -p ${pids//$'\n'/,} | cut -c1-160; exit 1
fi
echo "the host's services are stopped"
