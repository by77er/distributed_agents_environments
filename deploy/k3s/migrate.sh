#!/usr/bin/env bash
# Copy the platform's state on this machine into the K3s cluster's chart (deploy/chart/rollout, release `rollout`),
# reading the host's files and never changing them. Safe to run again: each run copies what is new, and replaces the
# cluster's ledger with a fresh copy of the host's.
#
#   deploy/k3s/migrate.sh --dry-run   say what would be copied and rewritten; change nothing anywhere
#   deploy/k3s/migrate.sh             copy, while the host's services still run (a rehearsal)
#   deploy/k3s/migrate.sh --final     the same, refused while a host service still runs (deploy/k3s/cutover.md)
#
# What it copies, through a pod that mounts the host's ~/.cache/rollout read-only at the same path:
#   1. run directories and other local state onto the volume every pod mounts (`state`, at /root/.cache/rollout):
#      runs/, evaluations/, datasets/, gsm8k-tinker/, minecraft/ (its temporary servers left out), jdk/; each run's
#      ledger.json then names the cluster's ledger;
#   2. the stores of files the ledger's records point into (gsm8k-tinker/blobs, datasets/blobs, and each run's
#      RUN/blobs) into the bucket, s3://rollout-blobs/blobs (python -m rollout_s3.copying);
#   3. the ledger: a consistent copy of ledger.db (SQLite's backup, read-only), its records rewritten so that the stores
#      that moved are named as the bucket and paths under ~/.cache/rollout as /root/.cache/rollout
#      (python -m rollout_train.relocating), then copied into a new, empty `rollout` database in Postgres
#      (rollout ledger copy). The launchers, the gateway and the monitors are scaled to zero meanwhile.
# Everything it writes on this machine is under ~/.cache/rollout-migration.
set -euo pipefail

mode=copy
case "${1:-}" in
  --dry-run) mode=dry-run ;;
  --final) mode=final ;;
  "") ;;
  *) echo "usage: $0 [--dry-run | --final]" >&2; exit 2 ;;
esac

export KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
kubectl() { k3s kubectl -n "$NAMESPACE" "$@"; }
NAMESPACE=${NAMESPACE:-rollout}
SOURCE=$HOME/.cache/rollout                      # the host's state, read only
WORK=$HOME/.cache/rollout-migration              # this script's own files
STATE=/root/.cache/rollout                       # where the pods mount the state volume
LEDGER_URL=postgresql://rollout@postgres.$NAMESPACE:5432/rollout
BUCKET=rollout-blobs
INTO='{"kind": "rollout_s3:S3BlobStore", "bucket": "rollout-blobs", "prefix": "blobs/"}'
IMAGE=${IMAGE:-localhost:30500/rollout-platform:dev}
COPIED=(runs evaluations datasets gsm8k-tinker minecraft jdk)
NAMED=("$SOURCE/gsm8k-tinker/blobs" "$SOURCE/datasets/blobs")   # stores the records name by their location
POD=migrate

say() { printf '\n== %s\n' "$*"; }

host_services() {  # this user's launchers, gateways, monitors and Ray processes, outside the cluster's containers
  local pid
  for pid in $(pgrep -u "$(id -u)" -f 'rollout(_train\.cli)? (launcher|gateway|monitor)|gcs_server|raylet' || true); do
    grep -q kubepods "/proc/$pid/cgroup" 2>/dev/null || ps -o pid=,args= -p "$pid" | cut -c1-160
  done
}
if [[ $mode == final ]]; then
  say "host services"
  if [[ -n $(host_services) ]]; then
    host_services >&2
    echo "these still run on the host: stop them first (deploy/k3s/cutover.md, step 1)" >&2
    exit 1
  fi
  echo "none"
fi

say "ledger: a consistent copy of $SOURCE/ledger.db"
mkdir -p "$WORK"
python3 - "$SOURCE/ledger.db" "$WORK/ledger.db" <<'EOF'
import sqlite3, sys
source = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)  # read only: the live ledger is never changed
target = sqlite3.connect(sys.argv[2] + ".new")
source.backup(target)
counts = {name: target.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
          for (name,) in target.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")}
target.close()
source.close()
import os
os.replace(sys.argv[2] + ".new", sys.argv[2])
print(", ".join(f"{count} rows of {name}" for name, count in counts.items()))
EOF

say "sizes on the host"
(cd "$SOURCE" && du -sh "${COPIED[@]}" 2>/dev/null) || true

say "the migration pod"
kubectl delete pod "$POD" --ignore-not-found --wait=true >/dev/null
kubectl apply -f - >/dev/null <<EOF
apiVersion: v1
kind: Pod
metadata: {name: $POD, labels: {app: migrate}}
spec:
  restartPolicy: Never
  containers:
    - name: migrate
      image: $IMAGE
      imagePullPolicy: Always
      command: [sleep, infinity]
      env:
        - {name: PGPASSWORD, valueFrom: {secretKeyRef: {name: stores, key: POSTGRES_PASSWORD}}}
        - {name: AWS_ENDPOINT_URL, value: "http://s3.$NAMESPACE:7070"}
        - {name: AWS_DEFAULT_REGION, value: us-east-1}
        - {name: AWS_ACCESS_KEY_ID, valueFrom: {secretKeyRef: {name: stores, key: ROOT_ACCESS_KEY_ID}}}
        - {name: AWS_SECRET_ACCESS_KEY, valueFrom: {secretKeyRef: {name: stores, key: ROOT_SECRET_ACCESS_KEY}}}
      volumeMounts:
        - {name: source, mountPath: $SOURCE, readOnly: true}
        - {name: work, mountPath: $WORK}
        - {name: state, mountPath: $STATE}
      resources: {requests: {cpu: "1", memory: 1Gi}, limits: {memory: 4Gi}}
  volumes:
    - {name: source, hostPath: {path: $SOURCE, type: Directory}}
    - {name: work, hostPath: {path: $WORK, type: Directory}}
    - {name: state, persistentVolumeClaim: {claimName: state}}
EOF
trap 'kubectl delete pod "$POD" --ignore-not-found --wait=false >/dev/null' EXIT
kubectl wait --for=condition=Ready "pod/$POD" --timeout=300s >/dev/null
inside() { kubectl exec -i "$POD" -- "$@"; }

say "local state onto the volume ($STATE)"
dry=()
[[ $mode == dry-run ]] && dry=(--dry-run)
for each in "${COPIED[@]}"; do
  [[ -e $SOURCE/$each ]] || continue
  echo "-- $each"
  inside rsync -aH "${dry[@]}" --stats --exclude '/minecraft/servers/*' "$SOURCE/$each" "$STATE/" \
    | grep -E '^(Number of (regular )?files transferred|Total transferred file size)' || true
done
if [[ $mode != dry-run ]]; then
  inside python - "$STATE" "$LEDGER_URL" <<'EOF'
import json, sys
from pathlib import Path
state, url = Path(sys.argv[1]), sys.argv[2]
pointed = 0
for path in [*state.glob("runs/*/ledger.json"), *state.glob("evaluations/*/ledger.json")]:
    said = json.loads(path.read_text())
    if str(said.get("url", "")).startswith("sqlite:"):
        path.write_text(json.dumps({"kind": "rollout_train.database:DatabaseLedger", "url": url}))
        pointed += 1
print(f"{pointed} run directories now name {url}")
EOF
fi

say "blobs into s3://$BUCKET/blobs"
stores=("${NAMED[@]}")
for each in "$SOURCE"/runs/*/blobs; do [[ -d $each ]] && stores+=("$each"); done
printf '   %s\n' "${stores[@]}"
inside python -m rollout_s3.copying "${stores[@]}" --to "s3://$BUCKET/blobs" "${dry[@]}"

say "the ledger's records rewritten (on the copy)"
relocate=(--into "$INTO" --uris "s3://$BUCKET/blobs/" --path "$SOURCE=$STATE" --home "$HOME")
for each in "${NAMED[@]}"; do relocate+=(--store "$each"); done
inside cp "$WORK/ledger.db" "$WORK/relocated.db"
inside python -m rollout_train.relocating "sqlite:///$WORK/relocated.db" "${relocate[@]}"

if [[ $mode == dry-run ]]; then
  say "dry run: nothing was changed in the cluster"
  exit 0
fi

say "the platform's processes scaled to zero"
workloads=$(kubectl get deployment -l app.kubernetes.io/part-of=rollout -o name)
declare -A replicas
for each in $workloads; do replicas[$each]=$(kubectl get "$each" -o jsonpath='{.spec.replicas}'); done
kubectl scale $workloads --replicas=0 >/dev/null
kubectl wait --for=delete pod -l 'app in (launcher, gateway, monitor)' --timeout=120s >/dev/null 2>&1 || true
restore() {
  for each in "${!replicas[@]}"; do kubectl scale "$each" --replicas="${replicas[$each]}" >/dev/null; done
  kubectl delete pod "$POD" --ignore-not-found --wait=false >/dev/null
}
trap restore EXIT

say "the ledger into a new, empty database: $LEDGER_URL"
inside python - <<'EOF'
import psycopg
with psycopg.connect("postgresql://rollout@postgres:5432/postgres", autocommit=True) as connection:
    connection.execute("DROP DATABASE IF EXISTS rollout WITH (FORCE)")
    connection.execute("CREATE DATABASE rollout")
EOF
inside rollout ledger copy "sqlite:///$WORK/relocated.db" "$LEDGER_URL"

say "the platform's processes scaled back"
restore
trap - EXIT
kubectl rollout status deployment -l app.kubernetes.io/part-of=rollout --timeout=300s 2>/dev/null \
  || kubectl get pods -l 'app in (launcher, gateway, monitor)'
say "done ($mode)"
