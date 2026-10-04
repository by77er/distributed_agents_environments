# Cutover: from the host's services to the K3s cluster

The platform runs in the K3s cluster from the chart `deploy/chart/rollout` (release `rollout`, namespace `rollout`),
installed as [README.md](README.md) says, with its Secrets made and `deploy/k3s/migrate.sh` run once while the host's
services still ran. These steps move the platform for good: stop the host's services, copy their state a last time,
use the cluster, check it, and how to go back. Run them from the main checkout, at the repository's root.

```sh
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
alias kubectl='k3s kubectl'
```

## 1. Stop the host's services

Nothing should be playing: `RAY_AUTH_MODE=token uv run ray job list --address http://127.0.0.1:8265` (in
`~/Code/distributed_agents_environments-run`) shows the Minecraft launcher's job and no `run-…` job running, and the
monitor's launches show none `running`. Then:

```sh
deploy/k3s/stop-host.sh
```

It asks the GSM8K launcher, the gateway on 127.0.0.1:8900, the monitors on 8765 and 8766, the Minecraft launcher,
and the host's Ray head with its workers (one process session) to stop, terminates what still runs after 30
seconds, and lists anything left. It leaves tests' own Ray
sessions (under `~/.cache/rollout/ray-tests`) and the cluster's containers running; `ray stop` would stop those too.

## 2. Run the final migration

```sh
deploy/k3s/migrate.sh --final
```

It refuses while a launcher, gateway, monitor or Ray process of yours runs outside the cluster. It copies what changed
since the last run onto the state volume and into the bucket, scales the launchers, the gateway and the monitors to
zero, replaces the cluster's ledger with a fresh copy of `~/.cache/rollout/ledger.db` (so whatever the cluster recorded
during rehearsals is gone), and scales them back. Note the row counts it prints for the ledger's copy: step 4 checks
them. `~/.cache/rollout` is only read.

## 3. Point things at the cluster

| Was | Is |
|---|---|
| the monitor at `http://localhost:8765` | `http://monitor.localhost` |
| the eval's monitor at `http://localhost:8766` | `http://astra.monitor.localhost` |
| the gateway at `http://127.0.0.1:8900` | `http://gateway.localhost` from this machine and Windows; `http://gateway.rollout:8900` in the cluster |
| Ray's dashboard at `http://127.0.0.1:8265` | `http://ray.localhost`, with the token `kubectl -n rollout get secret ray -o jsonpath='{.data.auth_token}' \| base64 -d` |
| `--ledger sqlite:///~/.cache/rollout/ledger.db` | `--cluster` in a pod, which finds the cluster config at `ROLLOUT_CLUSTER`: `kubectl -n rollout exec deploy/launcher-minecraft -- rollout checkpoints --cluster` |
| run directories in `~/.cache/rollout/runs` | `/root/.cache/rollout/runs` on the volume `state`, in every pod |
| blobs in `RUN/blobs`, `~/.cache/rollout/gsm8k-tinker/blobs`, `~/.cache/rollout/datasets/blobs` | `s3://rollout-blobs/blobs` at `http://s3.rollout:7070` |

New runs and evals are asked for from the monitor's **New run** form, as before; the launchers in the cluster start
them as Ray jobs. Leave `~/.cache/rollout` as it is until the cluster has run for a while: it is what step 5 goes back
to.

## 4. Check it

```sh
kubectl -n rollout get pods                     # postgres-0, s3-0, ray-head-…, launcher-minecraft-…, launcher-gsm8k-…,
                                                # gateway-…, monitor-main-…, monitor-astra-…: Running and ready
kubectl -n rollout exec postgres-0 -- psql -U rollout -c \
  "SELECT runner, now() - to_timestamp(at) AS ago FROM presence ORDER BY runner"
                                                # launcher/launcher-minecraft/minecraft, launcher/launcher-gsm8k/gsm8k
                                                # and gateway/…: each seconds ago
kubectl -n rollout exec postgres-0 -- psql -U rollout -c "SELECT count(*) FROM ledger_records"
                                                # what migrate.sh printed for ledger_records
kubectl -n rollout exec deploy/launcher-minecraft -- rollout cluster check
                                                # every secret and project resolves, except $TINKER_API_KEY where the
                                                # Secret tinker holds credentials.json (Tinker's SDK reads that instead)
kubectl -n rollout exec deploy/launcher-minecraft -- rollout checkpoints --cluster   # curriculum-9's 31 checkpoints
curl -s -H 'Host: gateway.localhost' http://127.0.0.1/v1/models          # {"object":"list","data":[{"id":"policy",…
curl -s -H 'Host: monitor.localhost' http://127.0.0.1/api/system | python3 -c \
  'import json,sys; print(sorted(run["name"] for run in json.load(sys.stdin)["runs"]))'
                                                # qwencraft-1 (curriculum-9) and gsm8k-tinker-base among them
curl -s -o /dev/null -w '%{http_code}\n' -H 'Host: astra.monitor.localhost' http://127.0.0.1/   # 200
```

In a browser: curriculum-9's groups and episodes, its checkpoints, the `math` suite's eval, and the Machines tab with
the two launchers and the gateway. A GPU job, to see the GPU group start (it takes the GPU while it runs):

```sh
kubectl -n rollout exec deploy/launcher-minecraft -- ray job submit --address http://ray-head-svc:8265 \
  --entrypoint-num-gpus 1 -- python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## 5. Roll back

The migration only read the host's files, so the host starts again where it stopped. What the cluster recorded after
the cutover stays in its ledger, bucket and volume and is not copied back.

```sh
kubectl -n rollout scale deployment -l app.kubernetes.io/part-of=rollout --replicas=0   # launchers, gateway, monitors
kubectl -n rollout get pods -l ray.io/node-type=worker   # no Ray job left on the GPU: the group goes to zero when idle
cd ~/Code/distributed_agents_environments-run
RAY_AUTH_MODE=token RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run ray start --head --num-gpus 1 \
  --object-store-memory 1000000000 --temp-dir ~/.cache/ray   # with the token in ~/.ray/auth_token, as before
uv run python -m rollout_train.cli launcher --ledger sqlite:///~/.cache/rollout/ledger.db \
  --profiles environments/minecraft/profiles --environment minecraft_team.environment:environment \
  --runs ~/.cache/rollout/runs --ray http://127.0.0.1:8265 --as-job
(cd implementations/rollout-verifiers && nohup uv run rollout launcher --ledger sqlite:///~/.cache/rollout/ledger.db \
  --profiles ~/.cache/rollout/launch-profiles/gsm8k --environment rollout_verifiers.environments:gsm8k \
  --runs ~/.cache/rollout/runs --at-once 2 --name gsm8k > ~/.cache/rollout/launcher-gsm8k.log 2>&1 &)
(cd implementations/rollout-tinker && nohup uv run rollout gateway ../rollout-verifiers/examples/gsm8k_tinker.toml \
  > ~/.cache/rollout/gateway-gsm8k-tinker.log 2>&1 &)
nohup uv run rollout monitor ~/.cache/rollout/runs/curriculum-9 --port 8765 > ~/.cache/rollout/monitor-8765.log 2>&1 &
(cd ~/Code/distributed_agents_environments && nohup uv run rollout monitor ~/.cache/rollout/evaluations/astra-t054u \
  --port 8766 > ~/.cache/rollout/evaluations/astra-t054u/monitor.log 2>&1 &)
```

To go forward again later, start from step 1: `migrate.sh --final` replaces the cluster's ledger with the host's.
