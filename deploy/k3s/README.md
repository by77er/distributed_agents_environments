# K3s on WSL2

A one-node Kubernetes cluster on the WSL2 machine, with KubeRay and the NVIDIA GPU reachable from pods, and the
platform in it: the chart `deploy/chart/rollout`. Nothing here locks the card: Kubernetes only accounts for it, and
processes outside the cluster keep using it.

| File | Does |
|---|---|
| `install-wsl.sh` | As root: installs NVIDIA's container toolkit with a CDI spec for WSL2's GPU (`/dev/dxg`), installs K3s with a kubeconfig your user can read, and moves containerd's stream server to port 9910, outside Ray's worker ports (10002-19999) |
| `device-plugin.yaml` | Values for NVIDIA's device plugin chart: the node advertises its card as one `nvidia.com/gpu`, for one Ray worker pod, and Ray shares it among its actors with fractional `num_gpus` |
| `build.yaml` | Building images in the cluster: a registry the node pulls from at `localhost:30500`, and BuildKit |
| `ray-smoke.yaml` | A Ray cluster whose GPU worker group sits at zero pods until a task asks for a GPU; the autoscaler removes the worker after a minute idle |
| `migrate.sh` | Copies the platform's state on this machine into the chart's stores and volume, reading the host's files only; `--dry-run` says what it would do |
| `cutover.md` | Moving from the services on the host to the cluster, step by step, and back |

## Setup

```sh
sudo bash deploy/k3s/install-wsl.sh
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
kubectl label node "$(hostname | tr A-Z a-z)" nvidia.com/gpu.present=true  # WSL2 has no PCI labels for the chart's affinity
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm upgrade --install nvidia-device-plugin nvdp/nvidia-device-plugin --version 0.20.1 \
  -n nvidia-device-plugin --create-namespace -f deploy/k3s/device-plugin.yaml
helm upgrade --install kuberay-operator kuberay/kuberay-operator --version 1.7.1 -n kuberay --create-namespace
kubectl apply -f deploy/k3s/build.yaml
```

## Images

`deploy/images/platform` is built in the cluster from the repository's root and pulled by the node as
`localhost:30500/rollout-platform:TAG`. It holds the workspace (`/opt/rollout/venv`) and rollout-verifiers, locked
apart, in a Python environment of its own (`/opt/rollout/verifiers`):

```sh
kubectl -n build port-forward statefulset/buildkit 1234:1234 &
buildctl --addr tcp://127.0.0.1:1234 build --frontend dockerfile.v0 --local context=. \
  --local dockerfile=deploy/images/platform \
  --output type=image,name=registry.build.svc.cluster.local:5000/rollout-platform:dev,push=true
```

## The platform

The chart installs into the namespace `rollout`, by role:

| Role | What runs | Reached at |
|---|---|---|
| Stores | StatefulSets `postgres` (the ledger) and `s3` (versitygw, the bucket `rollout-blobs`), each on its own volume | `postgresql://rollout@postgres.rollout:5432/rollout` (the password: `PGPASSWORD`), `http://s3.rollout:7070` |
| Ray cluster | RayCluster `ray`: a head that runs no tasks, a GPU group (`runtimeClassName: nvidia`, one GPU, 14 GiB) and a CPU group (4 GiB), each from zero to one pod by the autoscaler, with token auth | `http://ray-head-svc.rollout:8265`, `http://ray.localhost` |
| Launchers | Deployments `launcher-minecraft` (the workspace's Python, the GPU) and `launcher-gsm8k` (rollout-verifiers' Python, no GPU): each submits the runs it claims to the Ray cluster | their beats, in the monitor's Machines tab |
| Gateway | Deployment `gateway`, over the profile `gsm8k/gsm8k_tinker.toml` | `http://gateway.rollout:8900`, `http://gateway.localhost` |
| Monitor | Deployments `monitor-main` (over curriculum-9's directory, so its ledger and every run in it) and `monitor-astra` (over `evaluations/astra-t054u`) | `http://monitor.localhost`, `http://astra.monitor.localhost` |

Every pod of the platform mounts the same things:

- the volume `state` at `/root/.cache/rollout` (the code's `~/.cache/rollout`; the containers run as root): run
  directories, Minecraft's servers and worlds, the Hugging Face cache (`HF_HOME`), node-local scratch;
- the ConfigMap `rollout` at `/etc/rollout`: `cluster.toml` (the cluster config, [docs/guide/cluster.md](../../docs/guide/cluster.md);
  `ROLLOUT_CLUSTER` names it) and the profiles the commands still take, under `profiles/LAUNCHER/`, each naming the
  stores above. The chart's `files/` holds them;
- the Secret `gateway-keys` at `/etc/rollout-secrets/gateway`, and the Secret `tinker` at `/root/.tinker`.

Each process gets the stores' credentials from the Secret `stores` (`PGPASSWORD`, `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`), `AWS_ENDPOINT_URL`, and `TINKER_API_KEY` where the Secret `tinker` has one. Every container
has requests and a memory limit (`values.yaml`), which keep the machine's 23 GB from running out. The launchers run
under the ServiceAccount `launcher`, which may manage RayJobs and RayClusters.

### Install

The Secrets are made outside the chart, once:

```sh
kubectl create namespace rollout
kubectl -n rollout create secret generic stores --from-literal=POSTGRES_PASSWORD="$(openssl rand -hex 24)" \
  --from-literal=ROOT_ACCESS_KEY_ID=rollout --from-literal=ROOT_SECRET_ACCESS_KEY="$(openssl rand -hex 24)"
kubectl -n rollout create secret generic gateway-keys --from-file=gateway.keys=$HOME/.config/rollout/gsm8k-tinker.keys
kubectl -n rollout create secret generic tinker --from-file=credentials.json=$HOME/.tinker/credentials.json \
  --from-literal=TINKER_API_KEY="$(python3 -c 'import json, os; d = json.load(open(os.path.expanduser("~/.tinker/credentials.json"))); print(d["keys"][d["default"]]["key"], end="")')"
helm upgrade --install rollout deploy/chart/rollout -n rollout
```

The Secret `tinker` holds both what `tinker auth login` wrote, which Tinker's SDK reads from `/root/.tinker`, and its
default key as `TINKER_API_KEY`, which the cluster config names; `rollout cluster check` in a pod then finds every
secret resolved.

### Moving the host's state in

`deploy/k3s/migrate.sh` copies what the host's services kept, through a pod that mounts `~/.cache/rollout` read-only:

- the run directories, the evaluations' directories, `datasets/`, `gsm8k-tinker/`, Minecraft's files and the JDK onto
  the volume `state`; each run's `ledger.json` then names the cluster's ledger;
- the stores of files the ledger's records point into into the bucket (`python -m rollout_s3.copying`): those named by
  a location in a record (`gsm8k-tinker/blobs` by the GSM8K eval's start, `datasets/blobs` by the dataset), and each
  run's own `RUN/blobs` (curriculum-9's episodes and checkpoints, named by their blobs' URIs). Blobs are named by their
  SHA-256, so the stores merge into `s3://rollout-blobs/blobs` without a clash;
- the ledger: a consistent copy of `ledger.db`, rewritten so that the stores that moved are named as the bucket and
  their blobs by the bucket's URIs, and paths under `~/.cache/rollout` are under `/root/.cache/rollout`
  (`python -m rollout_train.relocating`), then copied into a new, empty database with `rollout ledger copy`. A run
  whose start names no store (curriculum-9) is read from its directory, as on the host.

The Hugging Face cache is not copied: pods download what they load into `HF_HOME` on the volume, once.

Undo: `sudo /usr/local/bin/k3s-uninstall.sh`, then `sudo apt remove nvidia-container-toolkit`.
