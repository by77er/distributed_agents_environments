# K3s on WSL2

A one-node Kubernetes cluster on a WSL2 machine, with KubeRay and the NVIDIA GPU reachable from pods, and the
platform in it: the chart `deploy/chart/rollout`. Nothing here locks the card: Kubernetes only accounts for it, and
processes outside the cluster keep using it. The deployment guide in [docs/deploy/](../../docs/deploy/README.md)
describes each step for any cluster: [preparing it](../../docs/deploy/kubernetes.md), the
[image](../../docs/deploy/image.md), the [chart](../../docs/deploy/helm.md) and [starting runs](../../docs/deploy/runs.md).

| File | Does |
|---|---|
| `install-wsl.sh` | As root: installs NVIDIA's container toolkit with a CDI spec for WSL2's GPU (`/dev/dxg`), installs K3s with a kubeconfig your user can read, and moves containerd's stream server to port 9910, outside Ray's worker ports (10002-19999) |
| `device-plugin.yaml` | Values for NVIDIA's device plugin chart: the node advertises its card as one `nvidia.com/gpu`, for one Ray worker pod, and Ray shares it among its actors with fractional `num_gpus` |
| `storage-class.yaml` | The StorageClass `local-path-retain`: K3s's local-path volumes, kept when their claims are deleted |
| `build.yaml` | Building images in the cluster: a registry the node pulls from at `localhost:30500`, and BuildKit |

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
kubectl apply --server-side -f https://github.com/kubernetes-sigs/kueue/releases/download/v0.19.7/manifests.yaml
kubectl -n kueue-system wait deploy/kueue-controller-manager --for=condition=available --timeout=5m
kubectl apply -f deploy/k3s/storage-class.yaml
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
| Ray cluster | RayCluster `ray`, where the monitors check environments imported from git: a head that runs no tasks, a GPU group (`runtimeClassName: nvidia`, one GPU, 14 GiB) and a CPU group (4 GiB), each from zero to one pod by the autoscaler, with token auth | `http://ray-head-svc.rollout:8265`, `http://ray.localhost` |
| Runs | A RayJob for each run asked for, made from `files/rayjob.yaml`: a Ray cluster of its own, one head pod sized from what the run needs (the card only for a run with a local trainer or engine host), where the run's driver, trainer and engine hosts run; with `kueue.enabled`, admitted by Kueue's queue `runs` once its quota has room; submitted again up to `rayjob.backoffLimit` times when its driver is lost, and removed `rayjob.ttlSeconds` after it ends | the monitor's Runs and Machines tabs |
| Gateway | Deployment `gateway`: `rollout gateway --cluster` | `http://gateway.rollout:8900`, `http://gateway.localhost` |
| Monitor | A Deployment `monitor-NAME` for each of `monitors` (`main`): `rollout monitor --cluster` over the cluster config's ledger, or over `monitors.NAME.where` where it names one (a run's directory on the state volume, or a ledger's URL), importing environments from git with the cluster config's blob store and Ray cluster; it asks for the token in the Secret `monitor-token`, which the chart makes | `kubectl -n rollout port-forward svc/monitor-main 8765:8765`, then `http://localhost:8765/login?token=TOKEN` once ([opening the monitor](../../docs/deploy/access.md#opening-the-monitor)) |
| Presets | The Job `presets`, a hook at every install and upgrade: `rollout preset load /etc/rollout/presets --cluster` saves each of `files/presets` as a preset, a new version only where its newest one holds other settings | |

A monitor asks for each run from its page as a RayJob, under the ServiceAccount `monitor` (`templates/rbac.yaml`: create,
get, list, watch and delete on `rayjobs` in the release's namespace, and with Kueue get, list and watch on
`workloads`), reads its status, and deletes it to stop the run.

Kueue 0.19.7 (installed above, after KubeRay, so its RayJob integration finds KubeRay's resources) admits runs' RayJobs
whole once the chart makes its queue: install or upgrade with `--set kueue.enabled=true`. The chart then makes the
ResourceFlavor `rollout`, the ClusterQueue `rollout` (12 CPUs, 16 GiB and one GPU for runs, `kueue.quota`) and the
LocalQueue `runs` in the namespace, and the cluster config names the queue and the quota
([Kueue](../../docs/deploy/helm.md#kueue)). A run waiting for admission says so on its launch tile;
`kubectl -n rollout get workloads` lists what Kueue holds.

Each role is given only what its code reads ([what each role is given](../../docs/deploy/helm.md#what-each-role-is-given)):

- the ConfigMap `rollout` at `/etc/rollout`, for every role but the long-lived Ray cluster: `cluster.toml` (the cluster
  config, [docs/guide/cluster.md](../../docs/guide/cluster.md); `ROLLOUT_CLUSTER` names it), `rayjob.yaml` (what its
  `[kubernetes] rayjob` names) and `presets/`, each naming the stores above. The chart's `files/` holds them;
- the volume `state` at `/root/.cache/rollout` (the code's `~/.cache/rollout`; the containers run as root), for the
  gateway, the monitors, the sandbox pools and runs: run directories, Minecraft's servers and worlds, the Hugging Face
  cache (`HF_HOME`), node-local scratch;
- of the Secrets, what each reads: the ledger's password (`stores`) for every role that reads the ledger, the blob
  store's keys for the gateway, the monitors and runs, the hosted APIs' keys (`providers`) for the gateway and runs,
  and Tinker's (`tinker`), the gateway's keys and the rest for runs alone. The long-lived Ray cluster, which checks
  environments imported from git, is given none.

NetworkPolicies let each role reach only the roles it uses ([network
policies](../../docs/deploy/helm.md#network-policies)). Every container has requests and a memory limit
(`values.yaml`), which keep the node's memory from running out.

### Install

The namespace is labelled for Pod Security, and the Secrets are made outside the chart, once
([Pod Security](../../docs/deploy/kubernetes.md#pod-security)):

```sh
kubectl create namespace rollout
kubectl label namespace rollout pod-security.kubernetes.io/enforce=baseline \
  pod-security.kubernetes.io/warn=restricted pod-security.kubernetes.io/audit=restricted
kubectl -n rollout create secret generic stores --from-literal=POSTGRES_PASSWORD="$(openssl rand -hex 24)" \
  --from-literal=ROOT_ACCESS_KEY_ID=rollout --from-literal=ROOT_SECRET_ACCESS_KEY="$(openssl rand -hex 24)"
kubectl -n rollout create secret generic gateway-keys --from-literal=gateway.keys="k1 $(openssl rand -hex 32)"
kubectl -n rollout create secret generic tinker --from-file=credentials.json=$HOME/.tinker/credentials.json \
  --from-literal=TINKER_API_KEY="$(python3 -c 'import json, os; d = json.load(open(os.path.expanduser("~/.tinker/credentials.json"))); print(d["keys"][d["default"]]["key"], end="")')"
kubectl -n rollout create secret generic providers --from-literal=OPENAI_API_KEY="$OPENAI_API_KEY" \
  --from-literal=ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
helm upgrade --install rollout deploy/chart/rollout -n rollout
```

The Secret `providers` holds the hosted APIs' keys (either may be left out), which the cluster config's
`[inference.openai]` and `[inference.anthropic]` name.

The Secret `tinker` holds both what `tinker auth login` wrote, which Tinker's SDK reads from `/root/.tinker`, and its
default key as `TINKER_API_KEY`, which the cluster config names. Runs' pods, which read them, are given both; each
role's pod is checked for what it reads with `rollout cluster check --role ROLE`.

### Volumes

Every volume in the cluster is a directory on the node under `/var/lib/rancher/k3s/storage`, made by K3s's local-path
provisioner: the ledger's (Postgres), the blob store's (S3), `state`, and the registry's and BuildKit's. The class
`local-path` deletes a volume's directory when its claim is deleted; `local-path-retain` keeps it. A fresh install
names the latter, at install and at every upgrade after it, since a claim's class cannot change once it is made:

```sh
helm upgrade --install rollout deploy/chart/rollout -n rollout --set storageClass=local-path-retain
```

An install made with `local-path` keeps its volumes once they are set to Retain by hand, after any install that makes
a new one:

```sh
for pv in $(kubectl get pv -o name); do kubectl patch $pv -p '{"spec":{"persistentVolumeReclaimPolicy":"Retain"}}'; done
```

## Undo

`sudo /usr/local/bin/k3s-uninstall.sh`, then `sudo apt remove nvidia-container-toolkit`.

**The uninstall script deletes `/var/lib/rancher`, and with it every volume: the ledger (Postgres), the blobs (S3) and
the state volume (run directories), whatever their reclaim policy.** Back them up first, for example:

```sh
kubectl -n rollout exec postgres-0 -- pg_dump -U rollout -Fc rollout > ~/rollout-ledger.dump
sudo cp -a /var/lib/rancher/k3s/storage ~/rollout-volumes   # every volume's directory, the bucket's and state's among them
```
