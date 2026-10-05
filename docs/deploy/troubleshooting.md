# Troubleshooting

Problems seen in real deployments, each with what you see, why it happens and how to fix it. This page is for whoever
deploys or operates the platform.

**Read first:** [Deploy the platform](README.md). **Next:** [Operate the platform](../operate/README.md).

Most problems show first in a pod's events or logs:

```bash
kubectl -n rollout get pods
kubectl -n rollout describe pod POD      # the Events at the end say why a pod is pending or failing
kubectl -n rollout logs POD --previous   # the log of the container before it restarted
```

## vLLM fails with "Failed to find C compiler"

**What you see.** A run's engine fails as it starts or on its first request, and its log ends in Triton's
`RuntimeError: Failed to find C compiler. Please specify via CC environment variable.`

**Why.** Triton and torch's compiler (inductor) build kernels while vLLM and the trainers run, and need a C compiler
and the C library's headers in the image.

**Fix.** The platform image installs `gcc` and `libc6-dev`. An image of your own needs them too: on Debian or Ubuntu,
`apt-get install -y gcc libc6-dev`.

## K3s does not start: a port is taken by Ray

**What you see.** On a machine that also runs Ray outside Kubernetes, K3s fails to start, and its log
(`journalctl -u k3s`) says containerd could not listen on `127.0.0.1:10010`: the address is already in use.

**Why.** Ray's workers take ports from 10002 to 19999 by default, from the bottom up. containerd's stream server,
which K3s starts, listens on port 10010, inside that range; whichever starts first takes it. K3s's other ports
(10248 to 10259) are in the range too, and collide once Ray runs enough workers.

**Fix.** Move one of them:

- Start Ray with its worker ports above K3s's, which fixes every collision:

    ```bash
    uv run ray start --head --min-worker-port 20000 --max-worker-port 29999 --temp-dir ~/.cache/ray
    ```

- Or move containerd's stream server. K3s renders containerd's configuration from a template: save the rendered
  `/var/lib/rancher/k3s/agent/etc/containerd/config.toml` as `config-v3.toml.tmpl` in the same directory, change
  `stream_server_port` there to a free port (such as `"9910"`), and restart K3s. `deploy/k3s/install-wsl.sh` does
  this.

## An upgrade fails on an immutable field

**What you see.** `helm upgrade` stops with an error such as
`StatefulSet.apps "postgres" is invalid: spec: Forbidden: updates to statefulset spec for fields other than ...` or
`PersistentVolumeClaim "state" is invalid: spec: Forbidden: spec is immutable after creation`.

**Why.** Kubernetes does not change some fields once a resource exists: a StatefulSet's volume claim templates (the
stores' `storageClass` and size), and a claim's class. A changed value in your values file, or values missing
because `--reuse-values` was passed, render a different spec.

**Fix.**

1. Compare what the release has with what you pass, and set the value back:

    ```bash
    helm get values rollout -n rollout
    ```

2. If the change is wanted, for a StatefulSet's template only: delete the StatefulSet but keep its pod and claim, then
   upgrade. The new StatefulSet adopts the pod, and its template applies to claims it makes from then on:

    ```bash
    kubectl -n rollout delete statefulset postgres --cascade=orphan
    helm upgrade --install rollout deploy/chart/rollout -n rollout -f rollout-values.yaml
    ```

3. A claim's class cannot change at all: back up its data, delete the claim, upgrade, and restore
   ([volumes and backups](backups.md)).

## A GPU pod stays pending

**What you see.** A Ray GPU worker pod stays `Pending`, and its events say `Insufficient nvidia.com/gpu`; or the pod
fails to start with `RuntimeClass "nvidia" not found`.

**Why, and the fix for each cause:**

- **No device plugin on the node.** `kubectl get nodes -o
  custom-columns='NODE:.metadata.name,GPUS:.status.allocatable.nvidia\.com/gpu'` shows no GPUs. Check that the node
  carries the label the plugin's chart selects (`nvidia.com/gpu.present=true`), and read the plugin pod's log in the
  namespace `nvidia-device-plugin`.
- **No `nvidia` RuntimeClass.** K3s registers it when it finds NVIDIA's container toolkit as it starts: install the
  toolkit, then restart K3s. On other clusters, the GPU Operator makes it.
- **Every GPU is taken.** Another pod holds the card: a run still on its GPU worker, or a process outside Kubernetes
  that the device plugin does not see. `ray.gpu.maxReplicas` caps the GPU workers at once.
- **Time-slicing not applied.** A node that should report several GPUs per card reports one: the device plugin did
  not read its configuration. Check its log, and that the upgrade passed `--set-file config.map.config=…`
  ([share GPUs](gpus.md)).

## Pods cannot pull from the in-cluster registry

**What you see.** Pods stay in `ImagePullBackOff`, with events such as
`http: server gave HTTP response to HTTPS client` or `no such host`.

**Why.** Images are pulled by each node's container runtime, which does not use the cluster's DNS and uses HTTPS for
every registry but one on `localhost`. So a node cannot pull `registry.build.svc.cluster.local:5000/…`, the name
BuildKit pushes to, nor a registry at another plain-HTTP address.

**Fix.**

- Name the image by the registry's NodePort on `localhost`, which every node can reach: the chart's default,
  `localhost:30500/rollout-platform`.
- Check that the image is there:

    ```bash
    curl http://localhost:30500/v2/rollout-platform/tags/list
    ```

- For a plain-HTTP registry at another address, tell each node's runtime about it (for K3s, a mirror with an `http://`
  endpoint in `/etc/rancher/k3s/registries.yaml`, then restart K3s).

## A run waits for resources

**What you see.** A run stays waiting and plays no [episode](../libraries/rollout-train/episodes.md).

**Why, and what to check:**

- **No Ray worker can start.** The autoscaler starts a GPU worker for a run that asks for a GPU, up to
  `ray.gpu.maxReplicas`. If one is already busy with another run, the new run waits for it. If the worker pod is
  pending, see [a GPU pod stays pending](#a-gpu-pod-stays-pending). Ray's own view:

    ```bash
    kubectl -n rollout exec "$(kubectl -n rollout get pod -l app=ray-head -o name)" -- ray status
    kubectl -n rollout exec "$(kubectl -n rollout get pod -l app=ray-head -o name)" -- ray job list
    ```

- **Not enough free memory.** The cluster config's `[guards]` say how much system memory must be free: short of
  `runs_gib`, a runner waits before it claims another episode; short of `training_gib`, a run whose trainer shares
  the engines' GPU stops before its next step, with `NotEnoughMemory`.
- **More than the cluster has.** A run that asks for more GPUs than the cluster has is refused when its settings are
  checked; one that asks for more than are free waits ([validation](../guide/cluster.md#validation)).
