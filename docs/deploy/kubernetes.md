# Prepare a Kubernetes cluster

What a Kubernetes cluster needs before the chart can be installed: the KubeRay operator, GPUs that pods can use, a
storage class, an ingress controller, and the namespace's Pod Security level. This page is for whoever sets up the cluster, on one machine with K3s or on
an existing cluster.

**Read first:** [Choose a setup](setups.md). **Next:** [Build the platform image](image.md).

## What the chart expects

- **Helm 3** on the machine you install from, and `kubectl` with access to the cluster.
- **The KubeRay operator**, which adds the `RayCluster` and `RayJob` resources and keeps the chart's Ray cluster
  running. These pages use operator 1.7.1; the chart runs Ray 2.59.0, the version the workspace locks.
- **NVIDIA GPUs that pods can request.** Each GPU node needs:
    - the NVIDIA driver;
    - the NVIDIA container toolkit, so that containerd can give a container the GPU;
    - a RuntimeClass named `nvidia`, which the chart's GPU workers name (`runtimeClassName: nvidia`);
    - NVIDIA's device plugin, which advertises each card as an `nvidia.com/gpu` resource.

    The NVIDIA GPU Operator installs all of these on a cluster that has none. K3s registers the `nvidia` RuntimeClass
    itself when it finds the container toolkit on the node.
- **A storage class** for the chart's volumes (`storageClass` in the values). Prefer one whose reclaim policy is
  `Retain`, so that deleting a claim keeps its data ([volumes and backups](backups.md)).
- **An ingress controller**, named by `ingress.className` (Traefik by default, which K3s includes).
- **A registry the nodes pull from**, for the platform image ([build the platform image](image.md)).
- **Kubernetes 1.30 or later**, for ValidatingAdmissionPolicies (the chart's `admission`), and a network plugin that
  enforces NetworkPolicies (the chart's `networkPolicies`; K3s's does).
- **The namespace labelled for Pod Security** ([below](#pod-security)).

## Install K3s on one machine

1. Install the NVIDIA driver and NVIDIA's container toolkit on the machine, following NVIDIA's instructions for your
   distribution. On WSL2, `deploy/k3s/install-wsl.sh` does this step and the next: it installs the toolkit with a
   Container Device Interface (CDI) specification for WSL2's GPU, then K3s.
2. Install K3s, with a kubeconfig your user can read:

    ```bash
    curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--write-kubeconfig-mode 644" sh -
    export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
    ```

3. Check that K3s found the GPU runtime:

    ```bash
    kubectl get runtimeclass nvidia
    ```

If anything on the machine runs Ray outside Kubernetes, read
[the port collision with Ray's workers](troubleshooting.md#k3s-does-not-start-a-port-is-taken-by-ray) before starting
K3s.

## Install the operators

1. Add the charts' repositories:

    ```bash
    helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
    helm repo add kuberay https://ray-project.github.io/kuberay-helm/
    helm repo update
    ```

2. Label each GPU node, unless Node Feature Discovery or the GPU Operator labels it. The device plugin's chart runs
   only on nodes labelled as having an NVIDIA GPU:

    ```bash
    kubectl label node NODE nvidia.com/gpu.present=true
    ```

3. Install the device plugin. `deploy/k3s/device-plugin.yaml` sets its RuntimeClass to `nvidia`; each card is then
   one `nvidia.com/gpu` ([share GPUs](gpus.md) says how to split a card):

    ```bash
    helm upgrade --install nvidia-device-plugin nvdp/nvidia-device-plugin --version 0.20.1 \
      -n nvidia-device-plugin --create-namespace -f deploy/k3s/device-plugin.yaml
    ```

4. Install the KubeRay operator:

    ```bash
    helm upgrade --install kuberay-operator kuberay/kuberay-operator --version 1.7.1 -n kuberay --create-namespace
    ```

5. Optionally, install Kueue, which admits each run's RayJob whole once its queue's quota holds all of it
   ([Kueue](helm.md#kueue)). Kueue 0.19.7 manages KubeRay RayJobs (`ray.io/rayjob` is among the integrations its
   default configuration enables), serves the `kueue.x-k8s.io/v1beta2` API the chart's objects use, and needs
   Kubernetes 1.29 or newer. Install it after KubeRay, so it finds the RayJob resource as it starts (if KubeRay comes
   later, restart `deploy/kueue-controller-manager`):

    ```bash
    kubectl apply --server-side -f https://github.com/kubernetes-sigs/kueue/releases/download/v0.19.7/manifests.yaml
    kubectl -n kueue-system wait deploy/kueue-controller-manager --for=condition=available --timeout=5m
    ```

    or with Helm:

    ```bash
    helm upgrade --install kueue oci://registry.k8s.io/kueue/charts/kueue --version 0.19.7 \
      -n kueue-system --create-namespace --wait
    ```

6. On K3s, add the storage class `local-path-retain`: K3s's local volumes, kept when their claims are deleted:

    ```bash
    kubectl apply -f deploy/k3s/storage-class.yaml
    ```

## Pod Security

The namespace the chart is installed into enforces Pod Security's `baseline` level, and warns of and audits what
`restricted` would refuse. Helm does not own the namespace, so label it yourself, once, when you make it:

```bash
kubectl create namespace rollout
kubectl label namespace rollout pod-security.kubernetes.io/enforce=baseline \
  pod-security.kubernetes.io/warn=restricted pod-security.kubernetes.io/audit=restricted
```

`baseline` refuses a pod that is privileged, shares the node's network, processes or IPC, mounts a directory of the
node (`hostPath`), takes a host port or adds capabilities, whoever makes it: a RayJob's pods too, which KubeRay makes
from whatever the RayJob says. Every pod of the platform passes it, the GPU pods (`runtimeClassName: nvidia`), the Ray
clusters' pods with what KubeRay adds to them (its autoscaler, its init containers), step-ca, Postgres, versitygw and
cloudflared among them. None passes `restricted`, which also asks every container to run as a user other than root
with no privilege escalation and its capabilities dropped: the platform's image runs as root.

The chart's admission policy (`admission.enabled`, `templates/admission.yaml`) holds what `baseline` does not: it
refuses a RayJob in the namespace that submits to a Ray cluster it does not make, exposes its Ray cluster outside the
namespace, runs KubeRay's autoscaler (whose account may make pods), sets the options that bring in Secrets of their own,
runs a pod under another account than the namespace's default, mounts another claim than the state volume, or reads a
Secret a run is not given (each run's pods read the Secrets [a run's job is given](helm.md#what-each-role-is-given), and
no other). The monitors' account may make RayJobs, and these two keep that from being a way onto the node or to every
Secret.

`rollout cluster check`, run in a monitor's pod (whose account may read its namespace), says when a label is missing
or names another level:

```bash
kubectl -n rollout exec deploy/monitor-main -- rollout cluster check --role monitor
```

## Check the cluster

Each command should list what it asks for:

```bash
kubectl get nodes -o custom-columns='NODE:.metadata.name,GPUS:.status.allocatable.nvidia\.com/gpu'
kubectl get crd rayclusters.ray.io rayjobs.ray.io
kubectl get crd clusterqueues.kueue.x-k8s.io localqueues.kueue.x-k8s.io   # with Kueue
kubectl get storageclass
kubectl get ingressclass
kubectl get namespace rollout --show-labels                                 # pod-security.kubernetes.io/…
```

A GPU node whose `GPUS` column is empty has no device plugin running on it: see
[a GPU pod stays pending](troubleshooting.md#a-gpu-pod-stays-pending).
