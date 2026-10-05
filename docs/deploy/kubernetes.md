# Prepare a Kubernetes cluster

What a Kubernetes cluster needs before the chart can be installed: the KubeRay operator, GPUs that pods can use, a
storage class and an ingress controller. This page is for whoever sets up the cluster, on one machine with K3s or on
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

## Check the cluster

Each command should list what it asks for:

```bash
kubectl get nodes -o custom-columns='NODE:.metadata.name,GPUS:.status.allocatable.nvidia\.com/gpu'
kubectl get crd rayclusters.ray.io rayjobs.ray.io
kubectl get crd clusterqueues.kueue.x-k8s.io localqueues.kueue.x-k8s.io   # with Kueue
kubectl get storageclass
kubectl get ingressclass
```

A GPU node whose `GPUS` column is empty has no device plugin running on it: see
[a GPU pod stays pending](troubleshooting.md#a-gpu-pod-stays-pending).
