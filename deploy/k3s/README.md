# K3s on WSL2

A one-node Kubernetes cluster on the WSL2 machine, with KubeRay and the NVIDIA GPU reachable from pods. Nothing here
locks the card: Kubernetes only accounts for it, and processes outside the cluster keep using it.

| File | Does |
|---|---|
| `install-wsl.sh` | As root: installs NVIDIA's container toolkit with a CDI spec for WSL2's GPU (`/dev/dxg`), installs K3s with a kubeconfig your user can read, and moves containerd's stream server to port 9910, outside Ray's worker ports (10002-19999) |
| `device-plugin.yaml` | Values for NVIDIA's device plugin chart: the node advertises its card as one `nvidia.com/gpu`, for one Ray worker pod, and Ray shares it among its actors with fractional `num_gpus` |
| `ray-smoke.yaml` | A Ray cluster whose GPU worker group sits at zero pods until a task asks for a GPU; the autoscaler removes the worker after a minute idle |

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
kubectl apply -f deploy/k3s/ray-smoke.yaml
```

Undo: `sudo /usr/local/bin/k3s-uninstall.sh`, then `sudo apt remove nvidia-container-toolkit`.
