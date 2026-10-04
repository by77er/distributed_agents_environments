#!/usr/bin/env bash
# Install K3s on WSL2 with the NVIDIA GPU reachable from pods, without taking it from the host.
#
# Run once, as root:   sudo bash deploy/k3s/install-wsl.sh
# Undo:                sudo /usr/local/bin/k3s-uninstall.sh   (and: apt remove nvidia-container-toolkit)
#
# What it does:
#   1. Installs NVIDIA's container toolkit from NVIDIA's apt repository, and writes a CDI spec for WSL2's GPU
#      (WSL2 exposes the card as /dev/dxg with libcuda under /usr/lib/wsl/lib, not as /dev/nvidia*).
#   2. Installs K3s as a systemd service, with a kubeconfig your user can read. K3s finds the NVIDIA runtime and
#      registers a RuntimeClass `nvidia` for pods that use the GPU.
#   3. Moves containerd's stream server from 127.0.0.1:10010 to 127.0.0.1:9910, outside Ray's worker ports
#      (10002-19999), by saving K3s's rendered containerd config as its template with that one port changed.
# Nothing here locks the GPU: Kubernetes only accounts for it, and processes outside the cluster keep using it.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "run as root: sudo bash $0" >&2
  exit 1
fi
if [[ ! -e /dev/dxg ]]; then
  echo "no /dev/dxg: this script is for WSL2 with a GPU" >&2
  exit 1
fi

STREAM_PORT=9910

echo "== NVIDIA container toolkit"
install -d -m 0755 /usr/share/keyrings /etc/cdi
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk cdi generate --mode=wsl --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list

echo "== K3s"
curl -sfL https://get.k3s.io | INSTALL_K3S_SKIP_START=true INSTALL_K3S_EXEC="--write-kubeconfig-mode 644" sh -

echo "== containerd stream server on port $STREAM_PORT"
containerd_dir=/var/lib/rancher/k3s/agent/etc/containerd
systemctl start k3s || true  # renders the containerd config; containerd may fail to bind 10010 the first time
for _ in $(seq 30); do
  [[ -s $containerd_dir/config.toml ]] && break
  sleep 1
done
sed -E "s/^([[:space:]]*stream_server_port[[:space:]]*=[[:space:]]*).*/\1\"$STREAM_PORT\"/" \
  "$containerd_dir/config.toml" > "$containerd_dir/config-v3.toml.tmpl"
grep -q "stream_server_port = \"$STREAM_PORT\"" "$containerd_dir/config-v3.toml.tmpl" \
  || { echo "no stream_server_port in $containerd_dir/config.toml" >&2; exit 1; }
systemctl restart k3s
for _ in $(seq 60); do
  k3s kubectl get nodes >/dev/null 2>&1 && break
  sleep 2
done
k3s kubectl get nodes -o wide
k3s kubectl get runtimeclass

echo "== done"
echo "kubeconfig: /etc/rancher/k3s/k3s.yaml (readable by your user); export KUBECONFIG=/etc/rancher/k3s/k3s.yaml"
