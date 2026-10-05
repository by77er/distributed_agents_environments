# Choose a setup

The same code runs in three setups: one machine without Kubernetes, one machine with K3s, and a Kubernetes cluster
of several nodes. This page compares them and says how to set up the first; it is for anyone deciding how to deploy.

**Read first:** [What runs where](roles.md). **Next:** [Prepare a Kubernetes cluster](kubernetes.md), for the other
two setups.

## Compare the setups

**One machine without Kubernetes.** Local processes and a local Ray head, with the
[ledger](../libraries/rollout-train/checkpoints.md#the-ledger) in SQLite and blobs in a directory of files.

- Use it to develop environments, to try the platform, and for one person's runs on their own GPU.
- It needs no container runtime and no cluster. Everything is in `~/.cache/rollout`, so backing up is copying a
  directory.
- One machine's memory and GPU are the limit, and nothing restarts a process that dies.

**One machine with K3s.** A one-node Kubernetes cluster ([K3s](https://k3s.io/)) running the Helm chart: Postgres, an
S3-compatible store, a Ray cluster kept by KubeRay, the [gateway](../libraries/rollout-train/gateway.md) and the
monitor.

- Use it for a lab machine that several people use, or to run the cluster setup before buying more nodes.
- Kubernetes restarts what dies, every container has a memory limit, and the stores are the ones a bigger cluster
  uses, so moving to more nodes changes values, not code.
- It costs a few GB of memory for Kubernetes and the stores, and the image has to be built and pushed somewhere the
  node pulls from.

**A Kubernetes cluster of several nodes.** The same chart on a cluster with GPU nodes, a managed or replicated
Postgres, and S3 or an S3-compatible service.

- Use it for a team, for several runs at once on several GPUs, and for anything that must keep running.
- The autoscaler adds Ray workers on GPU nodes when runs wait for them.
- Every pod of the chart mounts one state volume, so on several nodes it needs a storage class that several nodes
  can mount at once ([volumes](helm.md#volumes)).

## Set up one machine without Kubernetes

1. Install the workspace with its GPU packages (Linux and an NVIDIA GPU):

    ```bash
    uv sync --all-extras
    ```

2. Write the cluster config. `deploy/clusters/example.toml` describes one machine with one 16 GB GPU: the ledger in
   SQLite, blobs in files, a vLLM pool on the GPU, a LoRA (low-rank adaptation) trainer that shares it, Tinker, the
   Minecraft worlds and GSM8K. Copy it to where the commands look for it, and edit the paths and models ([the cluster
   config](../guide/cluster.md#the-cluster-config) describes every field):

    ```bash
    mkdir -p ~/.config/rollout
    cp deploy/clusters/example.toml ~/.config/rollout/cluster.toml
    ```

3. Check that every secret and project it names resolves on this machine. It exits with status 1 if one does not:

    ```bash
    uv run rollout cluster check
    ```

4. Start a Ray head, with its temporary directory on disk (Ray writes sessions and spilled objects there, and `/tmp`
   may be memory):

    ```bash
    uv run ray start --head --node-ip-address 127.0.0.1 --dashboard-host 127.0.0.1 --num-gpus 1 --temp-dir ~/.cache/ray
    ```

5. Serve the monitor over the ledger, with the cluster config's Ray and blob store for importing environments:

    ```bash
    uv run rollout monitor "sqlite:///$HOME/.cache/rollout/ledger.db" --cluster
    ```

    It listens on `127.0.0.1:8765` unless `--host` and `--port` say otherwise, and prints the link that signs a
    browser in (`ROLLOUT_MONITOR_TOKEN`, or one it makes up: [signing
    in](../libraries/rollout-train/monitor.md#signing-in)).

6. Start runs: see [Start runs and evals](runs.md).

`uv run ray stop` stops the Ray head and everything running on it.

A run's driver serves a gateway of its own for the run's channels, so one machine needs no gateway process. For
programs or harnesses elsewhere that sample runs' channels on providers the cluster reaches at addresses
(`vllm-servers`, RunPod pods), start a replica of the cluster's gateway:

```bash
uv run rollout gateway --cluster --listen 127.0.0.1:8900
```

When several machines or processes share the stores, move the ledger to Postgres and the blobs to S3 first:
[Postgres and S3](stores.md) says how, and [local Postgres and S3](../development/local-services.md) runs both on a
development machine.

## Set up Kubernetes

Follow the pages in order:

1. [Prepare a Kubernetes cluster](kubernetes.md): K3s on one machine, or the operators on an existing cluster.
2. [Build the platform image](image.md).
3. [Install the Helm chart](helm.md).
4. [Share GPUs](gpus.md), [Volumes and backups](backups.md) and [Ingress, TLS and sign-in](access.md) before other
   people use it.
