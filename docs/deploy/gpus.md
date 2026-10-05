# Share GPUs

How the platform's pods get GPUs: whole cards for Ray workers, and time-slicing when several pods share one card.
This page is for whoever sets up GPU nodes and sizes the chart's GPU workers.

**Read first:** [Prepare a Kubernetes cluster](kubernetes.md). **Next:** [Postgres and S3](stores.md).

## Whole cards for Ray workers

By default the device plugin advertises each card as one `nvidia.com/gpu`, and each Ray GPU worker pod asks for one
(`ray.gpu.resources.limits`). Inside that pod, Ray shares the card among the run's processes with fractional GPUs: a
LoRA (low-rank adaptation) trainer that shares the engines' card (`colocate_with` in the cluster config) and the vLLM
engine each take part of it, and the engines sleep while the trainer steps ([sharing a
GPU](../libraries/rollout-train/channels.md#share-a-gpu-between-engines-and-the-trainer)).

- One GPU worker pod holds one run that uses a local GPU. `ray.gpu.maxReplicas` is the most such runs at once, at
  most the number of cards.
- A run trained and sampled on Tinker uses no GPU, and runs on a CPU worker.

## Time-slicing a card between pods

When two pods must use one card at once (for example, a run's trainer and a shared inference pool's engine, on a node
with one GPU), the device plugin can advertise each card as several `nvidia.com/gpu` with time-slicing. Write the
plugin's configuration:

```yaml title="time-slicing.yaml"
version: v1
sharing:
  timeSlicing:
    resources:
      - name: nvidia.com/gpu
        replicas: 2
```

And install the device plugin with it:

```bash
helm upgrade --install nvidia-device-plugin nvdp/nvidia-device-plugin --version 0.20.1 \
  -n nvidia-device-plugin -f deploy/k3s/device-plugin.yaml --set-file config.map.config=time-slicing.yaml
```

The node then reports two `nvidia.com/gpu` per card.

**Time-slicing only divides the count Kubernetes schedules by.** It does not divide the card's memory or isolate the
pods: every pod sees the whole card, and one that takes all its memory makes the other fail. So each process caps its
own memory:

- a vLLM engine takes the share of the card's memory its `gpu_memory_utilization` option says, set per model in the
  cluster config (`options = { gpu_memory_utilization = 0.45 }`);
- the LoRA and full-weight trainers use at most the memory that is free when a step's process starts
  ([the memory bound](../implementations/rollout-lora.md#the-memory-bound)), so start them after the engines have
  taken their share.

Leave room for the desktop and other processes outside Kubernetes too: Kubernetes only counts the card, so it never
stops them from using it.

## Check what a pod sees

```bash
kubectl describe node NODE | grep nvidia.com/gpu
kubectl -n rollout exec POD -- nvidia-smi
```

[A GPU pod stays pending](troubleshooting.md#a-gpu-pod-stays-pending) lists what to check when a pod asking for a GPU
is never scheduled.
