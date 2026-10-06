# The trainer image

`ghcr.io/by77er/rollout-trainer`: a pod that takes the training steps of the run that holds it.

| Process | Listens on | Does |
|---|---|---|
| `pki.sh renew` | | Renews the pod's certificate at about two thirds of its life, and publishes each new one for Envoy |
| Envoy | `0.0.0.0:8443` (the pod's exposed TCP port), admin on `127.0.0.1:9901` | Ends mutual TLS; takes only the gateway's certificate; passes on `POST /v1/steps`, `GET /v1/steps/CHECKPOINT` and `GET /v1/trainer` and answers everything else 404; limits request sizes (1 MiB) and rates (20 a second, bursts of 40); times requests out; logs each request without its body |
| The training service (`python -m rollout_train.pods.training`) | `127.0.0.1:8001` (and `/healthz`, `/readyz` there) | Takes one step at a time, each idempotent by the checkpoint it makes: fetches the batch and the parent's files from the blob store, steps, keeps the new weights in the blob store and answers with their manifest and the step's metrics, while the trainer's processes keep the full state in the blob store from host memory (the answer completed once they have); beats with the pod's name, identity, address and the step it is taking |

The trainers (`rollout_lora:LoraTrainer`, `rollout_lora:FullTrainer`) start a process per GPU under torchrun (`python -m
torch.distributed.run --standalone --nproc-per-node N -m rollout_lora.workers`, the image's PyTorch), on a pod of one
GPU too, which hold the policy (on several GPUs, sharded over them) between steps and end when another run takes the
pod or its lease is released. On a host's pod whose vLLM sleeps while a step is taken (`ROLLOUT_SLEEP_VLLM`), they end
after each step, which gives vLLM back the GPU's memory. A run's trainer reaches the pod with `rollout_train.pods.RemoteTrainer`.
The service reads its lease: when a run takes the pod, it makes its trainer anew with the run's settings (the lease's)
and reads the ledger with the token the lease gives for that run, once no step runs; its beats then say it is ready for
that run.

## Variables

Beside those every pod reads ([deploy/images](../README.md#the-variables-both-pods-read)):

| Variable | Says |
|---|---|
| `ROLLOUT_TRAINER` | The trainer, by `module:name`: `rollout_lora:LoraTrainer` or `rollout_lora:FullTrainer` |
| `ROLLOUT_TRAINER_MODEL` | The model it trains (a Hugging Face id) |
| `ROLLOUT_TRAINER_SETTINGS` | Its settings, as a JSON object of `LoraSettings`' fields (default `{}`), until a run that holds the pod says its own |
| `ROLLOUT_TRAINER_GPUS` | The GPUs it steps on: the provider's `gpu_count`, which the pod is leased with; by default all the pod has |
| `ROLLOUT_SLEEP_VLLM` | On a pod that also serves (`runpod-host`): `1` to have the pod's vLLM sleep while a step is taken |
| `ROLLOUT_VLLM` | Where that vLLM listens (default `http://127.0.0.1:8000`) |
| `ROLLOUT_WORK` | Where steps' files and the answers of the steps made are kept (default `/workspace/rollout`, on the volume, so a step made before the pod started again is answered from it) |
| `ROLLOUT_LISTEN` | Where the training service listens (default `127.0.0.1:8001`) |

## Building

From the repository's root (CI builds it; this is the same command):

```bash
docker build -f deploy/images/trainer/Dockerfile -t rollout-trainer .
```

It starts from `TORCH_IMAGE`, PyTorch's image of the torch `uv.lock` pins
(`pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime`); the build fails if that image's torch is another
([deploy/images](../README.md#what-is-in-them)).
