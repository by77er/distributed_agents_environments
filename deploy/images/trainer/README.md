# The trainer image

`ghcr.io/by77er/rollout-trainer`: a pod that takes a run's training steps.

| Process | Listens on | Does |
|---|---|---|
| `pki.sh renew` | | Renews the pod's certificate at about two thirds of its life, and publishes each new one for Envoy |
| Envoy | `0.0.0.0:8443` (the pod's exposed TCP port), admin on `127.0.0.1:9901` | Ends mutual TLS; takes only the gateway's certificate; passes on `POST /v1/steps`, `GET /v1/steps/CHECKPOINT` and `GET /v1/trainer` and answers everything else 404; limits request sizes (1 MiB) and rates (20 a second, bursts of 40); times requests out; logs each request without its body |
| The training service (`python -m rollout_train.pods.training`) | `127.0.0.1:8001` (and `/healthz`, `/readyz` there) | Takes one step at a time, each idempotent by the checkpoint it makes: fetches the batch and the parent's files from the blob store, steps, keeps the new weights and state in the blob store, answers with their manifests and the step's metrics; beats with the pod's name, identity, address and the step it is taking |

The trainers (`rollout_lora:LoraTrainer`, `rollout_lora:FullTrainer`) run each step in a fresh process on the GPU, which
frees the GPU and the memory when it ends. A run's trainer reaches the pod with `rollout_train.pods.RemoteTrainer`.

## Variables

Beside those every pod reads ([deploy/images](../README.md#the-variables-both-pods-read)):

| Variable | Says |
|---|---|
| `ROLLOUT_TRAINER` | The trainer, by `module:name`: `rollout_lora:LoraTrainer` or `rollout_lora:FullTrainer` |
| `ROLLOUT_TRAINER_MODEL` | The model it trains (a Hugging Face id) |
| `ROLLOUT_TRAINER_SETTINGS` | Its settings, as a JSON object of `LoraSettings`' fields (default `{}`) |
| `ROLLOUT_WORK` | Where steps' files and the answers of the steps made are kept (default `/workspace/rollout`, on the volume, so a step made before the pod started again is answered from it) |
| `ROLLOUT_LISTEN` | Where the training service listens (default `127.0.0.1:8001`) |

## Building

From the repository's root (CI builds it; this is the same command):

```bash
docker build -f deploy/images/trainer/Dockerfile -t rollout-trainer .
```
