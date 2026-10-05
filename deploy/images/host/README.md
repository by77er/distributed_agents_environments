# The host image

`ghcr.io/by77er/rollout-host`: a pod that trains and samples for the run that holds it, on one GPU (`runpod-host`): the
one-GPU runs of a local machine, rented. vLLM takes its share of the GPU's memory (`--gpu-memory-utilization`, 0.42
unless the provider's `memory_fraction` says), and the trainer has the rest.

| Process | Listens on | Does |
|---|---|---|
| `pki.sh renew` | | Renews the pod's certificate at about two thirds of its life, and publishes each new one for Envoy |
| Envoy | `0.0.0.0:8443` (the pod's exposed TCP port), admin on `127.0.0.1:9901` | Ends mutual TLS; takes only the gateway's certificate; passes on `POST /v1/completions` and `GET /v1/models` to vLLM, `POST /v1/steps`, `GET /v1/steps/CHECKPOINT` and `GET /v1/trainer` to the training service, and answers everything else 404 |
| vLLM (`vllm serve`, the workspace's version) | `127.0.0.1:8000` | Samples, as on an inference pod |
| The training service (`python -m rollout_train.pods.training`) | `127.0.0.1:8001` | Takes the run's steps, as on a training pod, its trainer made for the run that holds the pod |
| The follower (`python -m rollout_train.pods.inference`) | `127.0.0.1:8081` (`/healthz`, `/readyz`) | Loads what the run's channel should serve into vLLM, and beats for the pod: ready once vLLM serves it and the training service holds the run's trainer |

A checkpoint the training service makes is kept in the bucket (so evals, the monitor and a run started again find it)
and on the pod's disk (`ROLLOUT_BLOB_CACHE`, default `/workspace/blobs`), where the follower loads it from, with no
round trip through the bucket. With `ROLLOUT_SLEEP_VLLM=1` (the provider's `sleep`), vLLM sleeps while a step is
taken, as local engines do beside a trainer on one card (`rollout_train.colocated`), for a GPU too small for both.

## Variables

Those of the [inference image](../inference/README.md#variables) and the [trainer image](../trainer/README.md#variables)
(`ROLLOUT_TRAINER` and `ROLLOUT_TRAINER_MODEL` are needed), those every pod reads
([deploy/images](../README.md#the-variables-both-pods-read)), and:

| Variable | Says |
|---|---|
| `ROLLOUT_BLOB_CACHE` | Where the pod keeps a copy of every blob its processes put or read (default `/workspace/blobs`) |
| `ROLLOUT_TRAINER_URL` | Where the follower asks the training service which run's trainer it holds (default `http://127.0.0.1:8001`) |
| `ROLLOUT_SLEEP_VLLM` | `1`: vLLM sleeps while a step is taken |

## Building

From the repository's root (CI builds it; this is the same command):

```bash
docker build -f deploy/images/host/Dockerfile --build-arg VLLM_VERSION=0.30.0 -t rollout-host .
```
