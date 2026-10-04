# The inference image

`ghcr.io/by77er/rollout-inference`: a pod that serves one run's channel.

| Process | Listens on | Does |
|---|---|---|
| `pki.sh renew` | | Renews the pod's certificate at about two thirds of its life, and publishes each new one for Envoy |
| Envoy | `0.0.0.0:8443` (the pod's exposed TCP port), admin on `127.0.0.1:9901` | Ends mutual TLS; takes only the gateway's certificate; passes on `POST /v1/completions` and `GET /v1/models` and answers everything else 404; limits request sizes (8 MiB) and rates (200 a second, bursts of 400); times requests out; logs each request without its body |
| vLLM (`vllm serve`, the workspace's version) | `127.0.0.1:8000` | Samples; adapters are loaded and unloaded while it runs (`VLLM_ALLOW_RUNTIME_LORA_UPDATING`), sampled tokens' logprobs are those of the distribution sampled from (`--logprobs-mode processed_logprobs`) |
| The follower (`python -m rollout_train.pods.inference`) | `127.0.0.1:8081` (`/healthz`, `/readyz`) | Reads what the run says the channel should serve, fetches the checkpoint's files from the blob store, loads them into vLLM as an adapter named by the checkpoint's id, beats with the pod's name, identity, address and readiness |

The pod is ready (`/readyz` answers 200, and its beats say `ready`) once vLLM answers and has what the channel should
serve now: the base model while the run says nothing else, then each checkpoint once it is loaded. The adapter before
stays loaded, so a turn begun under it ends under it. When any process ends, the others are ended and the container
exits; RunPod starts it again, and the follower loads what the channel should serve anew.

## Variables

Beside those every pod reads ([deploy/images](../README.md#the-variables-both-pods-read)):

| Variable | Says |
|---|---|
| `ROLLOUT_RUN` | The run whose channel the pod serves, by id |
| `ROLLOUT_CHANNEL` | The channel, by its name within the run (default `policy`) |
| `ROLLOUT_MODEL` | The model vLLM serves, by the name a request asks for it under (a Hugging Face id) |
| `ROLLOUT_VLLM` | Where the follower reaches vLLM (default `http://127.0.0.1:8000`) |
| `ROLLOUT_CHECKPOINTS` | Where checkpoints' files are kept while they are served (default `/workspace/checkpoints`) |
| `ROLLOUT_HEALTH` | Where `/healthz` and `/readyz` are served (default `127.0.0.1:8081`) |
| `VLLM_MAX_LORA_RANK` | The largest adapter rank vLLM takes (default 32) |
| `VLLM_MAX_LORAS` | Adapters one batch may mix (default 2): at least the sum of the windows of the runs it serves, each run's `max_lag + 1` |
| `VLLM_MAX_CPU_LORAS` | Adapters held in CPU memory above those, to load again quickly (default: vLLM's, as many as `VLLM_MAX_LORAS`) |
| `VLLM_ARGS` | More of `vllm serve`'s options, split on spaces (`--max-model-len 8192 --gpu-memory-utilization 0.9`) |

## Building

From the repository's root (CI builds it; this is the same command):

```bash
docker build -f deploy/images/inference/Dockerfile --build-arg VLLM_VERSION=0.30.0 -t rollout-inference .
```

`VLLM_VERSION` is the version `uv.lock` pins, which the workflow reads from it.
