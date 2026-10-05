# Remote providers

Training and sampling that happen outside the cluster: Tinker at Thinking Machines, GPU pods rented on RunPod, and
hosted model APIs. This page says what each needs from a deployment, and marks what is built and what is designed. It
is for whoever adds a provider to a cluster.

**Read first:** [the cluster config](../guide/cluster.md#inference-providers), where providers are declared.
**Next:** [Start runs and evals](runs.md).

## Tinker

**Built.** Tinker trains LoRA (low-rank adaptation) adapters and samples from them on Thinking Machines' GPUs. A run
can train on Tinker and sample there, or train on Tinker and be served on your own vLLM engines through the bridge
from Tinker's format to PEFT ([bridges](../guide/cluster.md#bridges)).

- **Install:** the `tinker` extra (`uv sync --extra tinker`); the platform image has it.
- **Key:** Tinker's SDK reads `TINKER_API_KEY`, or the `credentials.json` that `tinker auth login` writes under
  `~/.tinker`. On Kubernetes, put either in the Secret `tinker` ([make the Secrets](helm.md#make-the-secrets)); the
  chart passes the key to every pod and mounts the file at `/root/.tinker`.
- **Cluster config:** an inference provider and a trainer of kind `tinker`, with the models each offers and their
  costs:

    ```toml
    [inference.tinker]
    kind = "tinker"                     # Tinker's SDK reads TINKER_API_KEY
    [inference.tinker.models."Qwen/Qwen3.5-4B"]
    context = 65536
    cost = { input = 0.0, output = 0.0 }   # dollars per million tokens, from Tinker's price list

    [trainers.tinker-lora]
    kind = "tinker"
    models = ["Qwen/Qwen3.5-4B"]
    segment_tokens = 32768
    cost = { train = 0.0 }
    ```

- **Spend:** a billing error from Tinker ends the run, and the run setting `limits.spend` caps a run's estimated spend.
  [The Tinker trainer and engine](../implementations/rollout-tinker.md) says what a step costs and how to read the bill.

## GPU pods on RunPod

**In progress.** GPU pods rented by the hour serve a run's [channel](../libraries/rollout-train/channels.md) (the
inference image) or take its training steps (the trainer image).

Built:

- the images `deploy/images/inference` (vLLM, the follower that loads what the run's channel should serve, Envoy) and
  `deploy/images/trainer` (the training service over the LoRA and full-weight trainers, Envoy), built and pushed to
  GHCR by `.github/workflows/images.yml` on a version tag or when run by hand;
- the code on the pods, the client of RunPod's pods API (`rollout_runpod.RunPod`) and of step-ca
  (`rollout_runpod.StepCa`), and the provider kinds `runpod-inference` and `runpod-trainer` in the cluster config.

Designed, not built yet: starting and stopping pods for a run, and the [gateway](../libraries/rollout-train/gateway.md)
reaching them by their heartbeats ([RunPod pods as providers](../research/runpod-providers.md)).

How a pod is secured:

- **Mutual TLS (mTLS) through Envoy.** Envoy in each pod takes requests only from a client whose certificate chains to
  the cluster's root and carries the gateway's identity, `spiffe://rollout/gateway`. It passes on only the routes the
  pod serves (sampling, or steps) and answers everything else 404; vLLM and the training service listen on the pod's
  loopback interface. Use a raw TCP port on RunPod: its HTTPS proxy ends TLS and would drop the client's certificate.
- **Certificates from step-ca.** Each pod has a 24-hour certificate whose identity is `spiffe://rollout/pod/NAME`.
  It gets the first with a one-time token, renews it itself at about two thirds of its life, and a revoked one is
  not renewed and lapses within a day. `deploy/images/README.md` has the details.
- **Scoped credentials.** A pod is given a read-only bucket token and, once the
  [ledger](../libraries/rollout-train/checkpoints.md#the-ledger) service is built, a ledger token that can read its
  run's serving records and write its beats, nothing else.

What a deployment provides for pods:

- a RunPod API key, as `RUNPOD_API_KEY`, and RunPod console secrets for the blob store's credentials (and `HF_TOKEN`
  for a gated model), named in the cluster config;
- a bucket in S3 or R2 for the blob store, reachable from RunPod and from the cluster;
- a step-ca deployment reachable from RunPod, with a JWK provisioner whose certificates last 24 hours, its root's
  fingerprint, and the provisioner's key;
- the gateway's client certificate: on Kubernetes from cert-manager with step-issuer, a `Certificate` whose URI SAN
  is `spiffe://rollout/gateway`; on one machine from `step ca certificate` with a renewal daemon beside the gateway;
- the cluster config's `[tls]`: the cluster's root, and the gateway's certificate and key.

## Hosted model APIs

**Built.** OpenAI's models (the Responses API, `rollout-openai`) and Anthropic's (the Messages API,
[`rollout-anthropic`](../implementations/rollout-anthropic.md)) are inference providers of the kind `api`: the gateway
samples a channel on one through its endpoint, for an eval of a hosted model or a slot that is not trained (a judge, a
fixed opponent). Such turns are not token-exact, so they are never trained on; each records what it cost, and an
eval's `limits.spend` ends it once it spends that ([hosted APIs](../guide/cluster.md#hosted-apis)).

- **Install:** both packages are in the platform image (`rollout-openai`, `rollout-anthropic`).
- **Keys:** `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`, named by each provider's `api_key_env`. On Kubernetes, put them
  in the Secret `providers` ([provider keys](helm.md#provider-keys)): the chart passes them to the gateway and to each
  run's job, never to the ConfigMap.
- **Cluster config:** the chart's `files/cluster.toml` and `deploy/clusters/example.toml` declare `[inference.openai]`
  and `[inference.anthropic]`, metered, with each vendor's current models, their contexts and their prices per million
  tokens (input, cached input, output; thinking is billed as output), and the date the prices were checked. Update the
  prices there when a vendor changes them: an eval's estimate and its spend are counted from them.
- **Rate limits:** `concurrency` caps the requests each gateway (and each run's driver) sends a provider at once; a
  429 or an overloaded API is asked again with backoff.
