# Remote providers

Training and sampling that happen outside the cluster: Tinker at Thinking Machines, GPU pods rented on RunPod, and
hosted model APIs. This page says what each needs from a deployment. It is for whoever adds a provider to a
cluster.

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

**Built.** GPU pods rented by the hour on RunPod serve a run's [channel](../libraries/rollout-train/channels.md), take
its training steps, or both on one GPU. A run leases the pods it needs when it starts and releases them when it ends; a
released pod stays warm for a few minutes for the next run, and a reaper deletes what no run holds. No runs, no pods.

| Kind | Table | Its pods run | What a run leases |
|---|---|---|---|
| `runpod-inference` | `[inference.NAME]` | `ghcr.io/by77er/rollout-inference`: vLLM and the follower | A pod for each replica of a channel on it |
| `runpod-trainer` | `[trainers.NAME]` | `ghcr.io/by77er/rollout-trainer`: the training service | A pod for its steps |
| `runpod-host` | `[inference.NAME]`, with a `runpod-trainer` whose `colocate_with` names it | `ghcr.io/by77er/rollout-host`: both on one GPU | One pod for the trained channel and the steps |

### How a run uses its pods

- **Claimed when it starts.** After its placement group is reserved, the run's driver claims its pods. It first takes a
  warm pod: one no run holds, of the same provider, image and model, taken by compare-and-set so that two runs never
  take one pod. Else it starts one in a free slot of the provider's `max_pods`; with every slot taken, the run waits and
  its launch says so. A pod is ready once RunPod's API has said its public address, which its lease keeps, and its beat
  says it is ready for the run; a pod not ready within the provider's `start_timeout` (1200 seconds unless said) is
  deleted, and the run fails saying which pod and why.
- **Renewed while it runs.** Every 30 seconds the run renews each lease and the time it is charged for each pod. A lease
  not renewed for 5 minutes is stale: its run is taken to be gone (its driver died, or its cluster went away).
- **Released when it ends,** whether it finished, was stopped, failed or reached a limit. The pod stays up, warm, for
  the provider's `idle_stop` (600 seconds unless said). The next run with the same image and model takes it with no cold
  start, and the pod is reset to that run: its lease names the new run and channel and holds a ledger token for it, the
  follower drops the old run's adapters and follows the new one's, and the training service makes its trainer anew
  with the new run's settings.
- **Reaped.** `rollout pods reap` (the chart's CronJob with `runpod.reaper: true`, every minute) deletes the pods of
  idle leases past their `idle_stop`, of stale leases, and every pod RunPod lists with the cluster's tag
  (`rollout-CLUSTER-`) that no lease names. Deleting a pod revokes its certificate. `rollout pods list` shows every
  lease: who holds it, since when, at what price.

The gateway reaches a pod at its public IP and raw TCP port over mutual TLS, and only over `https`: the run's channel
finds its pods by their leases and beats (`rollout_train.pods.routing.LeasedServers`), at the address the lease holds,
each checked by the identity named for the pod. The lease's address is what RunPod's API says (the pod's public IP and
the public port 8443/tcp is mapped to), read by the run that leased the pod; a beat says only whether the pod is ready,
never where it is reached, so a pod's ledger token cannot send its traffic elsewhere. A pod reaches the cluster only
through the ledger service, step-ca and the bucket.

**Spend.** A pod's time is charged at its hourly price (RunPod's `costPerHr` for it, else the table's `price`), from
when the run asked for it or took it until the run released it, and then its warm time until another run takes it or
it is deleted. The warm time is charged to the run that last held the pod: after that run has ended, so it is recorded
on the run, and it cannot stop it. While a run runs, its pods' time counts toward its `limits.spend` (with its turns on
hosted APIs), and the run ends stopped once it reaches it. `limits.hours` ends a run once it has run that long; its
RayJob's `activeDeadlineSeconds` stops it half an hour later in any case. Before a run is asked for, one step's spend on
its pods is estimated as each pod's price for as long as a step of the same trainer and model took here lately; where
no such run made three checkpoints, the estimate says it is not known yet.

Pods are not the cluster's capacity: Kueue's quota counts only what runs in the cluster, and a run is refused only for
more pods than a provider's `max_pods`. The New run form shows each RunPod provider's GPU type and price, and the
monitor's queue shows every pod with its run and what it has cost.

### The provider's table

Beside what every provider has (`models`, `replicas`), a RunPod table says what its pods are:

| Key | Default | Says |
|---|---|---|
| `image` | | The image its pods run, by digest (the images workflow's summary has each pushed digest) |
| `gpu_types` | | RunPod's GPU type ids, in order of preference (`NVIDIA H100 80GB HBM3`) |
| `gpu_count` | 1 | GPUs a pod has |
| `cloud` | `secure` | RunPod's cloud tier: `secure` or `community` |
| `regions` | any | RunPod's data centers its pods may be in (`["US-KS-2"]`) |
| `price` | | Dollars an hour a pod is reckoned at before RunPod says its own: what estimates use |
| `max_pods` | 1 | The most pods of the provider at once, across runs: a cap on what it spends |
| `idle_stop` | 600 | Seconds a released pod stays warm before it is deleted (0: deleted when released) |
| `start_timeout` | 1200 | Seconds a pod may take to be ready for its run |
| `volume_gb`, `container_disk_gb` | 50, 50 | The pod's volume (models, checkpoints, certificates) and container disk |
| `store` | `[blobs]` | The blob store its pods read and write (`[stores.NAME]`) |
| `step_ca` | | `{ url, provisioner, key_file, root, trust }`: step-ca for the pods' certificates (`trust = "system"` behind a Cloudflare Tunnel) |
| `secrets` | | Variables whose values are RunPod console secrets, by the secret's name (`HF_TOKEN = "hf_token"`) |
| `api_key_env` | `RUNPOD_API_KEY` | The variable RunPod's API key is read from |
| `memory_fraction` | 0.42 on a host | vLLM's share of the GPU's memory (`--gpu-memory-utilization`) |
| `sleep` | false | On a host: vLLM sleeps while a step is taken, for a GPU too small for both |
| `max_logprobs` | 20 | The top-k logprobs its vLLM is started with |

A `runpod-trainer` also says `trainer` (`lora` or `full`) and its `models`; one that takes its steps on a host's pods
says `colocate_with` and nothing of pods.

### A pod that trains and samples

A `runpod-host` pod is a local machine's one-GPU run, rented: vLLM with its share of the GPU's memory, the follower, and
the training service with the rest of the memory, behind one Envoy. A run whose trainer is a `runpod-trainer` with
`colocate_with` naming the host, and whose trained channel is on the host, leases one pod for both, renewed, released,
kept warm and reaped as one, and charged at one GPU's price. A checkpoint the training service makes goes to the bucket
(so evals, the monitor and a run started again find it) and stays on the pod's disk, where the follower loads it from.
The pod is ready once vLLM serves what the run says and the training service holds the run's trainer.

One H100 SXM for Qwen3.5-9B with LoRA (about $2.69 an hour on the secure cloud):

```toml
[inference.h100]
kind = "runpod-host"
image = "ghcr.io/by77er/rollout-host@sha256:DIGEST"
gpu_types = ["NVIDIA H100 80GB HBM3"]
cloud = "secure"
price = 2.69
max_pods = 1
idle_stop = 600
memory_fraction = 0.42                         # vLLM: about 34 GB; the trainer has the rest
store = "r2"
volume_gb = 100
step_ca = { url = "https://ca.example.com", provisioner = "launcher", trust = "system", key_file = "/etc/rollout-secrets/step-ca/provisioner.jwk", root = "/etc/rollout-secrets/step-ca/root_ca.crt" }
[inference.h100.models."Qwen/Qwen3.5-9B"]
context = 8192
options = { max_lora_rank = 32, args = "--max-model-len 8192" }

[trainers.h100-lora]
kind = "runpod-trainer"
colocate_with = "h100"
trainer = "lora"
models = ["Qwen/Qwen3.5-9B"]
segment_tokens = 8000
```

A run on it says `trainer.provider = "h100-lora"`, `channels.policy.provider = "h100"`, and a `limits.hours` (and
`limits.spend`) that bounds it.

### What a deployment provides

1. **RunPod's API key**, in the Secret `runpod` (`RUNPOD_API_KEY`), which runs' drivers lease pods with and the reaper
   reaps with; and `runpod.reaper: true` in the chart's values.
2. **A bucket RunPod reaches** (an R2 bucket), as `[stores.r2]` beside the cluster's own `[blobs]`, with two keys in the
   Secret `r2`: one that reads and writes (`WRITER_ACCESS_KEY_ID`, `WRITER_SECRET_ACCESS_KEY`: the platform and
   trainer and host pods) and one that only reads (`READER_ACCESS_KEY_ID`, `READER_SECRET_ACCESS_KEY`: inference pods).
   R2 scopes a key to a bucket, not to a prefix, so a writer's key may write anywhere in the bucket; blobs are named by
   their hash and checked when read, so a wrong write is refused by every reader. A run whose trainer or servers are
   RunPod's writes its blobs there, and every reader finds each blob in the store its reference names:

    ```toml
    [stores.r2]
    kind = "rollout_s3:S3BlobStore"
    bucket = "rollout"
    prefix = "blobs/"
    endpoint_url = "https://ACCOUNT_ID.r2.cloudflarestorage.com"
    region = "auto"
    access_key_id_env = "R2_WRITER_ACCESS_KEY_ID"
    secret_access_key_env = "R2_WRITER_SECRET_ACCESS_KEY"
    reader = { access_key_id_env = "R2_READER_ACCESS_KEY_ID", secret_access_key_env = "R2_READER_SECRET_ACCESS_KEY" }
    ```

3. **step-ca**, for the pods' certificates: `stepCa.enabled: true` and its password in the Secret `step-ca-password`
   (`password`). It makes itself on its first start, with a JWK provisioner (`launcher`) whose certificates last 24
   hours, and `rollout pki publish` (a CronJob every six hours, and once at each install) writes its root and the
   provisioner's key to the Secret `step-ca`, which every pod of the platform mounts at `/etc/rollout-secrets/step-ca`.
   The provider's `step_ca` names them there. A step-ca of your own does as well: put its root (`root_ca.crt`) and the
   provisioner's decrypted key (`provisioner.jwk`) in the Secret `step-ca`.
4. **The gateway's certificate**, `spiffe://rollout/gateway`, which the gateway and runs' drivers present to pods: the
   same job writes a new one to the Secret `gateway-tls` (`tls.crt`, `tls.key`, `ca.crt`), mounted at
   `/etc/rollout-secrets/tls`, before the one before lapses; the cluster config's `[tls]` names the three files. With a
   step-ca of your own, cert-manager's step-issuer can make the same Secret.
5. **The ledger service, reachable from RunPod.** The chart serves it (the Deployment `ledger`); its token is the Secret
   `ledger` (`ROLLOUT_LEDGER_TOKEN`, a long random string), which pods' tokens are signed with. `ledger.public` is where
   pods reach it, given to each pod when it is leased. Expose it, and step-ca, in one of these ways:
    - **A Cloudflare Tunnel** (`tunnel.enabled: true`): cloudflared runs in the cluster with its token from the Secret
      `tunnel` (`token`), and carries `tunnel.hostnames.ledger` to the ledger service and `tunnel.hostnames.stepCa` to
      step-ca, with Cloudflare ending TLS at its edge with a publicly trusted certificate. Make the tunnel with
      `cloudflared tunnel create NAME`, route each hostname to it with `cloudflared tunnel route dns NAME HOST`, and put
      `cloudflared tunnel token NAME` in the Secret. Pods then reach step-ca by the system's roots: say
      `trust = "system"` in the provider's `step_ca`; their certificates still chain to the cluster's root, which each
      pod is given when it is leased, and they renew with a token rather than over mutual TLS.
    - **For a trial, a quick tunnel**: `cloudflared tunnel --url http://ledger.rollout:8840` (from a pod in the cluster,
      or with a port forward) prints a random `trycloudflare.com` host to use as `ledger.public`. It has no uptime
      guarantee, and its host changes whenever cloudflared starts again: a run started with one host breaks if it
      changes.
    - **An Ingress** (`ledger.ingress`, with a TLS Secret) or a Service of type LoadBalancer or NodePort
      (`ledger.service`).

    The gateway needs no tunnel: it connects out to each pod's public IP and raw TCP port.
6. **The images**, built and pushed by the images workflow (a `v*` tag, or run by hand), each table naming its image by
   the digest the workflow's summary gives.
7. Optionally, RunPod console secrets (an `HF_TOKEN` for a gated model), named in the table's `secrets`.

A run's pods are checked before it is asked for: more pods than a provider's `max_pods`, pods without `step_ca`, or a
cluster whose pods cannot reach the ledger service (no `[ledger] public` and token) is refused. The opt-in live test
(`tests/rollout_runpod/test_live.py`, [testing](../guide/testing.md#live-tests)) checks a deployment's key and bucket
with a few cents of a cheap GPU.

### How a pod is secured

- **Mutual TLS (mTLS) through Envoy.** Envoy in each pod takes requests only from a client whose certificate chains to
  the cluster's root and carries the gateway's identity, `spiffe://rollout/gateway`. It passes on only the routes the
  pod serves (sampling, steps, or both) and answers everything else 404; vLLM and the training service listen on the
  pod's loopback interface. RunPod maps a raw TCP port to it: its HTTPS proxy would end TLS and drop the client's
  certificate.
- **Certificates from step-ca.** Each pod has a 24-hour certificate whose identity is `spiffe://rollout/pod/NAME`. It
  gets the first with a one-time token minted when it is leased, renews it itself at about two thirds of its life, and
  a certificate revoked when its pod is deleted is not renewed and lapses within a day. `deploy/images/README.md` has
  the details.
- **Scoped credentials.** A pod's ledger token reads its run's serving records, starts and checkpoints, writes the
  pod's own beat and reads its own lease, nothing else, and only while its lease names that run
  ([the ledger over HTTP](../libraries/rollout-train/checkpoints.md#the-ledger-over-http)). An inference pod gets the
  bucket's read-only key; a trainer or host pod the key that writes.

A checkpoint made in the cluster's own store and served on RunPod (a run that starts from a local checkpoint, an eval
of one on pods) is not copied to the bucket the pods read yet: run such a run where its checkpoints are, or train it on
RunPod from the start ([RunPod pods as providers](../research/runpod-providers.md#later)).

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
