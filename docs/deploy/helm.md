# Install the Helm chart

The chart `deploy/chart/rollout` installs the platform into one namespace. This page covers its values, the Secrets
it reads, the cluster config it mounts, its ingresses and volumes, and how to install and upgrade it safely. It is for
whoever deploys on Kubernetes.

**Read first:** [Prepare a Kubernetes cluster](kubernetes.md) and [Build the platform image](image.md).
**Next:** [Share GPUs](gpus.md).

## What the chart installs

By role, in the namespace it is installed into (these pages use `rollout`):

- **The stores.** StatefulSets `postgres` (the [ledger](../libraries/rollout-train/checkpoints.md#the-ledger), Postgres
  17) and `s3` (versitygw, an S3 gateway whose buckets are directories), each on a volume of its own, behind Services at
  `postgres.rollout:5432` and `s3.rollout:7070`. A hook Job, `buckets`, makes the blob bucket after every install and
  upgrade if it is missing.
- **The Ray cluster.** A RayCluster `ray`: a head that runs no tasks, a GPU worker group and a CPU worker group, each
  started from zero by the autoscaler when work waits and removed after `ray.idleSeconds` without work. Clients
  present the cluster's token, which KubeRay keeps in a Secret named after the cluster. Its job server is at
  `ray-head-svc.rollout:8265`.
- **The gateway.** A Deployment of stateless replicas behind a Service at `gateway.rollout:8900` and an Ingress.
- **The ledger service.** A Deployment `ledger` (`rollout ledger serve --cluster`) behind a Service at
  `ledger.rollout:8840`, for pods outside the cluster; reachable from outside through an Ingress or a Service of
  another type when `ledger.ingress` or `ledger.service` asks ([the ledger over
  HTTP](../libraries/rollout-train/checkpoints.md#the-ledger-over-http)).
- **For RunPod's pods** ([GPU pods on RunPod](providers.md#gpu-pods-on-runpod)): with `runpod.reaper`, a CronJob
  `pods-reaper` (`rollout pods reap`, every minute); with `stepCa.enabled`, a StatefulSet `step-ca`, its Service at
  `step-ca.rollout:9000`, and a CronJob `pki-publish` and a hook Job (`rollout pki publish`) that write its root and
  provisioner key (the Secret `step-ca`) and a gateway certificate (`gateway-tls`), under a ServiceAccount `pki` that
  may write those two Secrets; with `tunnel.enabled`, a Deployment `tunnel` running cloudflared, which carries
  `tunnel.hostnames` to the ledger service and step-ca.
- **The sandbox pools.** For each kind `sandboxes` enables (`minecraft`), a Deployment of one pod running `rollout pool
  --kind KIND --cluster`, which holds at most `size` sandboxes and keeps their leases beside the ledger, behind the
  Service `sandboxes-KIND` (`sandboxes-minecraft.rollout:8710`); the cluster config's `[sandboxes.KIND]` names it as
  its `url`, and runs acquire from it there. Its requests are its sandboxes' memory, which runs' pods do not hold
  ([Where sandboxes run](../research/sandbox-placement.md)).
- **The monitor.** One Deployment per entry of `monitors`, each behind a Service (and an Ingress where
  `monitors.NAME.ingress` asks), and an Ingress for Ray's dashboard; and the Secret `monitor-token`, the token the
  monitors ask for, made where it is missing ([opening the monitor](access.md#opening-the-monitor)).
- **What each role is given.** Only what its code reads: the cluster config (the ConfigMap `rollout`), and of the
  state volume and the Secrets, what it uses ([what each role is given](#what-each-role-is-given)).
- **Network policies** (with `networkPolicies.enabled`, the default): nothing reaches a pod but the roles that use it
  ([network policies](#network-policies)).

- **Each run's job.** A RayJob made by a monitor when a run is asked for from its page (or by `rollout train
  --cluster` with Kubernetes credentials), from `files/rayjob.yaml`: a head pod of the platform's image sized from
  what the run needs (its driver, trainer, engine hosts and bridge, and room for Ray's own processes), within
  `rayjob.resources.limits`, with worker pods for its engine hosts where one pod cannot hold the run; started again up
  to `rayjob.backoffLimit` times when its driver is lost, and removed `rayjob.ttlSeconds` after it ends. The cluster
  config's `[kubernetes]` names the namespace and the template
  ([launching runs](../libraries/rollout-train/launching.md#a-rayjob)).
- **Kueue's queue** (with `kueue.enabled`): a ResourceFlavor, a ClusterQueue with `kueue.quota` and a LocalQueue in the
  namespace (`templates/kueue.yaml`), [below](#kueue).
- **The monitors' account.** A ServiceAccount `monitor` with a Role that may create, get, list, watch and delete
  `rayjobs`, and get its own namespace (whose Pod Security labels `rollout cluster check` reads); with Kueue, also get, list and watch `workloads` and get the chart's LocalQueue, and a ClusterRole
  (`NAMESPACE-monitor`) that may get the chart's ClusterQueue and its pending Workloads through Kueue's visibility API
  (`templates/rbac.yaml`, [what the monitor shows](#what-the-monitor-shows)).
- **The admission policy** (with `admission.enabled`, the default): a ValidatingAdmissionPolicy and its binding,
  `NAMESPACE-rayjobs`, bound to the release's namespace, which refuses a RayJob that asks for more than a run is
  given ([Pod Security](kubernetes.md#pod-security)). Its pods also meet the namespace's Pod Security level.
- **The presets.** A hook Job, `presets`, saves `files/presets/*.toml` beside the ledger after every install and
  upgrade (`rollout preset load /etc/rollout/presets --cluster`), a new version only where a preset's newest says
  otherwise.

The gateway runs `rollout gateway --cluster`: it samples every channel a run's start names on the providers it
reaches. Each monitor serves its page over the cluster config's ledger, or over `monitors.NAME.where` where it names
one (a ledger's URL, or a run's directory on the state volume), and asks for runs on the cluster config.

## The chart's values

`deploy/chart/rollout/values.yaml` holds every value with a comment. The ones a deployment usually sets:

| Value | Default | What it sets |
|---|---|---|
| `image.repository`, `image.tag` | `localhost:30500/rollout-platform`, `dev` | The platform image ([build the platform image](image.md)) |
| `image.pullPolicy` | `Always` | When nodes pull the image again |
| `storageClass` | `local-path` | The class of every volume the chart makes; cannot change once a claim exists ([volumes](#volumes)) |
| `state.size` | `200Gi` | The state volume |
| `state.path` | `/root/.cache/rollout` | Where pods mount the state volume: the code's `~/.cache/rollout` |
| `secrets.stores`, `secrets.gatewayKeys`, `secrets.tinker`, `secrets.providers` | `stores`, `gateway-keys`, `tinker`, `providers` | The names of the Secrets the chart reads ([secrets](#make-the-secrets)) |
| `secrets.ledger`, `secrets.runpod`, `secrets.r2`, `secrets.stepCa`, `secrets.gatewayTls` | `ledger`, `runpod`, `r2`, `step-ca`, `gateway-tls` | The Secrets for the ledger service's token, RunPod's key, a second bucket's two keys, step-ca's root and provisioner key, and the gateway's certificate ([secrets](#make-the-secrets)) |
| `ledger.public`, `ledger.ingress`, `ledger.service` | none, off, `ClusterIP` | Where pods outside the cluster reach the ledger service, and how it is exposed |
| `runpod.reaper` | `false` | The CronJob that deletes the pods no run holds |
| `stepCa.enabled`, `stepCa.passwordSecret`, `stepCa.dnsNames` | `false`, `step-ca-password`, none | The chart's step-ca, its password's Secret, more names for its certificate |
| `tunnel.enabled`, `tunnel.secret`, `tunnel.hostnames` | `false`, `tunnel`, none | A Cloudflare Tunnel to the ledger service and step-ca, its token's Secret, and its hostnames |
| `clusterExtra` | none | TOML appended to the cluster config: this deployment's own tables, such as `[stores.r2]`, `[tls]` and RunPod providers (`files/cluster.toml` has a commented example of each) |
| `stores.postgres.storage`, `stores.s3.storage` | `20Gi`, `200Gi` | The stores' volumes; cannot change once made |
| `stores.bucket`, `stores.prefix` | `rollout-blobs`, `blobs/` | Where blobs are kept in the S3 store |
| `ray.version` | `2.59.0` | Ray's version, which must be the image's |
| `ray.idleSeconds` | `300` | How long a Ray worker stays without work before the autoscaler removes it |
| `ray.gpu.maxReplicas`, `ray.cpu.maxReplicas` | `1`, `1` | The most GPU and CPU worker pods at once |
| `ray.gpu.resources` | 4 CPUs, 8 GiB requested, 14 GiB and one `nvidia.com/gpu` as limits | One GPU worker pod |
| `rayjob.resources` | 1 CPU, 4 GiB requested; 14 GiB and one `nvidia.com/gpu` as limits | The most one pod of a run's Ray cluster may have (its limits, one node's worth); its requests apply only where a run's demand is not given |
| `rayjob.ttlSeconds` | `30` | How long a finished run's Ray cluster stays (and holds what it asked for) |
| `kueue.enabled`, `kueue.queue`, `kueue.quota` | `false`, `runs`, 12 CPUs, 9 GiB, one GPU | Kueue's admission of runs ([Kueue](#kueue)) |
| `sandboxes.minecraft.enabled`, `.size`, `.resources` | `true`, `4`, 4 CPUs and 7.5 GiB requested, 10 GiB as the limit | The Minecraft worlds' pool: at most `size` worlds at once, each asking for 1.75 GiB and a CPU |
| `sandboxes.minecraft.onPods` | none | The section's `on_pods`, as TOML (`true`, or `'{ size = 8 }'`): the worlds are also served from the host pods a run leases whose provider (in `clusterExtra`) lists `sandboxes = ["minecraft"]`, with this pool behind them ([sandboxes on a host pod](providers.md#sandboxes-on-a-host-pod)) |
| `gateway.replicas`, `gateway.port`, `gateway.host` | `1`, `8900`, `gateway.localhost` | The gateway's replicas, port and Ingress host |
| `monitors.NAME.ingress`, `.host`, `.hosts` | `false`, `monitor.localhost`, none | Whether a monitor has an Ingress, its host, and more names the monitor answers under (beside `localhost`, `127.0.0.1` and its Service's names) |
| `secrets.monitor` | `monitor-token` | The monitors' token (`ROLLOUT_MONITOR_TOKEN`), which the chart makes where it is missing |
| `ingress.className`, `ingress.rayHost` | `traefik`, `ray.localhost` | The ingress controller, and the host of Ray's dashboard |
| `admission.enabled` | `true` | The admission policy that holds what a RayJob in the namespace may ask for ([Pod Security](kubernetes.md#pod-security)) |
| `networkPolicies.enabled`, `.egress` | `true`, `true` | The namespace's NetworkPolicies, and the egress limits of the long-lived Ray cluster's workers and the sandbox pools ([network policies](#network-policies)) |
| `networkPolicies.ingressController`, `.kuberay`, `.dns` | K3s's Traefik in `kube-system`, `kuberay`, K3s's CoreDNS | Where the ingress controller, KubeRay's operator and the cluster's DNS run |

Every container has `resources` with requests and a memory limit; [what runs where](roles.md#what-each-role-needs)
lists them.

Keep your settings in a values file of your own, outside the chart, and pass it at every install and upgrade:

```yaml title="rollout-values.yaml"
image:
  repository: registry.example.com/rollout-platform
  tag: "1.0.0"
storageClass: local-path-retain
gateway:
  host: gateway.example.com
ingress:
  rayHost: ray.example.com
```

## Kueue

With `kueue.enabled`, Kueue admits each run's RayJob whole. Kueue itself is installed outside the chart
([install the operators](kubernetes.md#install-the-operators)); the chart makes:

| Object | Name (values) | What it does |
|---|---|---|
| ResourceFlavor | `rollout` (`kueue.flavor`) | The node's resources, with no node labels: one kind of node |
| ClusterQueue | `rollout` (`kueue.clusterQueue`) | Takes RayJobs from the release's namespace only; its quota (`kueue.quota`: `cpus`, `memoryGib`, `gpus`) bounds the `cpu`, `memory` and `nvidia.com/gpu` every admitted run's pods ask for together |
| LocalQueue | `runs` (`kueue.queue`), in the namespace | What each run's RayJob names (`kueue.x-k8s.io/queue-name`) |

The cluster config then names the queue (`[kubernetes] queue`) and states the quota as its `[capacity]`. A run's
RayJob is made suspended, and Kueue starts it once the quota has room for its head pod, its worker pods and the pod
KubeRay starts to submit its job; until then the monitor's launch tile says it waits for admission, with Kueue's
reason. A run whose pods would ask for more than the quota is refused when it is asked for. The default quota is what
one node of 20 CPUs and 23 GiB with one GPU has beside the platform's own pods (the stores, the Ray cluster, the
gateway and the monitor, about 4 CPUs and 6.5 GiB) and the Minecraft pool's (4 CPUs, 7.5 GiB), which are outside the
queue; set it to what your nodes give runs:

```yaml title="rollout-values.yaml"
kueue:
  enabled: true
  quota: {cpus: 12, memoryGib: 9, gpus: 1}
```

The long-lived RayCluster `ray` is not in the queue: its GPU worker, when the autoscaler starts it, holds a card Kueue
does not count, and a run admitted meanwhile waits for that card as a pending pod.

### What the monitor shows

The Machines tab's Queue section ([the queue](../libraries/rollout-train/monitor.md#the-queue)) reads Kueue through the
API server with the monitor's account: the quota and what is reserved of it as a bar for each of CPU, memory and GPU,
split into each admitted run's share, and the runs that wait in Kueue's order, each with what it asks for, how long it
has waited and why. A waiting run's page and launch tile say its place in the queue. What the account may read, all of
it get only, and the ClusterQueue rules limited to the chart's ClusterQueue by name:

| Rule | API group | Resource | Scope | What the monitor reads |
|---|---|---|---|---|
| Role `monitor` | `kueue.x-k8s.io` | `workloads` (get, list, watch) | the namespace | each run's Workload: its RayJob, what it asks for, whether it is admitted and why it waits |
| Role `monitor` | `kueue.x-k8s.io` | `localqueues` (`kueue.queue`) | the namespace | which ClusterQueue the queue feeds |
| ClusterRole `NAMESPACE-monitor` | `kueue.x-k8s.io` | `clusterqueues` (`kueue.clusterQueue`) | the cluster | the quota and what is reserved of it |
| ClusterRole `NAMESPACE-monitor` | `visibility.kueue.x-k8s.io` | `clusterqueues/pendingworkloads` (`kueue.clusterQueue`) | the cluster | the order of the pending Workloads |

Kueue serves its visibility API by default (the `kueue-visibility-server` APIService, `v1beta2`). Where it is not
served, or not readable, the pending runs are listed in the order their Workloads were made, and `/api/queue` says so
(`"order": "created"`). To see what the account may read:

```bash
kubectl auth can-i get clusterqueues.kueue.x-k8s.io/rollout --as system:serviceaccount:rollout:monitor
kubectl get --raw /apis/visibility.kueue.x-k8s.io/v1beta2/clusterqueues/rollout/pendingworkloads \
  --as system:serviceaccount:rollout:monitor
```

## Make the Secrets

The chart reads the Secrets below and makes none of them, so that uninstalling the chart never deletes a credential.
The one it makes is the monitors' token, `monitor-token`, where it is missing, and it keeps that one when it is
uninstalled ([opening the monitor](access.md#opening-the-monitor)). Make the others once, before the first install:

1. The namespace, labelled for Pod Security ([Pod Security](kubernetes.md#pod-security)):

    ```bash
    kubectl create namespace rollout
    kubectl label namespace rollout pod-security.kubernetes.io/enforce=baseline \
      pod-security.kubernetes.io/warn=restricted pod-security.kubernetes.io/audit=restricted
    ```

2. **`stores`**: the Postgres password and the S3 store's root keys. Postgres reads its password only when it first
   creates its data directory, so changing this Secret later does not change the database's password.

    ```bash
    kubectl -n rollout create secret generic stores \
      --from-literal=POSTGRES_PASSWORD="$(openssl rand -hex 24)" \
      --from-literal=ROOT_ACCESS_KEY_ID=rollout \
      --from-literal=ROOT_SECRET_ACCESS_KEY="$(openssl rand -hex 24)"
    ```

3. **`gateway-keys`**: the secrets the gateway signs and checks keys with, one `KID SECRET` line each, the signing one
   first; each secret is 32 bytes at least ([keys](../libraries/rollout-train/gateway.md#keys)):

    ```bash
    printf 'k1 %s\n' "$(openssl rand -hex 32)" > gateway.keys
    kubectl -n rollout create secret generic gateway-keys --from-file=gateway.keys=gateway.keys
    rm gateway.keys
    ```

4. **`tinker`**, only to train or sample on Tinker: the API key, and optionally the `credentials.json` that
   `tinker auth login` writes:

    ```bash
    kubectl -n rollout create secret generic tinker --from-literal=TINKER_API_KEY="$TINKER_API_KEY"
    ```

5. <a id="provider-keys"></a>**`providers`**, only to sample hosted APIs (OpenAI's, Anthropic's): their API keys, under
   the names the cluster config's `api_key_env` say. Either key may be left out; a provider whose key is missing
   refuses each turn on it, saying which variable is not set.

    ```bash
    kubectl -n rollout create secret generic providers \
      --from-literal=OPENAI_API_KEY="$OPENAI_API_KEY" \
      --from-literal=ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
    ```

    The chart gives these keys only to what samples hosted APIs: the gateway's pods and each run's job (its RayJob's
    pods, through `files/rayjob.yaml`), as environment variables read from the Secret by reference
    (`rollout.providerEnv` in `templates/_helpers.tpl`, [what each role is given](#what-each-role-is-given)). They are never in the ConfigMap: the cluster config names
    them (`api_key_env`), and the RayJob template holds only the reference. To change a key, update the Secret and
    restart the gateway (`kubectl -n rollout rollout restart deploy/gateway`); runs started after read the new one.

6. **For RunPod's pods**, only to rent them ([GPU pods on RunPod](providers.md#what-a-deployment-provides)):

    ```bash
    kubectl -n rollout create secret generic ledger --from-literal=ROLLOUT_LEDGER_TOKEN="$(openssl rand -hex 32)"
    kubectl -n rollout create secret generic runpod --from-literal=RUNPOD_API_KEY="$RUNPOD_API_KEY"
    kubectl -n rollout create secret generic r2 \
      --from-literal=WRITER_ACCESS_KEY_ID="$R2_WRITER_ID" --from-literal=WRITER_SECRET_ACCESS_KEY="$R2_WRITER_SECRET" \
      --from-literal=READER_ACCESS_KEY_ID="$R2_READER_ID" --from-literal=READER_SECRET_ACCESS_KEY="$R2_READER_SECRET"
    kubectl -n rollout create secret generic step-ca-password --from-literal=password="$(openssl rand -hex 24)"
    kubectl -n rollout create secret generic tunnel --from-literal=token="$(cloudflared tunnel token rollout)"
    ```

    The chart gives `ROLLOUT_LEDGER_TOKEN` to the ledger service and runs' jobs, the `R2_*` keys to runs' jobs (and
    the writer's to the monitors, which read episodes kept there), and `RUNPOD_API_KEY` to runs' jobs and the reaper
    ([what each role is given](#what-each-role-is-given)). The Secrets `step-ca` and `gateway-tls` are written by the chart's
    `rollout pki publish`; with a step-ca of your own, make them yourself (`root_ca.crt`, `provisioner.jwk`; `tls.crt`,
    `tls.key`, `ca.crt`).

Another provider's key (a tool set's token, say) is named in the cluster config by environment variable (`token_env`,
`key_env`). Put each in a Secret, and add it to the environment of the roles that use it (`templates/_helpers.tpl`):
a run's job (`rollout.runEnv`), and the gateway for a provider it samples.

## What each role is given

Each role is given only what its code reads (`rollout_train.cluster.Cluster.secrets_of`, `templates/_helpers.tpl`),
each Secret by reference, so a role that is broken into holds no more than it needs. Code imported from anywhere is
checked on the long-lived Ray cluster, whose pods hold nothing at all: the monitor hands the job server the
environment's zip itself. A Secret that is a file is mounted read-only, readable by its owner only.

| Role | Variables from Secrets | Secrets mounted | Volumes |
|---|---|---|---|
| The ledger service (`ledger`) | `PGPASSWORD`; `ROLLOUT_LEDGER_TOKEN` (`ledger`) | none | the cluster config |
| The gateway | `PGPASSWORD`; `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (it records turns in the blob store); `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` (`providers`) | `gateway-keys`, `gateway-tls` (the certificate it reaches RunPod's pods with) | the state volume (tokenizers), the cluster config |
| Each monitor | `PGPASSWORD`; the blob store's keys (it reads episodes and stores imports); `R2_WRITER_ACCESS_KEY_ID`, `R2_WRITER_SECRET_ACCESS_KEY` (`r2`: episodes kept there); the Ray cluster's token; `ROLLOUT_MONITOR_TOKEN` (`monitor-token`) | none | the state volume, the cluster config |
| Each sandbox pool | `PGPASSWORD` (its leases are kept beside the ledger) | none | the state volume, the cluster config |
| The reaper (`pods-reaper`) | `PGPASSWORD`; `RUNPOD_API_KEY` (`runpod`) | `step-ca` (what a deleted pod's certificate is revoked with) | the cluster config |
| The presets Job | `PGPASSWORD` | none | the cluster config |
| The long-lived Ray cluster's head and workers | none (KubeRay gives them the cluster's token itself) | none | none |
| Each run's job (`files/rayjob.yaml`) | everything a run uses: `PGPASSWORD`, the blob store's keys, the four `R2_*` keys, `TINKER_API_KEY`, `ROLLOUT_LEDGER_TOKEN`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `RUNPOD_API_KEY` | `gateway-keys` (it signs its agents' keys), `tinker`, `step-ca` (it mints pods' one-time tokens), `gateway-tls` | the state volume, the cluster config |
| The pod that submits a run's job | none | none | none |
| Postgres, the S3 store, the `buckets` Job | their own (`stores`) | none | their volumes |
| step-ca, the `pki-publish` jobs, the tunnel | their own (`step-ca-password`, `tunnel`) | none | step-ca's volume |

`PGPASSWORD` is the Secret `stores`' `POSTGRES_PASSWORD` (the ledger's URL in the cluster config holds none), and the
blob store's keys its `ROOT_ACCESS_KEY_ID` and `ROOT_SECRET_ACCESS_KEY`, with `AWS_ENDPOINT_URL` and
`AWS_DEFAULT_REGION`. Every role but Ray's reads the cluster config at `/etc/rollout/cluster.toml` (`ROLLOUT_CLUSTER`)
from the ConfigMap `rollout`, and those that load models keep them under `HF_HOME` on the state volume. To see what a
role's pod is missing of what it reads, run `rollout cluster check --role ROLE` there
([the cluster config](#the-cluster-config)).

## Network policies

With `networkPolicies.enabled` (the default), nothing in the namespace reaches a pod unless a policy lets it
(`templates/networkpolicies.yaml`); K3s enforces NetworkPolicies itself. Each role is reached only by the roles that use
it, picked by the labels the chart sets (`app`, and KubeRay's `ray.io/cluster`):

| Pods | Reached by |
|---|---|
| Postgres (5432) | the gateway, the monitors, the ledger service, the sandbox pools, the presets Job, the reaper, runs |
| The S3 store (7070) | the gateway, the monitors, runs, the `buckets` Job |
| The gateway | runs, and harnesses inside sandboxes; the ingress controller (its Ingress) |
| The ledger service | the tunnel; the ingress controller with `ledger.ingress`; anything with a `ledger.service.type` other than `ClusterIP`; the roles `networkPolicies.ledgerFrom` names |
| step-ca (with `stepCa.enabled`) | the tunnel, the `pki-publish` jobs, runs, the reaper |
| The monitors (8765) | one another; the ingress controller where a monitor's Ingress is on |
| Each sandbox pool | runs, at the pool's port |
| The long-lived Ray cluster | its own pods and KubeRay's operator; the monitors and the ingress controller at its job server and dashboard (8265) |
| Each run's Ray cluster and the pod that submits its job | one another (every run's), KubeRay's operator, the gateway |

`networkPolicies.ingressController` and `networkPolicies.kuberay` say where the ingress controller's pods
(K3s's Traefik) and KubeRay's operator are. A kubelet's probes and `kubectl port-forward` reach a pod through the node,
not the pod network, and no policy stops them, so the monitor is still opened through a port-forward, and backups
through one ([backups](backups.md)).

With `networkPolicies.egress` (the default), the pods that run code from elsewhere reach out only where they must:
the long-lived Ray cluster's workers (where imported environments are checked) reach their own cluster, the cluster's
DNS and the internet (to build an environment's Python); the sandbox pools reach Postgres, the gateway (a harness inside a
sandbox samples through it), the DNS and the internet (Minecraft's server and JDK). Neither reaches any other address in `networkPolicies.privateRanges`: not the stores,
the gateway, a run, the Kubernetes API or the node. The other roles' egress is not limited.

## The cluster config

The chart's `files/cluster.toml` is the cluster config every pod reads, in the format
[the cluster config](../guide/cluster.md) describes. Helm renders it as a template, so it names the stores, the Ray
job server and the gateway at the addresses of the release it is installed with:

```toml title="files/cluster.toml (in part)"
[ledger]
url = "{{ include "rollout.ledgerUrl" . }}"  # the password is PGPASSWORD's

[blobs]
kind = "rollout_s3:S3BlobStore"               # the endpoint and the credentials come from the environment
bucket = "{{ .Values.stores.bucket }}"
prefix = "{{ .Values.stores.prefix }}"
```

Its other sections describe what this cluster offers: the inference providers and their models, the trainers, the
sandbox pools and the environments. Edit them to match your GPUs and models, then upgrade the chart. Pods read the
file when they start, so restart the gateway and the monitors after an upgrade that changes it. To check it, run
`rollout cluster check` in a role's pod, for what that role reads:

```bash
kubectl -n rollout exec deploy/gateway -- rollout cluster check --role gateway
kubectl -n rollout exec deploy/monitor-main -- rollout cluster check --role monitor
```

To use a managed Postgres or a cloud bucket in place of the chart's stores, change `[ledger]` and `[blobs]` in this
file and the S3 endpoint (`AWS_ENDPOINT_URL`) in `templates/_helpers.tpl`: see [Postgres and S3](stores.md).

## Ingresses

The chart makes an Ingress for the gateway (`gateway.host`), for each monitor whose `monitors.NAME.ingress` is on
(`monitors.NAME.host`) and for Ray's dashboard (`ingress.rayHost`), all of class `ingress.className`. A monitor is
opened through a port-forward otherwise ([opening the monitor](access.md#opening-the-monitor)). The default hosts end in `.localhost`, which
browsers send to the machine they run on. Set real host names, TLS and sign-in before anyone else reaches them:
[Ingress, TLS and sign-in](access.md).

## Volumes

The chart makes three kinds of volume, all of class `storageClass`:

- `data-postgres-0`, the ledger's;
- `data-s3-0`, the blob store's;
- `state`, which every pod mounts. Uninstalling the chart keeps it (`helm.sh/resource-policy: keep`).

A claim's class and the stores' sizes cannot change once the claim is made, so choose them before the first install.
On K3s, name `local-path-retain` there, so that deleting a claim keeps its directory on the node.

The state volume is `ReadWriteOnce`: every pod that mounts it must run on the same node. On a cluster of several
nodes, give it a storage class that several nodes can mount at once (`ReadWriteMany`, such as NFS or CephFS) and
change the claim's access mode in `templates/config.yaml`.

[Volumes and backups](backups.md) covers reclaim policies and backups.

## Install and upgrade

Install, or upgrade to a new chart or new values, with the same command:

```bash
helm upgrade --install rollout deploy/chart/rollout -n rollout -f rollout-values.yaml
```

**Never pass `--reuse-values`.** With it, Helm takes the values the release had and does not merge in the chart's
defaults, so a value the chart has added since renders empty, or the render fails. Pass your values file every time,
or nothing to take the chart's defaults.

Before an upgrade:

- see what you set before, and what the chart would make now:

    ```bash
    helm get values rollout -n rollout
    helm template rollout deploy/chart/rollout -n rollout -f rollout-values.yaml > rendered.yaml
    ```

- leave `storageClass`, `stores.postgres.storage` and `stores.s3.storage` as they were: Kubernetes refuses to change
  them ([an upgrade fails on an immutable field](troubleshooting.md#an-upgrade-fails-on-an-immutable-field)).

After installing, check that the stores and the Ray head are ready and the bucket was made:

```bash
kubectl -n rollout get pods
kubectl -n rollout get raycluster ray
kubectl -n rollout logs job/buckets
```

`helm uninstall rollout -n rollout` deletes the chart's Deployments, Services and the Ray cluster. It keeps the state
volume, and the stores' claims, which a StatefulSet never deletes; delete those by hand only once they are backed up.
