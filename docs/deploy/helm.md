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
- **The monitor.** One Deployment per entry of `monitors`, each behind a Service and an Ingress, and an Ingress for
  Ray's dashboard.
- **What every pod mounts.** The state volume, the ConfigMap `rollout` with the cluster config, and the Secrets for
  gateway keys and Tinker ([what every pod is given](#what-every-pod-is-given)).

<!-- Follow-up: describe the values that start runs, and what the gateway and the monitors serve (`launchers`,
`gateway.profile`, `monitors.NAME.where`), once runs are submitted from the monitor and the command line. -->

## The chart's values

`deploy/chart/rollout/values.yaml` holds every value with a comment. The ones a deployment usually sets:

| Value | Default | What it sets |
|---|---|---|
| `image.repository`, `image.tag` | `localhost:30500/rollout-platform`, `dev` | The platform image ([build the platform image](image.md)) |
| `image.pullPolicy` | `Always` | When nodes pull the image again |
| `storageClass` | `local-path` | The class of every volume the chart makes; cannot change once a claim exists ([volumes](#volumes)) |
| `state.size` | `200Gi` | The state volume |
| `state.path` | `/root/.cache/rollout` | Where pods mount the state volume: the code's `~/.cache/rollout` |
| `secrets.stores`, `secrets.gatewayKeys`, `secrets.tinker` | `stores`, `gateway-keys`, `tinker` | The names of the Secrets the chart reads ([secrets](#make-the-secrets)) |
| `stores.postgres.storage`, `stores.s3.storage` | `20Gi`, `200Gi` | The stores' volumes; cannot change once made |
| `stores.bucket`, `stores.prefix` | `rollout-blobs`, `blobs/` | Where blobs are kept in the S3 store |
| `ray.version` | `2.59.0` | Ray's version, which must be the image's |
| `ray.idleSeconds` | `300` | How long a Ray worker stays without work before the autoscaler removes it |
| `ray.gpu.maxReplicas`, `ray.cpu.maxReplicas` | `1`, `1` | The most GPU and CPU worker pods at once |
| `ray.gpu.resources` | 4 CPUs, 8 GiB requested, 14 GiB and one `nvidia.com/gpu` as limits | One GPU worker pod |
| `gateway.replicas`, `gateway.port`, `gateway.host` | `1`, `8900`, `gateway.localhost` | The gateway's replicas, port and Ingress host |
| `monitors.NAME.host` | `monitor.localhost` | Each monitor's Ingress host |
| `ingress.className`, `ingress.rayHost` | `traefik`, `ray.localhost` | The ingress controller, and the host of Ray's dashboard |

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

## Make the Secrets

The chart reads three Secrets and makes none of them, so that uninstalling the chart never deletes a credential.
Make them once, before the first install:

1. The namespace:

    ```bash
    kubectl create namespace rollout
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

Other providers' keys, such as an OpenAI key for a judge, are named in the cluster config by environment variable
(`key_env`). Put each in a Secret, and add it to the environment every pod is given (`rollout.env` in
`templates/_helpers.tpl`).

## What every pod is given

Every pod of the platform (the Ray head and workers, the gateway, the monitors) mounts the same things:

- the state volume at `state.path`: run directories, Minecraft's servers and worlds, the Hugging Face cache
  (`HF_HOME`), scratch space;
- the ConfigMap `rollout` at `/etc/rollout`, with the cluster config at `/etc/rollout/cluster.toml`, which
  `ROLLOUT_CLUSTER` names;
- the Secret `gateway-keys` at `/etc/rollout-secrets/gateway`, and the Secret `tinker` at `/root/.tinker`.

And has these environment variables:

| Variable | From |
|---|---|
| `ROLLOUT_CLUSTER` | `/etc/rollout/cluster.toml` |
| `PGPASSWORD` | the Secret `stores`, `POSTGRES_PASSWORD`; the ledger's URL holds no password |
| `AWS_ENDPOINT_URL`, `AWS_DEFAULT_REGION` | the S3 store's Service, and `us-east-1` |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | the Secret `stores`, `ROOT_ACCESS_KEY_ID` and `ROOT_SECRET_ACCESS_KEY` |
| `TINKER_API_KEY` | the Secret `tinker`, where it has one |
| `HF_HOME` | `huggingface` under the state volume |

The monitors also get `RAY_AUTH_MODE=token` and the Ray cluster's token, to reach its job server.

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
`rollout cluster check` in a pod:

```bash
kubectl -n rollout exec deploy/gateway -- rollout cluster check
```

To use a managed Postgres or a cloud bucket in place of the chart's stores, change `[ledger]` and `[blobs]` in this
file and the S3 endpoint (`AWS_ENDPOINT_URL`) in `templates/_helpers.tpl`: see [Postgres and S3](stores.md).

## Ingresses

The chart makes an Ingress for the gateway (`gateway.host`), for each monitor (`monitors.NAME.host`) and for Ray's
dashboard (`ingress.rayHost`), all of class `ingress.className`. The default hosts end in `.localhost`, which
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
