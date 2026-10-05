# Volumes and backups

Where the platform's data lives on Kubernetes, how to keep it when claims are deleted, and how to back it up and
restore it. This page is for whoever runs the platform; read it before deleting anything.

**Read first:** [Install the Helm chart](helm.md#volumes). **Next:** [Ingress, TLS and sign-in](access.md).

## What to keep

| Data | Where | Lost if it is lost |
|---|---|---|
| The [ledger](../libraries/rollout-train/checkpoints.md#the-ledger) | Postgres's volume (`data-postgres-0`) | Every run's record: groups, [episodes](../libraries/rollout-train/episodes.md), steps, checkpoints, evals, names |
| Blobs | the S3 store's volume (`data-s3-0`), or the bucket of a managed service | Checkpoints' weights, trajectories, datasets, imported environments |
| The state volume | `state` | Run directories and scratch; the Hugging Face cache and Minecraft's worlds can be made again |

The ledger and the blobs belong together: the ledger names blobs by their SHA-256, and a blob is useful only through
the records that name it.

## Keep volumes when their claims are deleted

A volume's reclaim policy decides what happens to it when its claim is deleted. K3s's `local-path` class deletes the
volume's directory; `local-path-retain` (`deploy/k3s/storage-class.yaml`) keeps it. Name a retaining class in
`storageClass` before the first install, since a claim's class cannot change once it is made.

To keep the volumes of an install made with a deleting class, set each volume's policy to `Retain` by hand, once,
after any install that made a new one:

```bash
for pv in $(kubectl get pv -o name); do
  kubectl patch "$pv" -p '{"spec":{"persistentVolumeReclaimPolicy":"Retain"}}'
done
```

`helm uninstall` keeps the state volume (the chart marks it `helm.sh/resource-policy: keep`) and the stores' claims,
which a StatefulSet never deletes.

## Back up

Take the ledger first, then the blobs: a blob is always stored before the record that names it, so every blob the
ledger's dump names is in the store by the time you copy it.

1. **The ledger**, with `pg_dump`:

    ```bash
    kubectl -n rollout exec postgres-0 -- pg_dump -U rollout -Fc rollout > rollout-ledger.dump
    ```

2. **The blobs**, with any S3 client. Blobs never change once stored, so `sync` copies only new ones:

    ```bash
    kubectl -n rollout port-forward svc/s3 7070:7070 &
    aws s3 sync s3://rollout-blobs ./rollout-blobs --endpoint-url http://127.0.0.1:7070
    ```

    The credentials are the Secret `stores`' `ROOT_ACCESS_KEY_ID` and `ROOT_SECRET_ACCESS_KEY`
    ([Postgres and S3](stores.md#move-an-existing-ledger-and-blob-store) shows how to read them).

3. **The run directories** on the state volume, if runs will go on from them, by copying them out of any pod that
   mounts it:

    ```bash
    pod="$(kubectl -n rollout get pod -l app=gateway -o jsonpath='{.items[0].metadata.name}')"
    kubectl -n rollout cp "$pod:/root/.cache/rollout/runs" ./rollout-runs
    ```

## Restore

Into a new install, before any run starts:

```bash
kubectl -n rollout exec -i postgres-0 -- pg_restore -U rollout -d rollout --clean --if-exists < rollout-ledger.dump
aws s3 sync ./rollout-blobs s3://rollout-blobs --endpoint-url http://127.0.0.1:7070
```

## Uninstalling K3s deletes every volume

On K3s, every volume is a directory on the node under `/var/lib/rancher/k3s/storage`: the ledger's, the blob store's,
the state volume, and the registry's and BuildKit's. **K3s's uninstall script deletes `/var/lib/rancher`, and with it
every volume, whatever its reclaim policy.** Back up the ledger and the blobs first, or copy the volumes' directories:

```bash
sudo cp -a /var/lib/rancher/k3s/storage ./rollout-volumes
```

Then uninstall:

```bash
sudo /usr/local/bin/k3s-uninstall.sh
```
