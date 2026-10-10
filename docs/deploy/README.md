# Deploy the platform

This section is for people who install and run rollout for themselves or a team: what runs where, which setup to
choose, how to install it on Kubernetes with the Helm chart, where its data lives, and how to fix what goes wrong.

**Read first:** [Start here](../start/README.md). **Next:** [What runs where](roles.md).

## Read in this order

1. [What runs where](roles.md): the platform's roles (the stores, runs as jobs, the Ray cluster, the sandbox pools,
   the gateway, the monitor, the ledger service) and the CPU, GPU, memory and storage each needs.
2. [Choose a setup](setups.md): one machine without Kubernetes, one machine with K3s, or a cluster of several nodes,
   and when to use each.
3. [Prepare a Kubernetes cluster](kubernetes.md): the KubeRay operator, the NVIDIA device plugin and GPU runtime, a
   storage class, and an ingress controller.
4. [Build the platform image](image.md): one image for every role, built with BuildKit or in CI.
5. [Install the Helm chart](helm.md): its values, the Secrets it reads, the cluster config it mounts, and installing
   and upgrading.
6. [Share GPUs](gpus.md): whole cards per Ray worker, or time-slicing when several pods share a card.
7. [Postgres and S3](stores.md): the ledger in Postgres, blobs in S3 or an S3-compatible store, and moving an existing
   ledger and blob store into them.
8. [Volumes and backups](backups.md): reclaim policies, backing up Postgres and S3, and what uninstalling K3s deletes.
9. [Ingress, TLS and sign-in](access.md): host names, TLS, authentication in front of the monitor, and what is never
   exposed.
10. [Remote providers](providers.md): Tinker, GPU pods on RunPod, and hosted model APIs.
11. [Start runs and evals](runs.md): what to do once the platform is up.
12. [Troubleshooting](troubleshooting.md): problems seen in real deployments, and how to fix them.

Also in this section:

- [Local Postgres and S3 for development](../development/local-services.md): both stores in Docker Compose on a
  development machine, for the tests and for a cluster config.
- [Deploy with the rollout command](../guide/deploying.md): asking for runs and evals from a shell, what a run's job
  starts, engines elsewhere, the gateway and Ray.

## Files in the repository

| Path | What it holds |
|---|---|
| `deploy/chart/rollout` | The Helm chart: the stores, a Ray cluster, the sandbox pools, the gateway, the monitors, the ledger service and what each run's RayJob is made from, in one namespace |
| `deploy/images/platform` | The image every pod of the platform runs |
| `deploy/images/inference`, `deploy/images/trainer`, `deploy/images/host` | Images for GPU pods rented elsewhere, reached over mutual TLS |
| `deploy/k3s` | A one-node K3s cluster with a GPU: values for the device plugin, a retaining storage class, an in-cluster registry and BuildKit |
| `deploy/clusters/example.toml` | A cluster config for one machine without Kubernetes |
| `deploy/local` | Postgres and an S3-compatible store in Docker Compose, for development |
