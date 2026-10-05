# Build the platform image

Every pod of the platform runs one image, built from `deploy/images/platform`. This page says what it holds and how
to build it and push it where the nodes pull from, for whoever deploys on Kubernetes.

**Read first:** [Prepare a Kubernetes cluster](kubernetes.md). **Next:** [Install the Helm chart](helm.md).

## What the image holds

- Every workspace package with every extra (vLLM, the trainers, Tinker, the renderers) in `/opt/rollout/venv`,
  installed exactly as `uv.lock` pins them, on a Python 3.13 of its own;
- `rollout-verifiers`, a project with its own lock, in a Python environment of its own at `/opt/rollout/verifiers`;
- the repository's source at `/opt/rollout/source`, the working directory of Ray's pods;
- a C compiler (`gcc`, `libc6-dev`) for the kernels Triton and torch compile while vLLM and the trainers run;
- a Java runtime for Minecraft's Paper servers, and Node.js for the Minecraft harness;
- `git`, for importing environments from git, and `rsync`.

The Ray cluster, the [gateway](../libraries/rollout-train/gateway.md) and the monitor all run it. It is large, because
of CUDA, torch and vLLM: leave tens of GB for the build cache and the registry.

## Build with BuildKit in the cluster

On a cluster without Docker, such as K3s, `deploy/k3s/build.yaml` runs a registry and BuildKit in the namespace
`build`. The registry is a NodePort at `localhost:30500` on every node, which containerd pulls from over plain HTTP;
BuildKit pushes to the same registry at its in-cluster name.

1. Start the registry and BuildKit:

    ```bash
    kubectl apply -f deploy/k3s/build.yaml
    ```

2. Install `buildctl` (BuildKit's client) on your machine, reach BuildKit through a port forward, and build from the
   repository's root:

    ```bash
    kubectl -n build port-forward statefulset/buildkit 1234:1234 &
    buildctl --addr tcp://127.0.0.1:1234 build --frontend dockerfile.v0 --local context=. \
      --local dockerfile=deploy/images/platform \
      --output type=image,name=registry.build.svc.cluster.local:5000/rollout-platform:dev,push=true
    ```

The chart's default image is `localhost:30500/rollout-platform:dev`, the same image by the name the nodes use.

## Build with Docker or in CI

Anywhere Docker with BuildKit runs (a workstation, a CI job), build from the repository's root and push to a registry
your nodes can pull from:

```bash
docker buildx build -f deploy/images/platform/Dockerfile -t registry.example.com/rollout-platform:1.0.0 --push .
```

Then name it in the chart's values:

```yaml
image:
  repository: registry.example.com/rollout-platform
  tag: "1.0.0"
```

The chart sets no image pull secret. A registry that needs credentials has to be configured on the nodes (for K3s,
in `/etc/rancher/k3s/registries.yaml`).

## Roll out a new image

- **A new tag.** Set `image.tag` and upgrade the chart ([installing and upgrading](helm.md#install-and-upgrade)).
- **The same tag.** The chart pulls with `pullPolicy: Always`, so restarting the platform's Deployments pulls the
  image again:

    ```bash
    kubectl -n rollout rollout restart deployment
    ```

    Ray's worker pods pull it whenever the autoscaler starts one. The Ray head pulls it when its pod is deleted and
    made again (`kubectl -n rollout delete pod -l app=ray-head`), which stops every run on the Ray cluster: do it
    when none is running.

The pods for GPUs rented elsewhere have images of their own, built in CI: see [remote providers](providers.md).
