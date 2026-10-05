# Ingress, TLS and sign-in

Who can reach what: the host names the chart's ingresses answer on, serving them over TLS, putting sign-in in front
of the monitor, and keeping the stores private. This page is for whoever opens a deployment to other people.

**Read first:** [Install the Helm chart](helm.md#ingresses). **Next:** [Remote providers](providers.md).

## What is reached from where

| Service | Reached by | Exposed through an Ingress |
|---|---|---|
| The monitor | people, in a browser | yes, behind sign-in |
| The gateway | runs in the cluster, at its Service; harnesses outside the cluster | yes, if harnesses outside the cluster use it |
| Ray's dashboard | operators | optional; it asks for the Ray cluster's token |
| Postgres (the [ledger](../libraries/rollout-train/checkpoints.md#the-ledger)) | the platform's pods only | **never** |
| The S3 store | the platform's pods only | **never** |

## Host names

The chart's Ingresses answer on `gateway.host`, each `monitors.NAME.host`, and `ingress.rayHost`. The defaults end in
`.localhost`, which a browser sends to its own machine, so they work only on the node itself. Set names your DNS
points at the ingress controller:

```yaml title="rollout-values.yaml (in part)"
gateway:
  host: gateway.example.com
monitors:
  main:
    host: monitor.example.com
ingress:
  className: traefik
  rayHost: ray.example.com
```

## TLS

The chart's Ingresses name hosts and no certificates, so TLS is the ingress controller's job. Either terminate TLS in
a load balancer in front of the controller, or give the controller a default certificate for your hosts (a wildcard
certificate covers them all). With Traefik, that is a TLSStore named `default`:

```yaml
apiVersion: traefik.io/v1alpha1
kind: TLSStore
metadata:
  name: default
  namespace: rollout
spec:
  defaultCertificate:
    secretName: rollout-tls   # a kubernetes.io/tls Secret, made by cert-manager or by hand
```

The gateway trusts the `X-Forwarded-*` headers of any proxy in front of it (the chart starts it with `--proxied '*'`),
so reach its pods only through the ingress controller.

## Sign-in in front of the monitor

The monitor has no sign-in of its own, and whoever reaches it can import code from git and start runs on the cluster.
Put an authenticating proxy in front of it before anyone else can reach it: the ingress controller's basic
authentication, or a single sign-on proxy such as oauth2-proxy.

With Traefik's basic authentication:

1. Make a file of users with `htpasswd` (from Apache's tools), and keep it in a Secret under the key `users`:

    ```bash
    htpasswd -nbB admin "$(openssl rand -hex 12)" > users
    kubectl -n rollout create secret generic monitor-users --from-file=users=users
    rm users
    ```

2. Make a Middleware that asks for them:

    ```yaml
    apiVersion: traefik.io/v1alpha1
    kind: Middleware
    metadata:
      name: monitor-auth
      namespace: rollout
    spec:
      basicAuth:
        secret: monitor-users
    ```

3. Attach it to each monitor's Ingress, and to Ray's dashboard if you expose it:

    ```bash
    kubectl -n rollout annotate ingress monitor-main \
      traefik.ingress.kubernetes.io/router.middlewares=rollout-monitor-auth@kubernetescrd
    ```

Helm keeps an annotation it does not manage across upgrades. Check that the page now asks for a password, from a
machine that is not the node.

## The gateway's keys

The gateway samples only for a request that carries a key signed with one of its secrets
([keys](../libraries/rollout-train/gateway.md#keys)), and each key is limited to one session of one run and expires.
Its Ingress is safe to expose to harnesses outside the cluster for that reason; leave it unexposed if nothing outside
uses it. Rotate a secret by putting a new one first in the Secret `gateway-keys`, restarting the gateway, and removing
the old one once the keys it signed have expired.

## Keep the stores private

Postgres and the S3 store have ClusterIP Services only. Never give them an Ingress, a NodePort or a LoadBalancer:
whoever reaches the ledger's database can rewrite any run's record. To reach them from your machine for a copy or a
backup, use `kubectl port-forward`, which goes through the Kubernetes API with your credentials. Processes outside the
cluster, such as GPU pods elsewhere, never connect to the database: they are given scoped access through the ledger
service the [runtime design](../research/runtime-design.md#decisions-after-review) describes.
