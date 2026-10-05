# Operate the platform

This section is for people who keep a deployed platform running: watching runs, changing them while they go, keeping
checkpoints, checking the cluster's configuration, and backing up what it records.

**Read first:** [Deploy the platform](../deploy/README.md). **Next:** [The
monitor](../libraries/rollout-train/monitor.md).

## Watch runs

[The monitor](../libraries/rollout-train/monitor.md) is a live page over a ledger and every run in it: steps, groups,
every episode's transcript, checkpoints, evals, environments, statistics, and the machines doing the work. It asks for
its token ([signing in](../libraries/rollout-train/monitor.md#signing-in)). On Kubernetes, open it through a
port-forward, and sign in once with the token in the Secret `monitor-token`
([opening the monitor](../deploy/access.md#opening-the-monitor)):

```bash
kubectl -n rollout port-forward svc/monitor-main 8765:8765 &
token=$(kubectl -n rollout get secret monitor-token -o jsonpath='{.data.ROLLOUT_MONITOR_TOKEN}' | base64 -d)
echo "http://localhost:8765/login?token=$token"
```

- [The machines](../libraries/rollout-train/monitor.md#the-machines) shows every runner, engine host and gateway
  replica by its heartbeat ([heartbeats](../libraries/rollout-train/rollouts.md#heartbeats)): one silent for 90
  seconds is taken to be gone.
- [Statistics](../libraries/rollout-train/monitor.md#statistics) charts a run's rewards and step metrics.

## Change a run while it goes

- **Pause, resume or stop it:** [pausing, resuming and
  stopping](../libraries/rollout-train/monitor.md#pausing-resuming-and-stopping) from the page, and [pausing and
  resuming](../libraries/rollout-train/training.md#pausing-and-resuming) for what the loop does.
- **Change its settings:** [changing a running run's
  settings](../libraries/rollout-train/training.md#changing-a-running-runs-settings) says which settings take effect
  from the next step, such as `max_lag` and `groups_per_step`.
- **If a process dies:** [dying and starting again](../libraries/rollout-train/training.md#dying-and-starting-again)
  says how a run goes on from the ledger.

## Keep and name checkpoints

- [Bookmarks](../libraries/rollout-train/checkpoints.md#bookmarks) name a checkpoint; a run can carry one forward.
- [The checkpoint commands](../libraries/rollout-train/checkpoints.md#the-command-line) list checkpoints, rename runs
  and move bookmarks.
- Saves thin out with age, and a checkpoint a bookmark names is kept
  ([checkpoints](../libraries/rollout-train/checkpoints.md#checkpoints)).

## Check the configuration

`rollout cluster check` reads the cluster config and says which of its secrets and projects do not resolve on the
node it runs on. On Kubernetes, run it in a platform pod:

```bash
kubectl -n rollout exec deploy/gateway -- rollout cluster check
```

[The cluster config](../guide/cluster.md) describes every field.

## Back up and recover

- [Volumes and backups](../deploy/backups.md): what to back up (the ledger, the blob store and the state volume) and
  how.
- [Troubleshooting](../deploy/troubleshooting.md): problems seen in real deployments, and their fixes.
- [Ledger guarantees](../research/ledger-guarantees.md): what the ledger promises when processes die or race, each
  promise with the test that holds it.
