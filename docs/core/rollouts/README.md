# Rollouts

Status: **Working** (2026-10-02) · Code: `rollout.rollouts` · See [episodes](../trajectories/README.md), [training](../training.md)

Runs a task's rows at scale and delivers the finished runs as one stream of episodes per job. The caller, typically
whoever trains, decides what to run, how often and how to group it: a job knows no algorithm.

```python
job = await jobs.start(program=program, binding=binding, in_flight=5, name="train")
ticket = await job.run({"word": "yes"}, labels={"group": "0001"}, count=4)
episodes = await ticket.episodes()                 # the ticket's four, once all have ended
async for episode in job.episodes(cursor):         # or the whole stream, while runs are still going
    ...
await job.acknowledge(episode.cursor)
version = await job.publish("policy", "step-3", "/adapters/step-3")
```

| | |
|---|---|
| `Jobs.start(program, binding, in_flight, name)` | A job that runs `program` (each ticket's row as its parameters) under `binding`, at most `in_flight` runs at a time. A named job finds its log again in a later process. |
| `Job.run(parameters, labels, count)` | Queues `count` runs of one row. They start together, when there is room for all of them; the labels go to the runs and their episodes. |
| `Job.episodes(cursor)` | Every episode after a cursor, then new ones as runs end. Episodes are numbered from 1 in the order they ended. |
| `Ticket.episodes()` | A ticket's episodes, once every one of its runs has ended. |
| `Job.acknowledge(cursor)` | The caller has consumed everything through the cursor: it need not be kept. |
| `Job.publish(channel, adapter, path)` | Serves new weights on a channel; returns its new version. |
| `Job.note(kind, payload)` | Puts something of the caller's (an update's statistics) where whoever watches the job sees it. |
| `Job.status()` | Runs queued and running, episodes finished, the cursor acknowledged. |

## Guarantees

- **Every run is an episode**, whatever its outcome: completed, failed (the program raised, or could not start),
  cancelled. Counts stay exact, and a group is complete when its count is.
- **Admission.** A ticket's runs start together or not at all, while runs in flight plus the ticket's count fit
  `in_flight`. With `in_flight` one more than a group, the next group starts when one episode of the group before is
  still running. A `guard` given to `RolloutJobs` is called before runs are admitted and raises to refuse them.
- **Asynchronous training** is reading the stream while runs are in flight. Nothing waits for a batch: each sampled
  token carries the weights version it was sampled at, and the caller decides how stale it tolerates data.
- **Identical starts** for the runs of a ticket come from the row: they are given the same parameters.
- **A caller that stops** goes on from its cursor: with a `log` directory, unacknowledged episodes are kept on disk.

## Where it runs

`RolloutJobs(runner, recorder)` runs jobs on any `Runner`: one that runs programs in this process, or a durable one
over a database. `rollout.rollouts.service.create_app(jobs)` serves them over HTTP and `RolloutClient(url)` is the same
`Jobs` for a caller on another machine. The training loop is tested under both.

## Catalog

What an environment offers to be trained on (`rollout.rollouts.Catalog`): the program, its rows (easiest first), and
how one start of a row is drawn. See [three ways in](../../guide/perspectives.md).

## Watching

`JobHooks.on_job(event)` receives what a job did at the level its caller thinks at: `ticket`, `admitted`, `episode`,
`published`, and the caller's notes. The [monitor](../monitor.md)'s feed is one.
