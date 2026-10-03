# Rollouts

Code: `rollout_train.rollouts` · See [episodes](episodes.md), [training](training.md),
[API reference](../../guide/reference.md#rollout_trainrollouts)

A rollout job runs a program's rows, many at a time, and delivers the finished runs as one stream of episodes. The
caller, typically whoever trains, decides what to run, how often and how to group it: a job knows no algorithm.

```python
job = await jobs.start(program=program, binding=binding, in_flight=5, name="train")
ticket = await job.run({"word": "yes"}, labels={"group": "0001"}, count=4)
episodes = await ticket.episodes()                 # the ticket's four, once all have ended
async for episode in job.episodes(cursor):         # or the whole stream, while runs are still going
    ...
await job.acknowledge(episode.cursor)
await job.publish("policy", "swarm@3", "/versions/swarm@3/weights", 3)
```

[`Jobs`](../../guide/reference.md#jobs), [`Job`](../../guide/reference.md#job) and
[`Ticket`](../../guide/reference.md#ticket) are protocols. A training loop written against them holds jobs in its
own process or jobs served elsewhere, and cannot tell which.

| | In this process | Over HTTP |
|---|---|---|
| Jobs | `RolloutJobs(runner, recorder, log=..., blobs=...)`, over any [`Runner`](../rollout/README.md#runner) | `RolloutClient(url, blobs)` |
| Served by | | `rollout_train.rollouts.service.create_app(jobs)`, which a profile's `serve` starts ([deploying](../../guide/deploying.md)) |

## A job

- **A job** runs one program under one binding, at most `in_flight` runs at a time. Its id is its `name`, or `j_`
  and ten hexadecimal digits.
- **A named job is found again.** `RolloutJobs.start` under the name of a job that is still open closes that job
  first, and the new job goes on over its log.
- **A ticket** is `count` runs of one row: `job.run(parameters, labels=..., count=...)`. Each run is the job's
  program with the row as its parameters, so the runs of a ticket start alike. A ticket's id is `t_` and ten
  hexadecimal digits.
- **A ticket asked for with a `key`** is `t_KEY`, and asking again is asking for the same ticket: a caller that
  died after asking gets it back, with whatever episodes it has.
- **Labels** go to the runs and to their episodes. The job adds `job`, `ticket` and `episode` (the run's number
  within its ticket).
- **The log** numbers episodes from 1 in the order their runs ended. That number is the episode's `cursor`.

## Guarantees

- **Every run is an episode**, whatever its outcome: completed, failed (the program raised, or the run could not
  start), cancelled. Counts stay exact, and a ticket is complete when its count is.
- **Admission.** Tickets are admitted in the order they were queued. A ticket's runs start together or not at all,
  when runs in flight plus the ticket's count fit `in_flight`, or when nothing is in flight. With `in_flight` one
  more than a group, the next group starts when one episode of the group before is still running.
- **Refusal.** A `guard` given to `RolloutJobs` is called before a ticket is admitted and raises to refuse it. A
  ticket that was queued when its job closed is refused too. `ticket.episodes()` then raises
  [`Refused`](../../guide/reference.md#refused), in this process and over HTTP alike. `RolloutTicket.refused`
  holds the reason.
- **A ticket's episodes** are returned by `ticket.episodes()` once every one of its runs has ended, in the order
  they ended. `RolloutTicket.ready(seconds)` waits that long at most and says whether the ticket is over.
- **Asynchronous training** is reading the stream while runs are in flight. Nothing waits for a batch: each sampled
  token carries the weights version it was sampled at, and the caller decides how stale it tolerates data.
- **Acknowledging** a cursor says that everything through it has been consumed. The job then lets go of every
  ticket whose episodes are all acknowledged. Until then `job.ticket(id)` finds a ticket by its id, as
  `jobs.job(id)` finds a job: the HTTP service holds nothing else.
- **A caller that stops** goes on from its cursor: a job started again under its name knows how far its caller got.
- **The recorder forgets a run** once its episode is assembled: the episode holds everything training needs of it.
- **Closing** a job stops admission, cancels its runs in the runner, refuses every ticket that is not over to
  whoever waits on it, and ends every `episodes` stream.

`job.publish(channel, adapter, path, version)` serves new weights on a channel and returns the version they are
served as: the number the caller's policy gives them ([channels](channels.md#publishing-weights),
[policies](policies.md)). `job.status()` counts the runs queued and running, the episodes
finished and the cursor acknowledged.

## The log

A job given a `log` directory keeps every episode, from the moment its run ends.

| Kept | Where | What |
|---|---|---|
| A [`Record`](../../guide/reference.md#record) per episode | one line of `log/JOB/episodes.jsonl` | Everything about the episode but its trajectories' segments: labels, outcome, result, rewards, tokens sampled by slot. It names two blobs |
| The trajectories | a blob | Each slot's segments, spans and logprobs |
| The run's events | a blob | Its tool calls and their results, observations and rewards, as the runner recorded them |
| What was asked for | one line of `log/JOB/tickets.jsonl` per ticket | The row, the labels and the count, as they were asked |
| How far the caller got | `log/JOB/acknowledged` | The cursor acknowledged |

- **Nothing is deleted.** Acknowledging records the caller's cursor and frees the job's memory. `job.episodes(cursor)`
  reads the log from any cursor, trajectories and all, so earlier episodes can be trained on again.
- **Blobs** go to the [`Blobs`](../../guide/reference.md#blobs) store the jobs are given, or to files under
  `log/blobs`. They are JSON, compressed. How long they are kept is the store's business.
- **A span names its sample.** `Span.effect_id` is the effect the run's events know the sample by, so a trajectory can be
  joined to what the action it sampled did. `events_of(record, blobs)` reads the events.
- **A job started again runs what it still owes.** A ticket some of whose runs have no episode gets those runs
  again, from the same row. A run the job itself cut short by closing is in the log as a cancelled episode whose
  `detail` says so, and is not one of its ticket's episodes: it is one of the runs still owed.
- **Without a log** a job holds episodes in memory until they are acknowledged, and then they are gone.

## Over HTTP

`RolloutClient` polls. A read of a ticket or of the stream waits on the server for news, up to a `wait` in seconds,
and then answers with what there is. The routes are listed in `rollout_train.rollouts.service`. Episodes cross as
their records, and the client reads their trajectories from the blob store, which both sides share. Jobs served over HTTP
therefore need a log.

## Catalog

What an environment offers to be trained on is a [`Catalog`](../../guide/reference.md#catalog): the program, its
rows (easiest first), and how one start of a row is drawn. `Catalog`, `Row` and `binding_for` live in
`rollout.catalog`, in the core library, so an environment needs the harness and nothing above it.
`binding_for(catalog, channel, tools)` binds every model slot of the catalog's program to one channel
([three ways in](../../guide/perspectives.md#building-an-environment)).

## Watching

[`JobHooks.on_job(event)`](../../guide/reference.md#jobhooks) receives what a job did, at the level its caller
thinks at. Every event has `kind`, `job` and `at`.

| `kind` | When | Also carries |
|---|---|---|
| `ticket` | a ticket is queued | `ticket`, `labels`, `count` |
| `admitted` | a ticket's runs have started | `ticket`, `runs` |
| `episode` | a run ended | `cursor`, `ticket`, `run`, `labels`, `outcome`, `detail`, `reward`, `info`, and `sampled`: tokens sampled by slot |
| `published` | weights were published | `channel`, `adapter`, `version` |
| anything else | the caller's `job.note(kind, payload)` | the payload. The training loop notes `iteration`. |

The [monitor](monitor.md)'s feed is one such hook.
