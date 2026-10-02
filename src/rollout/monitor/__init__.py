"""A monitor for runs: watch each run, and each model in it, as it happens (docs/core/monitor.md).

- `RunFeed`: hooks for a runner and for rollout jobs that write every run's events and model samples, and what the
  job did, to a directory.
- `create_app(directory)`: a web page over such a directory: the job, runs by group, and for each run every slot's
  turns: what the model was sent, what it thought, what it did and what came back. `rollout monitor DIRECTORY`
  serves it.
"""

from rollout.monitor.feed import FeedReader, RunFeed, plain

__all__ = ["FeedReader", "RunFeed", "plain"]
