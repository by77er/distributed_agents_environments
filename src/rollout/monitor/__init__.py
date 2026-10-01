"""A monitor for runs: watch each run, and each model in it, as it happens (docs/core/monitor.md).

- `RunFeed`: hooks for a runner (`LocalRunner(hooks=[RunFeed(directory)])`) that write every run's events and model
  samples to a directory.
- `create_app(directory)`: a web page over such a directory: runs by group, and for each run every slot's turns: what
  the model was sent, what it thought, what it did and what came back. `python -m rollout.monitor DIRECTORY` serves it.
"""

from rollout.monitor.feed import FeedReader, RunFeed, plain

__all__ = ["FeedReader", "RunFeed", "plain"]
