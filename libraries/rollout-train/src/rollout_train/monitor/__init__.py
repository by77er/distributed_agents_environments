"""A monitor for runs: watch each run, and each model in it, as it happens (docs/libraries/rollout-train/monitor.md).

- `RunFeed`: hooks for a runner and for rollout jobs that write every run's events and model samples, and what the
  job did, to a directory.
- `System`: where a run stands, read from its directory: the groups in flight and their stages, the policies'
  versions, the jobs, the engines, the machine.
- `create_app(directory)`: a web page over a run's directory: the system, runs by group, and for each run every
  slot's turns: what the model was sent, what it thought, what it did and what came back. `rollout monitor
  DIRECTORY` serves it.
"""

from rollout_train.monitor.feed import FeedReader, RunFeed, plain
from rollout_train.monitor.system import System

__all__ = ["FeedReader", "RunFeed", "System", "plain"]
