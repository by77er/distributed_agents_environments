"""A monitor for runs: watch each run, and each model in it, as it happens (docs/libraries/rollout-train/monitor.md).

- `RunFeed`: hooks for a runner and for rollout jobs that write every run's events and model samples, and what the
  job did, to a directory.
- `System`: a run's directory read as the run is laid out: where it stands, its steps (each with the groups that went
  into it), each group (its episodes, its step, its outcome) and each episode (what it reported, and its rollouts: one
  per agent, each a trajectory to train on).
- `create_app(directory)`: a web page over a run's directory, organised the same way: the run, its steps, their
  groups, the groups' episodes and each episode's rollouts, turn by turn, beside the policy and the machine.
  `rollout monitor DIRECTORY` serves it.
"""

from rollout_train.monitor.feed import FeedReader, RunFeed, plain
from rollout_train.monitor.system import System

__all__ = ["FeedReader", "RunFeed", "System", "plain"]
