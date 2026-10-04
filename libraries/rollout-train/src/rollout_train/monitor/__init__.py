"""A monitor for runs: watch each run, and each model in it, as it happens (docs/libraries/rollout-train/monitor.md).

- `RunFeed`: hooks for a runner and for rollout jobs that write every run's events and model samples, and what the
  job did, to a directory.
- `System`: a ledger read with every run in it: where each run stands and where its episodes are read (its
  directory on this machine, or the monitor on its own), its steps (each with the groups that went into it), each
  group (its episodes, its step, its outcome), each episode (what it reported, and its rollouts: one per agent, each
  a trajectory to train on), the policies as a graph (`rollout_train.monitor.lineage`) and statistics across the runs
  (`rollout_train.monitor.statistics`).
- `create_app(where)`: a web page over a ledger (or a run's directory and its ledger), in three pages: the runs,
  each organised the same way, down to each episode's rollouts, turn by turn; the policies; and statistics, with the
  machine. `rollout monitor WHERE` serves it.
"""

from rollout_train.monitor.feed import FeedReader, RunFeed, plain
from rollout_train.monitor.system import System

__all__ = ["FeedReader", "RunFeed", "System", "plain"]
