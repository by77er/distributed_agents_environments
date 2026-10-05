"""Where a run keeps what, under its directory: the monitor and the report read there.

A run's driver keeps its feed under its directory on its node (`rollout_train.jobs.run_directory`). A run's directory
from before keeps its ledger and its blobs there too, unless its start names other places."""

RUN = "run.json"
"""Which run the directory is: its id in the registry (`rollout_train.registry`)."""
LEDGER = "ledger"
"""The run's tables, the checkpoints and the fences, in files (unless the run's start names another place)."""
FEED = "feed"
"""The monitor's feed."""
BLOBS = "blobs"
"""Episodes' trajectories and checkpoints' files (unless the run's start names a blob store)."""
