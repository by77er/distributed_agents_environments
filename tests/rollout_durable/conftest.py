"""Multi-runner tests share `scenarios` and `cluster` with the runner processes they start (see peer.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
