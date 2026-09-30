"""The session tests share `session_scripts` with the server processes they start."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
