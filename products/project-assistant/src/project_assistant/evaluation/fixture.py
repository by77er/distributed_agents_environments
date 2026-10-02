"""The evaluation repository: an invented codebase with a fixed git history, so every question has a known answer."""

import os
import subprocess
from pathlib import Path

FILES_BY_COMMIT: list[tuple[str, str, str, dict[str, str]]] = [
    (
        "Ana Souza",
        "2026-03-02T10:00:00+00:00",
        "Initial billing service",
        {
            "README.md": "# Tidepool\n\nInvoicing for marine research stations.\n",
            "src/tidepool/__init__.py": "",
            "src/tidepool/billing.py": (
                "TAX_RATE = 0.0825\n\n\n"
                "def invoice_total(line_items: list[float], tax_rate: float = TAX_RATE) -> float:\n"
                '    """Sum the line items and add tax."""\n'
                "    return round(sum(line_items) * (1 + tax_rate), 2)\n\n\n"
                "def apply_discount(total: float, percent: float) -> float:\n"
                "    return round(total * (1 - percent / 100), 2)\n"
            ),
        },
    ),
    (
        "Kofi Mensah",
        "2026-04-11T15:30:00+00:00",
        "Add session tokens to auth",
        {
            "src/tidepool/auth.py": (
                "SESSION_TIMEOUT_MINUTES = 30\n\n\n"
                "def verify_token(token: str) -> bool:\n"
                '    return token.startswith("tp_") and len(token) == 35\n'
            ),
        },
    ),
    (
        "Kofi Mensah",
        "2026-05-20T09:15:00+00:00",
        "Raise session timeout to 45 minutes",
        {
            "src/tidepool/auth.py": (
                "SESSION_TIMEOUT_MINUTES = 45\n\n\n"
                "def verify_token(token: str) -> bool:\n"
                '    return token.startswith("tp_") and len(token) == 35\n'
            ),
        },
    ),
    (
        "Ana Souza",
        "2026-06-01T12:00:00+00:00",
        "Default currency to EUR",
        {
            "src/tidepool/config.py": 'DEFAULTS = {"currency": "EUR", "retry_attempts": 4}\n',
            "CHANGELOG.md": (
                "# Changelog\n\n- 2026-06-01: default currency is EUR\n- 2026-05-20: sessions last 45 minutes\n"
            ),
        },
    ),
]


def create_fixture_repository(root: Path) -> Path:
    """Create the tidepool repository at `root` (which must not exist) and return it."""
    root.mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    for author, date, message, files in FILES_BY_COMMIT:
        for path, content in files.items():
            file = root / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(content)
        _git(root, "add", ".")
        email = author.lower().replace(" ", ".") + "@tidepool.example"
        environment = {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
        _git(root, "-c", f"user.name={author}", "-c", f"user.email={email}", "commit", "-qm", message, env=environment)
    return root


def _git(root: Path, *arguments: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(["git", "-C", str(root), *arguments], check=True, env={**os.environ, **(env or {})})


def ground_truth(root: Path) -> str:
    """The repository's files, with numbered lines, and its history: everything a judge needs to check a reply."""
    listing = subprocess.run(["git", "-C", str(root), "ls-files"], check=True, capture_output=True, text=True).stdout
    sections: list[str] = []
    for path in listing.split():
        lines = (root / path).read_text().splitlines()
        numbered = "\n".join(f"{number:>3}  {line}" for number, line in enumerate(lines, start=1))
        sections.append(f"--- {path}\n{numbered}")
    history = subprocess.run(
        ["git", "-C", str(root), "log", "--stat", "--format=commit %h %ad %an: %s", "--date=short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return "\n\n".join(sections) + "\n\n--- git log\n" + history
