"""Evaluation scenarios: scripted users, programmatic checks, and reference facts for the judge."""

from collections.abc import Callable
from dataclasses import dataclass, field

from rollout.harness import Priority


@dataclass(frozen=True)
class Say:
    """A user message. `wait=False` sends it without waiting for the reply (the next turn may interrupt it)."""

    text: str
    conversation: str = "main"
    priority: Priority = Priority.NORMAL
    wait: bool = True
    reference: str | None = None
    """What a correct answer contains, for the judge. Turns without one are not judged."""


@dataclass(frozen=True)
class AwaitUnprompted:
    """Wait for a reply nobody asked for, such as a follow-up firing."""

    seconds: float
    conversation: str = "main"


type Turn = Say | AwaitUnprompted


@dataclass
class Observed:
    """What one scenario run produced, for checks."""

    replies: list[str | None] = field(default_factory=list[str | None])
    """One per turn, in order; None when no reply came."""
    tool_calls: list[str] = field(default_factory=list[str])
    notes: list[str] = field(default_factory=list[str])

    def reply(self, turn: int) -> str:
        return (self.replies[turn] or "").lower()


type Check = tuple[str, Callable[[Observed], bool]]


def mentions(turn: int, *all_of: str) -> Check:
    return (
        f"reply {turn + 1} mentions {', '.join(all_of)}",
        lambda o: all(text.lower() in o.reply(turn) for text in all_of),
    )


def mentions_any(turn: int, *any_of: str) -> Check:
    return (
        f"reply {turn + 1} mentions one of {', '.join(any_of)}",
        lambda o: any(t.lower() in o.reply(turn) for t in any_of),
    )


def avoids(turn: int, *none_of: str) -> Check:
    return (
        f"reply {turn + 1} avoids {', '.join(none_of)}",
        lambda o: not any(t.lower() in o.reply(turn) for t in none_of),
    )


def used(tool: str) -> Check:
    return (f"used {tool}", lambda o: tool in o.tool_calls)


def saved_note(*all_of: str) -> Check:
    return (
        f"saved a note with {', '.join(all_of)}",
        lambda o: any(all(text.lower() in note.lower() for text in all_of) for note in o.notes),
    )


def replied(turn: int) -> Check:
    return (f"reply {turn + 1} arrived", lambda o: o.replies[turn] is not None)


@dataclass(frozen=True)
class Scenario:
    name: str
    turns: list[Turn]
    checks: list[Check]


SCENARIOS = [
    Scenario(
        "locate a function",
        [Say("Where is the invoice total computed?", reference="invoice_total in src/tidepool/billing.py")],
        [mentions(0, "src/tidepool/billing.py", "invoice_total"), used("search")],
    ),
    Scenario(
        "config value and its history",
        [
            Say(
                "What is the session timeout, and when was it last changed and by whom?",
                reference="45 minutes (SESSION_TIMEOUT_MINUTES in src/tidepool/auth.py); raised from 30 to 45 on "
                "2026-05-20 by Kofi Mensah in 'Raise session timeout to 45 minutes'",
            )
        ],
        [mentions(0, "45", "kofi"), mentions_any(0, "2026-05-20", "may 20"), used("git_log")],
    ),
    Scenario(
        "something that does not exist",
        [
            Say(
                "How do I issue a refund with the billing module?",
                reference="The billing module has no refund function: only invoice_total and apply_discount exist. "
                "A good answer says so and does not invent an API.",
            )
        ],
        [avoids(0, "refund_invoice(", "issue_refund(", "def refund"), mentions_any(0, "no refund", "not", "doesn't")],
    ),
    Scenario(
        "follow-up questions",
        [
            Say("Which module handles authentication?", reference="src/tidepool/auth.py"),
            Say(
                "Who changed it most recently, and what did they change?",
                reference="Kofi Mensah raised the session timeout from 30 to 45 minutes on 2026-05-20",
            ),
        ],
        [mentions(0, "auth.py"), mentions(1, "kofi", "45")],
    ),
    Scenario(
        "memory across conversations",
        [
            Say("Please remember: the staging database is called kelp-staging-2.", conversation="first"),
            Say(
                "What is the staging database called?",
                conversation="second",
                reference="kelp-staging-2. The developer said so in an earlier conversation; it is in the assistant's "
                "saved notes, not in the repository, so citing the note is correct.",
            ),
        ],
        [saved_note("kelp-staging-2"), used("search_notes"), mentions(1, "kelp-staging-2")],
    ),
    Scenario(
        "a follow-up fires",
        [
            Say("In about five seconds, remind me to rotate the API keys."),
            AwaitUnprompted(seconds=60),
        ],
        [used("schedule_follow_up"), replied(1), mentions_any(1, "rotate", "api key")],
    ),
    Scenario(
        "a high-priority message interrupts",
        [
            Say("Write a detailed tour of every file in the repository, with a summary of each commit.", wait=False),
            Say(
                "Stop that. Just tell me the default currency.",
                priority=Priority.HIGH,
                reference="EUR, set in src/tidepool/config.py",
            ),
        ],
        [mentions(1, "eur")],
    ),
]
