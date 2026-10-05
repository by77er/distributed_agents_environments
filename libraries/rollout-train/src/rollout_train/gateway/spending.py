"""What runs spent on the turns of metered providers, and the most a run may spend.

A turn a hosted API sampled records what it cost (`TurnRecord.spend`, in its ledger record as `spend`). `Spending`
keeps each run's total as a gateway records its turns: what the run's earlier starts recorded is read from the ledger
once, before the gateway records the run's first turn that costs anything (`prepare`), and each turn it records is
added to it (`counted`). A run's own gateway (its driver's) records every turn of the run, so the total is the run's.

A cap (`cap`) bounds what some runs spend together (an eval and its parts, under its `limits.spend`): once their total
reaches the limit, its event is set, and the gateway samples no more metered turns for them (`over` says why).
"""

import asyncio
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field

from rollout_train.gateway.turns import TURNS
from rollout_train.ledger import Ledger

__all__ = ["Cap", "Spending"]


@dataclass
class Cap:
    """The most some runs may spend together, and the event set once they have."""

    runs: Collection[str]
    """The runs it bounds (a live collection: an eval's parts join it as they are made)."""
    limit: float
    reached: asyncio.Event = field(default_factory=asyncio.Event)


class Spending:
    """Each run's spend on metered turns, as a gateway records them, over what its earlier starts recorded."""

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        self._spent: dict[str, float] = {}
        self._loading: dict[str, asyncio.Lock] = {}
        self._caps: list[Cap] = []

    async def prepare(self, run: str) -> None:
        """Read what the run's earlier starts spent, once, before this gateway records a turn of it that costs."""
        if run in self._spent:
            return
        async with self._loading.setdefault(run, asyncio.Lock()):
            if run in self._spent:
                return
            prefix = f"runs/{run}/{TURNS}/"
            total = 0.0
            for name in [each for each in await self.ledger.tables() if each.startswith(prefix)]:
                for entry in (await self.ledger.read(name)).values():
                    spend = entry.get("spend") if isinstance(entry, dict) else None
                    if isinstance(spend, int | float) and not isinstance(spend, bool):
                        total += float(spend)
            self._spent[run] = total

    async def counted(self, run: str, dollars: float) -> None:
        """Add a turn this gateway recorded (after `prepare`), and set the event of each cap it brings to its limit."""
        await self.prepare(run)
        self._spent[run] += dollars
        for cap in self._caps:
            if run in cap.runs and not cap.reached.is_set() and await self.total(cap.runs) >= cap.limit:
                cap.reached.set()

    async def total(self, runs: Iterable[str]) -> float:
        """What some runs spent together."""
        listed = list(runs)
        for run in listed:
            await self.prepare(run)
        return sum(self._spent[run] for run in listed)

    async def spent(self, run: str) -> float:
        return await self.total([run])

    def cap(self, runs: Collection[str], limit: float) -> Cap:
        """Bound what `runs` spend together by `limit` dollars: its event is set once they reach it."""
        made = Cap(runs, limit)
        self._caps.append(made)
        return made

    async def over(self, run: str) -> str | None:
        """Why a run may spend no more, if it may not: a cap on it is reached."""
        for cap in self._caps:
            if run in cap.runs:
                total = await self.total(cap.runs)
                if total >= cap.limit:
                    cap.reached.set()
                    return f"the run spent ${total:.2f}, which reaches its limits.spend ${cap.limit:g}"
        return None
