"""The relay: delivers the coordination outbox through a runner.

It runs outside any run, so it can start runs and send messages freely. Each delivery uses the outbox entry's key as
its idempotency key, so a delivery repeated after a crash (sent, but not yet marked delivered) is deduplicated by the
runner. On start it delivers whatever a crash left pending. Several processes sharing a database may each run a relay:
one delivers, and another takes over when it stops.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from agent_sessions.coordination.store import CoordinationStore, Delivery

type Deliver = Callable[[Delivery], Awaitable[None]]

logger = logging.getLogger(__name__)


class Relay:
    def __init__(self, store: CoordinationStore, deliver: Deliver, *, retry_seconds: float = 2.0) -> None:
        self.store = store
        self.deliver = deliver
        self.retry_seconds = retry_seconds
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        store.on_outbox(self._wake.set)

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="coordination relay")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def drain(self) -> None:
        """Deliver everything pending now."""
        for delivery in self.store.pending():
            await self.deliver(delivery)
            self.store.delivered(delivery.id)

    async def _run(self) -> None:
        # One relay delivers at a time, among every process sharing the store's database; the others wait to
        # take over if it stops.
        async with self.store.database.lock("coordination relay", poll_seconds=1.0):
            await self._relay()

    async def _relay(self) -> None:
        while True:
            self._wake.clear()
            try:
                await self.drain()
            except Exception:  # a failed delivery stays pending and is retried
                logger.exception("delivering the coordination outbox failed; retrying")
                await asyncio.sleep(self.retry_seconds)
                continue
            try:
                async with asyncio.timeout(self.retry_seconds):
                    await self._wake.wait()
            except TimeoutError:
                pass
