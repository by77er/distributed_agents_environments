"""The relay: delivers the coordination outbox through a runner.

It runs outside any run, so it can start runs and send messages freely. Each delivery uses the outbox entry's key as
its idempotency key, so a delivery repeated after a crash (sent, but not yet marked delivered) is deduplicated by the
runner. On start it delivers whatever a crash left pending.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from rollout.coordination.store import CoordinationStore, Delivery

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
