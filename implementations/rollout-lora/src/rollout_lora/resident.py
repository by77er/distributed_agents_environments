"""A trainer's processes on several GPUs, kept between steps: `Workers` starts them under torchrun (one per GPU:
`rollout_lora.workers`), hands each step to all of them, and ends them.

The processes hold the sharded policy and optimizer from one step to the next, so a step from the checkpoint the last
one made loads nothing (`rollout_train.trainer.Resident`; each step's state names what they hold, `HELD`). They are
started at the first step, and again after one failed: a failure in one process leaves the others waiting at a
collective, so a step that fails ends them all, and raises `StepFailed` with the failing process's traceback. They end
when the trainer is closed (`close`: on a training pod, when its lease is released or another run takes it), when the
trainer's process ends (each process ends when its connection to the trainer closes, or when its parent does), and when
a step is cancelled.

The processes reach the trainer at a local address with a key of its own (`multiprocessing.connection`), and are sent
each step's settings with it, so a setting changed between steps (`Changeable`) reaches every process.
"""

import asyncio
import contextlib
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from multiprocessing.connection import Connection, Listener, wait
from pathlib import Path
from typing import Any, cast

from rollout.processes import end_with_parent
from rollout_lora.settings import LoraSettings
from rollout_train.trainer import Files, Item, StepFailed

__all__ = ["START_TIMEOUT", "Workers", "visible_gpus"]

START_TIMEOUT = 600.0
"""Seconds the processes may take to start and join each other."""
STOPPING = 10.0
"""Seconds the processes have to end once told to stop, before torchrun is ended."""


def visible_gpus(environ: Mapping[str, str] | None = None) -> int:
    """The GPUs this process is given: those `CUDA_VISIBLE_DEVICES` names (Ray names an actor's), else the machine's."""
    said = (os.environ if environ is None else environ).get("CUDA_VISIBLE_DEVICES")
    if said is not None:
        return len([each for each in said.split(",") if each.strip() and not each.strip().startswith("-")])
    try:
        import torch
    except ImportError:
        return 0
    return torch.cuda.device_count()


class Workers:
    """`count` processes stepping a trainer of `checkpoint` (`weights`: `lora` or `full`) with `settings` (the trainer
    sets them anew between steps), on `device` (`cuda`, a GPU each; `cpu`, on gloo)."""

    def __init__(
        self, checkpoint: str, settings: LoraSettings, weights: str, count: int, *, device: str = "cuda"
    ) -> None:
        if count < 1:
            raise ValueError("a trainer steps in one process at least")
        self.checkpoint = checkpoint
        self.settings = settings
        self.weights = weights
        self.count = count
        self.device = device
        self.holding: str | None = None
        """What the processes hold, by the name the last step gave it (none: nothing, or no processes)."""
        self._process: subprocess.Popen[bytes] | None = None
        self._connections: list[Connection] = []
        self._lock = asyncio.Lock()

    async def step(self, segments: Sequence[Item], *, seed: int, parent: Files | None, into: Path) -> dict[str, float]:
        """Train one step in every process, from `parent` (what they hold, where it is the checkpoint they made last),
        leaving the weights in `into/weights` and the state in `into/state`; the step's metrics."""
        async with self._lock:
            try:
                if self._process is None or self._process.poll() is not None:
                    listener = self._spawn()  # (from this thread, which lasts: torchrun ends when its starter does)
                    await asyncio.to_thread(self._join, listener)
                return await asyncio.to_thread(self._step, list(segments), seed, parent, into)
            except asyncio.CancelledError:  # (whoever waited is gone: the processes are not left stepping for nobody)
                self.close()
                raise

    def _step(self, segments: list[Item], seed: int, parent: Files | None, into: Path) -> dict[str, float]:
        from rollout_lora.workers import Asked  # (here: the package does not import what torchrun runs)

        name = f"{into.name}:{secrets.token_hex(8)}"
        asked = Asked(self.checkpoint, self.settings, self.weights, segments, seed, parent, into, name)
        self.holding = None  # (until they say they made it)
        try:
            for connection in self._connections:
                connection.send(("step", asked))
        except OSError as error:
            self.close()
            raise StepFailed(f"the trainer's processes could not be reached: {error}") from None
        answers = self._answers()
        self.holding = name
        return answers[0]

    def _answers(self) -> list[dict[str, float]]:
        """Every process's answer, by rank; `StepFailed` (the processes ended) when one fails or ends."""
        pending = dict(enumerate(self._connections))
        found: dict[int, dict[str, float]] = {}
        while pending:
            ready = cast(list[Connection], wait(list(pending.values()), timeout=1.0))
            for connection in ready:
                rank = next(index for index, each in pending.items() if each is connection)
                try:
                    kind, payload = cast(tuple[str, Any], connection.recv())
                except (EOFError, OSError):
                    kind, payload = "error", f"process {rank} ended without an answer"
                if kind == "error":
                    code = self._ended()
                    raise StepFailed(f"the trainer's process {rank} failed (torchrun exited {code}):\n{payload}")
                found[rank] = cast(dict[str, float], payload)
                del pending[rank]
            if not ready and self._process is not None and self._process.poll() is not None:
                code = self._ended()
                raise StepFailed(f"the trainer's processes exited without an answer (torchrun exited {code})")
        return [found[rank] for rank in range(self.count)]

    def _ended(self) -> int | None:
        code = self._process.poll() if self._process is not None else None
        self.close()
        return code

    def _spawn(self) -> Listener:
        """Start the processes under torchrun (ended when the thread that starts them ends, as Linux's parent-death
        signal is); where they connect."""
        self.close()
        key = secrets.token_bytes(32)
        listener = Listener(("127.0.0.1", 0), authkey=key)
        host, port = listener.address
        environ = {**os.environ, "ROLLOUT_WORKERS": f"{host}:{port}", "ROLLOUT_WORKERS_KEY": key.hex(),
                   "ROLLOUT_WORKERS_DEVICE": self.device}  # fmt: skip
        environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        environ.setdefault("OMP_NUM_THREADS", "1")
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={self.count}",
                   "-m", "rollout_lora.workers"]  # fmt: skip
        self._process = subprocess.Popen(command, env=environ, preexec_fn=end_with_parent, start_new_session=True)
        return listener

    def _join(self, listener: Listener) -> None:
        """Wait until each process has connected and joined the others."""
        assert self._process is not None
        accepted: list[Connection] = []
        failed: list[BaseException] = []

        def accept() -> None:
            try:
                while len(accepted) < self.count:
                    accepted.append(listener.accept())
            except BaseException as error:  # (the listener closed: given up)
                failed.append(error)

        accepting = threading.Thread(target=accept, name="trainer-workers", daemon=True)
        accepting.start()
        given_up = time.monotonic() + START_TIMEOUT
        while accepting.is_alive() and self._process.poll() is None and time.monotonic() < given_up:
            accepting.join(timeout=0.5)
        listener.close()
        if len(accepted) < self.count:
            for each in accepted:
                each.close()
            code = self._ended()
            raise StepFailed(f"the trainer's processes did not start (torchrun exited {code}): {failed[:1]}")
        ranks: dict[int, Connection] = {}
        for connection in accepted:
            try:
                kind, payload = connection.recv() if connection.poll(START_TIMEOUT) else ("error", "no answer")
            except (EOFError, OSError):
                kind, payload = "error", "it ended"
            if kind != "ready":
                self._connections = accepted
                code = self._ended()
                raise StepFailed(f"a trainer's process did not start (torchrun exited {code}):\n{payload}")
            ranks[int(payload)] = connection
        if sorted(ranks) != list(range(self.count)):
            self._connections = accepted
            code = self._ended()
            raise StepFailed(f"the trainer's processes did not all join (torchrun exited {code})")
        self._connections = [ranks[rank] for rank in range(self.count)]

    def close(self) -> None:
        """End the processes (and what they hold)."""
        self.holding = None
        for connection in self._connections:
            with contextlib.suppress(OSError):
                connection.send(("stop", None))
            with contextlib.suppress(OSError):
                connection.close()
        self._connections = []
        process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        with contextlib.suppress(subprocess.TimeoutExpired):  # (told to stop, they end, and torchrun with them)
            process.wait(timeout=STOPPING)
        if process.poll() is not None:
            return
        _signal(process, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _signal(process, signal.SIGKILL)
            process.wait(timeout=30)

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


def _signal(process: "subprocess.Popen[Any]", number: int) -> None:
    """Signal torchrun's process group: its agent, and the processes it started."""
    with contextlib.suppress(OSError):
        os.killpg(process.pid, number)
