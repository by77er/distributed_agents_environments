"""A trainer's processes: `Workers` starts them under torchrun (one per GPU: `rollout_lora.workers`), hands each step
to all of them, and ends them.

Kept between steps (a trainer with its GPUs to itself), the processes hold the policy and optimizer from one step to
the next (on several GPUs, each its shard), so a step from the checkpoint the last one made loads nothing
(`rollout_train.trainer.Resident`; each step's state names what they hold, `HELD`). They are started at the first
step, and again after one failed: a failure in one process leaves the others waiting at a collective, so a step that
fails ends them all, and raises `StepFailed` with the failing process's traceback. They end when the trainer is closed
(`close`: on a training pod, when its lease is released or another run takes it), when the trainer's process ends
(each process ends when its connection to the trainer closes, or when its parent does), and when a step is cancelled.

Beside an engine on the same GPU (`kept = False`), the processes end after each step, which gives the engine back the
GPU's memory and the machine's: a trainer parked in system memory between steps, next to a sleeping engine's offloaded
weights, can exhaust a small machine. An engine's client libraries can also change how transformers builds models in
the process that uses them (vLLM swaps in its own configuration classes), which processes of their own avoid.

A step whose parent's state left its full state out, which the processes do not hold, is refused before they are
asked (`rollout_lora.workers.refusal`). The processes reach the trainer at a local address with a key of its own
(`multiprocessing.connection`), and are sent each step's settings with it, so a setting changed between steps
(`Changeable`) reaches every process.

Told where to keep the full state (`keep`, a blob store's location), the processes keep each step's after answering
it (`rollout_lora.workers`): `keeping` says which steps' state they are keeping, and `kept_state` waits for what every
process kept of one, merged, or raises `StateLost` where one could not keep its share or the processes ended first.
Whatever reads from the processes (a step waiting for its answers, `kept_state` waiting for a state) reads every
message that arrives, so each is noted whoever reads it: how far the step being taken has got among them (`progress`,
which `watcher` is told too, from the thread that read it). Closing waits for the states being kept, for up to
`KEEP_PATIENCE` seconds, before it ends the processes.
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
from collections.abc import Callable, Mapping, Sequence
from multiprocessing.connection import Connection, Listener, wait
from pathlib import Path
from typing import Any, cast

from rollout.processes import end_with_parent
from rollout_lora.settings import LoraSettings
from rollout_train.trainer import Files, Item, Progress, StateLost, StepFailed

__all__ = ["KEEP_PATIENCE", "START_TIMEOUT", "Workers", "visible_gpus"]

START_TIMEOUT = 600.0
"""Seconds the processes may take to start and join each other."""
STOPPING = 10.0
"""Seconds the processes have to end once told to stop, before torchrun is ended."""
KEEP_PATIENCE = 1800.0
"""Seconds closing waits for the full states the processes are still keeping."""


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
    sets them anew between steps), on `device` (`cuda`, a GPU each; `cpu`, on gloo); `kept` between steps, or ended
    after each."""

    def __init__(
        self,
        checkpoint: str,
        settings: LoraSettings,
        weights: str,
        count: int,
        *,
        device: str = "cuda",
        kept: bool = True,
    ) -> None:
        if count < 1:
            raise ValueError("a trainer steps in one process at least")
        self.checkpoint = checkpoint
        self.settings = settings
        self.weights = weights
        self.count = count
        self.device = device
        self.kept = kept
        self.keep: Mapping[str, Any] | None = None
        """Where the processes keep each step's full state after answering it (a blob store's location); none: they
        write it into the step's state before answering."""
        self._holding: str | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._connections: list[Connection] = []
        self._lock = asyncio.Lock()
        self._reading = threading.Lock()
        self._answered: dict[int, tuple[str, Any]] = {}
        """What each process answered the step being taken, by rank."""
        self._ended_ranks: set[int] = set()
        self._keeping: dict[str, int] = {}
        """The steps whose full state the processes keep, by their directories' names: how many processes keep it."""
        self._reports: dict[str, dict[int, dict[str, Any]]] = {}
        """What each process said it kept of a step's full state (or why it could not), by rank."""
        self.progress: Progress | None = None
        """How far the step being taken has got, as the processes last said (none between steps)."""
        self.watcher: Callable[[Progress], None] | None = None
        """Told each `progress` as it arrives."""

    @property
    def holding(self) -> str | None:
        """What the processes hold, by the name the last step gave it (none: nothing, or no processes running)."""
        if self._process is None or self._process.poll() is not None:
            return None
        return self._holding

    async def step(self, segments: Sequence[Item], *, seed: int, parent: Files | None, into: Path) -> dict[str, float]:
        """Train one step in every process, from `parent` (what they hold, where it is the checkpoint they made last),
        leaving the weights in `into/weights` and the state in `into/state`; the step's metrics."""
        from rollout_lora.workers import refusal  # (here: the package does not import what torchrun runs)

        async with self._lock:
            refused = refusal(parent, self.holding, self.settings.state_every)
            if refused is not None:
                raise StepFailed(refused)
            try:
                if self._process is None or self._process.poll() is not None:
                    listener = self._spawn()  # (from this thread, which lasts: torchrun ends when its starter does)
                    await asyncio.to_thread(self._join, listener)
                return await asyncio.to_thread(self._step, list(segments), seed, parent, into)
            except asyncio.CancelledError:  # (whoever waited is gone: the processes are not left stepping for nobody)
                self.close()
                raise
            finally:
                if not self.kept:  # (their memory given back to the engine beside them)
                    await asyncio.to_thread(self.close)

    def _step(self, segments: list[Item], seed: int, parent: Files | None, into: Path) -> dict[str, float]:
        from rollout_lora.workers import Asked

        name = f"{into.name}:{secrets.token_hex(8)}" if self.kept else None
        every = self.settings.state_every if self.kept else 1
        asked = Asked(self.checkpoint, self.settings, self.weights, segments, seed, parent, into, name, every,
                      keep=self.keep)  # fmt: skip
        self._holding = None  # (until they say they made it)
        self._answered = {}
        self.progress = None
        try:
            for connection in self._connections:
                connection.send(("step", asked))
        except OSError as error:
            self.close()
            raise StepFailed(f"the trainer's processes could not be reached: {error}") from None
        try:
            answers = self._answers()
        finally:
            self.progress = None
        self._holding = name
        if self.keep is not None and answers[0].get("full_state") == 1.0:
            self._keeping[into.name] = self.count
        return answers[0]

    def _answers(self) -> list[dict[str, float]]:
        """Every process's answer, by rank; `StepFailed` (the processes ended) when one fails or ends."""
        while True:
            for rank, (kind, payload) in sorted(self._answered.items()):
                if kind == "error":
                    code = self._ended()
                    raise StepFailed(f"the trainer's process {rank} failed (torchrun exited {code}):\n{payload}")
            if len(self._answered) == self.count:
                return [cast(dict[str, float], self._answered[rank][1]) for rank in range(self.count)]
            if not self._pump(1.0) and self._process is not None and self._process.poll() is not None:
                code = self._ended()
                raise StepFailed(f"the trainer's processes exited without an answer (torchrun exited {code})")

    def _pump(self, timeout: float) -> bool:
        """Read what the processes send within `timeout` seconds: their answers to the step being taken, and what each
        kept of a step's full state; whether anything arrived. A process whose connection ends answers that it
        ended."""
        with self._reading:
            connections = {rank: each for rank, each in enumerate(self._connections) if rank not in self._ended_ranks}
            if not connections:
                time.sleep(min(timeout, 0.05))
                return False
            ready = cast(list[Connection], wait(list(connections.values()), timeout=timeout))
            for connection in ready:
                rank = next(index for index, each in connections.items() if each is connection)
                message: tuple[Any, ...]
                try:
                    message = cast(tuple[Any, ...], connection.recv())
                except (EOFError, OSError):
                    self._ended_ranks.add(rank)
                    message = ("error", f"process {rank} ended without an answer")
                if message[0] == "kept":
                    self._reports.setdefault(str(message[1]), {})[rank] = cast(dict[str, Any], message[2])
                elif message[0] == "progress":
                    self.progress = cast(Progress, message[1])
                    if (watcher := self.watcher) is not None:
                        watcher(self.progress)
                else:
                    self._answered[rank] = (str(message[0]), message[1])
            return bool(ready)

    def keeping(self, name: str) -> bool:
        """Whether the processes keep the full state of the step that wrote into a directory called `name`, after it."""
        return name in self._keeping

    def kept_state(self, name: str) -> dict[str, Any]:
        """What every process kept of the full state of the step that wrote into a directory called `name` (its files'
        blob references, as JSON, by their paths within the state), once all have. Raises `StateLost` where one could
        not keep its share, or the processes ended before they all said, and `KeyError` for a step they do not keep."""
        if name not in self._keeping:
            raise KeyError(f"the processes keep no state of {name}")
        try:
            while True:
                reports = dict(self._reports.get(name, {}))
                if failed := [each["error"] for each in reports.values() if "error" in each]:
                    raise StateLost(f"a trainer's process did not keep its share of the state: {failed[0]}")
                if len(reports) >= self._keeping[name]:
                    return {path: reference for each in reports.values() for path, reference in each["files"].items()}
                if not self._pump(0.5) and (self._process is None or self._process.poll() is not None):
                    raise StateLost("the trainer's processes ended before they kept the state")
        finally:
            self._reports.pop(name, None)
            self._keeping.pop(name, None)

    def _settle(self, patience: float) -> None:
        """Wait, for up to `patience` seconds, while the processes keep a step's full state."""
        given_up = time.monotonic() + patience
        while self._process is not None and self._process.poll() is None and time.monotonic() < given_up:
            pending = [name for name, count in list(self._keeping.items())
                       if len(self._reports.get(name, {})) < count]  # fmt: skip
            if not pending:
                return
            self._pump(0.5)

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
        """End the processes (and what they hold), once they have kept the full states they were keeping (for up to
        `KEEP_PATIENCE` seconds)."""
        self._settle(KEEP_PATIENCE)
        self._holding = None
        for connection in self._connections:
            with contextlib.suppress(OSError):
                connection.send(("stop", None))
            with contextlib.suppress(OSError):
                connection.close()
        self._connections = []
        self._ended_ranks = set()
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
    """Signal torchrun's process group: its agent, which leads a session of its own. The processes it starts are in
    sessions of their own (torchrun starts each so), out of this group: they end when the agent does (each asks the
    kernel to end it with its parent, `end_with_parent`), and when their connection to the trainer closes."""
    with contextlib.suppress(OSError):
        os.killpg(process.pid, number)
