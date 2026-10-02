"""Engines for the recorder: tokens in; tokens, behavior logprobs and a finish reason out.

`VllmEngine` drives vLLM's async engine, whose engine core runs in its own process: when it sleeps, its GPU memory
is free for a trainer. Sleeping drops the weights (they are read again from the checkpoint on waking, about 3 s from
the file cache) rather than parking them in system memory, where 8 GiB of them sat next to the trainer.

LoRA adapters are registered by name (`load_adapter`); a recorder channel names the adapter each sample uses. Entry
points that start an engine must guard `if __name__ == "__main__":` (vLLM starts its process with `spawn`).
"""

import contextlib
import itertools
import json
import os
import signal
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rollout.recorder.recorder import Generation


class VllmEngine:
    def __init__(
        self,
        model: str,
        *,
        gpu_memory_utilization: float = 0.72,
        max_model_len: int = 8192,
        max_num_seqs: int = 32,
        max_num_batched_tokens: int = 4096,
        max_lora_rank: int = 32,
        max_loras: int = 2,
        language_model_only: bool = True,
        seed: int = 0,
    ) -> None:
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        arguments = AsyncEngineArgs(
            model=model,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_lora=True,
            max_lora_rank=max_lora_rank,  # pyright: ignore[reportArgumentType] (a Literal of allowed ranks)
            max_loras=max_loras,
            enable_sleep_mode=True,
            language_model_only=language_model_only,
            logprobs_mode="processed_logprobs",  # the distribution actually sampled from (after temperature)
            seed=seed,
        )
        self.model = model
        self.max_model_len = max_model_len
        """The longest sequence (prompt and completion) the engine accepts; a recorder channel reads it."""
        self._engine: Any = AsyncLLM.from_engine_args(arguments)
        self._requests = itertools.count()
        self._adapters: dict[str, Any] = {}
        self._adapter_ids = itertools.count(1)

    async def generate(
        self,
        prompt: Sequence[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_token_ids: Sequence[int],
        adapter: str | None,
    ) -> Generation:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_token_ids=list(stop_token_ids),
            logprobs=0,
            detokenize=False,
            skip_special_tokens=False,
        )
        lora = self._adapters[adapter] if adapter is not None else None
        final: Any = None
        async for output in self._engine.generate(
            TokensPrompt(prompt_token_ids=list(prompt)), params, f"r{next(self._requests)}", lora_request=lora
        ):
            final = output
        completion = final.outputs[0]
        tokens = list(completion.token_ids)
        entries: list[Any] = list(completion.logprobs or [])
        missing = len(entries) != len(tokens) or any(
            entry is None or token not in entry for token, entry in zip(tokens, entries, strict=False)
        )
        if missing:  # (it could not be trained on, and as NaN it would undo the adapter)
            raise RuntimeError("the engine returned a sampled token without its logprob")
        logprobs = [float(entry[token].logprob) for token, entry in zip(tokens, entries, strict=True)]
        finish = "length" if completion.finish_reason == "length" else "stop"
        return Generation(tokens=tokens, logprobs=logprobs, finish_reason=finish)

    async def load_adapter(self, name: str, path: str) -> None:
        """Register a LoRA adapter (a PEFT directory) under `name`; samples name it to use it."""
        from vllm.lora.request import LoRARequest

        request = LoRARequest(name, next(self._adapter_ids), path)
        await self._engine.add_lora(request)
        self._adapters[name] = request

    async def remove_adapter(self, name: str) -> None:
        request = self._adapters.pop(name, None)
        if request is not None:
            await self._engine.remove_lora(request.lora_int_id)

    async def sleep(self, *, keep_weights: bool = False) -> None:
        """Free the GPU: the cache is discarded, and the weights dropped (or, with `keep_weights`, moved to system
        memory: waking is then a second faster and costs the weights' size in memory meanwhile)."""
        await self._engine.reset_prefix_cache()
        self._dropped = not keep_weights
        await self._engine.sleep(level=1 if keep_weights else 2)

    async def wake(self) -> None:
        if getattr(self, "_dropped", False):
            await self._engine.wake_up(tags=["weights"])
            await self._engine.collective_rpc("reload_weights")
            await self._engine.wake_up(tags=["kv_cache"])
            self._dropped = False
        else:
            await self._engine.wake_up()

    def close(self) -> None:
        self._engine.shutdown()


ENGINE_PROCESS = "VLLM::Engine"
"""How an engine core's process is named (the start of it: the kernel keeps fifteen characters)."""


def engine_processes(parent: int | None = None) -> list[int]:
    """The engine core processes started by `parent` (this process, by default)."""
    parent = os.getpid() if parent is None else parent
    found: list[int] = []
    for status in Path("/proc").glob("[0-9]*/status"):
        try:
            fields = dict(line.split(":\t", 1) for line in status.read_text().splitlines() if ":\t" in line)
        except OSError:
            continue  # it ended meanwhile
        if fields.get("Name", "").startswith(ENGINE_PROCESS) and int(fields.get("PPid", "0")) == parent:
            found.append(int(status.parent.name))
    return sorted(found)


def note_engines(record: Path) -> None:
    """Write down this process's engine cores, so that a later process can end them if this one dies without doing
    so (a killed process cannot shut its engine down, and an engine left behind holds the GPU)."""
    record.write_text(json.dumps({"owner": os.getpid(), "engines": engine_processes()}))


def end_orphaned_engines(record: Path) -> list[int]:
    """End the engine cores an earlier process noted in `record`, if that process is gone and they are not; returns
    the ones ended."""
    try:
        noted = json.loads(record.read_text())
        owner, engines = int(noted["owner"]), [int(pid) for pid in noted["engines"]]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    if owner == os.getpid() or Path(f"/proc/{owner}").exists():
        return []
    ended: list[int] = []
    for pid in engines:
        try:
            name = Path(f"/proc/{pid}/comm").read_text().strip()
        except OSError:
            continue
        if name.startswith(ENGINE_PROCESS):  # (a process id may have been given to something else since)
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
                ended.append(pid)
    return ended
