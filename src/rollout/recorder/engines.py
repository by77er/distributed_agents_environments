"""Engines for the recorder: tokens in; tokens, behavior logprobs and a finish reason out.

`VllmEngine` drives vLLM's async engine, whose engine core runs in its own process: when it sleeps, its GPU memory
is free for a trainer in this process. LoRA adapters are registered by name (`load_adapter`); a recorder channel
names the adapter each sample uses. Entry points that start one must guard `if __name__ == "__main__":` (vLLM
starts its process with `spawn`).
"""

import itertools
import math
import os
from collections.abc import Sequence
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
        logprobs = [
            entry[token].logprob if entry is not None and token in entry else math.nan
            for token, entry in zip(tokens, completion.logprobs or [None] * len(tokens), strict=True)
        ]
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

    async def sleep(self) -> None:
        """Free the GPU: weights to CPU memory, cache discarded."""
        await self._engine.reset_prefix_cache()
        await self._engine.sleep(level=1)

    async def wake(self) -> None:
        await self._engine.wake_up()

    def close(self) -> None:
        self._engine.shutdown()
