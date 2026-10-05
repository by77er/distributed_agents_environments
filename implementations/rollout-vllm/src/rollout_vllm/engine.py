"""vLLM as an engine.

`VllmEngine` drives vLLM's async engine, whose engine core runs in its own process: when it sleeps, its GPU memory
is free for a trainer. Sleeping drops the weights (they are read again from the checkpoint on waking, about 3 s from
the file cache) rather than parking them in system memory, where 8 GiB of them sat next to the trainer.

LoRA adapters are registered by name (`load_adapter`); a request names the adapter it samples from. Entry points
that start an engine must guard `if __name__ == "__main__":` (vLLM starts its process with `spawn`).
"""

import itertools
import os
from collections.abc import Mapping, Sequence
from typing import Any

from rollout.processes import children
from rollout_train.inference.channel import Generation, Scores, most_likely, scored_range


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
        speculative: Mapping[str, Any] | None = None,
        quantization: str | None = None,
        seed: int = 0,
        max_logprobs: int = 20,
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
            speculative_config=dict(speculative) if speculative else None,
            quantization=quantization,  # (`fp8`: a bfloat16 checkpoint's weights quantized as they load)
            seed=seed,
            max_logprobs=max_logprobs,
        )
        self.model = model
        self.max_logprobs = max_logprobs
        """The most tokens a request may ask for at each position, with their logprobs (`top`)."""
        self.max_model_len = max_model_len
        """The longest sequence (prompt and completion) the engine accepts; a channel reads it."""
        self._engine: Any = AsyncLLM.from_engine_args(arguments)
        self._requests = itertools.count()
        self._adapters: dict[str, Any] = {}
        self._weights: str | None = None
        """The full checkpoint's files the engine serves, if it serves one in place of the model's own."""
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
        top: int = 0,
    ) -> Generation:
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_token_ids=list(stop_token_ids),
            logprobs=self._top(top),
            detokenize=False,
            skip_special_tokens=False,
        )
        completion = (await self._final(prompt, params, adapter)).outputs[0]
        tokens = list(completion.token_ids)
        entries: list[Any] = list(completion.logprobs or [])
        missing = len(entries) != len(tokens) or any(
            entry is None or token not in entry for token, entry in zip(tokens, entries, strict=False)
        )
        if missing:  # (it could not be trained on, and as NaN it would undo the adapter)
            raise RuntimeError("the engine returned a sampled token without its logprob")
        logprobs = [float(entry[token].logprob) for token, entry in zip(tokens, entries, strict=True)]
        finish = "length" if completion.finish_reason == "length" else "stop"
        top_tokens, top_logprobs = tops(entries, tokens, top) if top else ([], [])
        return Generation(tokens, logprobs, finish, top_tokens=top_tokens, top_logprobs=top_logprobs)

    async def score(
        self, tokens: Sequence[int], *, start: int, end: int | None = None, top: int = 0, adapter: str | None
    ) -> Scores:
        """The logprobs the model (or `adapter`) gives the tokens at positions `start` to `end` of `tokens`, each given
        those before it, and the `top` most likely tokens at each: vLLM's prompt logprobs of `tokens` up to `end`, with
        one token generated (vLLM generates at least one) and dropped. The scores are of the model's own distribution
        (prompt logprobs skip temperature). The sequence must leave room for that token (`max_model_len`). vLLM
        computes the logits of every position before `end`, 1,024 positions at a time, so scoring a long sequence takes
        about 0.6 GiB of the GPU beyond the engine's own share (Qwen3's vocabulary of 152K), however long it is."""
        from vllm import SamplingParams

        end = scored_range(len(tokens), start, end)
        params = SamplingParams(
            max_tokens=1, temperature=0.0, prompt_logprobs=self._top(top), detokenize=False, skip_special_tokens=False
        )
        prompt = list(tokens[:end])
        entries: list[Any] = list((await self._final(prompt, params, adapter)).prompt_logprobs or [])
        if len(entries) != end or any(entries[at] is None or prompt[at] not in entries[at] for at in range(start, end)):
            raise RuntimeError("the engine returned a scored token without its logprob")
        logprobs = [float(entries[at][prompt[at]].logprob) for at in range(start, end)]
        top_tokens, top_logprobs = tops(entries[start:end], prompt[start:end], top) if top else ([], [])
        return Scores(start, logprobs, top_tokens=top_tokens, top_logprobs=top_logprobs)

    def _top(self, top: int) -> int:
        """How many logprobs vLLM is asked for at each position, beside the token's own: `top`, which vLLM caps at
        `max_logprobs`."""
        if not 0 <= top <= self.max_logprobs:
            raise ValueError(f"top is 0 to {self.max_logprobs} (max_logprobs), not {top}")
        return top

    async def _final(self, prompt: Sequence[int], params: Any, adapter: str | None) -> Any:
        """vLLM's final output for a request of `prompt`'s tokens, sampled from `adapter` (None: the weights held)."""
        from vllm.inputs import TokensPrompt

        lora = self._adapters[adapter] if adapter is not None else None
        final: Any = None
        async for output in self._engine.generate(
            TokensPrompt(prompt_token_ids=list(prompt)), params, f"r{next(self._requests)}", lora_request=lora
        ):
            final = output
        return final

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

    async def load_weights(self, path: str) -> None:
        """Serve the full weights in `path` (a checkpoint's files, in the model's own layout) in place of the ones
        held, from now on: they are read into the model as it is, and read again from there on waking. (vLLM warns
        that `ParallelLMHead` failed to load where the output layer is tied to the embeddings: it shares them, and
        serves the new ones.)"""
        await self._engine.collective_rpc("reload_weights", kwargs={"weights_path": path})
        await self._engine.reset_prefix_cache()  # (what was cached was computed with the weights before)
        self._weights = path

    async def sleep(self) -> None:
        """Free the GPU: the cache is discarded and the weights dropped (they are read again on waking)."""
        await self._engine.reset_prefix_cache()
        await self._engine.sleep(level=2)

    async def wake(self) -> None:
        await self._engine.wake_up(tags=["weights"])
        served = {"weights_path": self._weights} if self._weights is not None else {}
        await self._engine.collective_rpc("reload_weights", kwargs=served)
        await self._engine.wake_up(tags=["kv_cache"])

    @property
    def processes(self) -> list[int]:
        return children(ENGINE_PROCESS)

    def close(self) -> None:
        self._engine.shutdown()


def tops(
    entries: Sequence[Mapping[int, Any]], tokens: Sequence[int], top: int
) -> tuple[list[list[int]], list[list[float]]]:
    """The `top` most likely tokens at each position and their logprobs, from vLLM's logprobs there (each position's
    own token among them, or one more)."""
    found = [
        most_likely({int(each): float(value.logprob) for each, value in entry.items()}, token, top)
        for entry, token in zip(entries, tokens, strict=True)
    ]
    return [ids for ids, _ in found], [values for _, values in found]


ENGINE_PROCESS = "VLLM::Engine"
"""How an engine core's process is named (the start of it: the kernel keeps fifteen characters)."""
