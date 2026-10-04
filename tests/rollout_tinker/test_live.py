# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Tinker itself, on `Qwen/Qwen3.5-4B` (the 9B's family at half the price): the tokenizer agrees with our renderer,
sampling keeps its contract, its logprobs agree with a training pass, a step round-trips to a new sampler, and its
adapter, downloaded and converted, serves on this machine's vLLM. Each test deletes the checkpoints it made.

Run only when asked (`ROLLOUT_TINKER=1`), and skipped without a key (`TINKER_API_KEY`, or `tinker auth login`). It
spends well under ten cents. The last test also starts vLLM on the GPU, and the model's files (about 9 GB) are fetched
from the Hub the first time:

    ROLLOUT_TINKER=1 flock ~/.cache/rollout/gpu.lock uv run pytest -s tests/rollout_tinker/test_live.py

What it finds is printed and written to `~/.cache/rollout/tinker-smoke.json` (no key, no secrets: what was measured).
"""

import hashlib
import json
import os
import statistics
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from tinker import ModelInput, SamplingParams

from rollout.contracts import Message, ToolSpecification
from rollout_qwen import qwen35
from rollout_tinker import TinkerEngine, TinkerTrainer
from rollout_tinker.bridges import converted
from rollout_tinker.data import datum
from rollout_tinker.service import Service, connected, has_key
from rollout_tinker.weights import downloaded, pointer, ranks
from rollout_train.recorder import Renderer, Segment, Span
from rollout_train.trainer import STATE, WEIGHTS, Weighted

pytestmark = [
    pytest.mark.skipif(os.environ.get("ROLLOUT_TINKER") != "1", reason="calls Tinker (ROLLOUT_TINKER=1)"),
    pytest.mark.skipif(not has_key(), reason="no Tinker key: set TINKER_API_KEY or run `tinker auth login`"),
]

MODEL = "Qwen/Qwen3.5-4B"
REPORT = Path("~/.cache/rollout/tinker-smoke.json").expanduser()
TOOL = ToolSpecification(
    name="mine",
    description="Mine a block you can see.",
    input_schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
)
CONVERSATION = [
    Message.system("You are a miner in a Minecraft world. Think briefly, then act."),
    Message.user("A diamond ore is at x=4. Mine it."),
]
FOUND: dict[str, Any] = {}


def noted(key: str, value: Any) -> None:
    FOUND[key] = value
    print(f"{key}: {value}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(FOUND, indent=2) + "\n")


@pytest.fixture(scope="module")
def service() -> Iterator[Service]:
    made = connected()
    yield made
    closing: Any = made
    closing.close("success").result()


@pytest.fixture(scope="module")
def renderer() -> Renderer:
    return qwen35(MODEL)


async def sampled(
    service: Service, renderer: Renderer, count: int, max_tokens: int = 64
) -> list[tuple[list[int], Any]]:
    """Prompts and what the base model sampled after each."""
    sampler = await service.create_sampling_client_async(base_model=MODEL)
    prompt = renderer.render(CONVERSATION, [TOOL])
    found: list[tuple[list[int], Any]] = []
    for _ in range(count):
        parameters = SamplingParams(max_tokens=max_tokens, temperature=1.0, top_p=1.0, stop=renderer.stop_token_ids())
        response = await sampler.sample_async(ModelInput.from_ints(prompt), 1, parameters)
        found.append((prompt, response.sequences[0]))
    return found


def segments_of(samples: list[tuple[list[int], Any]]) -> list[Weighted]:
    made: list[Weighted] = []
    for index, (prompt, sequence) in enumerate(samples):
        tokens = [*prompt, *sequence.tokens]
        segment = Segment(tokens, [Span(len(prompt), len(tokens), 0)], [float(each) for each in sequence.logprobs])
        made.append(Weighted(segment, 1.0 if index % 2 == 0 else -1.0, source=f"smoke/1/{index}/ada/0"))
    return made


async def forgotten(service: Service, into: Path) -> None:
    """Delete the Tinker checkpoints a step's files point at."""
    rest = service.create_rest_client()
    for path in {pointer(into / WEIGHTS, "sampler"), pointer(into / STATE, "state")}:
        if path is not None:
            await rest.delete_checkpoint_from_tinker_path_async(path)


async def test_the_tokenizer_agrees_with_our_renderer(service: Service, renderer: Renderer) -> None:
    sampler: Any = await service.create_sampling_client_async(base_model=MODEL)
    theirs = sampler.get_tokenizer()
    ours: Any = getattr(renderer, "tokenizer")  # noqa: B009
    ids = renderer.render(CONVERSATION, [TOOL])
    assert theirs.encode(renderer.decode(ids), add_special_tokens=False) == ids

    def digest(tokenizer: Any) -> str:
        return hashlib.sha256(json.dumps(sorted(tokenizer.get_vocab().items())).encode()).hexdigest()[:16]

    noted("vocabulary_digest", {"tinker": digest(theirs), "hub": digest(ours)})
    assert digest(theirs) == digest(ours)


async def test_sampling_keeps_the_engines_contract(service: Service, renderer: Renderer) -> None:
    ((prompt, sequence),) = await sampled(service, renderer, 1, max_tokens=64)
    assert sequence.stop_reason in ("stop", "length")
    assert sequence.logprobs is not None and len(sequence.logprobs) == len(sequence.tokens)
    stops = set(renderer.stop_token_ids())
    if sequence.stop_reason == "stop":
        assert sequence.tokens[-1] in stops  # the stop token is among the tokens returned
    noted("sampling", {"stop_reason": sequence.stop_reason, "tokens": len(sequence.tokens),
                       "stop_token_last": sequence.tokens[-1] in stops})  # fmt: skip
    engine = TinkerEngine(MODEL, service=service)
    made = await engine.generate(prompt, max_tokens=16, temperature=1.0, top_p=1.0, stop_token_ids=sorted(stops),
                                 adapter=None)  # fmt: skip
    assert made.tokens and len(made.logprobs) == len(made.tokens)


async def test_sampled_logprobs_agree_with_a_training_pass(service: Service, renderer: Renderer) -> None:
    samples = await sampled(service, renderer, 2, max_tokens=48)
    client = await service.create_lora_training_client_async(base_model=MODEL, rank=32, train_unembed=False)
    data = [datum([*prompt, *sequence.tokens], [], {"weights": []}) for prompt, sequence in samples]
    out = await (await client.forward_async(data, "cross_entropy"))
    differences: list[float] = []
    for (prompt, sequence), found in zip(samples, out.loss_fn_outputs, strict=True):
        trained = found["logprobs"].data[len(prompt) - 1 :]
        differences += [abs(float(a) - float(b)) for a, b in zip(sequence.logprobs, trained, strict=True)]
    noted("mean_mismatch", round(statistics.mean(differences), 5))
    noted("max_mismatch", round(max(differences), 5))
    assert statistics.mean(differences) < 0.2


async def test_a_step_round_trips_to_a_new_sampler(service: Service, renderer: Renderer, tmp_path: Path) -> None:
    samples = await sampled(service, renderer, 4, max_tokens=48)
    trainer = TinkerTrainer(MODEL, service=service, tokens_per_step=10**6, learning_rate=1e-4)
    into = tmp_path / "smoke"
    try:
        step = await trainer.step(segments_of(samples), seed=0, parent=None, into=into)
        noted("step", {key: round(value, 5) for key, value in step.metrics.items()})
        noted("minibatches", (into / STATE / "minibatches.jsonl").read_text().splitlines())
        assert step.metrics["optimizer_steps"] == 1 and abs(step.metrics["mean_ratio"] - 1) < 1e-6
        engine = TinkerEngine(MODEL, service=service)
        await engine.load_adapter("smoke", str(into / WEIGHTS))
        made = await engine.generate(samples[0][0], max_tokens=16, temperature=1.0, top_p=1.0,
                                     stop_token_ids=renderer.stop_token_ids(), adapter="smoke")  # fmt: skip
        assert made.tokens and len(made.logprobs) == len(made.tokens)
        sampler = pointer(into / WEIGHTS, "sampler")
        assert sampler is not None
        archive = await service.create_rest_client().get_checkpoint_archive_url_from_tinker_path_async(sampler)
        adapter = await downloaded(archive.url, tmp_path / "archive")
        from safetensors import safe_open

        with safe_open(str(adapter / "adapter_model.safetensors"), framework="pt") as opened:
            keys = sorted(opened.keys())
        noted("archive", {
            "config": json.loads((adapter / "adapter_config.json").read_text()),
            "megabytes": round((adapter / "adapter_model.safetensors").stat().st_size / 2**20, 1),
            "keys": len(keys), "first_layer": [key for key in keys if ".layers.0." in key or ".layers.3." in key],
        })  # fmt: skip
    finally:
        await forgotten(service, into)


async def test_its_adapter_serves_on_this_machines_vllm(service: Service, renderer: Renderer, tmp_path: Path) -> None:
    pytest.importorskip("vllm")
    from rollout_vllm import VllmEngine

    samples = await sampled(service, renderer, 4, max_tokens=48)
    trainer = TinkerTrainer(MODEL, service=service, tokens_per_step=10**6, learning_rate=1e-4)
    into, peft = tmp_path / "made", tmp_path / "peft"
    try:
        await trainer.step(segments_of(samples), seed=0, parent=None, into=into)
        await converted(into / WEIGHTS, peft, MODEL, service)  # (Tinker's bridge)
    finally:
        await forgotten(service, into)
    largest = ranks(peft)
    noted("peft", {"largest_rank": largest, "config": json.loads((peft / "adapter_config.json").read_text())})
    engine = VllmEngine(MODEL, gpu_memory_utilization=0.8, max_model_len=4096, max_num_seqs=4,
                        max_lora_rank=next(rank for rank in (32, 64, 128, 256) if rank >= largest))  # fmt: skip
    try:
        await engine.load_adapter("smoke", str(peft))
        made = await engine.generate(samples[0][0], max_tokens=8, temperature=1.0, top_p=1.0,
                                     stop_token_ids=renderer.stop_token_ids(), adapter="smoke")  # fmt: skip
        noted("local_vllm", {"tokens": made.tokens, "text": renderer.decode(made.tokens)})
        assert made.tokens and len(made.logprobs) == len(made.tokens)
    finally:
        engine.close()
