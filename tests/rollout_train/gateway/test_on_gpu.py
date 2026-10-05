"""The gateway over a real model: a replica started by `rollout gateway PROFILE` serves Qwen3-0.6B on vLLM, the
official OpenAI client talks to it, and the turn it records holds the tokens the model sampled, with logprobs. Run only
when asked (`-m live`), holding the machine's GPU lock (`~/.cache/rollout/gpu.lock`)."""

import contextlib
import fcntl
import math
import os
import subprocess
import sys
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from rollout.harness.blobs import FileBlobStore
from rollout_train.gateway import TurnStore
from rollout_train.ledger import FileLedger
from rollout_train.testing import SECRETS, keyring
from tests.rollout_train.gateway.support import grant

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parents[3]
MODEL = os.environ.get("ROLLOUT_SMALL_MODEL", "Qwen/Qwen3-0.6B")
PORT = 8834
PROFILE = """
directory = "{directory}/run"
ledger = "{directory}/ledger"

[channels.policy]
model = "{model}"
renderer = "rollout_qwen:qwen3"
engine = "rollout_vllm:VllmEngine"
engines = [{{ gpu_memory_utilization = 0.4, max_model_len = 4096 }}]
thinking_tokens = 64
answer_tokens = 64

[gateway]
keys = "{directory}/keys"
"""


@contextlib.contextmanager
def gpu() -> Generator[None]:
    """The machine's GPU, held while the block runs."""
    path = Path.home() / ".cache" / "rollout" / "gpu.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def test_a_replica_serves_a_real_model_and_records_what_it_sampled(tmp_path: Path) -> None:
    import asyncio

    import openai

    profile = tmp_path / "profile.toml"
    profile.write_text(PROFILE.format(directory=tmp_path, model=MODEL))
    (tmp_path / "keys").write_text("".join(f"{name} {secret}\n" for name, secret in SECRETS))
    ledger = FileLedger(tmp_path / "ledger")
    key = keyring().mint(asyncio.run(grant(ledger)))
    command = [sys.executable, "-m", "rollout_train.cli", "gateway", str(profile), "--listen", f"127.0.0.1:{PORT}"]
    with gpu():
        replica = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})
        try:
            deadline = time.monotonic() + 600
            while True:
                with contextlib.suppress(httpx.TransportError):
                    if httpx.get(f"http://127.0.0.1:{PORT}/readyz", timeout=2).status_code == 200:
                        break
                assert replica.poll() is None and time.monotonic() < deadline, "the replica did not start"
                time.sleep(1)
            client = openai.OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key=key)
            answer = client.chat.completions.with_raw_response.create(
                model=MODEL,
                messages=[{"role": "user", "content": "Name a colour."}],
                extra_headers={"Idempotency-Key": "gpu-1"},
            )
            reply = answer.parse()
            assert answer.headers["x-rollout-checkpoint"] == MODEL and reply.choices[0].message is not None
        finally:
            replica.terminate()
            replica.wait(60)
    store = TurnStore(ledger, FileBlobStore(tmp_path / "run" / "blobs"))
    (turn,) = asyncio.run(store.turns("train", "r_1"))
    sampled = [value for value, kept in zip(turn.logprobs, turn.mask, strict=True) if kept]
    assert sampled and all(value <= 0 and not math.isnan(value) for value in sampled)
    (segment,) = asyncio.run(store.sessions("train", "r_1"))["policy"]
    from rollout_qwen import qwen3

    text = qwen3(MODEL).decode(segment.tokens)
    assert "Name a colour." in text and cast(Any, segment.spans[0]).version == 0
