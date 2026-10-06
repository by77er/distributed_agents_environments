"""Where a model's files are on this machine."""

from pathlib import Path

import pytest


def test_a_model_not_on_this_machine_is_downloaded_before_its_files_are_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # (a training pod starts with an empty Hugging Face cache)
    import huggingface_hub
    import huggingface_hub.constants

    from rollout_lora.models import fetched

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    asked: list[str] = []

    def downloaded(model: str) -> str:
        asked.append(model)
        snapshot = tmp_path / "models--org--tiny" / "snapshots" / "abc"
        snapshot.mkdir(parents=True)
        (snapshot / "model.safetensors").write_bytes(b"x")
        (tmp_path / "models--org--tiny" / "refs").mkdir()
        (tmp_path / "models--org--tiny" / "refs" / "main").write_text("abc")
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", downloaded)
    assert (fetched("org/tiny") / "model.safetensors").exists() and asked == ["org/tiny"]
    assert fetched("org/tiny").name == "abc" and asked == ["org/tiny"]  # (cached now: not asked again)
