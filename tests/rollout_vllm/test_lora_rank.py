import pytest

from rollout_vllm.engine import lora_rank


def test_an_engine_takes_the_next_rank_vllm_accepts() -> None:
    assert [lora_rank(asked) for asked in (1, 16, 48, 64, 96, 300)] == [1, 16, 64, 64, 128, 320]
    with pytest.raises(ValueError, match="above the largest"):
        lora_rank(600)
