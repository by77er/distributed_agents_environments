"""Bridges' declarations: the path from a checkpoint's format to what a provider loads, the pairs refused and why, the
rank a provider sees, and a checkpoint's format read from its files."""

from rollout_train.bridges import BRIDGES, Bridge, NoBridge, format_of, path, rank_factor


def names(found: tuple[Bridge, ...] | NoBridge) -> list[str]:
    assert not isinstance(found, NoBridge), found
    return [each.name for each in found]


def test_each_pair_takes_its_bridge() -> None:
    assert names(path("tinker", {"tinker"})) == ["none"]
    assert names(path("tinker", {"peft", "full"})) == ["peft-from-tinker"]
    assert names(path("peft", {"peft", "full"})) == ["verbatim"]
    assert names(path("full", {"peft", "full"})) == ["full-reload"]
    assert names(path("peft", {"peft"})) == ["verbatim"]


def test_merge_quantize_only_when_asked() -> None:
    assert isinstance(path("peft", {"full"}), NoBridge)
    assert names(path("peft", {"full"}, wanted="merge-quantize")) == ["merge-quantize"]
    assert names(path("peft", {"peft", "full"}, wanted="merge-quantize")) == ["merge-quantize"]
    assert names(path("tinker", {"full"}, wanted="merge-quantize")) == ["peft-from-tinker", "merge-quantize"]
    refused = path("tinker", {"tinker"}, wanted="merge-quantize")
    assert isinstance(refused, NoBridge) and "goes through merge-quantize" in refused.reason


def test_the_refused_pairs_say_why() -> None:
    for source in ("peft", "full"):
        refused = path(source, {"tinker"})
        assert isinstance(refused, NoBridge)
        assert refused.reason == "Tinker samples only checkpoints Tinker trained: there is no upload"
    refused = path("full", {"peft"})
    assert isinstance(refused, NoBridge) and "not an adapter" in refused.reason
    nothing = path("peft", set())
    assert isinstance(nothing, NoBridge) and "base models only" in nothing.reason
    wrong = path("peft", {"peft"}, wanted="sideways")
    assert isinstance(wrong, NoBridge) and "auto or merge-quantize" in wrong.reason


def test_tinkers_adapters_for_qwen35_triple_their_rank() -> None:
    found = path("tinker", {"peft"})
    assert not isinstance(found, NoBridge)
    assert rank_factor(found, "Qwen/Qwen3.5-9B") == 3 and rank_factor(found, "Qwen/Qwen3-0.6B") == 1
    verbatim = path("peft", {"peft"})
    assert not isinstance(verbatim, NoBridge) and rank_factor(verbatim, "Qwen/Qwen3.5-9B") == 1


def test_bridges_declare_their_needs_and_tasks() -> None:
    by_name = {each.name: each for each in BRIDGES}
    assert by_name["none"].task is None
    assert by_name["peft-from-tinker"].network and by_name["peft-from-tinker"].cpus == 2
    assert by_name["merge-quantize"].explicit and by_name["merge-quantize"].memory_gib == 48
    assert by_name["verbatim"].task == by_name["full-reload"].task == "rollout_train.resharding:verbatim"


def test_a_checkpoints_format_is_read_from_its_files() -> None:
    assert format_of(["weights/tinker.json"]) == {"tinker"}
    assert format_of(["adapter_config.json", "adapter_model.safetensors"]) == {"peft"}
    assert format_of(["config.json", "model-00001-of-00002.safetensors", "tokenizer.json"]) == {"full"}
    both = ["weights/tinker.json", "weights/adapter_config.json", "weights/adapter_model.safetensors"]
    assert format_of(both) == {"tinker", "peft"}
    assert format_of(["adapter_config.json"]) == set()
