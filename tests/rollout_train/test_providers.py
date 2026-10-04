"""Providers' declarations: what each kind can do, the trainers' settings read from their dataclasses, and how each
way of reaching a provider becomes a client's connection settings."""

import pytest

from rollout_train.providers import (
    INFERENCE_KINDS,
    TRAINER_KINDS,
    Auth,
    Secret,
    SettingSpec,
    Tls,
    is_local,
    settings_of,
)
from rollout_train.recorder import TOKEN_LEVEL

TLS = Tls(ca="~/ca.pem", certificate="~/gateway.crt", key="~/gateway.key")


def can_be_trained_on(kind: str) -> bool:
    offered = INFERENCE_KINDS[kind].capabilities
    return offered.token_exact and offered.sampled_logprobs and offered.honours_sampling


def test_the_capability_table() -> None:
    assert {kind for kind in INFERENCE_KINDS if can_be_trained_on(kind)} == {
        "vllm", "vllm-servers", "tinker", "runpod-inference",
    }  # fmt: skip
    loads = {kind: each.capabilities.loads for kind, each in INFERENCE_KINDS.items()}
    assert loads == {
        "vllm": {"peft", "full"}, "vllm-servers": {"peft"}, "tinker": {"tinker"}, "api": set(),
        "runpod-inference": {"peft"},
    }  # fmt: skip
    assert [kind for kind, each in INFERENCE_KINDS.items() if each.capabilities.full_reload] == ["vllm"]
    assert INFERENCE_KINDS["api"].capabilities.streaming and not INFERENCE_KINDS["api"].capabilities.adapters
    assert INFERENCE_KINDS["runpod-inference"].capabilities.bills == "hours"
    tinker = INFERENCE_KINDS["tinker"].capabilities
    assert tinker.bills == "tokens" and tinker.unchecked == {"prompt_logprobs", "top_logprobs"}  # (checked live first)
    assert INFERENCE_KINDS["runpod-inference"].auths == ("mtls",)
    assert INFERENCE_KINDS["tinker"].auths == ("vendor",)
    sampled_with = {kind: each.capabilities.sampled_with for kind, each in INFERENCE_KINDS.items()}
    assert sampled_with.pop("api") == () and set(sampled_with.values()) == {TOKEN_LEVEL}  # (as a turn records it)


def test_the_trainers_table() -> None:
    made = {kind: (each.capabilities.produces, each.capabilities.format) for kind, each in TRAINER_KINDS.items()}
    assert made == {
        "lora": ("lora", "peft"), "full": ("full", "full"), "tinker": ("lora", "tinker"),
        "runpod-trainer": ("lora", "peft"),
    }  # fmt: skip
    assert TRAINER_KINDS["full"].capabilities.starts_from == {"full"}
    assert TRAINER_KINDS["lora"].capabilities.starts_from == {"peft", "full"}
    assert TRAINER_KINDS["tinker"].capabilities.starts_from == {"tinker"}
    assert all(each.capabilities.scores for each in TRAINER_KINDS.values())


def test_a_trainers_settings_are_read_from_its_dataclass() -> None:
    lora = {each.key: each for each in settings_of(TRAINER_KINDS["lora"])}
    assert lora["trainer.rank"] == SettingSpec("trainer.rank", ("int",), 32, False)
    assert lora["trainer.learning_rate"] == SettingSpec("trainer.learning_rate", ("float",), 5e-5, True)
    assert lora["trainer.truncate"].types == ("float", "null") and lora["trainer.truncate"].changeable
    assert lora["trainer.segment_tokens"] == SettingSpec("trainer.segment_tokens", ("int", "null"), None, False)
    assert lora["trainer.objective"].default == "policy_gradient"
    full = {each.key for each in settings_of(TRAINER_KINDS["full"])}
    assert "trainer.rank" not in full and "trainer.learning_rate" in full  # (a full-weight trainer has no adapter)


def test_a_setting_says_what_it_accepts() -> None:
    spec = SettingSpec("trainer.truncate", ("float", "null"), 2.0, True)
    assert spec.accepts(1) and spec.accepts(1.5) and spec.accepts(None)
    assert not spec.accepts(True) and not spec.accepts("2")
    assert not SettingSpec("trainer.rank", ("int",), 32, False).accepts(1.5)


def test_mutual_tls_with_the_clusters_ca_a_client_certificate_and_the_pods_identity() -> None:
    pods = Auth("mtls", identity="beats")
    connection = pods.connection(TLS, identity="spiffe://rollout/pod/inference-run-1")
    assert (
        connection.ca == "~/ca.pem" and connection.certificate == "~/gateway.crt" and connection.key == "~/gateway.key"
    )
    assert connection.identity == "spiffe://rollout/pod/inference-run-1" and connection.token_env is None
    with pytest.raises(ValueError, match="from its heartbeat"):
        pods.connection(TLS)
    with pytest.raises(ValueError, match="cluster's CA"):
        pods.connection(None, identity="spiffe://rollout/pod/x")
    fixed = Auth("mtls").connection(TLS)
    assert fixed.identity is None  # (a public endpoint: its host name is checked)


def test_a_bearer_token_over_the_systems_cas_or_the_clusters() -> None:
    public = Auth("bearer", token=Secret(env="ENGINES_TOKEN")).connection(TLS)
    assert public.token_env == "ENGINES_TOKEN" and public.ca is None and public.certificate is None
    assert public.identity is None
    ours = Auth("bearer", token=Secret(file="~/token"), trust="cluster").connection(TLS)
    assert ours.token_file == "~/token" and ours.ca == "~/ca.pem" and ours.certificate is None


def test_no_auth_and_vendor_auth_need_no_connection_settings() -> None:
    assert Auth("none").connection() == Auth("vendor", key=Secret(env="TINKER_API_KEY")).connection()


@pytest.mark.parametrize(
    ("kwargs", "says"),
    [
        ({"kind": "bearer"}, "names its token"),
        ({"kind": "none", "token": Secret(env="X")}, "only bearer auth has a token"),
        ({"kind": "bearer", "token": Secret(env="X"), "identity": "spiffe://x"}, "only over mutual TLS"),
        ({"kind": "mtls", "key": Secret(env="X")}, "only vendor auth names a key"),
        ({"kind": "password"}, "auth is one of"),
    ],
)
def test_an_auth_that_says_too_little_or_too_much_is_refused(kwargs: dict[str, object], says: str) -> None:
    with pytest.raises(ValueError, match=says):
        Auth(**kwargs)  # pyright: ignore[reportArgumentType]


def test_a_secret_is_one_name_resolved_on_use(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_TOKEN", "value")
    assert Secret(env="SOME_TOKEN").resolve() == "value"
    assert Secret(env="UNSET_TOKEN").resolve() is None
    assert Secret(file="/nowhere/at/all").resolve() is None
    assert "value" not in repr(Secret(env="SOME_TOKEN")) and str(Secret(env="SOME_TOKEN")) == "$SOME_TOKEN"
    with pytest.raises(ValueError, match="one environment variable or one file"):
        Secret(env="A", file="b")


def test_what_is_on_this_machine() -> None:
    assert is_local("http://127.0.0.1:8000") and is_local("localhost") and is_local("http://[::1]:80")
    assert is_local("127.0.0.1")
    assert not is_local("http://gpu-1:8000") and not is_local("0.0.0.0") and not is_local("https://10.0.0.2")
