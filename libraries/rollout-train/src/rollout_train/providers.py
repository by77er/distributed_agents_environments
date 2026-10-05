"""Inference providers and trainers, declared: what each kind can do, how it is reached, and what a cluster's
deployment of it offers.

An **inference provider** samples a channel. Its kind (`INFERENCE_KINDS`: `vllm`, `vllm-servers`, `tinker`, `api`,
`runpod-inference`) declares its `Capabilities`: whether it is token-exact, returns sampled-token, prompt and top-k
logprobs, honours sampling parameters, serves adapters by name or reloads full weights, streams, how it bills, and the
checkpoint formats it loads. The cluster config (`rollout_train.cluster`) adds what this deployment has: its models
(`ModelOffer`: context, the base a quantized model was made from, the highest LoRA rank, cost per token class), its
GPUs and replicas, and, for a provider shared by several runs, its adapter slots (`SharedPool`).

A **trainer** makes checkpoints. Its kind (`TRAINER_KINDS`: `lora`, `full`, `tinker`, `runpod-trainer`) declares its
`TrainerCapabilities`: what it produces (`lora` or `full`), the format its files are in (`peft`, `full`, `tinker`), the
objectives it takes, whether it scores given tokens, and the formats a run may start from. The cluster config adds its
models, the longest segment this hardware trains on, its GPUs, colocation and cost. A trainer's own settings (its
rank, learning rate, clips) are read from its settings dataclass by `settings_of`, without importing torch.

Every provider declares how it is reached (`Auth`): `mtls` (the cluster's CA, a client certificate, the server's SPIFFE
identity checked), `bearer` (a token named by an environment variable or a file), `vendor` (the vendor's SDK reads its
own key, as Tinker's does) or `none` (allowed only for an endpoint on this machine). `Auth.connection` turns that into
the connection settings a client uses (`rollout_train.inference.remote.Connection`): the system's CAs or the
cluster's, the host name checked for a public endpoint or a SPIFFE identity for a pod reached by IP, a client
certificate only for `mtls`.

A channel may be served by several providers at once (`Routing`): `spill` fills the first and sends the rest to the
next; `weighted` shares turns by weight.

Everything here is a declaration: nothing is imported, started or reached.
"""

import dataclasses
import types
import typing
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit

from pydantic import JsonValue

from rollout_train.inference.remote import Connection
from rollout_train.recorder.segments import TOKEN_LEVEL

__all__ = [
    "AUTHS",
    "INFERENCE_KINDS",
    "OBJECTIVES",
    "ROUTING",
    "TRAINER_KINDS",
    "Auth",
    "Capabilities",
    "InferenceKind",
    "InferenceProvider",
    "ModelOffer",
    "Routing",
    "Secret",
    "SettingSpec",
    "SharedPool",
    "Tls",
    "TrainerCapabilities",
    "TrainerKind",
    "TrainerProvider",
    "is_local",
    "settings_of",
]

type AuthKind = Literal["mtls", "bearer", "vendor", "none"]
AUTHS: tuple[AuthKind, ...] = ("mtls", "bearer", "vendor", "none")
"""How a provider is reached: mutual TLS with the cluster's CA, a bearer token, the vendor's SDK, or nothing (only on
this machine)."""
BEATS = "beats"
"""An `Auth.identity` that says each server's SPIFFE identity is the one its heartbeat names (a RunPod pod's)."""
OBJECTIVES = ("policy_gradient/token", "policy_gradient/segment", "likelihood")
"""The objectives a trainer may take: the clipped policy gradient with a ratio per token (PPO) or per segment (GSPO),
and likelihood (imitation)."""
ROUTING = ("spill", "weighted")
"""How turns are shared among a channel's providers: fill the first and spill the rest over to the next, or by
weight."""


@dataclass(frozen=True)
class Secret:
    """A secret, named: an environment variable (`env`) or a file (`file`). The value is read where it is used, at the
    moment it is needed (`resolve`), and never kept."""

    env: str | None = None
    file: str | None = None

    def __post_init__(self) -> None:
        if (self.env is None) == (self.file is None):
            raise ValueError("a secret is named by one environment variable or one file")

    def resolve(self, environ: Mapping[str, str] | None = None) -> str | None:
        """The secret's value, read now (none where the variable is unset or empty, or the file is missing)."""
        import os
        from pathlib import Path

        if self.env is not None:
            return (os.environ if environ is None else environ).get(self.env) or None
        path = Path(str(self.file)).expanduser()
        return (path.read_text().strip() or None) if path.exists() else None

    def __str__(self) -> str:
        return f"${self.env}" if self.env is not None else f"file {self.file}"


@dataclass(frozen=True)
class Tls:
    """The cluster's own certificate authority and the client certificate its gateway and launcher present
    (the cluster config's `[tls]`): paths, never the keys themselves."""

    ca: str | None = None
    """The cluster CA's root certificate."""
    certificate: str | None = None
    key: str | None = None
    """The client certificate and its key's file, for providers reached over mutual TLS."""
    identity: str = "spiffe://rollout/gateway"
    """The SPIFFE identity the client certificate carries."""


@dataclass(frozen=True)
class Auth:
    """How a provider is reached. `token` for `bearer`; `key` for `vendor` (what the vendor's SDK reads, named so that
    `rollout cluster check` can say whether it resolves); `trust` says whose CAs verify the server (`system` or
    `cluster`; by default the cluster's for `mtls`, the system's otherwise); `identity` is the SPIFFE identity the
    server's certificate must carry (`beats`: each server's own, from its heartbeat), in place of its host name."""

    kind: AuthKind = "none"
    token: Secret | None = None
    key: Secret | None = None
    trust: Literal["system", "cluster"] | None = None
    identity: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in AUTHS:
            raise ValueError(f"auth is one of {', '.join(AUTHS)}, not {self.kind!r}")
        if self.kind == "bearer" and self.token is None:
            raise ValueError("bearer auth names its token: token_env or token_file")
        if self.kind != "bearer" and self.token is not None:
            raise ValueError(f"only bearer auth has a token (this is {self.kind})")
        if self.kind != "vendor" and self.key is not None:
            raise ValueError(f"only vendor auth names a key for its SDK (this is {self.kind})")
        if self.identity is not None and self.kind != "mtls":
            raise ValueError("a server's SPIFFE identity is checked only over mutual TLS (auth mtls)")
        if self.trust not in (None, "system", "cluster"):
            raise ValueError(f"trust is system or cluster, not {self.trust!r}")

    @property
    def trusts(self) -> Literal["system", "cluster"]:
        """Whose CAs verify the server."""
        return self.trust or ("cluster" if self.kind == "mtls" else "system")

    def connection(self, tls: Tls | None = None, *, identity: str | None = None) -> Connection:
        """The settings a client reaches the provider's servers with: the cluster's CA or the system's; a client
        certificate only for `mtls`; the server's SPIFFE identity checked in place of its host name where one is said
        (`identity`, the server's own from its heartbeat when `self.identity` is `beats`); else its host name. Raises
        `ValueError` where the cluster has no `[tls]` it needs."""
        if self.kind in ("none", "vendor"):
            return Connection()
        if self.trusts == "cluster" and (tls is None or tls.ca is None):
            raise ValueError("the cluster's CA is needed ([tls] ca) to verify this provider's servers")
        ca = tls.ca if self.trusts == "cluster" and tls is not None else None
        checked = identity if self.identity == BEATS else self.identity
        if self.identity == BEATS and identity is None:
            raise ValueError("each server's identity comes from its heartbeat: pass the one it names")
        if self.kind == "bearer":
            assert self.token is not None
            return Connection(token_env=self.token.env, token_file=self.token.file, ca=ca, identity=checked)
        if tls is None or tls.certificate is None:
            raise ValueError("mutual TLS needs the cluster's client certificate ([tls] certificate and key)")
        return Connection(ca=ca, certificate=tls.certificate, key=tls.key, identity=checked)


def is_local(address: str) -> bool:
    """Whether an address (a URL, or a host) is on this machine: `localhost` or a loopback address."""
    host = urlsplit(address).hostname if "://" in address else address.rsplit(":", 1)[0].strip("[]")
    return host is not None and (host == "localhost" or host == "::1" or host.startswith("127."))


@dataclass(frozen=True)
class Capabilities:
    """What an inference provider's kind can do."""

    token_exact: bool
    """Takes token ids and returns the exact sampled ids."""
    sampled_logprobs: bool
    """Each sampled token's logprob, under the distribution it was sampled from."""
    prompt_logprobs: bool
    """Logprobs of given tokens (scoring without sampling)."""
    top_logprobs: int
    """How many of the top logprobs it can return per position (0: none)."""
    honours_sampling: bool
    """Temperature and top-p are applied, so recorded logprobs describe what was sampled."""
    adapters: bool
    """Serves LoRA adapters by name."""
    full_reload: bool
    """Serves new full weights under a checkpoint's name."""
    streaming: bool
    """Replies as a stream."""
    loads: frozenset[str]
    """The checkpoint formats it serves (`peft`, `full`, `tinker`); none: base models only."""
    bills: Literal["none", "tokens", "hours"] = "none"
    """What it costs by: nothing, tokens (per model and token class), or hours of pods."""
    unchecked: frozenset[str] = frozenset()
    """Capabilities declared as the SDK says but not yet confirmed by a live test: nothing relies on them until then
    (Tinker's prompt and top-k logprobs)."""

    @property
    def sampled_with(self) -> tuple[str, ...]:
        """What its turns are sampled with, as a turn records it (`rollout_train.recorder.segments.TOKEN_LEVEL`)."""
        return tuple(each for each in TOKEN_LEVEL if getattr(self, each))


@dataclass(frozen=True)
class ModelOffer:
    """A model a provider serves here."""

    model: str
    context: int
    """The longest sequence it takes, prompt and reply."""
    base: str | None = None
    """The model it was quantized from, if any: an adapter trained over the base can be served on it."""
    max_lora_rank: int | None = None
    """The highest adapter rank it loads (none: adapters are not limited here, or not served)."""
    cost: Mapping[str, float] = field(default_factory=dict[str, float])
    """Dollars per million tokens by token class (`input`, `cached_input`, `output`, `thinking`), or per hour
    (`hour`)."""
    options: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """What its engines are started with (`gpu_memory_utilization`, `max_num_seqs`, …)."""


@dataclass(frozen=True)
class SharedPool:
    """A provider shared by several runs: its servers hold every bound run's live checkpoints as adapters side by
    side. A run joins when its rank fits the model's `max_lora_rank` and the pool has free adapter slots for each
    channel it serves there (`max_lag + 1` for the trained channel, 2 for one following it, 1 for a fixed checkpoint);
    turns are shared among its runs by their `share`."""

    adapter_slots: int | None = None
    """Adapters its servers hold at once (none: not limited)."""
    max_runs: int | None = None
    """Runs bound to it at once (none: as many as slots allow)."""


@dataclass(frozen=True)
class Routing:
    """How a channel served by several providers shares its turns among them."""

    rule: Literal["spill", "weighted"] = "spill"
    providers: tuple[str, ...] = ()
    """In order: for `spill`, the first is filled before the next is asked."""
    weights: Mapping[str, float] = field(default_factory=dict[str, float])
    """For `weighted`: each provider's weight."""


@dataclass(frozen=True)
class InferenceKind:
    """What a kind of inference provider is, whatever cluster it is in."""

    name: str
    capabilities: Capabilities
    auths: tuple[AuthKind, ...]
    """The ways it may be reached."""
    auth: Auth | None
    """How it is reached when the cluster config does not say (none: the config must say)."""
    fields: tuple[str, ...]
    """The settings of its `[inference.NAME]` table beyond those every provider has."""
    secrets: tuple[str, ...] = ()
    """The secrets its table may name (`NAME_env`, `NAME_file`), beyond its auth's."""
    implementation: str | None = None
    """`module:name` of what samples it, where one module does."""
    shared: bool = False
    """Whether runs share its servers as a pool (`SharedPool`)."""
    remote: bool = False
    """Whether its servers are reached at addresses (and so need an auth other than `none` unless local)."""


def _token_level(
    *,
    prompt_logprobs: bool,
    top_logprobs: int,
    full_reload: bool,
    loads: frozenset[str],
    bills: Literal["none", "tokens", "hours"] = "none",
    unchecked: frozenset[str] = frozenset(),
) -> Capabilities:
    """The capabilities of a provider that samples tokens: token-exact, with sampled-token logprobs, honouring
    sampling, adapters by name, not streamed."""
    return Capabilities(
        token_exact=True,
        sampled_logprobs=True,
        prompt_logprobs=prompt_logprobs,
        top_logprobs=top_logprobs,
        honours_sampling=True,
        adapters=True,
        full_reload=full_reload,
        streaming=False,
        loads=loads,
        bills=bills,
        unchecked=unchecked,
    )


INFERENCE_KINDS: Mapping[str, InferenceKind] = {
    each.name: each
    for each in (
        InferenceKind(
            "vllm",
            _token_level(
                prompt_logprobs=True,  # (`VllmEngine.score`)
                top_logprobs=20,  # (vLLM's default `max_logprobs`; the provider's `max_logprobs` starts its engines)
                full_reload=True,
                loads=frozenset({"peft", "full"}),
            ),
            auths=("none", "bearer", "mtls"),
            auth=Auth("none"),
            fields=("engine", "listen", "max_logprobs", "pool"),
            implementation="rollout_vllm:VllmEngine",
            shared=True,
        ),
        InferenceKind(
            "vllm-servers",
            _token_level(
                prompt_logprobs=True,
                top_logprobs=20,
                full_reload=False,
                loads=frozenset({"peft"}),
            ),
            auths=("none", "bearer", "mtls"),
            auth=None,
            fields=("addresses", "via", "loader", "max_logprobs", "pool"),
            implementation="rollout_train.inference:RemoteEngine",
            shared=True,
            remote=True,
        ),
        InferenceKind(
            "tinker",
            _token_level(
                # (its SDK, 0.32, takes `include_prompt_logprobs` and a top k at prompt and sampled positions, the
                # width bounded by its server, which the SDK does not state)
                prompt_logprobs=True,
                top_logprobs=20,
                full_reload=False,
                loads=frozenset({"tinker"}),
                bills="tokens",
                unchecked=frozenset({"prompt_logprobs", "top_logprobs"}),
            ),
            auths=("vendor",),
            auth=Auth("vendor", key=Secret(env="TINKER_API_KEY")),
            fields=("project",),
            secrets=("project",),
            implementation="rollout_tinker:TinkerEngine",
        ),
        InferenceKind(
            "api",
            Capabilities(
                token_exact=False,
                sampled_logprobs=False,
                prompt_logprobs=False,
                top_logprobs=0,
                honours_sampling=False,
                adapters=False,
                full_reload=False,
                streaming=True,
                loads=frozenset(),
                bills="tokens",
            ),
            auths=("vendor", "bearer"),
            auth=None,
            fields=("endpoint",),
        ),
        InferenceKind(
            "runpod-inference",
            _token_level(
                prompt_logprobs=True,
                top_logprobs=20,
                full_reload=False,
                loads=frozenset({"peft"}),
                bills="hours",
            ),
            auths=("mtls",),
            auth=Auth("mtls", identity=BEATS),
            fields=(
                "image",
                "gpu_types",
                "pods",
                "idle_stop",
                "volume_gb",
                "secrets",
                "step_ca",
                "max_logprobs",
                "pool",
            ),
            secrets=("api_key",),
            implementation="rollout_train.pods.inference:InferencePod",
            shared=True,
            remote=True,
        ),
    )
}
"""Every kind of inference provider, by name."""


@dataclass(frozen=True)
class InferenceProvider:
    """An inference provider as a cluster deploys it (`[inference.NAME]`)."""

    name: str
    kind: str
    capabilities: Capabilities
    """The kind's, with this deployment's `max_logprobs` as its top-k logprobs: what a `vllm` provider's engines are
    started with, and what a `vllm-servers` or `runpod-inference` provider's servers were (`--max-logprobs`)."""
    models: Mapping[str, ModelOffer]
    auth: Auth
    gpus: float = 0
    """Per replica."""
    replicas: int = 1
    """Per run channel, unless the run asks for more (`channels.NAME.replicas`)."""
    pool: SharedPool | None = None
    """Set for a provider whose servers runs share."""
    endpoints: tuple[str, ...] = ()
    """Where its servers are reached (addresses, a router, where local engines listen)."""
    settings: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """The rest of its table: its kind's own settings, none of them a secret."""
    secrets: Mapping[str, Secret] = field(default_factory=dict[str, Secret])
    """The secrets its table names, beyond its auth's (RunPod's API key, Tinker's project)."""

    @property
    def local(self) -> bool:
        """Whether every endpoint it is reached at is on this machine."""
        return all(is_local(each) for each in self.endpoints)


@dataclass(frozen=True)
class TrainerCapabilities:
    """What a kind of trainer makes and takes."""

    produces: Literal["lora", "full"]
    format: str
    """The checkpoint format its files are in: `peft`, `full` or `tinker`."""
    objectives: frozenset[str]
    """Among `OBJECTIVES`."""
    scores: bool
    """Can compute logprobs of given tokens (for distillation, and supervised data without behaviour logprobs)."""
    starts_from: frozenset[str]
    """Checkpoint formats a run may start from (besides the base model)."""


@dataclass(frozen=True)
class TrainerKind:
    """What a kind of trainer is, whatever cluster it is in."""

    name: str
    capabilities: TrainerCapabilities
    implementation: str
    """`module:name` of the trainer."""
    settings: str
    """`module:name` of its settings dataclass: each field a setting `trainer.FIELD`, the module's `CHANGEABLE` the
    ones it takes between steps."""
    auths: tuple[AuthKind, ...]
    auth: Auth
    fields: tuple[str, ...] = ()
    """The settings of its `[trainers.NAME]` table beyond those every trainer has."""
    secrets: tuple[str, ...] = ()
    not_settings: Mapping[str, str] = field(default_factory=dict[str, str])
    """Fields of its settings dataclass a run does not set, and why."""


_EVERY_OBJECTIVE = frozenset(OBJECTIVES)
_LORA = TrainerCapabilities("lora", "peft", _EVERY_OBJECTIVE, True, frozenset({"peft", "full"}))
_FULL = TrainerCapabilities("full", "full", _EVERY_OBJECTIVE, True, frozenset({"full"}))
TRAINER_KINDS: Mapping[str, TrainerKind] = {
    each.name: each
    for each in (
        TrainerKind(
            "lora",
            _LORA,
            "rollout_lora:LoraTrainer",
            "rollout_lora.settings:LoraSettings",
            auths=("none",),
            auth=Auth("none"),
        ),
        TrainerKind(
            "full",
            _FULL,
            "rollout_lora:FullTrainer",
            "rollout_lora.settings:LoraSettings",
            auths=("none",),
            auth=Auth("none"),
            not_settings={"rank": "a full-weight trainer has no adapter"},
        ),
        TrainerKind(
            "tinker",
            TrainerCapabilities("lora", "tinker", _EVERY_OBJECTIVE, True, frozenset({"tinker"})),
            "rollout_tinker:TinkerTrainer",
            "rollout_tinker.settings:TinkerSettings",
            auths=("vendor",),
            auth=Auth("vendor", key=Secret(env="TINKER_API_KEY")),
            fields=("project",),
            secrets=("project",),
            not_settings={"project": "the cluster config says it ([trainers.NAME] project)"},
        ),
        TrainerKind(
            "runpod-trainer",
            _LORA,  # (as the trainer it runs: `trainer = "full"` gives `_FULL`'s)
            "rollout_train.pods:RemoteTrainer",
            "rollout_lora.settings:LoraSettings",
            auths=("mtls",),
            auth=Auth("mtls", identity=BEATS),
            fields=("trainer", "image", "gpu_types", "pods", "idle_stop", "volume_gb", "secrets", "step_ca"),
            secrets=("api_key",),
        ),
    )
}
"""Every kind of trainer, by name."""


@dataclass(frozen=True)
class TrainerProvider:
    """A trainer as a cluster deploys it (`[trainers.NAME]`)."""

    name: str
    kind: str
    capabilities: TrainerCapabilities
    models: tuple[str, ...]
    """The models it trains here."""
    auth: Auth
    segment_tokens: int | None = None
    """The longest segment this hardware trains on (none: any)."""
    gpus: float = 0
    colocate_with: str | None = None
    """A `vllm` provider whose GPU it shares: that provider's engines sleep while it steps."""
    cost: Mapping[str, float] = field(default_factory=dict[str, float])
    """Dollars per million tokens trained (`train`), or per hour (`hour`)."""
    settings: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """The rest of its table: its kind's own settings, none of them a secret."""
    secrets: Mapping[str, Secret] = field(default_factory=dict[str, Secret])

    @property
    def runs(self) -> TrainerKind:
        """The kind of trainer its steps are taken by (for `runpod-trainer`, the one it names)."""
        if self.kind == "runpod-trainer":
            return TRAINER_KINDS[str(self.settings.get("trainer", "lora"))]
        return TRAINER_KINDS[self.kind]


@dataclass(frozen=True)
class SettingSpec:
    """One setting a trainer takes, read from its settings dataclass."""

    key: str
    """As a run names it: `trainer.FIELD`."""
    types: tuple[str, ...]
    """The JSON types it takes: `int`, `float`, `bool`, `str`, `null`."""
    default: JsonValue
    changeable: bool
    """Whether a running run takes it from its next step on."""

    def accepts(self, value: JsonValue) -> bool:
        """Whether `value` is of a type it takes (a whole number is a float too; a bool is neither)."""
        if value is None:
            return "null" in self.types
        if isinstance(value, bool):
            return "bool" in self.types
        if isinstance(value, int):
            return "int" in self.types or "float" in self.types
        if isinstance(value, float):
            return "float" in self.types
        if isinstance(value, str):
            return "str" in self.types
        return False


_JSON_TYPES: Mapping[object, str] = {int: "int", float: "float", bool: "bool", str: "str", type(None): "null"}


def settings_of(kind: TrainerKind) -> tuple[SettingSpec, ...]:
    """The settings a kind of trainer takes, by `trainer.FIELD`, with their types, defaults and whether they are
    changeable: read from its settings dataclass and the `CHANGEABLE` beside it (importing neither the trainer nor
    torch). Raises `ImportError` where the trainer's package is not installed here."""
    import importlib

    module_name, _, class_name = kind.settings.partition(":")
    module = importlib.import_module(module_name)
    settings = getattr(module, class_name)
    changeable: Collection[str] = getattr(module, "CHANGEABLE", ())
    hints = typing.get_type_hints(settings)
    found: list[SettingSpec] = []
    for each in dataclasses.fields(settings):
        if each.name in kind.not_settings:
            continue
        hint = hints[each.name]
        members = typing.get_args(hint) if isinstance(hint, types.UnionType) else (hint,)
        found.append(
            SettingSpec(
                f"trainer.{each.name}",
                tuple(_JSON_TYPES[member] for member in members if member in _JSON_TYPES),
                each.default if each.default is not dataclasses.MISSING else None,
                each.name in changeable,
            )
        )
    return tuple(found)
