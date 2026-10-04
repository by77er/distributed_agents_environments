"""A run's settings, as a schema: every key a run takes, its type and default, whether it is fixed when the run starts
or changeable from its next step on, and which kinds of run take it.

A run's settings are a flat mapping of dotted keys (`trainer.provider`, `channels.policy.model`, `evals.suite`).
Keys whose meaning did not change keep their names (`groups_per_step`, `max_lag`, `evals.*`,
`channels.NAME.thinking_tokens`), so a start written before this schema compares key by key. A key may have a part
that names something (`channels.NAME.provider`, `slots.SLOT`), written `*` in the schema (`KEYS`). Keys under
`trainer.` other than `trainer.provider`, `trainer.model` and `trainer.channel` are the trainer's own settings, which
its settings dataclass declares (`rollout_train.providers.settings_of`): fixed (`trainer.rank`) or changeable
(`trainer.learning_rate`).

Settings are given in layers, each over the last (`layered`): the schema's defaults, then a preset
(`rollout_train.presets`), then a file (`from_file`: TOML or JSON, dotted keys or tables), then the command line
(`from_flags`: `--set KEY=VALUE`, and `shortcuts` for `--model`, `--provider`, `--renderer`, `--trainer`). `RunSettings`
holds the result and answers with defaults; `diff` says what changed between two. Whether the settings are right is
`rollout_train.validation.check`'s to say, with every other rule.
"""

import json
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue

from rollout_train.providers import ROUTING, SettingSpec

__all__ = [
    "KEYS",
    "KINDS",
    "Change",
    "Key",
    "RunSettings",
    "diff",
    "flattened",
    "from_file",
    "from_flags",
    "is_trainers",
    "key_of",
    "layered",
    "recorded",
    "shortcuts",
]

KINDS = ("train", "eval", "imitate", "check")
"""The kinds of run: training, an eval of one subject, supervised steps on a dataset, an environment's check."""
TRAINED = frozenset({"train"})
TRAINING = frozenset({"train", "imitate"})
SAMPLING = frozenset({"train", "eval", "check"})
EVERY = frozenset(KINDS)


@dataclass(frozen=True)
class Key:
    """One key of the schema."""

    pattern: str
    """Dotted, `*` for a part that names something (`channels.*.model`)."""
    types: tuple[str, ...]
    """The JSON types it takes: `int`, `float`, `bool`, `str`, `null`, `list`, `table`."""
    default: JsonValue
    changeable: bool
    kinds: frozenset[str]
    says: str
    least: float | None = None
    """The smallest number it takes."""
    choices: tuple[str, ...] = ()
    """The strings it takes, where it takes only some."""
    above: bool = False
    """`least` itself is not taken (a share above 0)."""

    def problem(self, value: JsonValue) -> str | None:
        """What is wrong with `value` for this key, in words; none when nothing is."""
        if not _typed(value, self.types):
            return f"is {_spoken(self.types)}, not {json.dumps(value)}"
        if isinstance(value, int | float) and not isinstance(value, bool) and self.least is not None:
            if value < self.least or (self.above and value == self.least):
                return f"is {'above' if self.above else 'at least'} {_number(self.least)}, not {_number(value)}"
            if "int" in self.types and "float" not in self.types and int(value) != value:
                return f"is a whole number, not {value}"
        if isinstance(value, str) and self.choices and value not in self.choices:
            return f"is one of {', '.join(self.choices)}, not {value!r}"
        return None


def _typed(value: JsonValue, types: Sequence[str]) -> bool:
    if value is None:
        return "null" in types
    if isinstance(value, bool):
        return "bool" in types
    if isinstance(value, int):
        return "int" in types or "float" in types
    if isinstance(value, float):
        return "float" in types or ("int" in types and value.is_integer())
    if isinstance(value, str):
        return "str" in types
    if isinstance(value, list):
        return "list" in types
    return "table" in types


_SPOKEN = {
    "int": "a whole number", "float": "a number", "bool": "true or false", "str": "text", "null": "null",
    "list": "a list", "table": "a table",
}  # fmt: skip


def _spoken(types: Sequence[str]) -> str:
    named = [_SPOKEN[each] for each in types if not (each == "int" and "float" in types)]
    return " or ".join(named)


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


_I, _F, _S, _N, _B = ("int",), ("float",), ("str",), ("null",), ("bool",)
KEYS: tuple[Key, ...] = (
    # Fixed: decided when the run starts.
    Key("kind", _S, "train", False, EVERY, "The kind of run", choices=KINDS),
    Key("name", _S + _N, None, False, EVERY, "What the run is called (the launch's name); never kept in a preset"),
    Key("environment", _S + _N, None, False, SAMPLING, "The environment, `module:name`"),
    Key("groups", _I, 100, False, TRAINED | {"check"}, "Groups it plays", least=1),
    Key("seed", _I, 0, False, EVERY, "The seed its draws start from", least=0),
    Key("start", _S + _N, None, False, EVERY, "The checkpoint it trains from or evaluates; none: the base model"),
    Key("bookmark", _S + _N, None, False, TRAINING, "A bookmark it moves to each checkpoint it makes"),
    Key("episodes_at_once", _I, 6, False, SAMPLING, "Episodes it keeps work waiting for", least=1),
    Key("trainer.provider", _S + _N, None, False, TRAINING, "The trainer, a `[trainers.NAME]` of the cluster"),
    Key("trainer.channel", _S, "policy", False, TRAINED, "The trained channel"),
    Key("trainer.model", _S + _N, None, False, TRAINING, "What the trainer trains over; none: the trained channel's"),
    Key("channels.*.provider", _S + _N, None, False, SAMPLING, "What samples the channel, an `[inference.NAME]`"),
    Key("channels.*.providers", ("list", "null"), None, False, SAMPLING, "Several providers serving it, in order"),
    Key("channels.*.routing", _S, "spill", False, SAMPLING, "How turns are shared among them", choices=ROUTING),
    Key("channels.*.weights", ("table", "null"), None, False, SAMPLING, "Each provider's weight, for `weighted`"),
    Key("channels.*.model", _S + _N, None, False, SAMPLING, "The model it serves, among its providers'"),
    Key("channels.*.renderer", _S + _N, None, False, SAMPLING, "The renderer, `module:name`"),
    Key("channels.*.thinking_tokens", _I + _N, None, False, SAMPLING, "Thinking budget per turn", least=1),
    Key("channels.*.answer_tokens", _I + _N, None, False, SAMPLING, "Room for the answer after it", least=1),
    Key("channels.*.replicas", _I + _N, None, False, SAMPLING, "Engine hosts; none: the provider's", least=1),
    Key("channels.*.bridge", _S, "auto", False, SAMPLING, "The bridge", choices=("auto", "merge-quantize")),
    Key(
        "channels.*.mode",
        _S + _N,
        None,
        False,
        SAMPLING,
        "`fixed` or `follows`; none: the trained channel serves what the run trains, another serves `fixed`",
        choices=("fixed", "follows"),
    ),
    Key("channels.*.checkpoint", _S + _N, None, False, SAMPLING, "What a `fixed` channel serves; none: the base model"),
    Key("channels.*.follows", _S + _N, None, False, SAMPLING, "The channel a `follows` channel follows"),
    Key("channels.*.lag", _I, 0, False, SAMPLING, "How many checkpoints behind it follows", least=0),
    Key("slots.*", _S, None, False, SAMPLING, "The channel a program's slot samples"),
    Key("distill.channel", _S + _N, None, False, TRAINED, "The teacher's channel, for distillation"),
    Key("distill.k", _I + _N, None, False, TRAINED, "Top-k logprobs matched; none: the teacher scores", least=1),
    Key("eval.suite", _S + _N, None, False, frozenset({"eval"}), "The suite an eval plays, by name or `NAME@N`"),
    Key("eval.episodes", _I + _N, None, False, frozenset({"eval"}), "Episodes of each start", least=1),
    Key("check.episodes", _I, 1, False, frozenset({"check"}), "Scripted episodes a check plays", least=1),
    Key("imitation.dataset", _S + _N, None, False, frozenset({"imitate"}), "The dataset, by name or id"),
    Key("imitation.limit", _I + _N, None, False, frozenset({"imitate"}), "At most this many segments", least=1),
    Key("imitation.passes", _I, 1, False, frozenset({"imitate"}), "Passes over the dataset", least=1),
    Key("imitation.warmup", _I, 0, False, frozenset({"imitate"}), "Warm-up updates", least=0),
    Key("imitation.resume_optimizer", _B, False, False, frozenset({"imitate"}), "Go on from the start's optimizer"),
    Key("imitation.without", ("list",), [], False, frozenset({"imitate"}), "Datasets whose segments are left out"),
    # Changeable: taken from the next step on.
    Key("groups_per_step", _I, 4, True, TRAINED, "Groups a step waits for", least=1),
    Key("max_lag", _I, 1, True, TRAINED, "Checkpoints behind the newest a turn may begin", least=0),
    Key("evals.suite", _S + _N, None, True, TRAINED, "The suite its checkpoints play, by name or `NAME@N`"),
    Key("evals.every", _I, 1, True, TRAINED, "Every this many steps", least=1),
    Key("evals.episodes", _I + _N, None, True, TRAINED, "Episodes of each start; none: the suite's", least=1),
    Key(
        "limits.spend",
        ("float", "null"),
        None,
        True,
        TRAINING,
        "Dollars: the run ends once its estimate reaches this",
        least=0,
    ),
    Key("share", ("float",), 1.0, True, SAMPLING, "Its weight in a shared pool's fair shares", least=0, above=True),
)
"""Every key a run takes, beside the trainer's own (`trainer.FIELD`)."""
TRAINER = "trainer."
_TRAINER_OWN = ("trainer.provider", "trainer.channel", "trainer.model")


def key_of(key: str) -> Key | None:
    """The schema's key that `key` is (`channels.policy.model` is `channels.*.model`); none for a key that is not in
    it (a trainer's own setting among them)."""
    parts = key.split(".")
    for each in KEYS:
        pattern = each.pattern.split(".")
        matches = len(pattern) == len(parts) and all(p in ("*", part) for p, part in zip(pattern, parts, strict=True))
        if matches and all(parts):
            return each
    return None


def is_trainers(key: str) -> bool:
    """Whether `key` is one of the trainer's own settings (`trainer.rank`)."""
    return key.startswith(TRAINER) and key not in _TRAINER_OWN and key.count(".") == 1


@dataclass(frozen=True)
class RunSettings:
    """A run's settings, as given (`values`), answering with the schema's defaults for what was not."""

    values: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])

    def __getitem__(self, key: str) -> JsonValue:
        if key in self.values:
            return self.values[key]
        found = key_of(key)
        if found is None:
            raise KeyError(key)
        return found.default

    def get(self, key: str, default: JsonValue = None) -> JsonValue:
        """A key's value: as given, else the schema's default, else `default` (a trainer's own setting)."""
        try:
            return self[key]
        except KeyError:
            return default

    @property
    def kind(self) -> str:
        return str(self["kind"])

    @property
    def trained(self) -> str | None:
        """The trained channel, for a training run."""
        return str(self["trainer.channel"]) if self.kind == "train" else None

    @property
    def channels(self) -> list[str]:
        """Every channel the settings name, in the order first named."""
        names: list[str] = []
        for key in self.values:
            if key.startswith("channels.") and key.count(".") == 2 and (name := key.split(".")[1]) not in names:
                names.append(name)
        if self.trained is not None and self.trained not in names:
            names.insert(0, self.trained)
        return names

    def providers(self, channel: str) -> tuple[str, ...]:
        """The providers of a channel, in order (`providers`, or the one `provider`)."""
        several = self[f"channels.{channel}.providers"]
        if isinstance(several, list):
            return tuple(str(each) for each in several)
        one = self[f"channels.{channel}.provider"]
        return (str(one),) if one is not None else ()

    @property
    def trainer_model(self) -> str | None:
        """What the trainer trains over: `trainer.model`, else the trained channel's model."""
        said = self["trainer.model"]
        if said is not None:
            return str(said)
        trained = self["trainer.channel"]
        model = self.get(f"channels.{trained}.model")
        return str(model) if model is not None else None

    def mode(self, channel: str) -> str:
        """`trained` for the trained channel, else its `mode` (`fixed` by default)."""
        if channel == self.trained:
            return "trained"
        return str(self[f"channels.{channel}.mode"] or "fixed")

    def split(self, trainer: Sequence[SettingSpec] = ()) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
        """The settings in full (every key of the schema for the run's kind and channels, with defaults, and the
        trainer's own with theirs), as fixed ones and changeable ones."""
        fixed: dict[str, JsonValue] = {}
        changeable: dict[str, JsonValue] = {}
        for each in KEYS:
            if self.kind not in each.kinds:
                continue
            if each.pattern.startswith("channels.*."):
                keys = [each.pattern.replace("*", name, 1) for name in self.channels]
            elif "*" in each.pattern:
                keys = [key for key in self.values if key_of(key) is each]
            else:
                keys = [each.pattern]
            for key in keys:
                (changeable if each.changeable else fixed)[key] = self[key]
        if self.kind in TRAINING:
            for spec in trainer:
                (changeable if spec.changeable else fixed)[spec.key] = self.values.get(spec.key, spec.default)
        for key, value in self.values.items():  # (what the schema does not know is kept, for validation to say)
            if key not in fixed and key not in changeable:
                fixed[key] = value
        return fixed, changeable


@dataclass(frozen=True)
class Change:
    """One key that differs between two settings: `before` or `after` is absent where the key was not given."""

    key: str
    before: JsonValue = None
    after: JsonValue = None
    added: bool = False
    removed: bool = False


def diff(before: Mapping[str, JsonValue], after: Mapping[str, JsonValue]) -> list[Change]:
    """What changed from `before` to `after`, key by key, in key order."""
    changes: list[Change] = []
    for key in sorted(set(before) | set(after)):
        if key not in before:
            changes.append(Change(key, after=after[key], added=True))
        elif key not in after:
            changes.append(Change(key, before=before[key], removed=True))
        elif before[key] != after[key]:
            changes.append(Change(key, before[key], after[key]))
    return changes


def layered(*layers: Mapping[str, JsonValue] | None) -> RunSettings:
    """Settings given in layers, each over the ones before (a preset's, a file's, the flags'): a key of a later layer
    replaces the same key of an earlier one."""
    values: dict[str, JsonValue] = {}
    for layer in layers:
        values.update(layer or {})
    return RunSettings(values)


def from_flags(given: Sequence[str]) -> dict[str, JsonValue]:
    """`--set KEY=VALUE` flags, in order (a later one wins). A value is read as JSON (`null`, `3e-5`, `true`,
    `["a", "b"]`, `"text"`), else as TOML (`{ a = 1 }`), else as the text it is (`rollout_qwen:qwen35`)."""
    values: dict[str, JsonValue] = {}
    for each in given:
        key, equals, value = each.partition("=")
        if not equals or not key.strip():
            raise ValueError(f"--set {each!r}: it should be KEY=VALUE")
        values[key.strip()] = _value(value.strip())
    return values


def _value(text: str) -> JsonValue:
    try:
        return json.loads(text)
    except ValueError:
        pass
    try:
        return tomllib.loads(f"value = {text}")["value"]
    except tomllib.TOMLDecodeError:
        return text


def from_file(path: Path) -> dict[str, JsonValue]:
    """Settings from a file: JSON (`.json`, where `null` unsets a key) or TOML, of dotted keys (`"trainer.rank" =
    16`) or tables (`[trainer] rank = 16`), or both."""
    text = path.read_text()
    loaded: Any = json.loads(text) if path.suffix == ".json" else tomllib.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: settings are a table of keys")
    return flattened(cast(dict[str, Any], loaded))


def flattened(table: Mapping[str, Any], prefix: str = "") -> dict[str, JsonValue]:
    """Nested tables as dotted keys, down to a key the schema takes a table for (`channels.NAME.weights`)."""
    flat: dict[str, JsonValue] = {}
    for name, value in table.items():
        key = f"{prefix}{name}"
        found = key_of(key)
        if isinstance(value, dict) and not (found is not None and "table" in found.types):
            flat |= flattened(cast(dict[str, Any], value), f"{key}.")
        else:
            flat[key] = value
    return flat


def shortcuts(
    *,
    model: str | None = None,
    provider: str | None = None,
    renderer: str | None = None,
    trainer: str | None = None,
    channel: str = "policy",
) -> dict[str, JsonValue]:
    """What `--model`, `--provider`, `--renderer` and `--trainer` set, for the channel `--channel` names."""
    said: dict[str, JsonValue] = {}
    for name, value in (("model", model), ("provider", provider), ("renderer", renderer)):
        if value is not None:
            said[f"channels.{channel}.{name}"] = value
    if trainer is not None:
        said["trainer.provider"] = trainer
    return said


def recorded(
    settings: RunSettings, trainer: Sequence[SettingSpec] = (), preset: str | None = None
) -> dict[str, JsonValue]:
    """What a run's start records of its settings: a full copy, fixed and changeable (every key with its value,
    defaults included), and, as provenance only, the preset version they came from (`NAME@N`)."""
    fixed, changeable = settings.split(trainer)
    return {"fixed": fixed, "changeable": changeable, "preset": preset}
