"""A program's model slots, bound to a run's channels: which channel each slot samples, and the bindings a run may not
make.

A run binds a slot to a channel by name (`slots.SLOT`, a run setting). A trained slot the settings do not bind samples
the run's subject channel (`subject`: the trained channel, `trainer.channel`). A slot that is not trained (a judge, a
fixed opponent: `ModelSlot.trained`) has no such default: the run binds it by name, or it is refused. A judge
(`ModelSlot.judge`) bound to a channel that serves the run's own checkpoints (the trained channel, or one that follows
it) judges the policy by itself: the run must say so (`self_judging`). `problems` says what a run's settings break of
these, as `rollout_train.validation` refuses them.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass

from rollout.harness import ModelSlot
from rollout_train.run_settings import RunSettings

__all__ = ["Declared", "bound", "problems", "serving", "subject"]


@dataclass(frozen=True)
class Declared:
    """The slots a program declares: their names, those that are not trained, and those that judge."""

    names: frozenset[str]
    untrained: frozenset[str] = frozenset()
    judges: frozenset[str] = frozenset()

    @classmethod
    def of(cls, slots: Mapping[str, ModelSlot]) -> "Declared":
        return cls(
            frozenset(slots),
            frozenset(name for name, slot in slots.items() if not slot.trained),
            frozenset(name for name, slot in slots.items() if slot.judge),
        )


def subject(settings: RunSettings) -> str:
    """The channel a trained slot the settings do not bind samples: the trained channel."""
    return str(settings["trainer.channel"])


def bound(settings: RunSettings, declared: Declared) -> dict[str, str]:
    """The channel each declared slot samples: the one `slots.SLOT` names, else, for a trained slot, the subject
    channel (an untrained slot the settings do not bind is left out: `problems` refuses it)."""
    found: dict[str, str] = {}
    for slot in sorted(declared.names):
        named = settings.get(f"slots.{slot}")
        if isinstance(named, str):
            found[slot] = named
        elif slot not in declared.untrained:
            found[slot] = subject(settings)
    return found


def serving(settings: RunSettings) -> set[str]:
    """The channels that serve the run's own checkpoints: the trained channel, and those that follow it (directly or
    through others)."""
    trained = settings.trained
    if trained is None:
        return set()
    found: set[str] = set()
    for channel in settings.channels:
        seen: set[str] = set()
        at = channel
        while at not in seen and at != trained and settings.mode(at) == "follows":
            seen.add(at)
            at = str(settings[f"channels.{at}.follows"])
        if at == trained:
            found.add(channel)
    return found


def problems(settings: RunSettings, declared: Declared, own: Collection[str] | None = None) -> list[tuple[str, str]]:
    """What a run's bindings of the declared slots break, as `(key, reason)`: an untrained slot left unbound; a channel
    a slot is bound to with no provider or model (`settings.providers` where the run names providers); a judge bound
    to a channel serving the run's own checkpoints (`own`, by default `serving`) without `self_judging`."""
    found: list[tuple[str, str]] = []
    channels = bound(settings, declared)
    for slot in sorted(declared.untrained - set(channels)):
        found.append((f"slots.{slot}", f"slot {slot} is not trained, so it samples no channel by default: bind it to "
                      f"one (slots.{slot})"))  # fmt: skip
    providing = any(settings.providers(each) for each in settings.channels)  # (settings naming none: as given)
    for channel, slot in {channel: slot for slot, channel in sorted(channels.items(), reverse=True)}.items():
        if channel == subject(settings) or channel not in settings.channels:
            continue  # (the subject's are required of every run; a channel the settings lack is refused for that)
        if providing and not settings.providers(channel):
            found.append((f"channels.{channel}.provider", f"channel {channel}, which slot {slot} samples, needs a "
                          "provider"))  # fmt: skip
        if settings.get(f"channels.{channel}.model") is None:
            found.append((f"channels.{channel}.model", f"channel {channel}, which slot {slot} samples, needs a model"))
    own = serving(settings) if own is None else own
    for slot in sorted(declared.judges & set(channels)):
        channel = channels[slot]
        if channel in own and settings["self_judging"] is not True:
            found.append((f"slots.{slot}", f"slot {slot} judges, and channel {channel} serves the run's own "
                          "checkpoints: the policy would judge itself (self_judging allows it)"))  # fmt: skip
    return found
