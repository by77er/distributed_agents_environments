"""Things named in configuration as `module:name`."""

import importlib
from typing import Any


def named(reference: str) -> Any:
    """What `module:name` names (`name` may be dotted: an attribute of an attribute)."""
    module, _, name = reference.partition(":")
    if not name:
        raise ValueError(f"{reference!r} names nothing: it should be module:name")
    value: Any = importlib.import_module(module)
    for part in name.split("."):
        value = getattr(value, part)
    return value
