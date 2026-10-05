"""How the ledger service and `HttpLedger` say things to each other: each operation is `POST /v1/STORE/OPERATION` with
`{"args": {...}, "request": ID}`, answered with `{"result": ...}`, or an error `{"error": CODE, "message": ...}` whose
code is one of `ERRORS`. Records are JSON as the stores beside a ledger keep them."""

import json
from dataclasses import asdict
from typing import Any

from pydantic import JsonValue

from rollout.harness.sandboxes import Lease
from rollout_train.launches import Launch, as_launch
from rollout_train.ledger import Fenced
from rollout_train.ledger_service.scopes import Forbidden
from rollout_train.presence import Beat
from rollout_train.presets import Preset
from rollout_train.published import EnvironmentVersion, as_version
from rollout_train.registry import Bookmark, Entry, Named, SuiteName, Taken
from rollout_train.settings import Desired

__all__ = ["ERRORS", "Conflict", "code_of", "decoded", "encoded"]


class Conflict(Exception):
    """A compare-and-set that found the row changed since it was read."""


ERRORS: dict[str, type[Exception]] = {
    "fenced": Fenced,
    "taken": Taken,
    "missing": KeyError,
    "invalid": ValueError,
    "forbidden": Forbidden,
    "conflict": Conflict,
}
"""Each error code, and the error the client raises for it."""
STATUS = {"fenced": 409, "taken": 409, "missing": 404, "invalid": 400, "forbidden": 403, "conflict": 409}


def code_of(error: BaseException) -> str | None:
    """The code an error is answered with; None for one that is the service's own failure."""
    return next((code for code, kind in ERRORS.items() if isinstance(error, kind)), None)  # (Taken before ValueError)


def message_of(error: BaseException) -> str:
    """What an error says (a `KeyError`'s message without the quotes `str` puts around it)."""
    return str(error.args[0]) if isinstance(error, KeyError) and error.args else str(error)


def encoded(value: Any) -> JsonValue:
    """A store's answer as JSON."""
    if isinstance(value, list):
        return [encoded(each) for each in value]  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, Lease):
        return value.model_dump(mode="json")
    if isinstance(value, EnvironmentVersion):
        return value.to_json()
    if isinstance(value, Launch):
        return json.loads(json.dumps(asdict(value)))
    if isinstance(value, Beat | Desired | Preset | Entry | Bookmark | Named | SuiteName):
        return json.loads(json.dumps(asdict(value)))
    if hasattr(value, "__dataclass_fields__"):
        return json.loads(json.dumps(asdict(value)))
    return value


def decoded[T](kind: type[T], value: Any) -> T:
    """JSON as one of the stores' records."""
    if kind is Lease:
        return Lease.model_validate(value)  # pyright: ignore[reportReturnType]
    if kind is EnvironmentVersion:
        return as_version(value)  # pyright: ignore[reportReturnType]
    if kind is Launch:
        return as_launch(value)  # pyright: ignore[reportReturnType]
    return kind(**value)
