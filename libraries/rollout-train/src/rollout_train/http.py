"""What an HTTP service's error answer says, whichever service (an engine's server, the gateway) it is."""

from typing import Any, cast

import httpx


def error_of(response: httpx.Response) -> dict[str, Any]:
    """The `error` of an answer's JSON as an object (its `type`, `message` and what else it says; an error given as
    text is its `message`); empty for an answer that says none, or is no JSON object (a proxy's own page, say)."""
    try:
        said: Any = response.json()
    except ValueError:
        return {}
    error: Any = cast(dict[str, Any], said).get("error") if isinstance(said, dict) else None
    if isinstance(error, dict):
        return cast(dict[str, Any], error)
    return {"message": str(error)} if error else {}
