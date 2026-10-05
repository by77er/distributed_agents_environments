"""What an HTTP service's answer says, whichever service (an engine's server, the gateway, a trainer pod) it is: its
JSON object (`answer_of`) and its error (`error_of`)."""

from typing import Any, cast

import httpx


def answer_of(response: httpx.Response) -> dict[str, Any]:
    """An answer's JSON object; empty for an answer that is no JSON object (a proxy's own page, say)."""
    try:
        said: Any = response.json()
    except ValueError:
        return {}
    return cast(dict[str, Any], said) if isinstance(said, dict) else {}


def error_of(response: httpx.Response) -> dict[str, Any]:
    """The `error` of an answer's JSON as an object (its `type`, `message` and what else it says; an error given as
    text is its `message`); empty for an answer that says none, or is no JSON object (a proxy's own page, say)."""
    error: Any = answer_of(response).get("error")
    if isinstance(error, dict):
        return cast(dict[str, Any], error)
    return {"message": str(error)} if error else {}
