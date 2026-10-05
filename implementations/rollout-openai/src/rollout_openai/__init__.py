"""A model endpoint for the OpenAI Responses API, on an API key or a Codex login: an implementation of
`rollout.contracts.ModelEndpoint`, and what a cluster's `api` provider samples OpenAI's models with (`hosted`).
Nothing it serves is trained on."""

from rollout_openai.responses import (
    ApiKey,
    CodexLogin,
    Credentials,
    ResponsesContract,
    ResponsesEndpoint,
    codex_provider,
    hosted,
)

__all__ = ["ApiKey", "CodexLogin", "Credentials", "ResponsesContract", "ResponsesEndpoint", "codex_provider", "hosted"]
