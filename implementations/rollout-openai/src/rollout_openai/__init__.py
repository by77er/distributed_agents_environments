"""A model endpoint for the OpenAI Responses API, on an API key or a Codex login: an implementation of
`rollout.contracts.ModelEndpoint`. Nothing it serves is recorded for training."""

from rollout_openai.responses import (
    ApiKey,
    CodexLogin,
    Credentials,
    ResponsesContract,
    ResponsesEndpoint,
    codex_provider,
)

__all__ = ["ApiKey", "CodexLogin", "Credentials", "ResponsesContract", "ResponsesEndpoint", "codex_provider"]
