"""A model endpoint for Anthropic's Messages API: an implementation of `rollout.contracts.ModelEndpoint`, and what a
cluster's `api` provider samples Claude models with (`hosted`). Nothing it serves is trained on."""

from rollout_anthropic.messages import MessagesEndpoint, MessagesOptions, hosted

__all__ = ["MessagesEndpoint", "MessagesOptions", "hosted"]
