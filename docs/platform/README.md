# Platform

Status: **Proposed** · Layer: platform (optional)

The platform layer turns the core and durability layers into a shared, multi-tenant service at fleet scale. None of
it is needed for the local or single-cluster profiles.

| Document | Defines |
|---|---|
| [cells.md](cells.md) | Cells: the unit of scale and failure; what is per cell, per region, global |
| [trust-tiers.md](trust-tiers.md) | Isolation of task code by who writes it |
| [control-api/](control-api/README.md) | The external API: deployments, runs, conversations, rollout jobs, event streams |
| [tool-router/](tool-router/README.md) | Imported tools as a service; the MCP facade for unmanaged harnesses |

Connectors (Slack, email, UIs) deliver `output.emitted` events and turn inbound messages into
`Runner.send` calls. The ephemeral token-streaming channel (deltas keyed by `effect_id`, not durable) is served
alongside the Control API's durable event streams.
