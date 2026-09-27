# 0005 — Agent = model + tools; tools are spec + binding; MCP is one binding

Status: **Accepted**, amended by [0012](0012-task-agent-loop.md) · Date: 2026-09-26

## Context

Early sketches made MCP the backbone (envd as an MCP server, all tools as MCP). MCP lacks durable delivery,
idempotency, stable tool sets and failover-safe sessions, and it conflated three roles: the model-facing tool
contract, environment control, and third-party tool access. Requirement R2/R3: the environment backing a tool
must not matter to the agent.

## Decision

- An agent is a model plus the tools exposed to it.
- A tool is a `ToolSpecification` (what the model sees; shape borrowed from MCP's data model) plus a **binding** (how it
  executes): `env | mcp | http | native | agent | human`.
- Tools reference environments only through named slots; RunBindings fill slots.
- Environment control is its own API (envd), not MCP.
- MCP appears in exactly two places: as an import binding (third-party and in-environment tool servers) and as an
  outbound facade for unmanaged harnesses.
- Resolved tool sets are pinned in the log; changes are explicit events.

## Consequences

- The same task runs against Firecracker in RL, a VPS in production, a mock in tests.
- Durability semantics (idempotency, retry classes, outcome-unknown) are ours, uniformly.
- The MCP ecosystem remains usable, conservatively (hints ignored from untrusted servers).

## Alternatives considered

- **MCP everywhere**: simplest mental model; wrong durability and data-transfer semantics.
- **No MCP**: loses the ecosystem.

## Amendment ([0012](0012-task-agent-loop.md))

- Environment slots and the `env` / `native` binding kinds are removed. Tools a task defines are `@tool` methods
  that run in the task host and reach environments through handles; environment-provided tools are added by task
  code. The agent still sees only `ToolSpecification`s, so the principle — the environment backing a tool does not
  matter to the agent — is unchanged.
- Bindings remain for **imported** external tools only: `mcp`, `http`, `agent`, `human`.
