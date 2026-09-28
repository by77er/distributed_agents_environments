# Trust boundaries

Status: **Proposed** · What is trusted, what crosses each boundary, and where enforcement lives. Environment
isolation is part of the [environment system](../environments/README.md) and is designed there; task-code isolation
by tenant is in [trust tiers](../platform/trust-tiers.md).

## Zones

| Zone | Members | Trust | Holds credentials? |
|---|---|---|---|
| Z0 Control | Control API, policy registry, rollout service (platform layer) | trusted | platform credentials |
| Z1 Execution | runners (the durable pump), recorder, tool bindings / tool router, weight update controller | trusted | credential *references*; tool bindings inject credentials where they call out |
| Z1t Task hosts | program, task and agent code (possibly tenant-authored) | **untrusted code** under the durable runner, sandboxed per [trust tier](../platform/trust-tiers.md) | never; no network except the local socket to the pump |
| Z2 Environments | whatever the environment system runs | **untrusted** | never ([environments](../environments/README.md)) |
| Z3 External tool servers | MCP / HTTP services we call | semi-trusted | their own |
| Z4 Model output | assistant messages, tool arguments | **untrusted content** everywhere | n/a |

In the local profile everything runs in one trusted process; these zones apply when code or tenants are not
trusted.

## Rules

1. **Enforcement outside the untrusted code (P8).** Task hosts are constrained by their sandbox and by the pump's
   validation of every effect; environments by the environment system's host-side enforcement.
2. **Task code acts only through validated effects.** The pump checks schemas, sizes, budgets and resource ownership
   for every effect request; a forged environment or run identifier gets nothing (effects rule 4).
3. **Secrets never reach untrusted code (P9).** Credentialed external access goes through imported tools, whose
   bindings resolve credentials at call time. Nothing secret is written to run events, the recorder or tool results.
4. **Model output is data.** Tool arguments are validated against the pinned `input_schema` before dispatch. Tool
   results and third-party tool descriptions are untrusted content: a `RunBinding` may override descriptions, and
   results carry `untrusted: true` provenance.
5. **Scoring does not trust what the agent controlled.** Rewards should be computed from state the agent could not
   tamper with (e.g. a clean environment built from the same template). The core cannot enforce this; task authors
   must.
6. **Shared inference caches are per tenant** from trust tier T2 upward (prefix-cache salt), so tenants cannot probe
   each other's prompts through cache timing.
7. **External tool servers** are isolated per tenant, receive credentials only through bindings, and their MCP
   annotations are ignored unless the server is operator-trusted.

## Threats in scope

| Threat | Mitigation |
|---|---|
| Malicious or buggy task code | task-host sandbox and pooling per trust tier; effect validation; ownership checks |
| Credential theft | credentials only in trusted bindings (rule 3) |
| Reward hacking by tampering with scoring state | scoring from untampered state (rule 5) |
| Prompt injection via tool results or descriptions | untrusted provenance; description overrides; per-run tool allowlists |
| Cross-tenant prompt inference through caches | per-tenant cache salt (rule 6) |
| Environment escape, exfiltration, attacks on the control plane | the environment system's design |
