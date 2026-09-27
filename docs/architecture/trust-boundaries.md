# Trust boundaries

Status: **Proposed** · Environments run untrusted code (R7). This document fixes what is trusted, what crosses
each boundary, and where enforcement lives.

## Zones

| Zone | Members | Trust | Holds credentials? |
|---|---|---|---|
| Z0 Control plane | Control API, Global Router, Environment Manager, Policy Registry, Rollout Controller | trusted | platform credentials |
| Z1 Execution plane | Runtime, Tool Router, Recorder | trusted | **no** tool/guest credentials; only credential *references* and attachment tokens |
| Z1t Task hosts | task and agent code (possibly tenant-authored) | **untrusted code**, sandboxed | never; no network except the runtime socket |
| Z2 Env host | envlet, egress proxy, credential broker | trusted, hardened | **yes** — the only zone that materializes tool credentials |
| Z3 Guest | everything inside an environment, including envd | **untrusted** | never |
| Z4 External tool servers | MCP / HTTP services we call | semi-trusted | their own |
| Z5 Model output | assistant messages, tool arguments | **untrusted content** everywhere | n/a |

## Rules

1. **Enforcement outside the guest (P7).** Isolation, network policy, resource limits and credential injection are
   implemented by the VMM, host networking (per-VM tap + nftables), and the egress proxy. Nothing in Z3 is relied on.
2. **Control over vsock only.** The Environment API is reachable only via envlet → vsock. The guest network has no
   route to any Z0/Z1/Z2 service. `network: none` means no network at all.
3. **envd is untrusted.** envlet enforces response size caps, rate limits and timeouts on everything envd returns.
   A compromised guest can lie about *its own* tool results; it cannot affect any other run.
4. **Attachment tokens.** A runtime call into an environment presents a token scoped to
   `(environment_id, run_id, attachment)`. envlet rejects calls from runs not attached to the environment. Task
   code holds environment identifiers only; the runtime rejects requests for environments the run neither owns nor
   is attached to, so a forged identifier gets nothing.
5. **Secrets never enter guests (P8).** A `SecurityProfile` lists credential *grants*; the egress proxy injects the
   secret into matching outbound requests. The guest sees only the effect (an authenticated response).
6. **Model output is data.** Tool arguments are validated against the pinned `input_schema` before dispatch.
   Tool results and third-party tool descriptions are untrusted content: RunBinding MAY override descriptions,
   and results carry `untrusted: true` provenance for agents that want to treat them differently.
7. **Scoring doesn't trust the agent's environment.** `Task.score` should compute rewards in a scratch environment
   (`run.environments.scratch`) using artifacts extracted from the agent's environment, never by running checks
   inside an environment the agent controlled. The SDK makes this the easy path; it cannot enforce it. See
   [task](../components/harness/task.md#environments).
9. **Task hosts are sandboxes.** Task and agent code runs in a sandbox (gVisor) with no network except the socket to
   the runtime, no credentials, and no persistent filesystem. It can act only through effects the runtime validates.
   The same sandbox enforces determinism ([durability](../components/harness/durability.md#determinism-rules)).
   Credentialed external access goes through imported tools, whose bindings hold broker references.
8. **Z4 servers are network-isolated per tenant**, receive credentials only via the broker, and their MCP
   annotations (`readOnlyHint`, `idempotentHint`, …) are ignored unless the server is operator-trusted.

## Credential flow

```
RunBinding / SecurityProfile:  grant { destination: "api.github.com", scheme: bearer, secret_ref: "vault://…" }
                                   │
Environment Manager ──compiles──▶ envlet: nftables allowlist + proxy route with secret_ref
                                   │
guest: curl https://api.github.com/…  ──▶  egress proxy (Z2): match grant → fetch secret → inject header → forward
```

- Secrets are resolved by the broker at injection time; they are never written to the guest, the run log,
  the recorder, or tool results.
- Destination matching is by SNI / CONNECT host for TLS; header injection requires the proxy to terminate TLS for
  granted destinations only (guest trusts a per-env CA installed at boot). Non-granted TLS is passed through
  or blocked per `network` mode, never terminated.

## Threats explicitly in scope

| Threat | Mitigation |
|---|---|
| Guest escape via kernel | microVM isolation (hardware virtualization) for `isolation ≥ microvm`; jailer; seccomp on VMM |
| Guest exfiltration | `network` mode + allowlist enforced on host; egress byte limits |
| Guest attacks control plane | no route; vsock only; envlet input validation |
| Cross-run interference in shared env | attachment tokens; per-run process users (optional); envs default to single-run attachment |
| Reward hacking via env tampering | scoring in scratch environments (rule 7) |
| Malicious task code | task host sandbox; effect validation; environment ownership checks (rules 4, 9) |
| Prompt injection via tool results / descriptions | untrusted provenance; description overrides; tool allowlists per run |
| Credential theft | broker + injection (rule 5) |
