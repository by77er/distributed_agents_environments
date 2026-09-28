# 0021 — Isolate task code by trust tier, in one execution architecture

Status: **Proposed** (to be explored further) · Date: 2026-09-27

## Context

Who writes task code (in-house, partners, the public) was an open question that looked like it constrained the
substrate. With the pump / task-host split (ADR-0017), task code never holds substrate credentials on any tier, so
the question only decides how task hosts are isolated.

## Decision

Keep one execution architecture and vary three settings per trust tier — sandbox (separate process → gVisor →
microVM), pooling (shared → one tenant per process → one run per process), placement (shared → dedicated cell) — as
laid out in [trust tiers](../platform/trust-tiers.md). From tier T2 upward, a task host process never mixes tenants,
and inference prefix caches are salted per tenant.

## Consequences

- Promoting a tenant between tiers is configuration.
- Per-tenant pools cost density; pools must scale to zero and start quickly.

## Open

Which tiers are needed first, and whether the public tier is in scope.
