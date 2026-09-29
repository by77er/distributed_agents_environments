# 0025 — Agent sessions, and the pieces they are built from

Status: **Accepted** · Date: 2026-09-28 · Builds on [0024](0024-product-before-rl.md)

## Context

The next product is a set of **independent agent sessions**: each has its own isolated environment (a shell, with
internet access), sessions can create more sessions and message each other, a person manages them, and a shared
board carries one-to-many communication. The system must stay usable for RL: nothing here may be specific to this
product where a task, a swarm or a training run would need the same thing.

## Decision

Build general pieces in the library, and a thin product on top.

| Piece | Where | Also used by |
|---|---|---|
| The environment protocol: `Environments`, `Environment`, `EnvironmentSpecification`, `ExecutionResult`; `run.environments`; environment operations as effects | `rollout.core` | any task that needs a computer, including RL tasks |
| A first environment backend: **unprivileged Linux namespaces** (user, mount, PID) over a per-environment copy of a **minimal image** (Alpine mini root filesystem), network shared with the host | `rollout.environments.namespaces` | any runner; Firecracker microVMs come later behind the same protocol |
| **Attempt markers**: an effect that is side-effecting and whose receiver cannot deduplicate (a shell command) is guarded; after a crash it completes as `OUTCOME_UNKNOWN` instead of running twice | `rollout.durable` | every durable run |
| **Coordination tool sets**: sessions (list, create, message) and a board (channels of notes and tasks; atomic claims; subscriptions that push new posts as low-priority messages) | `rollout.coordination` | swarms and multi-agent RL tasks, as imported tools |
| The **agent sessions** product: the session task and agent, the service, an HTTP API and an `agents` CLI | `agent_sessions` | — |

Choices:

- **Isolation: namespaces now.** Each environment gets its own root filesystem, process tree and working directory;
  it shares the host's network, so the internet works without root. This isolates processes and files; it is not a
  security boundary against hostile code. Firecracker (available on the development machine, with KVM) replaces it
  behind the same protocol when stronger isolation is needed.
- **Base system: a minimal image.** Alpine's mini root filesystem (about 4 MB), downloaded once, checksummed and cached;
  each environment gets its own copy and can install packages.
- **Management: a CLI and an HTTP API**; a web page can sit on the same API later.
- **Environment identity from effect identity.** Creating an environment is an effect whose `effect_id` determines
  the environment's id, so a replayed or retried creation finds the same environment.

## Consequences

- RL tasks gain environments through the same `run.environments` call, and swarms gain the coordination tools; the
  environment system's full design (templates, snapshots, placement) still comes later and extends this protocol.
- `docs/development/plan.md` gains milestone P2 for agent sessions.
