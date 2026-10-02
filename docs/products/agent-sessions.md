# Agent sessions

Status: **Working** (2026-10-02) · Code: `products/agent-sessions/src/agent_sessions`, `agent_sessions.coordination`

Independent agent sessions, each with its own computer, that create and message each other and share a board. You
manage them from the terminal.

## Running it

```bash
uv sync --extra assistant --extra durable
uv run agents serve                                   # http://127.0.0.1:8421; state in ~/.local/state/agent-sessions

uv run agents new lead "Create two workers, post three tasks to the board channel 'jobs', report to me when done."
uv run agents list                                    # working ●, waiting ○, sleeping z, starting ◌, stopped -
uv run agents tail lead -f                            # messages in, replies out, tool calls, as they happen
uv run agents send lead "Also check the licence" [--urgent]
uv run agents board [--channel jobs] [--status open|claimed|done]
uv run agents post jobs "Title" "Body" [--task]       # post to the board as the operator: a note, or a task
uv run agents inbox                                   # messages sessions sent you
uv run agents stop w1                                 # its computer is destroyed
```

The client commands talk to the server at `$AGENTS_URL` (default `http://127.0.0.1:8421`).

| `agents serve` option | Default | Meaning |
|---|---|---|
| `--state DIR` | `~/.local/state/agent-sessions` | runs, environments, the coordination store and images |
| `--host`, `--port` | `127.0.0.1`, `8421` | where the server listens |
| `--model`, `--reasoning-effort` | `gpt-6-astra`, `low` | the model, on the local Codex login; effort is `low`, `medium` or `high` |
| `--environment` | `namespaces` | where sessions' commands run (below) |
| `--evict-after SECONDS` | `300` | how long a session waits before it is unloaded from memory and shown as sleeping; `0` keeps every session loaded ([evicting idle runs](../durability/eviction.md)) |
| `--in-memory` | off | run on the `LocalRunner`: nothing survives a restart |
| `--database URL`, `--runner-id NAME` | SQLite in `--state` | several servers on one Postgres ([several runners](../durability/runners.md)) |
| `--blobs s3://bucket/prefix` | files in `--state` | where images are kept ([local services](../development/local-services.md)) |

Unless `--in-memory` is given, sessions run on the `DurableRunner` and survive restarts of the server
([durability](../durability/README.md)).

| `--environment` | Each session gets | Isolation |
|---|---|---|
| `namespaces` | a private copy of Alpine Linux in unprivileged namespaces, root inside, with internet access | its own files and process tree; not a security boundary against hostile code |
| `local` | a workspace directory on this machine (`$WORKSPACE`); commands run as you, with your programs, files and environment variables | none |

A session is told which computer it has. On `local` it is asked to stay in its workspace and not to install software
system-wide unless asked; nothing enforces that. A state directory keeps the backend it started with: the server
refuses to serve it with the other one. The backends are described in [environments](../environments/README.md).

## What a session is

| Piece | Built from |
|---|---|
| The session | a conversation of the deployment `agents/session`, keyed by the session's name. It waits for messages and never ends until stopped |
| Its computer | an environment created in the task's `setup` and destroyed in `teardown`, with the `ComputerTools`: `shell`, `read_file`, `write_file`, `edit_file`, `read_image` |
| Sessions | `SessionTools`: `list_sessions`, `create_session`, `send_message` (urgent messages interrupt) |
| The board | `BoardTools`: `post`, `read_board`, `claim_task`, `resolve_task`, `subscribe`, `unsubscribe`. Channels hold notes and tasks; a claim is exclusive; subscribers get new posts as low-priority messages |
| Delivery | the tools only write to the `CoordinationStore`, each write recorded against its `effect_id`. A `Relay` delivers the store's outbox through the runner, with the outbox key as idempotency key, so each message arrives once |
| You | the `operator` participant. Sessions reach you with `send_message(to="operator")`, which fills your inbox. Your messages steer a session, or interrupt it with `--urgent` |

Messages from other sessions reach a session as `[message from NAME] …`, and board notices as `[board #CHANNEL] …`.

## HTTP API

| Endpoint | Does |
|---|---|
| `GET /sessions` | every session, with its status |
| `POST /sessions` | `{name, instructions}`: starts a session |
| `GET /sessions/{name}/activity` | messages, replies and tool calls, in order |
| `GET /sessions/{name}/stream` | the same as server-sent events, following new activity |
| `POST /sessions/{name}/messages` | `{text, urgent?}`: messages a session |
| `POST /sessions/{name}/stop` | stops a session and destroys its environment |
| `GET /board?channel=&status=` | posts, newest first |
| `POST /board` | `{channel, title, body, kind?}`: posts as the operator; `kind` is `note` or `task` |
| `GET /inbox` | messages sessions sent to the operator |

## Durability under faults

```bash
uv run agents faults --kills 3 --seed 1 [--environment local] [--servers 3]
```

The evaluation runs a fan-out: `lead` creates two workers subscribed to the channel `jobs` and posts three tasks; the
workers claim them, do them on their own computers and resolve them; `lead` sends the operator a summary. Without
faults this takes about 55 seconds. The evaluation kills the server with SIGKILL at `--kills` seeded random moments
and restarts it. Then it waits until every session is evicted (with a 5-second threshold), kills the server while
they sleep, messages `lead` to have `w1` write the answers to a file, and kills the server again while they wake.

A model call in flight at a kill is sampled again and may decide differently, so the checks are the system's
guarantees, made once every session is waiting or sleeping:

- every outbox message reached its recipient exactly once;
- no command ran twice: a command in flight at a kill is reported to the session as "may or may not have run";
- model calls repeat at most once per session per kill;
- every task was resolved exactly once, with no duplicate posts or sessions;
- every event stream is gapless, and each session has one environment;
- there were at least as many evictions as sessions, the file is in an environment with the answers, and the
  operator got a correct summary and the confirmation.

| Run (`gpt-6-astra`) | Kills | Evictions | Messages | Commands with unknown outcome | Repeated model calls | Result |
|---|---|---|---|---|---|---|
| seed 11 | 5 | 4 | 19, each once | 1 | 9 | passed in 113 s |
| seed 12 | 5 | 4 | 16, each once | 1 | 6 | passed in 94 s |
| seed 13 | 5 | 4 | 15, each once | 2 | 5 | passed in 110 s |
| seed 21, `--environment local` | 5 | 5 | 17, each once | 0 of 10 | 10 of 67 | passed in 123 s |

The result with three servers sharing a Postgres is in [several runners](../durability/runners.md#fault-evaluation).
