# Agent sessions

Status: **Working** (milestone P2) · See [ADR-0025](../decisions/0025-agent-sessions.md)

Independent agent sessions, each with its own computer, that create and message each other and share a board. You
manage them from the terminal.

## Running it

```bash
uv sync --extra assistant --extra durable
uv run agents serve                                   # http://127.0.0.1:8421; state in ~/.local/state/agent-sessions

uv run agents new lead "Create two workers, post three tasks to the board channel 'jobs', report to me when done."
uv run agents list                                    # sessions: working ●, waiting ○, starting ◌, stopped -
uv run agents tail lead -f                            # messages in, replies out, tool calls, as they happen
uv run agents send lead "Also check the licence" [--urgent]
uv run agents board --channel jobs
uv run agents inbox                                   # messages sessions sent you
uv run agents stop w1                                 # its computer is destroyed
```

Sessions survive restarts of the server: they run on the `DurableRunner`, and their computers are directories.

## What a session is

| Piece | Built from |
|---|---|
| The session | a durable conversation of the deployment `agents/session`, keyed by its name; it waits for messages and never ends until stopped |
| Its computer | an environment from `run.environments`: a private copy of Alpine Linux in unprivileged namespaces, root inside, with internet access. Tools: `shell`, `read_file`, `write_file`. A command interrupted by a crash reports that it may or may not have run instead of running twice |
| Sessions | `SessionTools` from `rollout.coordination`: `list_sessions`, `create_session`, `send_message` (urgent messages interrupt) |
| The board | `BoardTools`: channels of notes and tasks; `claim_task` is exclusive; `subscribe` pushes new posts to subscribers as low-priority messages |
| Delivery | tools write to the coordination store's outbox; a relay delivers through the runner, exactly once |
| You | the `operator` participant: sessions can `send_message(to="operator")`; your messages to a session steer it, or interrupt it with `--urgent` |
| Model | `gpt-6-astra` on the local Codex login, reasoning effort `low` |

Source: `src/agent_sessions/` (the product), composed from `rollout.environments`, `rollout.coordination` and
`rollout.durable`.

## A first run (2026-09-29)

`lead` was told to create two workers subscribed to `jobs`, post three tasks, and report when they were resolved. In
55 seconds: `lead` created `w1` and `w2`; both subscribed; `lead` posted the three tasks; the workers claimed them
(one claim lost to the other worker, as intended), did them on their own computers (installing Python with `apk`,
fetching a web page with `wget` after finding `curl` missing, hashing a string), and resolved them; `lead` sent the
operator a correct summary.

## Not yet

- A fault evaluation for sessions (the server killed during a fan-out), as the project assistant has.
- Isolation beyond namespaces (Firecracker microVMs), resource limits, and control over network access.
- Unloading idle sessions from memory; context compaction for sessions that run for hours.
- A web page on the same API.
