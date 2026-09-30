# Agent sessions

Status: **Working** (milestone P2) · See [ADR-0025](../decisions/0025-agent-sessions.md)

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
uv run agents board --channel jobs
uv run agents inbox                                   # messages sessions sent you
uv run agents stop w1                                 # its computer is destroyed
```

Sessions survive restarts of the server: they run on the `DurableRunner`, and their computers are directories.

Where sessions' commands run is a server option (a state directory keeps the one it started with):

| `--environment` | Each session gets | Isolation |
|---|---|---|
| `namespaces` (default) | a private copy of Alpine Linux in unprivileged namespaces, root inside | its own files and process tree; not a security boundary against hostile code |
| `local` | a workspace directory on this machine (`$WORKSPACE`); commands run as you, with your programs, files and environment variables | none: for uses where convenience matters more than a sandbox |

Sessions are told which they have: on `local` they are asked to stay in their workspace and not to install software
system-wide unless asked, but nothing enforces it.
Sessions idle for 5 minutes (`--evict-after SECONDS`) are unloaded from memory and shown as sleeping; a message or a
scheduled wake-up brings them back ([evicting idle runs](../durability/eviction.md)).

### Several servers

Servers can share a Postgres database, each with its own runner; any of them serves any request, and when one dies
another takes over its sessions ([several runners](../durability/runners.md)). They must share `--state` too, since
environments are files; images can go to S3 with `--blobs s3://bucket/prefix` ([local services](../development/local-services.md)):

```bash
uv run agents serve --database postgresql://…/sessions --runner-id server-0 --state /shared/sessions --port 8421
uv run agents serve --database postgresql://…/sessions --runner-id server-1 --state /shared/sessions --port 8422
```

## What a session is

| Piece | Built from |
|---|---|
| The session | a durable conversation of the deployment `agents/session`, keyed by its name; it waits for messages and never ends until stopped |
| Its computer | an environment from `run.environments`: by default a private copy of Alpine Linux in unprivileged namespaces, root inside, with internet access; or a workspace on the host (`--environment local`). Tools, from `ComputerTools`: `shell` (long output keeps its end; the rest is saved to a file), `read_file` (paged), `write_file`, `edit_file` (exact replacements) and `read_image` (images are scaled to 2000 px and kept in `state/blobs`). A command interrupted by a crash reports that it may or may not have run instead of running twice |
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

## Durability under faults

```bash
uv run agents faults --kills 3 --seed 1
```

The same fan-out, with the server killed by SIGKILL at seeded random moments and restarted. A model call in flight at
a kill may be re-sampled and decide differently, so the checks are the system's guarantees, not the model's choices:
every outbox message reaches its recipient exactly once, no command runs twice, model calls repeat at most once per
session per kill, every task is resolved exactly once with no duplicate posts or sessions, event streams stay
gapless, each session has one environment, and the operator gets a correct summary.

| Run (2026-09-29) | Kills (seconds in) | Messages | Commands (outcome unknown) | Model calls (repeated) | Result |
|---|---|---|---|---|---|
| seed 1 | 3 (9.5, 35.3, 38.7) | 14, each once | 6 (1) | 57 (8) | passed in 71 s |
| seed 2 | 5 (6.3 … 43.2) | 14, each once | 4 (0) | 46 (8) | passed in 61 s |
| seed 3 | 5 (13.8 … 29.7) | 11, each once | 8 (2) | 54 (10) | passed in 67 s |

A command in flight at a kill is reported to the session as "may or may not have run", and is never run again.

Since eviction (2026-09-29), the evaluation has a second phase: after the fan-out it waits until every session is
evicted (`sleeping`, with a 5-second threshold), kills the server while they sleep, messages `lead` to have `w1`
write the answers to a file, and kills the server again while they wake. It then also checks that the file exists
in `w1`'s environment with the answers, that the confirmation reached the operator, and that every session was
evicted; checks run once the system is quiet (every session waiting or sleeping).

| Run | Kills | Evictions | Messages | Commands with unknown outcome | Repeated model calls | Result |
|---|---|---|---|---|---|---|
| seed 11 | 5 | 4 | 19, each once | 1 | 9 | passed in 113 s |
| seed 12 | 5 | 4 | 16, each once | 1 | 6 | passed in 94 s |
| seed 13 | 5 | 4 | 15, each once | 2 | 5 | passed in 110 s |

The evaluation also runs on the local backend (`agents faults --environment local`): seed 21 passed in 123 s with
5 kills, 5 evictions, 17 messages each delivered once, 10 commands (none with an unknown outcome) and 10 repeated
model calls of 67.

With three servers sharing a Postgres (`agents faults --servers 3`), kills hit random servers and the first killed
server stays down until another takes its sessions over: seed 31 passed in 101 s with 6 kills, every session evicted
and woken, 14 messages each delivered once and no command run twice ([several runners](../durability/runners.md)).

One earlier run (seed 9) reported a message that never reached its recipient; the harness then stopped the server
three seconds after the confirmation, which could leave a message in flight to a waking session. The harness now
waits for quiet and reports any missing message with its recipient's state; five runs since passed, but the cause
of that one miss is not proven.

## Not yet

- Isolation beyond namespaces (Firecracker microVMs), resource limits, and control over network access.
- Context compaction for sessions that run for hours.
- A web page on the same API.
