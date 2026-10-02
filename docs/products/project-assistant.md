# Project assistant

Status: **Working** (2026-10-02) · Code: `products/project-assistant/src/project_assistant`

A long-lived conversational agent about one code repository, served over HTTP. It answers questions grounded in the
repository, remembers decisions across conversations, and follows up when asked.

## Running it

```bash
uv sync --extra assistant --extra durable
uv run project-assistant serve --repository /path/to/repo      # http://127.0.0.1:8420, on the local Codex login

curl -s localhost:8420/conversations/dev-1/messages -d '{"text": "Where is billing implemented?"}'
curl -s localhost:8420/conversations/dev-1/transcript
```

| `serve` option | Default | Meaning |
|---|---|---|
| `--repository PATH` | the current directory | the repository the assistant answers about |
| `--model`, `--reasoning-effort` | `gpt-6-astra`, `low` | the model; effort is `low`, `medium` or `high` |
| `--host`, `--port` | `127.0.0.1`, `8420` | where the server listens |
| `--state DIR` | none | run on the `DurableRunner` with its state here, so conversations survive restarts ([durability](../durability/README.md)); without it they live in memory, on the `LocalRunner` |
| `--notes PATH` | `.rollout/notes.sqlite` in the repository | the notes database |
| `--ledger PATH` | none | append every model call to this file |

| Endpoint | Does |
|---|---|
| `POST /conversations/{key}/messages` | `{text, priority?, idempotency_key?, wait? = true, timeout_seconds? = 300}`. With `wait`, returns the reply to this message, or 504 when none came in time; without, returns 202 |
| `GET /conversations/{key}/transcript` | user messages and assistant replies, in order |
| `GET /conversations/{key}/events?from_seq=0` | the live run's events as server-sent events |
| `POST /conversations/{key}/cancel` | cancels the conversation's live run |
| `GET /health` | liveness and the deployment name |

`priority` is `low`, `normal` (the default) or `high`: queue for the next wait, steer the current turn, or interrupt
it ([conversations](../guide/conversations.md)).

## How it maps onto the library

| Piece | Built from |
|---|---|
| The conversation | one run of the deployment `assistant/{repository name}`, keyed by the conversation key. It waits for messages with `WaitFor` and never ends on its own |
| Answers | `ProjectAgent` samples the `policy` slot with a system prompt naming the repository and the time. Replies leave the run with `run.emit("reply", …)` |
| Repository tools | `RepositoryTools`, an imported tool set: `list_files`, `search`, `read_file`, `git_log`, confined to the repository. As effects, their results are recorded, so a durable run replays them |
| Notes | `NotesStore`, an imported tool set in SQLite: `save_note`, `search_notes`, `list_notes`. Every conversation shares it; saves deduplicate by `effect_id` |
| Follow-ups | the `schedule_follow_up` `@tool` records a due time from `run.now()`. The task's `WaitFor` times out at the next one and wakes the assistant with a follow-up observation |
| Model | the Responses API adapter on the local Codex login |

On the `DurableRunner`, a conversation idle for 5 minutes is unloaded from memory and woken by its next message or
follow-up ([evicting idle runs](../durability/eviction.md)).

## Evaluations

```bash
uv run project-assistant evaluate --repeats 3        # all scenarios; results as JSON in .rollout/evaluations/
uv run project-assistant evaluate --scenario "locate a function" --no-judge
uv run project-assistant evaluate --durable          # on the DurableRunner, one scenario at a time
```

Scenarios run against **tidepool**, an invented repository created fresh for each evaluation with a fixed git history,
so every question has a known answer. Each scenario is a scripted user (possibly across several conversations) plus
programmatic checks. A judge model grades replies for correctness, grounding and concision against key facts, with
the whole repository and its history as ground truth. Cost and latency come from the runs' events.

| Scenario | What it exercises |
|---|---|
| locate a function | search and read tools, citing files |
| config value and its history | reading code together with `git_log` |
| something that does not exist | saying "there is none" instead of inventing an API |
| follow-up questions | context across turns of one conversation |
| memory across conversations | `save_note` in one conversation, `search_notes` in another |
| a follow-up fires | `schedule_follow_up` and a `WaitFor` timeout waking the assistant unprompted |
| a high-priority message interrupts | a `HIGH`-priority message cancelling the reply in progress |

Results with `gpt-6-astra`, reasoning effort `low`, 3 repeats, on the `LocalRunner`:

| Scenario | Success | Correctness | Grounding | Concision | Seconds per turn | Model calls | Tokens in / out |
|---|---|---|---|---|---|---|---|
| locate a function | 3/3 | 5.0 | 5.0 | 5.0 | 14.0 | 3.0 | 2,472 / 256 |
| config value and its history | 3/3 | 4.0 | 5.0 | 5.0 | 11.9 | 3.0 | 2,516 / 225 |
| something that does not exist | 3/3 | 5.0 | 5.0 | 5.0 | 16.7 | 4.0 | 3,625 / 326 |
| follow-up questions | 3/3 | 4.5 | 5.0 | 4.7 | 10.0 | 5.0 | 4,503 / 329 |
| memory across conversations | 3/3 | 5.0 | 5.0 | 5.0 | 11.8 | 6.7 | 5,283 / 398 |
| a follow-up fires | 3/3 | n/a | n/a | n/a | 7.8 | 5.0 | 3,970 / 159 |
| a high-priority message interrupts | 3/3 | 5.0 | 5.0 | 5.0 | 5.1 | 3.7 | 2,930 / 117 |
| **all** | **21/21** | **4.7** | **5.0** | **4.9** | **10.2** | **4.3** | **3,614 / 259** |

Tokens are per scenario run. The suite is saturated: it cannot tell two versions of the assistant apart. On the
`DurableRunner` (one repeat) all 7 scenarios passed, at 10.7 seconds and 4.1 model calls per turn.

## Durability under faults

```bash
uv run project-assistant faults --kills 3 --seed 1
```

A scripted client holds a six-message conversation (questions, and two facts to remember) with the server running
on the `DurableRunner`. While a reply is being produced, the client kills the server with SIGKILL at a random moment,
restarts it, and resends the message with the same idempotency key. It passes when no message is lost, no reply or
note is duplicated, no model call repeats except the one in flight at each kill, and the last reply recalls both
facts.

| Run | Kills | Lost messages | Duplicate replies | Duplicate notes | Repeated model calls | Recalls both | Seconds from restart to reply |
|---|---|---|---|---|---|---|---|
| seed 1 | 3 | 0 | 0 | 0 | 3 (one per kill) | yes | 12.1, 10.1, 9.9 |
| seed 2 | 5 | 0 | 0 | 0 | 5 (one per kill) | yes | 11.2, 14.4, 15.5, 10.4, 13.0 |

The seconds after a restart include starting the server, DBOS recovery, and finishing the interrupted reply.
