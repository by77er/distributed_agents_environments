# Project assistant

Status: **In progress** (milestone P1) · See [ADR-0024](../decisions/0024-product-before-rl.md)

A long-lived conversational agent about one code repository, built on the `rollout` core and served over HTTP. It
answers questions grounded in the repository, remembers decisions across conversations, and follows up when asked.
It exists to exercise the system end to end on a third-party model before RL work resumes.

## Running it

```bash
uv sync --extra assistant
uv run project-assistant serve --repository /path/to/repo      # http://127.0.0.1:8420, on the local Codex login

curl -s localhost:8420/conversations/dev-1/messages -d '{"text": "Where is billing implemented?"}'
curl -s localhost:8420/conversations/dev-1/transcript
```

| Endpoint | Does |
|---|---|
| `POST /conversations/{key}/messages` | `{text, priority?, idempotency_key?, wait? = true, timeout_seconds? = 300}`. With `wait`, returns the reply to this message; otherwise 202 |
| `GET /conversations/{key}/transcript` | user messages and assistant replies, in order |
| `GET /conversations/{key}/events?from_seq=0` | the live run's events as server-sent events |
| `POST /conversations/{key}/cancel` | cancels the conversation's live run |
| `GET /health` | liveness and the deployment name |

`priority` is `low`, `normal` or `high`: queue for the next wait, steer the current turn, or interrupt it.

## How it maps onto the library

| Piece | Built from |
|---|---|
| The conversation | one run of the deployment `assistant/{repository}`, keyed by the conversation key; it waits for messages with `WaitFor` and never ends on its own |
| Answers | `ProjectAgent` samples the `policy` slot with a system prompt naming the repository and the time; replies leave the run with `run.emit("reply", …)` |
| Repository tools | `RepositoryTools`, an **imported** tool set: `list_files`, `search`, `read_file`, `git_log`, confined to the repository. Imported rather than `@tool` so that a durable run replays recorded results |
| Notes | `NotesStore`, an imported tool set in SQLite (`.rollout/notes.sqlite`): `save_note`, `search_notes`, `list_notes`. Shared by every conversation; saves deduplicate by `effect_id` |
| Follow-ups | the `schedule_follow_up` `@tool` records a due time with `run.now()`; the task's `WaitFor` times out at the next one and wakes the assistant with a follow-up observation |
| Model | `gpt-6-astra` through the Responses API adapter on the local Codex login, reasoning effort `low` by default |
| Runner | `LocalRunner` today; `DurableRunner` in M2, unchanged task code |

Source: `src/project_assistant/`.

## Next

- The evaluation harness: scenarios, task-success checks, judged quality, cost and latency.
- M2: the same product on the durable runner, and the durability-under-faults evaluation.
