# Monitor

Status: **Working** (2026-10-01) · Code: `rollout.monitor`

The monitor is a web page for watching runs as they happen: every run, grouped; and for each run, every model slot's
turns: what the model was sent, what it thought, what it did and what came back.

```python
from rollout.monitor import RunFeed

runner = LocalRunner(hooks=[RunFeed(directory)])     # write the feed
```

```bash
rollout-monitor DIRECTORY --port 8765                 # read it: http://localhost:8765
```

| Piece | What it does |
|---|---|
| `RunFeed(directory, keep=200)` | [Hooks](harness/hooks.md) that append one JSON line per run event and per model sample to `<run_id>.jsonl`. A sample's line holds every message sent (text, reasoning, tool calls, tool results), the tools offered and the reply. Only the newest `keep` runs are kept. |
| `rollout-monitor`, `create_app(directory)` | A Starlette app over a feed directory. It only reads; the process running the runs writes. `/api/runs` summarises every run; `/api/runs/{run_id}?after=N` returns a run's lines from index N. |
| The page | Runs on the left, grouped by the label `group` and titled by `title` or `task` (labels are given to `runner.start(specification, labels=...)`; `episode` names a run within its group). For the chosen run: one column per slot, a slider over turns, and for the turn shown: **Sees** (the last message sent, or the whole context), **Thinks** (reasoning), **Does** (tool calls), **Result** (what the call returned, from the next turn). Other effects (tool calls the program itself made) are listed below. The page asks for news every two seconds. |

A grid of single characters in what a model sees (a map) is drawn with its symbols colored.

The feed is a copy for people to read. It is not the run's record: the run's events are, and under a durable runner
they are in its store.
