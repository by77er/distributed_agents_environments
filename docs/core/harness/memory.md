# Memory

Status: **Working** (2026-10-02) · Code: `rollout.core.harness.memory`

A long episode outgrows any model's context. `Memory` keeps a context that fits, whatever the model: the agent's
recent turns as they were, and its own summary of everything older.

```python
memory = Memory(prompt="The turns above are about to leave your memory. Write what you need to remember ...")
reply = await memory.sample(model, system=system, current=[observation_in_full], tools=tools)
memory.remember(observation_in_brief, reply)       # how the turn looks once it is no longer the current one
memory.answer("mined: stone")                      # how the reply went: its tool result, or a note if it called nothing
if memory.crowded(model):
    await memory.compact(model, system)
```

Code that uses it says nothing about tokens or limits. It says what a turn should look like once it is no longer
the current one, and what to ask when turns must go.

- **When memory is full** is measured: the model reports what its last prompt took, memory tracks how much a turn
  has been seen to add, and `crowded` says when one more turn might leave the model less than its full room to reply.
- **A compaction** shows the agent its oldest turns once more, with its earlier summary, and asks what to remember;
  its answer replaces them, and the newest third stay as they are. It is a sample like any other on the agent's own
  model slot, with room for the summary and none to think it over.
- **A context the model refuses** as too long is compacted and tried again (`sample`).
- **Several agents** that share a turn should compact in the same one: a turn takes as long as its slowest agent.

`CompactingAgent` is the same for the task loop: `agent_program(MyTask, CompactingAgent)`.
