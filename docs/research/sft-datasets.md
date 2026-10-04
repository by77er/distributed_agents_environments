# SFT datasets by rejection sampling

A proposal, measured on curriculum-9's ledger on 4 October 2026. Nothing here is built except a prototype script in
the working notes; the dataset record and the training command below are designs.

The recommended first dataset is **best-of-group, actions that worked**: from each group with a solved episode, the
best solved episode, and of it every agent's turn whose action came back ok. Today that is 26 episodes, 13 tasks,
1,966 turns, 437K sampled tokens to learn from (9.7M tokens of context). It can be built from the ledger alone, by a
rule that is easy to state and to rerun.

## What there is to sample from

| Source | Episodes | Solved | Token-level trajectories | Notes |
|---|---|---|---|---|
| curriculum-9, groups 28–72 | 180 (163 completed, 17 failed) | 65 | yes: segments with sampled spans and logprobs, in blobs | the only run with episode records in the ledger |
| curriculum-9, groups 1–27 | none recorded | (results only) | no | their results are in the ledger; their episodes predate episode records |
| curriculum-1 to curriculum-8 | 12 summary lines, a few text transcripts | – | no | not usable for training |
| codex-ceiling (gpt-6-astra) | 2 | 2 (28 and 12 diamonds) | no: plain messages in the feed, 424 turns | a frontier model's play, in its own format |
| astra-t054u (gpt-6-astra) | 1 | 0 (mined 25, died 28 times) | no: plain messages, 2,228 turns | partial progress only |

Curriculum-9's completed episodes hold 46,156 agent turns (283 per episode) and 10.8M sampled tokens. Every turn is
one segment: the agent's whole prompt (about 4,900 tokens) and what it sampled (about 220). Teams were 1 to 4
agents (20, 24, 24 and 95 episodes).

How the turns went, read from each agent's next observation (`last_action`):

- **Half of all actions failed** (23,179 of 46,156 turns). In solved episodes too: 41% of actions worked, against 38%
  in unsolved ones.
- **Mining fails most**: 25% of mines worked in solved episodes, 19% in unsolved. The commonest failures are mining
  diamond ore or deepslate without a good enough pickaxe (about 3,900) and walking into walls (about 1,000).
- **Repairs are common**: 5,158 times an agent's action failed and its next action worked.

## Candidates

All of these are rejection sampling over the policy's own episodes except the frontier one. "Sampled tokens" is
what an SFT step learns from; every example also carries its prompt as context.

| Dataset | Rule | Size today | Value | Risks |
|---|---|---|---|---|
| **Solved, all** | completed and solved episodes, every turn | 65 episodes, 13 tasks, 2.88M sampled tokens | the most data | half its turns are failed actions; the easy tasks dominate (t018 9, t004 7 episodes) |
| **Best-of-group** | per group, the solved episode with the highest reward, then the shortest | 26 episodes, 13 tasks, 1.26M sampled tokens | the policy's best play for each start, one per group; it is what GRPO's advantage already prefers | small; still half failed actions |
| **Solved, capped per task** | solved, at most 3 episodes per task, highest reward first | 39 episodes, 13 tasks, 1.54M sampled tokens | evens out the tasks | thinner on the hard tasks than their share of play |
| **Guided to unguided** (exists: `rollout imitate`) | solved episodes that were guided; the `way` guidance cut from their prompts | 40 episodes, 9 tasks, 1.56M sampled tokens | teaches what guidance showed, without it | only guided tasks; a cut that cannot be made exactly drops the turn |
| **Actions that worked** (a turn filter) | of any of the above, only turns whose action came back ok | about 50% of their turns | stops the model imitating its own mistakes | "ok" is not "useful": a move or a mine that worked can still be aimless |
| **Repairs** | a failed turn, then the same agent's next turn that worked | 5,158 pairs | teaches recovering from a failure, with the failure in the prompt | as SFT it is a subset of "actions that worked"; as pairs it needs a preference loss we do not have |
| **Progress turns** (proposed filter, not measured) | turns after which the team's inventory or score moved toward the task (a diamond, a tool it needed) | – | the turns that mattered | needs a per-task notion of progress from the observations |
| **Team agreement** (proposed, not measured) | turns where agents' plans agreed and the agreed action worked | – | coordination | hard to define from the logs; chat is short and rare (650 chats in 46K turns) |
| **Frontier distillation** | a frontier model's solved episodes, rendered with our renderer into our model's format | 2 episodes, 424 turns | play far above our policy (28 diamonds in t013) | another model's style and reasoning; its prompts were made by the same harness, but rendered differently; tiny |

What the risks mean here:

- **Mode collapse**: 13 tasks, and a few of them carry most turns (t018 is 32% of the first dataset). Cap turns per
  task, or weight them, before training on more than one pass.
- **Reward hacking**: "solved" is the task's own test, and best-of-group's tie-break is speed. Neither rewards a
  shortcut that the task itself would not count.
- **Distribution shift**: the turns were sampled by checkpoints @18 to @29; a step on them from @31 trains on an
  older policy's play. Small for one step; it grows if datasets are reused across many checkpoints.
- **Guidance leakage**: guided episodes' prompts carry the way to the goal. The existing `imitation.without` cuts it
  token-exactly; a turn where the cut cannot be made is left out. The `teamwork` guidance (103 episodes) is advice
  about coordinating, not the way, and is kept unless asked otherwise.

## A dataset as a record

Proposed: a dataset is an artifact like a checkpoint, made once and never changed.

- **Its record**, in the ledger table `datasets`, keyed by a random id like a checkpoint's: the selection rule and its
  parameters (runs, episode filter, turn filter, guidance kinds cut, caps), the counts (episodes, tasks, turns,
  sampled and context tokens, what was left out and why), and when and by whom it was made.
- **Its manifest**, a blob: one line per example, naming its source (`RUN/GROUP/EPISODE/SLOT/INDEX`) and what the
  filters saw (the action, whether it worked, the depth it was sampled at). The examples themselves are made from the
  episodes' blobs when a step trains on them, as `imitation.examples` does, so a dataset costs a few kilobytes per
  thousand examples, not a copy of the trajectories.
- **Its name** is optional, in the registry beside run names and bookmarks.
- **More data later is a new dataset**, its record naming the one it extends.
- **An SFT step on a dataset makes a checkpoint** whose first parent is the checkpoint it trained from, and whose
  other parents are the checkpoints that sampled the data (the depths in the manifest, mapped to the sampling run's
  checkpoints: @18 to @29 for the first dataset). The graph then shows "learned from" edges to where the data came
  from, the way a distillation's teachers are shown. The step's batch already records each example's source; its
  record adds the dataset's id.
- **Both trainers can take one**: the LoRA trainer's likelihood objective now, and the full-weight trainer
  (`rollout_lora:FullTrainer`), through the same `Weighted` segments.

A command for it, proposed: `rollout dataset make RULE --run RUN [filters]` writes the record and the manifest;
`rollout imitate PROFILE --dataset ID` (or the New run form's start version plus a dataset) takes the step.

## The first dataset

**Best-of-group, actions that worked**, over run curriculum-9:

1. Episodes: completed and solved. Per group, the one with the highest reward, then the shortest (`duration`), then
   the first by number. 26 of the 45 recorded groups have one.
2. Turns: every agent's turns whose action came back ok in that agent's next observation. Of 5,841 turns, 2,969 were
   left out because the action failed, and 906 because no outcome was seen (an agent sampling twice before acting,
   when it summarises its old turns, and its last turn).
3. Guidance: the `way` guidance cut from guided episodes' prompts when the examples are made.

The result is 1,966 turns over 13 tasks: 437K sampled tokens and 9.7M context tokens, sampled at depths 18 to 29.
By action: move 922, mine 783, idle 85, chat 73, craft 42. By task: t018 629, t015 418, t013u 216, t009u 182, t005
133, the rest under 110.

A step on all of it costs about what five of curriculum-9's steps did. Its step 31 took 38 minutes for 384 segments,
about 6 s a segment, so 1,966 segments take about 3.2 hours on the one GPU.

The prototype, `sft_datasets.py` in the working notes, measures every candidate above and writes this dataset's
manifest (`sft-best-of-group.jsonl`, 1,966 lines and a header). It reads the ledger read-only and one episode's
blobs at a time, in about 20 seconds.

Two things it had to work out from the data, which a real implementation should record instead:

- **Which observation goes with which turn.** A model sample's effect names neither its slot nor its agent. The
  segment's span names the sample, and the agent's next `observe` reports the action's outcome.
- **Which agent is which slot.** From group 39 on, model slots are `agent-1` to `agent-4` and agents have random
  in-game names. The episode's result lists the team's names in slot order (`info.team`), and slot `agent-N` is the
  Nth. Recording the slot in each `act` would make this exact.
