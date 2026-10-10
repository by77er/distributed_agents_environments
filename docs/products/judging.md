# Judging

Code: `environments/judging`

**Read first:** [Example environments](README.md) and
[more model slots](../guide/tasks.md#more-model-slots-judges-and-other-players). **Next:**
[Suites and evals](../libraries/rollout-train/evals.md).

The policy answers open-ended requests that have no exact answer: explain a concept to a ten-year-old in under 80
words, or summarize a short passage in at most two sentences. A second model, the judge, scores each answer against a
rubric the environment holds, and the score is the reward. The judge is a model slot of its own that is never trained:
its turns are recorded through the gateway, shown in the monitor and counted in what each episode sampled, and no step
trains on them.

The environment is the package `judging` (import `judging`), which depends on `rollout` alone. It needs no sandbox, no
tool set and no GPU of its own, but a run needs a channel for the judge.

```bash
uv run rollout env check judging.environment:environment                           # without a model
uv run rollout train judging.environment:environment --settings judged.toml --name judged   # judged.toml: below
```

## The requests

Two rows, one for each kind of request:

| Row | The policy is asked to | Rubric |
|---|---|---|
| `explain` | `Explain CONCEPT to a ten-year-old in under 80 words.` | `explain-1` |
| `summarize` | `Summarize this passage in at most two sentences and under 50 words.`, then the passage | `summarize-1` |

A start of a row is one request, drawn with its seed from the row's training list in `judging.data`: 16 concepts and
10 passages of about 70 words. The eval data, `judging-eval`, is every request of the eval lists: 6 concepts and 4
passages that training never draws. A start's parameters are the kind, what it is about (the concept, or the
passage's title), the request as the policy sees it, and how many times the judge is asked.

## The episode

1. The policy is given a short system prompt and the request, and answers once.
2. The judge is given the rubric's instructions as its system prompt, and the request and the answer.
3. It must reply with the rubric's verdict: one JSON object and nothing else. A reply that cannot be read is answered
   with what was wrong (`Your verdict could not be read: …`), and the judge replies again. A second reply that cannot
   be read ends the episode with reward 0, and the result says why.
4. The reward is the score on a scale of 0 to 1: `(score - 1) / 9`.

`judging.environment:twice` asks the judge twice about each answer, each time afresh, and averages the two scores. A
verdict that cannot be read, in either judgement, ends the episode as above.

## The rubric and the verdict

A rubric (`judging.rubric.Rubric`) has a version, says what the answer was asked to do, and lists criteria. The judge
scores each criterion and the answer as a whole from 1 (fails it entirely) to 10 (could not be better):

| Rubric | Criteria |
|---|---|
| `explain-1` | `accuracy` (nothing wrong or misleading), `clarity` (a ten-year-old follows it), `completeness` (it explains how or why), `length` (under 80 words; 1 if not) |
| `summarize-1` | `faithfulness` (nothing invented or changed), `coverage` (the main events and the ending), `concision` (two sentences, under 50 words; 1 if longer), `fluency` |

```json
{"criteria": {"accuracy": 8, "clarity": 9, "completeness": 7, "length": 10}, "score": 8, "reason": "Clear and true, but it only names chlorophyll."}
```

The verdict is read strictly (`Rubric.verdict`): the reply is one JSON object with nothing around it (no code fence),
its keys are exactly `criteria`, `score` and `reason`, `criteria` scores exactly the rubric's criteria, every score is a
whole number from 1 to 10, and the reason says something. A rubric's version changes whenever what the judge is told
changes, and every result records it, so a score always says which rubric gave it.

## Results

An episode's result reports:

- `kind` and `rubric` (the rubric's version);
- `words`: the answer's length in words;
- `verdicts`: every reply the judge gave, as it gave it, in order;
- `scores`: the score read from each judgement, and `score`, their average (none when the answer was not judged);
- `judged`, and `why` when it was not: `the judge's verdict could not be read after 2 replies: …`.

The environment's description says rewards fall in `[0, 1]`, and results report neither `solved` nor a duration.

## Binding the judge

The program declares two slots: `policy`, trained, and `judge`, declared `ModelSlot(trained=False, judge=True)`. A run
binds `judge` to a channel by name; it has no default. Every turn of the judge's slot records that it is not trained
(`trained: false`), its segments are kept in the episode marked so, and the algorithm and datasets leave them out.

### In a run's settings

The run's settings say the judge's channel, its provider and model, and that it serves a fixed model:

```toml
environment = "judging.environment:environment"
"trainer.provider" = "local-lora"
"slots.judge" = "judge"

[channels.policy]
provider = "local-vllm"
model = "Qwen/Qwen3.5-4B"
renderer = "rollout_qwen:qwen35"

[channels.judge]
provider = "lab"                 # any provider: a judge's turns need no logprobs
model = "Qwen/Qwen3.5-9B"
renderer = "rollout_qwen:qwen35"
mode = "fixed"                   # the base model; `checkpoint = "RUN:STEP"` pins a checkpoint instead
```

Asked for with these settings (`--settings judged.toml`, a preset, or the New run form's rows), the run's driver serves
the judge's channel beside the policy's: on engine hosts of its own for a `vllm` provider, at the servers of one reached
at addresses ([launching runs](../libraries/rollout-train/launching.md#the-driver)). The run's start records them, the
cluster's gateway builds the judge's channel from that start too ([the
gateway](../libraries/rollout-train/gateway.md#which-checkpoint)), and the runner's key for the `judge` slot routes to
it. Validation refuses a run that leaves `judge` unbound, binds it to a channel without a provider or a model, or to
a provider that does not offer the model, and a run that binds it to the trained channel, or to a channel that follows
it, unless the run says `self_judging = true`. A judge that follows the policy some checkpoints behind is a snapshot
of the policy judging it:

```toml
"self_judging" = true

[channels.judge]
provider = "local-vllm"
model = "Qwen/Qwen3.5-4B"
renderer = "rollout_qwen:qwen35"
mode = "follows"
follows = "policy"
lag = 5                          # the policy's checkpoint five serving records back
```

## Without a model

`rollout env check` answers every slot with one scripted reply. The judge's reply cannot be read twice, so the episode
ends with reward 0 and says why: the check passes, since that is a reward in range and a result that says what the
description says. The tests answer the policy and the judge with scripts of their own: a well-formed verdict, a
malformed one and then a well-formed one, two malformed ones, and a judge asked twice.

## The pieces

Paths are under `environments/judging/`.

| Piece | Where | What it does |
|---|---|---|
| Requests | `judging/data.py` | The concepts and passages, each list split into what training draws and what only the eval asks |
| Rubrics | `judging/rubric.py` | The two versioned rubrics, the judge's instructions, the strict reading of its verdict (`MalformedVerdict` says why), and the score normalized |
| Episode | `judging/episode.py` | The program: the policy's answer, the judge asked (again once after a verdict it cannot read; twice with `judgements = 2`), the reward and the result |
| Environment | `judging/environment.py` | The two rows, a start of one, the eval data `judging-eval`, and `environment` and `twice` |
| Tests | `tests/` | Verdicts read and refused, whole episodes through the runner with scripted policy and judge, the train and eval split, the slots' bindings, and `rollout env check` |
