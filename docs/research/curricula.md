# Curricula and evaluation suites

Code: `rollout.curriculum` · See [training](../libraries/rollout-train/training.md#the-curriculum),
[the Minecraft team](../products/minecraft-team.md#tasks-and-curriculum), [evals](../libraries/rollout-train/evals.md)

To train, draw rows by the chance that their next group will hold both solved and unsolved episodes, estimated from
recent evidence. Unlock rows on enough evidence and never lock them again, and stop playing rows that cannot be built.
To measure progress, use frozen, stratified suites of starts on world seeds that training never sees. Every tenth
checkpoint plays each start once, and checkpoints are compared start by start. The training curriculum never reads a
suite's results.

**A proposal.** What exists today is `Curriculum` (below), and suites and evals
([evals](../libraries/rollout-train/evals.md)): versioned lists of starts, played by a checkpoint or another model,
by hand or on a run's schedule, with each episode's result recorded. Everything else marked *proposed* is a design:
no code does it. The figures come from the run `curriculum-9` (72 groups, 31 checkpoints) and from three episodes
played by a team of `gpt-6-astra`.

## Recommendation

In order of what each is worth per hour of work:

| # | Build | Why (from the run) | Expected effect |
|---|---|---|---|
| 1 | **Redraw the world when a row cannot be built**, keep a record of which rows each world can host, and sweep every row through every world once | 5 of 72 groups never started, costing 99 minutes. t012u and t019u found no trees in their worlds. t022 failed the same way in two worlds ("an ore would be exposed", at 8 places each), which points at its builder. While a row keeps failing, it keeps the untried weight 1.0 for `FAILED_GROUPS` groups: the rule draws broken rows the most | About 7% of groups back; no row drawn for being broken; builder bugs found before a run |
| 2 | **`counts_for` credits only successes** | A guided twin is credited with its unguided row's failures. Groups 1–35 were played before ways existed and are filed under unguided rows, so t011, t014, t016 and t021 got only failures (weight 0.05 against up to 1.05) and the run never drew them | Guided rows are tried on their own evidence; fading guidance gets its first rung back |
| 3 | **Unlock on evidence and never relock; an unguided row waits for its guided twin** | Replayed through today's environment, the frontier moved on single groups of four: t013u at 3 of 4 unlocked up to row 34, t017u at 2 of 4 up to row 38. One group of t017u at 0 of 4 (group 68) locked 11 rows again. 6 of the 10 groups in which nobody scored anything were first tries of newly unlocked rows | Fewer wasted first tries; the frontier stops oscillating |
| 4 | **Weight by the chance of a mixed group**, from an age-discounted Beta posterior (formula below) | Untried rows weigh 1.0 against at most 1.05 for any row: with the run's final records and 38 rows open, 5 untried rows get 24% of draws. t019 (0 of 16 solved, only partial credit) and t007u (16 of 16) get 4% each | Same informative share at steady state (0.67 against 0.67 at the run's end); better during unlocks. The literature reports 2–3× fewer steps, but from large pools of cheap prompts |
| 5 | **A frozen `core` suite** of 24 starts on held-out worlds, played once each by every 10th checkpoint; a `frontier` suite of 8 long starts for every 40th checkpoint and outside models | The training solve rate fell from 63% to 36% as rows unlocked, the team size changed and the harness changed. Nothing in training can separate those from learning | A measured trend: two checkpoints compared on 48 paired episodes show a change of about 0.19; a fit over many checkpoints shows less |
| 6 | **Records** for curricula, suites and buildability in the shared ledger, and sections of the monitor's statistics page for each | A fold cannot be audited or replayed without its parameters and the checkpoint each group was played by | Offline comparison of sampling rules; one buildability table for every run |

## What the run shows

| Measure | Value |
|---|---|
| Groups decided / played to the end / never built | 72 / 67 / 5 |
| Groups with no difference in task reward | 28 of 72: 5 never built, 18 skipped by the algorithm, 5 trained only on the speed bonus |
| Played groups by solved count: none / some / all | 20 / 24 / 23. No group with some solved was skipped |
| Same, groups 1–35 → 43–72 | none 9 → 10, some 6 → 14, all 18 → 3 |
| Episodes solved | 52% overall (267 episodes). Groups 1–35: 63%. Groups 43–72: 36% |
| Rows played both before and after group 43 | 54% → 42% solved (84 and 59 episodes) |
| Guided / unguided rows | 57% solved, 4 of 39 groups skipped / 46% solved, 14 of 28 groups skipped |
| Team of 1 / 2 / 3 / 4 (groups 41–72) | 45% / 33% / 33% / 45% solved (20, 24, 24 and 47 episodes): too few to separate from the rows drawn |
| Hardest row solved half the time | t017u (difficulty 6.0), on one group of four |
| Never solved | t012u, t019, t019u, t023 (crafting chains from nothing); t011u, t014u, t018u (unguided ore out of sight); t016u, t020 (split kits) |
| Wall time per group | median 64 minutes from decision to last episode. Groups 43–72: 30 groups in 15.4 hours with steps between them |
| Starts within a row | intraclass correlation of solving 0.46 (16 rows with two or more groups). Between-start variance 0.091, within-start 0.107 |

The intraclass correlation includes everything that changed between a row's groups (checkpoints, team sizes, the
harness), so it overstates how much starts differ. It still says that episodes of one start are far from independent.

Two corrections to the run's own evidence:

- Crafting outcomes from before the table-crafting fix are wrong: about half of table crafts failed silently. The
  rows' estimates (t019 at 0 of 16, say) should be dropped from the fold, not carried forward.
- The ore rows count as solved at one diamond per player, so a row's difficulty depends on the team drawn. Laid-out
  rows need more than half of what was laid out, whatever the team. Row-level estimates mix team sizes.

## Training curricula

### What `Curriculum` does today

Rows unlock in environment order: the first `start` (3), and `reach` (4) past the hardest row whose moving-average solve
rate is at least ½. A row's weight is a moving average (`smoothing` 0.5) of whether its groups' task rewards
differed, plus `floor` (0.05). Untried rows weigh 1.0. A group counts for its row and for every row in `counts_for`.
A group in which no episode completed counts for nothing until `FAILED_GROUPS` (3) of them in a row.

| Right | Wrong |
|---|---|
| It samples for signal, not for success: a group-relative update learns nothing from a group whose scores are all equal, which is DAPO's filter applied before playing instead of after | "Rewards differed" is a weak proxy. It counts a group where everyone solved and only the diamond count varied the same as a group with both solved and unsolved episodes. The easy rows t003, t005 and t006 drew 17 of 72 groups |
| It is a fold over the ledger, replayable and robust to environment changes | `smoothing` 0.5 makes the last group carry half the estimate. With 4 episodes per group, one group decides unlocking |
| Pending rows go last, so draws are not made on stale evidence | Unlocks can be revoked, which wastes the groups that explored the newly opened rows |
| A guided row and its unguided variant share evidence | Failures are shared too, though failing without help says little about doing it with help |
| Unguided variants sit two steps later in difficulty, so guidance fades row by row | The environment's order is one line through a partial order: an unguided row can open before its guided twin is solved |
| | A row that cannot be built is the most drawn row while it fails |
| | The moving average is clocked by the row's own groups: a row drawn rarely keeps an old estimate whatever the policy has learnt since |

### What the literature does

| Work | Rule | What applies here |
|---|---|---|
| DAPO (Yu et al., 2025, [2503.14476](https://arxiv.org/abs/2503.14476)) | Over-sample prompts and drop groups whose accuracy is 0 or 1 until the batch is full; +8 points on AIME24 in its ablation | Exists here as the algorithm's skip. Over-sampling is unaffordable when a group costs 30–60 minutes, so the filter has to act before playing: predict which rows will give mixed groups |
| LILO (Foster et al., 2025, [2502.12272](https://arxiv.org/abs/2502.12272)) and SFL (Rutherford et al., 2024, [2408.15099](https://arxiv.org/abs/2408.15099)) | Sample by learnability p(1−p) of success. LILO reports reaching final accuracy in a third of the steps, across PPO, VinePPO and GRPO | The weight should peak at p = ½ and vanish at 0 and 1. That is the target of the proposed weight |
| Online difficulty filtering (Bae et al., 2025, [2504.03380](https://arxiv.org/abs/2504.03380)) | Keep prompts whose pass rate is in [0.2, 0.8]. The policy improvement is bounded below by p(1−p) | Same target, as a band rather than a weight |
| DOTS (Sun et al., 2025, [2506.05316](https://arxiv.org/abs/2506.05316)) | P(q) ∝ exp(−\|d̂ − ½\|/τ). The expected squared gradient norm is ∝ p(1−p)(1 − 1/G). 23–62% less fine-tuning time | The factor (1 − 1/G) is why groups of 4 lose more than groups of 16: with G = 4, a row at p = 0.1 gives a mixed group only 34% of the time |
| GRESO (Zheng et al., 2025, [2506.02177](https://arxiv.org/abs/2506.02177)) | Skip a prompt with probability 1 − p_e^z after z zero-variance rounds in a row, keeping at least 5% exploration | Zero-variance rows tend to stay so. Park them but keep a fixed small share of draws to notice when they wake |
| SEC (Chen et al., 2025, [2505.14970](https://arxiv.org/abs/2505.14970)) | A bandit over categories, rewarded by mean \|advantage\|, updated with Q ← αr + (1−α)Q, α 0.2–0.5, softmax at temperature τ | Close to today's moving average, with a better signal (\|advantage\| ∝ √(p(1−p)) for binary rewards) |
| Prioritized Level Replay (Jiang et al., 2021, [2010.03934](https://arxiv.org/abs/2010.03934)) | Rank levels by mean \|GAE\|, P ∝ (1/rank)^(1/β), β = 0.1, mixed with staleness at ρ = 0.1 | Rank-based weights are robust to the scale of the score. Staleness is a principled floor: revisit what has not been seen for long |
| PAIRED ([2012.02096](https://arxiv.org/abs/2012.02096)), Robust PLR ([2110.02439](https://arxiv.org/abs/2110.02439)), ACCEL ([2203.01302](https://arxiv.org/abs/2203.01302)) | Regret-based environment design: an adversary or a curated buffer chooses levels of high regret. Robust PLR updates only on replayed levels; ACCEL edits high-regret levels | No value function or antagonist here, and building is slow. SFL finds that the usual regret proxies track success rate anyway. ACCEL's "edit a level that is almost right" is what the environment's kit ladder does by hand |
| Teacher-Student Curriculum Learning (Matiisen et al., 2017, [1707.00183](https://arxiv.org/abs/1707.00183)) | A bandit over tasks rewarded by the absolute slope of the learning curve, so forgetting draws practice too | Learning progress needs many groups per row to estimate a slope; the run has 1–8. It suits row families more than rows |
| ALP-GMM (Portelas et al., 2019, [1910.07224](https://arxiv.org/abs/1910.07224)) | Absolute learning progress \|r_new − r_old\| against the nearest earlier task, a GMM over task parameters, 20% uniform sampling | Fits continuous task parameters; the environment is discrete. The 20% uniform share is the same idea as a floor |
| Self-paced learning (Kumar et al., 2010, [NeurIPS](https://papers.nips.cc/paper/3923-self-paced-learning-for-latent-variable-models)) | Train on examples whose loss is below a threshold that is relaxed over time | Today's unlocking is a self-paced schedule over the environment's order |
| Reverse curriculum generation (Florensa et al., 2017, [1707.05300](https://arxiv.org/abs/1707.05300)); Backplay ([1807.06919](https://arxiv.org/abs/1807.06919)); R³ (Xi et al., 2024, [2402.05808](https://arxiv.org/abs/2402.05808)) | Start near the goal and move the start back. Florensa keeps starts whose success is in [0.1, 0.9] and replays old ones | The environment is already a reverse curriculum over the tech tree: kits iron → ingots → raw iron → stone → wooden → nothing, and progress tasks from the end portal back to the surface. What is missing is moving along it by a row's own success band instead of a global order |
| QuestA ([2507.13266](https://arxiv.org/abs/2507.13266)), Guide-GRPO ([2506.13923](https://arxiv.org/abs/2506.13923)), StepHint ([2507.02841](https://arxiv.org/abs/2507.02841)) | Hints or partial solutions in the prompt for problems the policy fails, faded over stages (QuestA: 50% then 25% of the solution). Guide adds hints only when all k rollouts fail, and corrects with π(y\|x)/π_old(y\|x̃) toward the unhinted prompt | The guided twin is a hint at row granularity. Mixing guided and unguided episodes in one group, with the importance correction, is a later step: the trainer would score sampled tokens under the unguided prompt |

### Proposed sampling rule

For row *i*, when a row is drawn while the newest checkpoint is at depth *v*:

- **Evidence.** Each episode *e* of a group of row *i* counts with age weight a_e = 2^(−(v − v_e)/H), where v_e is
  the depth of the checkpoint that played it and H = 8 checkpoints (about 32 groups, 16–20 hours at the run's pace).
  The clock is the policy's checkpoints, not the row's own groups, so a rarely drawn row's estimate ages as the policy
  changes.
  s_i = Σ a_e·solved_e and f_i = Σ a_e·(1 − solved_e). A row that `counts_for` row *i* adds its solved episodes to
  s_i only.
- **Estimate.** p_i ~ Beta(½ + s_i, ½ + f_i).
- **Chance of a mixed group.** m_i = 1 − E[p_i^G] − E[(1 − p_i)^G], with G the group size (4), and
  E[p^k] = ∏_{j<k} (α + j)/(α + β + j). This is the probability that the row's next group has both solved and
  unsolved episodes, which is what DAPO's filter keeps. It is learnability p(1−p) corrected for groups of four, and
  for how much is known.
- **Partial credit.** d_i is the age-weighted share of the row's groups in which every episode had the same solved
  outcome but task rewards differed (steps of a chain, diamonds beyond the threshold). Those groups still train.
- **Weight.** w_i = m_i + λ·d_i, with λ = 0.25: a group with partial credit only is worth a quarter of a group
  with mixed outcomes.
- **Parked rows.** A row with m_i < 0.15 and d_i = 0 is parked. Parked rows share a fixed 5% of draws, spread
  evenly (GRESO's floor), and the rest are drawn by w_i.

What m_i gives:

| Evidence (solved of played, at full weight) | m_i |
|---|---|
| untried | 0.45 |
| 0 of 4 | 0.28 |
| 0 of 12 | 0.13 (parked) |
| 0 of 24 | 0.07 (parked) |
| 2 of 4 | 0.74 |
| 8 of 16 | 0.83 |
| 4 of 4 | 0.28 |
| 16 of 16 | 0.10 (parked) |

Untried rows are drawn eagerly but not ahead of rows known to be at the frontier (0.45 against up to 0.875). By m_i
alone, with the run's final records and 38 rows open, the 5 untried rows would take 14% of draws instead of 24%,
and t019 and t007u 0.7% each instead of 4%. The expected share of groups with mixed outcomes is about the same
(0.67 under either rule, taking each row's observed solve rate as true): the gain is in where the rest goes.

### Proposed unlocking

| Rule | Value |
|---|---|
| A row is solved when its posterior mean (½ + s)/(1 + s + f) is at least ½ and it has at least 8 episodes of evidence (s + f ≥ 8, two groups) | instead of a moving average at ½ from one group |
| Open: the first `start` rows, and `reach` past the hardest row ever solved | an unlock is never revoked: m_i lowers the weight of a row that got harder |
| A row may name rows it waits for (`Row.requires`, proposed): it is not drawn before each is solved | the environment fills it: an unguided row waits for its guided twin, a one-kit or split row for the kitted row of the same situation |
| `start`, `reach` | 3, 4, as today |

`requires` makes guidance fade where it has done its work: the team gets a situation without the way once it can
do it with the way, which is the reverse curriculum's rule (move the start back once success is high).

### Rows that cannot be built, and rows that give no signal

- **Cannot be built.** Buildability belongs to a row in a world, not to the policy. A `BuildError` before the first
  turn redraws the start in another world (up to three) instead of failing the group. Each outcome is recorded
  per (row, world) (proposed table `environments/ENVIRONMENT/buildable`), and `Teams.start` draws only from worlds where the
  row has been built or not yet tried. A row that no world can host is retired: weight 0, listed by the monitor.
  With the ledger shared by runs, this is learnt once. A sweep that builds every row in every world without playing
  (1,200 builds) would fill the table before a run, and tells a builder's bug (t022: the same error everywhere)
  from a world that lacks something (no trees).
- **No signal: unsolved.** Parked (5% shared). It comes back on its own as evidence ages (H) or as a twin succeeds.
  For an unguided row its guided twin is the hint; for a guided row the next easier kit is the nearer start.
- **No signal: always solved.** Parked too. The speed bonus still makes such groups train; that is the algorithm's
  business, not the curriculum's.
- **Wrong evidence.** The fold takes a filter (proposed, in the curriculum record): results to ignore, by rows and
  before a time. That drops the crafting rows' results from before the harness fix.

### Outside the curriculum

- **Reward scale.** Advantages are not divided by the group's spread, so a row's reward scale weighs its groups in
  the step: diamond counts reach 28 in a group, a chain's steps up to its total weight. A row's rewards divided by
  its cap (the diamonds laid out, the chain's total, the milestone's weight) would make rows count alike. That is
  a change to the algorithm, and it interacts with the curriculum's weights.
- **Cost.** Rows differ in budget from 3 to 66 game minutes, and a step takes at most 384 segments, so a long
  group's segments are thinned. Dividing w_i by the square root of a row's expected wall time would favour cheap
  signal. It is untested; record the times first.
- **Team size.** Keep row-level estimates until there are about 100 groups, then test a team-size offset
  (proposed: key the evidence by row and players, pooled through a shared prior).

## Evaluation suites

An evaluation suite is a frozen list of starts: row, world seed, layout seed and team. A subject plays it: a
checkpoint, or an outside model. Suites, their versions and the evals that play them exist
([evals](../libraries/rollout-train/evals.md)); what follows proposes which suites to make and how to read them.

### How many starts, how many episodes

With intraclass correlation 0.46, two episodes of one start are worth 1.37 episodes of different starts (the design
effect 1 + (2 − 1)·0.46): a second episode adds about a third of what a new start adds. Since the cost is the
episode (each builds its own world), spend episodes on starts: one episode per start.

| Starts × episodes | Standard error of a solve rate, one subject | Paired difference between two subjects on the same starts (lower bound) | Smallest difference seen at 80% power |
|---|---|---|---|
| 24 × 1 | 0.091 | 0.094 | 0.26 |
| 32 × 1 | 0.079 | 0.082 | 0.23 |
| 48 × 1 | 0.064 | 0.067 | 0.19 |
| 24 × 4 | 0.070 | 0.047 | 0.13 |

Two subjects on different starts would differ with √2 times the first column's error (0.129 for 24 starts). The
paired column assumes each start is equally hard for both subjects, so that only within-start noise remains. It is
the reason to keep starts fixed: comparing checkpoints start by start (Miller, 2024,
[2411.00640](https://arxiv.org/abs/2411.00640)) removes the variance between starts, the larger part. A single
evaluated checkpoint detects only large changes. A trend is read from a fit of solved against depth over all
evaluated checkpoints, and standard errors are clustered by start. tinyBenchmarks (Maia Polo et al., 2024,
[2402.14992](https://arxiv.org/abs/2402.14992)) estimates a 14,000-question benchmark from 100 questions, using
item response theory fitted on many models' results. That needs results from many subjects first. Once checkpoints and
outside models have played a suite, the same fit can say which starts discriminate and which to replace.

### The suites (proposed)

| Suite | Starts | Rows | Played by | Cost |
|---|---|---|---|---|
| `core` | 24 | skills and short survival, game budget at most 25 minutes | every 10th checkpoint, once per start; the base model once | about 6 group-equivalents (3–4 hours at the run's pace) |
| `core-seen` | 8 | the anchors of `core`, on training worlds with new layout seeds | with `core` | 2 group-equivalents. With `core`, 8 for every 40 groups of training: about 20% of play |
| `frontier` | 8 | long survival (t044–t054u), the nether and end stages; no game rows (240 minutes) | every 40th checkpoint; outside models | 4–8 hours a subject (the astra episode of t054u took 2 hours 23 minutes) |

`core` by stratum. Players rotate 1, 2, 3, 4 within each stratum (at least 2 for a split kit):

| Stratum | Starts | Rows (examples) | Anchor |
|---|---|---|---|
| Laid out | 2 | t001, t004 | t001: a working harness solves it; a drop means something broke |
| Ore, guided | 4 | t007, t011, t015, t018 | t018 |
| Ore, unguided | 4 | t007u, t013u, t017u, t018u | t013u, t017u |
| Coordination | 4 | t009 (one kit), t013 (one kit), t016 and t020 (split) | t016 |
| Crafting | 4 | t012, t019, t012u, t023u | t019 |
| Survival, short | 4 | t032, t033, t034, t036 | t032 |
| Beyond the frontier | 2 | t024u, t029 | |

- **Held-out worlds.** Training draws worlds from indices 0–11 of `world-{seed}-{index}`. `core` and `frontier`
  use indices 12 to 15. Every evaluation so far ran on training worlds: the astra start of t054u used world
  332881211, which training used for 8 groups, and both earlier ceiling starts used training worlds too.
  `core-seen` against the same anchors in `core` measures how much the policy depends on the 12 worlds.
- **Anchors.** Starts that every later suite repeats exactly. A changed suite is a new suite. Its anchors tie it to
  the old one, so that a trend continues across suites (common items, as tests are equated).
- **Certified starts.** Before a suite is frozen, an outside model plays each start once. A start that cannot be
  built is replaced. A start nobody solves stays only in `frontier`.
- **What is scored.** Solved (primary); reward divided by the row's cap; and, besides the end state, the peak held
  and deaths. The astra team on t054u mined 25 diamonds, held 10 at turn 434, died 28 times and ended with none:
  reward 0. Only the peak tells "cannot find" from "cannot keep".

### The frontier-model reference

| Start | Subject | Outcome | Trained policy on the row |
|---|---|---|---|
| t013, before ways existed, 4 players | `gpt-6-astra` (medium effort) | solved, 28 diamonds, 17 minutes of wall time | t013u: 42% solved, mean reward 3.0 (24 episodes) |
| t018, before ways existed, 4 players | `gpt-6-astra` | solved, 12 diamonds, 12 minutes | t018u: 0 of 4; t018 guided: 60%, mean 4.7 |
| t054u, 4 players | `gpt-6-astra` | reached diamonds in 64 turns, held 10 at turn 434, ended with 0 after 28 deaths; 2 hours 23 minutes of wall time | never unlocked |

One episode per start is a weak reference, for a frontier model as for a checkpoint. On `frontier`, an outside model
should play two episodes per start where the budget allows. Its results are a per-stratum ceiling drawn beside the
checkpoints' trend, not a target the curriculum reads.

### When to evaluate

- Every 10th checkpoint for `core` and `core-seen` (about every 40 groups). Every 40th for `frontier`.
- The base model, before training, once per suite.
- A checkpoint due for evaluation keeps its files until its evaluation ends. On a single machine, where evaluation
  episodes take room from training, every 20th checkpoint is the cheaper schedule.

### Should the curriculum read evaluation results?

No. A sampler that reads held-out results trains on them, and the suite stops measuring generalisation. PLR, SFL
and the reverse curriculum score only training levels. Evaluations feed back through people:

- a stratum that falls by more than two paired standard errors between evaluated checkpoints is flagged (forgetting,
  or a broken harness, which an anchor like t001 shows first);
- a stratum where checkpoints trail the outside model by most is where to add environment rows or guidance;
- a start that certification could not build is evidence for the buildability table.

## How curricula and suites are kept (proposed)

Everything lives in the ledger every run shares (`DatabaseLedger`). Records are appended, never changed.

| Table | Key | Record |
|---|---|---|
| `runs/RUN/curriculum` | the group number from which it applies | `kind` (`learnability`); `group_size`; `prior` (½, ½); `half_life` (8 checkpoints); `partial_credit` (0.25); `parked_below` (0.15); `parked_share` (0.05); `start`, `reach`, `unlock_mean` (½), `unlock_episodes` (8), `relock` (false); `ignore` (rows and a time before which their results are left out); `environment` (its reference) |
| `runs/RUN/groups` | group number | as today, and `depth` (the newest checkpoint's when it was decided), `chance` (the probability the row had), `evidence` (α, β, d of the row then) |
| `environments/ENVIRONMENT/buildable` | `ROW/WORLD` | `built`, `error`, `at` |
| `evaluations/SUITE/suite` | the version's number | as [evals](../libraries/rollout-train/evals.md) keep it, and `purpose`, `worlds` (held-out indices), `strata` (name: what it covers), `derived_from` (the suite whose anchors it repeats); each start with `stratum`, `anchor`, `players` |
| `evaluations/SUITE/EVAL/results` | `START-EPISODE` | as evals keep them, and `peak` (the most of the objective held at any turn), `deaths` |

A curriculum is an environment's rows, a sampling rule and unlock rules. Since it is a fold, its record and the run's
results reproduce every draw. A changed rule is a new record, keyed by the group from which it applies. With
`chance` logged on each group, another rule can be judged offline: reweigh the logged groups by the ratio of the
new rule's probability to `chance`.

A suite keeps its versions under its name (`core@1`, `core@2`); a suite made from another names what it
`derived_from` and repeats its anchors.

The monitor's statistics page ([`#/statistics`](../libraries/rollout-train/monitor.md#the-pages)) already draws
`outcomes` (solve rate and reward over groups), `rows` (each row's groups, share solved, and the same by team size)
and `pace` (what was done with each group, and why groups gave nothing to train on). Three sections would be added:

| Section (proposed) | Reads | Shows |
|---|---|---|
| `curriculum` | `curriculum`, `groups`, `results` | rows by checkpoint as a heat map of the posterior mean; the unlocked count over groups; each step's groups by kind (none solved, mixed, all solved, partial credit only, not built); parked and retired rows; each row's chance when drawn |
| `suites` | `evaluations/*` | per suite, starts by subjects (solved, peak); per stratum the paired difference to the previous evaluated checkpoint with its interval; solve rate against depth, with the outside models as reference lines |
| `buildable` | `environments/*/buildable` | rows by worlds, built or the error |

## Open questions

- **Half-life.** Eight checkpoints is a guess. Too short brings back today's one-group decisions; too long hides
  forgetting. Replaying logged groups under several values would choose it.
- **Families.** Learning progress per row family (a kit ladder, a coordination variant) would have enough data for
  TSCL's slope. It is not modelled here.
- **Hints inside a group.** Guide-GRPO's mixed groups would give unsolved unguided rows a signal without leaving
  the row. That needs the trainer to score tokens under a prompt the episode did not see.
- **Evaluation on one machine.** Evaluation episodes inside the run's `episodes_at_once`, or on workers of their
  own. The schedule above takes about 20% of play; every 20th checkpoint halves it.
