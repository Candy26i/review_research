# E1-H: robustness and audit notes

Companion to `results/e1h_summary.md` (the planned analysis). Everything here was
computed on the final logs: 1,308 logged attempts, 426 usable runs in each arm
(419 "clean" runs in B1). Run dates: 2026-11-12 and 2026-11-13 (UTC).

## 1. What the three arms are

All arms use DeepSeek `deepseek-flash` with thinking on, effort high, max_tokens 256,000,
no temperature setting, and the same reviewing prompt (`src/review/prompts/review_v1.md`).

| arm | request |
|---|---|
| A `direct` | One API call. A single user message = review rules + paper. No system prompt, no tools. |
| B0 `harness` | The same string piped into DeepSeek Harness (`dsh` 0.2.2-rc.1, headless, lock-down patch `src/review/harness_lockdown.yml`). |
| B1 `harness_sub` | `dsh`, two turns. Turn 1: the paper, with "reply READY". Turn 2: a required process — call `subagent_fork` once, wait for the subagent's review report, then output the final review JSON. The process is verified by program from the `dsh` session log. |

What `dsh` adds to the request in B0 (read from the session logs, identical in all 426 runs):

- its own 906-character system prompt, which tells the model it is a coding agent;
- 10 tool definitions (`create_goal`, `get_goal`, `interrupt_agent`, `list_agents`,
  `send_message`, `skill`, `subagent`, `subagent_fork`, `todo_write`, `update_goal`);
- a 599-character runtime-context block placed after the paper (working directory,
  read-only file policy, approvals disabled);
- request metadata: harness headers, a plugin list, and delivery of the session log to
  DeepSeek.

The first three are a constant 2,182 extra input tokens for every paper-version. The task
text recorded by `dsh` equals the string sent in the direct arm in 426 of 426 B0 sessions.
The system prompt and tool list are what the lock-down patch leaves, not the stock `dsh`
profile (web, shell, file and workflow tools are removed). That `dsh` sends the same
thinking and effort settings as the direct call rests on one request captured through a
local proxy before the runs and on the `maxTokens` / `reasoningEffort` values in the
session logs; the direct arm's request body is not saved.

**Consequence for interpretation:** the B0 − A contrast is the effect of this whole bundle
against a bare request with no system prompt at all. These data cannot say which part
matters (a follow-up ablation adding one part at a time is in
`results/e1h_ablation_notes.md`), whether any system prompt would do the same, or whether the server treats
requests carrying harness metadata differently. It is a statement about `dsh` 0.2.2-rc.1
in this locked-down headless configuration, not about agent harnesses in general.

## 2. Headline results (planned analysis)

50 papers, 142 paper-versions, 3 replicates per paper-version per arm (in the primary B1
set, 135 paper-versions have 3 clean replicates and 7 have 2). Slopes are rating points
per revision step, pooled within paper (Sxx-weighted). Intervals are 95%.

| arm | pooled slope | mean rating (paper level) |
|---|---|---|
| A direct | +0.184 [+0.071, +0.302] | 5.003 |
| B0 harness | +0.116 [−0.011, +0.247] | 4.520 |
| B1 harness + forced subagent review | +0.094 [−0.008, +0.187] | 4.246 |

| comparison | slope difference | sign-flip p | level difference | sign-flip p |
|---|---|---|---|---|
| B0 − A (primary) | −0.068 [−0.200, +0.048] | 0.27 | −0.483 [−0.615, −0.359] | < 0.0001 |
| B1 − A | −0.090 [−0.244, +0.060] | 0.29 | −0.757 [−0.908, −0.615] | < 0.0001 |
| B1 − B0 | −0.023 [−0.165, +0.113] | 0.81 | −0.274 [−0.378, −0.168] | < 0.0001 |

Unweighted per-paper slope differences (secondary estimate): −0.068 (p = 0.53),
−0.003 (p = 0.97), +0.065 (p = 0.44).

Reading:

- No slope difference between arms is detectable. The intervals for the differences run
  from "no slope at all under the harness" to "a slope about a quarter larger than the
  direct arm's", so the data show neither that the slopes are equal nor that they differ.
- Only the direct arm's own interval excludes zero. That is not evidence of a difference
  between arms; the tests of the differences give p = 0.27 and 0.29.
- The three pooled slopes should not be read as a falling sequence. The unweighted
  per-paper means are +0.205, +0.136 and +0.201, and three long papers carry 45% of the
  pooled weight.
- The B0 − A and B1 − A level differences are clear and present in both run sessions:
  about half a point and about three quarters of a point. The further quarter point
  between B1 and B0 is not stable across sessions (section 7).
- The arms rank the papers almost identically (paper-level Spearman correlation: A with
  B0 0.92, A with B1 0.88, B0 with B1 0.94). The harness changes how harsh the scores
  are, not which papers are preferred.

## 3. Independent recomputation

`src/analysis/e1h_checks/recompute.py` was written from a prose specification by an agent
instructed not to open `analyze_e1h.py` or any earlier result. On the final logs it
reproduces every estimate above to the third decimal (pooled differences −0.0676, −0.0902,
−0.0225; level differences −0.4832, −0.7571, −0.2740).

A second checker enumerated the sign-flip distribution exactly: p = 0.273, 0.288 and
0.808. This exposed a tie-handling defect in the planned script's sign-flip test (its
tolerance was smaller than the rounding applied to its inputs, so part of the exactly tied
sign patterns were dropped). The tolerance was corrected after the data were in; the
script's Monte-Carlo p-values moved from 0.270 / 0.289 / 0.799 to 0.275 / 0.291 / 0.809.
No estimate, interval or conclusion changed.

## 4. Robustness of the slope comparison

From `src/analysis/e1h_checks/regression.py`. Estimates are points per step unless noted.

| model | B0 − A | B1 − A | B1 − B0 |
|---|---|---|---|
| planned pooled estimate, sign-flip | −0.068, p = 0.27 | −0.090, p = 0.29 | −0.023, p = 0.81 |
| fixed-effects regression on replicate rows, SE clustered by paper (CR2) | −0.068 [−0.194, +0.058], p = 0.26 | −0.087 [−0.257, +0.082], p = 0.28 | −0.020 [−0.185, +0.145], p = 0.80 |
| unweighted mean of per-paper differences, paired t | −0.068 [−0.287, +0.150], p = 0.53 | −0.003 [−0.210, +0.203], p = 0.97 | +0.065 [−0.101, +0.231], p = 0.44 |
| mixed model (random intercept and slope by paper, paper-version component) | −0.027 [−0.120, +0.066], p = 0.57 | −0.052 [−0.145, +0.040], p = 0.27 | −0.023 [−0.112, +0.066], p = 0.61 |
| rating recoded to its ordinal index (index units) | −0.040 [−0.118, +0.038], p = 0.28 | −0.073 [−0.174, +0.028], p = 0.14 | −0.033 [−0.135, +0.069], p = 0.50 |
| first vs last version only, per step, paired t | −0.062, p = 0.57 | −0.002, p = 0.98 | +0.060, p = 0.47 |
| leave-one-paper-out range of the planned estimate | −0.082 to −0.033 | −0.121 to −0.041 | −0.058 to +0.037 |
| B1 using every verified run (`--b1-all`) | — | −0.093, p = 0.27 | −0.025, p = 0.79 |

- Joint test that both harness arms have the direct arm's slope: F(2, 48) = 0.95,
  p = 0.40 (wild cluster bootstrap p = 0.47).
- Precision is limited by a few long papers: the three papers with 6–7 versions carry 45%
  of the pooled weight, and the effective number of papers is about 12 of 50. The
  small-sample-corrected interval for the primary difference, [−0.194, +0.058], is wider
  than the direct arm's slope itself (width 0.25 against a slope of 0.18).
- One exploratory result goes the other way and should be read as a lead, not a finding:
  when the rating is recoded to "6 or higher" (which is exactly the accept recommendation
  in all 1,271 usable runs), the share rises with version in A (+0.065 per step) and B0
  (+0.052) but is flat in B1 (+0.007). B1 − A: −0.059 [−0.112, −0.005], p between 0.02 and
  0.06 depending on the test; B1 − B0: −0.046, p between 0.03 and 0.07. These are the only
  2 of 61 slope-difference rows with any p below 0.05, among 131 p-values with no
  multiplicity correction, and nothing similar appears at the "5 or higher" or "8 or
  higher" thresholds.
- The mixed models needed derivative-free optimizers in some fits and several are at the
  boundary (a variance component at zero); their fixed effects are stable across optimizers.

### Length control

Later versions are longer (about 7% per step within a paper), and in the original E1 the
DeepSeek version effect was absorbed by document length (`docs/FINDINGS.md`). The same
happens here. In a paper-fixed-effects regression of the paper-version mean rating on
position, adding log text length changes the slope from +0.184 to +0.098 (p = 0.06) in A,
from +0.116 to 0.000 in B0, and from +0.094 to +0.013 in B1. With a common length term the
differences between arms stay undetectable (B0 − A −0.068, p = 0.24; B1 − A −0.090,
p = 0.23). This check was added after the data were in.

## 5. The level difference

| | A direct | B0 harness | B1 harness_sub |
|---|---|---|---|
| mean rating over runs | 5.026 | 4.566 | 4.274 |
| share of ratings equal to 3 | 26.1% | 39.7% | 46.8% |
| share equal to 5 | 30.5% | 31.9% | 34.6% |
| share equal to 6 | 37.8% | 24.6% | 17.4% |
| share equal to 8 | 5.6% | 3.8% | 1.2% |
| accept = true (same as rating 6 or higher in every run) | 43.4% | 28.4% | 18.6% |
| paper-versions where all replicates are 3 | 15 of 142 | 32 of 142 | 43 of 142 |
| papers where every rating of every version is 3 | 2 of 50 | 8 of 50 | 11 of 50 |

- Ratings 1 and 10 never occur in 1,271 usable runs, so 3 is the effective floor.
- B0 is below A in 38 of 50 papers (4 above, 8 equal); B1 is below A in 45 of 50
  (1 above, 4 equal); B1 is below B0 in 33 of 50 (6 above, 11 equal).
- The B0 − A gap is a move into rating 3 more than a uniform shift: 39% of it is 5 → 3,
  31% is 6 → 3, 22% is 6 → 5.
- It is not a parsing artifact: every usable row was re-derived by two independent scripts
  (from the raw output, or from the `dsh` session log for the few B1 rows whose saved
  stream holds no complete review) with zero rating mismatches and no coerced values.
- It does not vary detectably with version position (−0.43, −0.41, −0.56 at positions
  0, 1, 2+), with the human decision (accepted −0.45, rejected −0.48), or with retries.
- In regression form (level at the mean position, CR2): B0 − A −0.460 [−0.605, −0.315],
  B1 − A −0.746 [−0.900, −0.592], B1 − B0 −0.286 [−0.393, −0.179].
- Floor: a paper whose every rating is 3 has a slope of zero by construction, and the
  harness arms have more of them. A floor can only make harness slopes look smaller, so
  it is a reason not to read the lower point estimates as lower sensitivity to revisions.
  These papers are mostly short two-version chains with little pooled weight.
- B0 uses 1.76 times the output tokens and thinking length of A on the same input.
  Adjusting for output length moves the gap from −0.460 to −0.406, but length is itself an
  outcome of the arm, so this does not separate the two.
- Exploratory: B0 runs fall into two groups by how the reasoning opens. In 276 of 426 it
  opens with "We need …"; these runs produce about twice the output tokens of a direct call
  (12,751) and score 0.57 below the direct arm on the same paper-version. The other 150
  produce about the direct arm's amount (7,167) and score 0.26 below. Both groups are
  lower. The direct arm's reasoning text was not stored, so the same split cannot be made
  there. (Later ablation, `results/e1h_ablation_notes.md`: a bare call opens with "We
  need …" in every run, so that opening is the plain-call default, not a harness mode.)

## 6. Data integrity

- Keys: 1,278 distinct runs, none outside the 142 × 3 design, no run counted twice.
  Every run that failed once was later completed (7 in B0, 23 in B1).
- Configuration: prompt hash, lock-down patch hash, resolved `dsh` configuration hash,
  `dsh` version, model and effort are constant within each arm and equal across arms.
  One B1 run (2308.16512v3 r2) ran in a different working directory, so its
  runtime-context block differs from the other 425.
- Inputs: one text hash per paper-version across all arms and attempts, equal to the file
  on disk. No truncated or coerced rows.
- Scores: all 1,278 parsed rows re-derive identically on all six score fields.
- B0 behaviour: 426 of 426 runs finished in a single model step with no tool call.
- Sleep: the laptop went through two sleep episodes during the runs (about 10:06–16:02 UTC
  and from about 22:26 UTC on 11-12). All 30 failed attempts fall in the second episode.
  Eight usable rows (6 direct, 1 B0, 1 B1) were produced during brief wake-ups of the
  first; dropping them changes no planned estimate by more than 0.004. `latency_s` is
  taken from a monotonic clock and does not show a sleep.
- Dropping every row with a retry or failed attempt (8 / 19 / 70 rows), or every row
  completed within three minutes of a wake-up, leaves the level differences within 0.02
  of the full-sample values and all slope differences inside their reported intervals.

## 7. Two run sessions

Almost all runs fall into two sessions: 2026-11-12 21:49–22:26 UTC (off-peak pricing) and
2026-11-13 02:56–03:53 UTC (peak pricing). Session and price period cannot be separated.
B1 has a larger share of its runs in the second session (58%) than A (37%) and B0 (42%).

| level difference (mean over matched paper-versions) | first session only | second session only | all runs |
|---|---|---|---|
| B0 − A | −0.57 [−0.74, −0.41] | −0.33 [−0.54, −0.11] | −0.46 |
| B1 − A | −0.73 | −0.64 | −0.75 |
| B1 − B0 | −0.11 [−0.27, +0.06] | −0.41 [−0.59, −0.23] | −0.29 |

- All three gaps are negative in both sessions, and the B0 − A and B1 − A gaps exclude
  zero in both.
- The gaps differ by session more than chance alone would suggest (nominal, uncorrected
  tests: B0 − A differs by 0.24, B1 − B0 by 0.30). The B1 − B0 gap is not distinguishable
  from zero in the first session. The intervals in section 2 do not include this
  session-to-session variation.
- Within an arm, the same paper-version was rated differently in the two sessions (second
  minus first: A −0.18, p = 0.05; B0 +0.10; B1 −0.12; pooled −0.08, p = 0.16), without a
  consistent direction.
- The primary slope difference is +0.07 on first-session runs alone and +0.00 on
  second-session runs alone (−0.068 overall); these halves are noisy and both inside the
  reported interval. Version position is not detectably associated with session.

## 8. Placebo: versions with identical text

Nine of the 92 adjacent version steps have byte-identical text, so the reviewer received
the same input for both versions. On those steps the expected rating change is zero.
Observed later-minus-earlier change: A +0.52, B0 +0.11, B1 +0.30.

- Permuting replicate labels within each step: p = 0.065, 0.79 and 0.27; the three arms
  pooled +0.31, p = 0.046.
- Flipping signs over the nine steps: p = 0.016 for A (seven steps positive, none
  negative), 0.81 for B0, 0.51 for B1.

The prompt carries no version label and the inputs are identical, so no mechanism is
known and this is most likely noise; but for the direct arm it is borderline rather than
clearly compatible with noise. The apparent change on identical text in A (+0.52) is
almost three times that arm's estimated slope per step (+0.18). This is one more reason
not to read the direct arm's positive slope as proof that the scores track the text.
Pooling replicates across identical versions changes the slopes to +0.167 / +0.113 /
+0.084 and the differences to −0.055 / −0.083 / −0.028.

## 9. What the forced subagent step did (B1)

- Process: 426 of 426 planned runs were verified (subagent had 100% of the paper, its
  report was delivered, the final review came after it). No process failures; 23 failed
  attempts (technical) were re-run. Two runs needed a reminder and five wrote a review
  before the report; these seven are excluded from the primary B1 set (`--b1-all`
  includes them and changes nothing).
- An independent decoder of the session logs agrees with the runner's verdict on every run.
- The instruction also said to call no other tool. In 85 of 426 runs the main agent did:
  `list_agents` (71 runs), `send_message` to nudge the subagent (13), `get_goal` (3), a
  second `subagent_fork` that was rejected (1). This was not part of the planned "clean"
  filter. Dropping these runs gives B1 − A level −0.756, slope difference −0.070; B1 − B0
  level −0.261, slope difference −0.012.
- The main agent passes the subagent's score through. In the 377 of 426 runs where the
  subagent's report states a single rating, the final rating is identical in 362 (96.0%);
  the main agent raised it in 14 runs and lowered it in 1. In the other 49 runs the report
  gave a range or no number, so the comparison cannot be made.
- The subagent's prompt is written by the main agent and differs in every run (median about
  1,200 characters). 70% contain directed fault-finding terms and 17% an explicit stance
  word such as "critical" or "skeptical". Whether this wording lowers scores is not
  established: within the same paper-version, runs with such wording score 0.16 lower,
  [−0.39, +0.08].
- The subagent saw turn 1 (the paper) and its own prompt, never the turn-2 process
  instruction.
- Cost of the step: 7.2 model calls per run (5.2 by the main agent, 2.0 by the subagent).
  A B1 run produces 4.2 times the output tokens of a direct call (2.4 times B0); the
  subagent itself produces 76% of that. Input tokens are about 10 times a direct call's,
  mostly cache reads.
- B1 − B0 bundles the two-turn delivery with the subagent step. Separating them needs a
  control arm with the same two turns and no subagent, which was not run.

## 10. Human scores as an outside reference (descriptive)

The manifest holds the ICLR reviewers' mean score for each paper (mean 5.99 on the same
scale). Human scores belong to the submitted version, which need not match any arXiv
version.

| | A direct | B0 harness | B1 harness_sub |
|---|---|---|---|
| LLM paper mean minus human mean | −0.99 [−1.34, −0.63] | −1.47 [−1.83, −1.10] | −1.74 [−2.07, −1.41] |
| mean absolute difference | 1.31 | 1.67 | 1.82 |
| Spearman correlation with human mean | +0.20 (p = 0.16) | +0.26 (p = 0.07) | +0.27 (p = 0.06) |
| AUC, accepted vs rejected | 0.675 | 0.679 | 0.671 |

Every arm rates below the human reviewers, and the harness arms are further below. The
rank correlation with human scores is weak in every arm, and no difference between arms
is detectable (the bootstrap intervals for the differences all include zero). Since the
arms rank the papers almost identically (section 2), they could hardly differ here.

## 11. Cost and time

| | A direct | B0 harness | B1 harness_sub |
|---|---|---|---|
| mean output tokens per run | 6,133 | 10,785 | 25,949 |
| mean input tokens per run (incl. cache reads) | 17,323 | 19,505 | 172,363 |
| median latency | 37 s | 62 s | 133 s |
| mean cost per run at list price | ¥0.042 | ¥0.069 | ¥0.195 |

Costs mix off-peak and peak prices (37% / 42% / 58% of runs at peak), so the cost ratios
overstate B1 relative to A; at a uniform price a B1 run costs about 3.9 times a direct
call. The list-price estimate over all usable runs is about ¥130; by account-balance
readings taken before and after the runs the three arms cost about ¥125.

## 12. Limits

- 50 papers; about 12 effective papers for the pooled slope. The design can detect a slope
  difference of roughly 0.2–0.3 points per step. That is as large as the direct arm's
  whole slope (+0.18): a complete loss of the slope would probably have been detected, a
  halving would have been missed about two times out of three.
- One model, one prompt, one `dsh` version and configuration, three replicates, two run
  sessions.
- The cause of the level difference is not isolated (section 1). What can be said is that
  it is not tool use or extra agent steps: B0 had no tools that could change a review (no
  web, shell or files) and used none of the remaining ones. This experiment therefore says
  nothing about a harness that is allowed to search or run code.
- B1 − B0 is confounded with two-turn delivery (section 9) and is the comparison most
  affected by session (section 7).
- A larger slope means later versions are scored higher. It does not show the scores are
  more accurate; part of it goes with length (section 4), and identical texts also showed
  an apparent increase (section 8).

## 13. Reproduce

```
python3 src/analysis/analyze_e1h.py                 # planned analysis -> results/e1h_summary.md
python3 src/analysis/analyze_e1h.py --b1-all --out /tmp/e1h_b1all.md
python3 src/analysis/e1h_checks/recompute.py   --logs data/processed/e1h_reviews_log.jsonl data/processed/e1h_sub_reviews_log.jsonl
python3 src/analysis/e1h_checks/regression.py  --logs data/processed/e1h_reviews_log.jsonl data/processed/e1h_sub_reviews_log.jsonl
python3 src/analysis/e1h_checks/audit_integrity.py --logs data/processed/e1h_reviews_log.jsonl data/processed/e1h_sub_reviews_log.jsonl --out /tmp/e1h_integrity
python3 src/analysis/e1h_checks/level_audit.py --logs data/processed/e1h_reviews_log.jsonl data/processed/e1h_sub_reviews_log.jsonl
python3 src/analysis/e1h_checks/b1/audit_b1.py --logs data/processed/e1h_sub_reviews_log.jsonl --main-log data/processed/e1h_reviews_log.jsonl --out-dir /tmp/e1h_b1
```

The scripts under `src/analysis/e1h_checks/` were written by independent audit agents on a
frozen interim snapshot and then re-run on the final logs. The numbers in sections 2–11
were afterwards checked figure by figure by three further agents. `data/` and `.tools/`
are git-ignored, so the logs, manifest, texts, raw outputs and `dsh` session logs that
these commands read exist only on this machine.
