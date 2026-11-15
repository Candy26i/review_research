# E1-H ablation: reading of the results

Companion to `results/e1h_ablation_summary.md` (output of the pre-specified analysis,
`src/analysis/analyze_e1h_ablation.py`). The plan and the reading rule are in that script's
docstring. They were fixed before the main run; three smoke tests (38 rated runs on one to
three papers) had been run before that to test the code. Main run: 2026-11-14,
07:52–08:26 UTC, one invocation, off-peak pricing.

## Question

E1-H found that the same review request is rated about half a point lower through
DeepSeek Harness (`dsh` 0.2.2-rc.1, locked down) than as a bare API call, although the
model never used a tool. `dsh` adds three things to the request content: a 906-character
system prompt, 10 tool definitions, and a 599-character runtime-context block after the
paper. Which of them lowers the rating?

## Design

142 paper-versions of 50 papers, same model and settings as E1-H. Six arms in one queue,
blocked by paper-version (all arms of a paper-version started within half a minute of each
other):

| arm | request | replicates |
|---|---|---|
| bare | plain API call (the E1-H direct arm) | 2 |
| sys | bare + the `dsh` system prompt | 1 |
| tools | bare + the 10 tool definitions | 1 |
| ctx | bare + the runtime-context block | 1 |
| replay | bare + all three, sent by our script | 1 |
| dsh | the real harness | 1 |

The three parts were captured from a real `dsh` request through a local proxy
(`src/review/dsh_request_template.json`). Checks that the arms are what they claim to be:

- The replay request equals the captured request body except for two `dsh` metadata
  fields. By the provider's own token counts, on every one of the 142 paper-versions the
  system prompt adds 185 input tokens, the tool list 1,880 and the context block 117, and
  the replay has exactly the same input size as the real harness and reads the same cache
  entry.
- In all 142 real-harness sessions the system prompt, the tool definitions and the context
  block equal the template, and there was one model step with no tool call.
- Before the queue, every request prefix was sent once with 8 output tokens, so that all
  arms read their input from the provider's prompt cache. 100% of runs in every arm did.

994 runs; 990 usable. Four answers were not valid JSON (an unescaped quote; one each in
sys, tools, replay, dsh) and are left out; including the ratings they contain moves no
contrast by more than 0.01. No run stopped to call a tool.

## Results

Level difference against the concurrent bare call (rating points, 95% interval):

| | difference | |
|---|---|---|
| bare + system prompt | −0.30 [−0.47, −0.14] | Holm p = 0.001 |
| bare + tool definitions | −0.42 [−0.60, −0.25] | Holm p < 0.001 |
| bare + runtime-context block | −0.11 [−0.26, +0.04] | p = 0.17 |
| bare + all three (replay) | −0.44 [−0.59, −0.31] | p < 0.0001 |
| real harness | −0.31 [−0.49, −0.13] | p = 0.002 |

| comparison | difference |
|---|---|
| system prompt vs tool definitions | +0.11 [−0.06, +0.27] |
| system prompt vs context block | −0.17 [−0.37, +0.01] |
| tool definitions vs context block | −0.30 [−0.51, −0.09] |
| replay vs system prompt alone | −0.14 [−0.32, +0.03] |
| replay vs tool definitions alone | −0.03 [−0.22, +0.16] |
| replay vs context block alone | −0.33 [−0.50, −0.17] |
| replay vs real harness | −0.13 [−0.31, +0.04] |
| replay effect minus the sum of the three single effects | +0.43 [+0.10, +0.76], p = 0.016 |
| for scale: one bare replicate minus the other (identical requests) | −0.08 [−0.28, +0.10] |

## Reading, by the rule fixed in advance

1. **The harness effect is present in this session.** The real harness rates 0.31 lower
   than the concurrent bare call [−0.49, −0.13]. That is a third smaller than in E1-H
   (−0.48 overall; −0.57 and −0.33 in its two sessions), not distinguishably so
   (+0.17 [−0.05, +0.40]). The bare arm sits at E1-H's direct level (paper-level mean 4.98
   against 5.00).
2. **Sending the same content without the harness also lowers the rating.** The replay
   rates 0.44 lower than bare. Replay against the real harness: no difference detected
   (−0.13 [−0.31, +0.04]). What the interval leaves open is one-sided: it excludes the
   real harness lowering the rating by more than 0.04 beyond the content; it does not
   exclude the real harness ending up as much as 0.3 higher than the replay. Two identical
   bare requests differ by −0.08 in this run, so −0.13 is within what identical requests
   produce.
3. **Adding the system prompt alone lowers the rating, and so does adding the tool
   definitions alone.** No difference between the two was detected (+0.11 [−0.06, +0.27]),
   so neither meets the rule for "enough on its own" and neither can be said to matter
   more. By the same rule the system prompt cannot be said to matter more than the context
   block either (−0.17 [−0.37, +0.01]).
4. **Adding the runtime-context block alone: no effect detected** (−0.11 [−0.26, +0.04]);
   a lowering of about 0.25 is not excluded. The tool definitions lower the rating more
   than the context block does (−0.30 [−0.51, −0.09], unadjusted).
5. **The parts do not add up.** On the 139 paper-versions usable in all five API arms the
   three single effects sum to −0.88 [−1.25, −0.52], while all three together give −0.45;
   the difference is +0.43 [+0.10, +0.76], p = 0.016. This is one unadjusted test in one
   session, and it is fragile: computed with only the first bare replicate it is
   +0.36 [−0.07, +0.80]. Replay against the tool definitions alone: no difference detected
   (−0.03 [−0.22, +0.16]); against the system prompt alone: no difference detected
   (−0.14 [−0.32, +0.03]); a further lowering of 0.2–0.3 is not excluded in either case.
   No arm combined the system prompt and the tool list without the context block, so this
   run cannot say how much one adds to the other.

## What this does and does not show

- Request content alone, sent by a script, is enough to produce a drop at least as large
  as the real harness's. Within the content, the system prompt alone and the tool list
  alone each lower the rating; for the context block alone no effect was detected.
- It does not show *what about* them matters. The system prompt was not compared with a
  neutral system prompt, nor the `dsh` tool list with an unrelated one, so "being told it
  is a coding agent" is not established; any system prompt, or any tool list, might do the
  same. That sentence is 59 of the system prompt's 906 characters.
- The system prompt and the tool list are not independent pieces of content: 793 of the
  906 characters of the system prompt are instructions for using four of the ten tools
  (the goal tools and the subagent tools). The system-prompt arm therefore already tells
  the model about tools, which in that arm are not defined. Some overlap between the two
  parts is built into the design. Beyond that, the non-additivity is compatible with both
  acting as a cue that this is an agent session rather than a plain question; that is a
  hypothesis, not a result.
- The non-additivity is not produced by the floor at rating 3 (rating 1 is never used).
  It shows at the accept threshold (share rated 6 or higher: +0.17 [+0.05, +0.28]), and it
  remains on the 127 paper-versions that E1-H's direct arm had rated above 3
  (+0.36 [+0.00, +0.72]).
- The single-part arms measure the effect of adding one part to a bare request. They do
  not measure what removing that part from the harness request would do; no
  leave-one-out arm was run. A request with tools and no system prompt is not something
  the harness sends.
- Tool availability changed the scores without any tool being used: 0 tool calls in the
  423 usable runs that had the tool list (tools, replay, real harness).
- Longer reasoning does not account for the drop, as far as this design can tell (length
  is itself an outcome of the arm). On the same paper-version the tool list changes
  reasoning length by +2% [−9%, +13%] while lowering the rating by 0.42; the context block
  lengthens it by 28% [+19%, +37%] with no rating change detected. The system prompt
  lengthens it by 23% and the replay by 78%. Median reasoning length in the real harness,
  read from its session logs, is 35,014 characters against 36,494 for the replay.
- The bare call opens its reasoning with "We need …" in 284 of 284 runs, and so does the
  context-block arm (142 of 142). That opening is therefore the plain-call default. What
  appears only with the system prompt or the tool list are other openings, mostly "Let
  me …" (38% of runs with the system prompt, 60% with the tool list, 33% in the replay,
  40% in the real harness). Both opening groups rate below bare, so the
  exploratory split in `results/e1h_robustness.md` section 5 does not mark a
  harness-specific mode.
- Exploratory, post hoc, unadjusted — a lead, not a finding: the full effect does not vary
  with the human decision (replay −0.43 for the 26 accepted and −0.46 for the 24 rejected
  papers; real harness −0.33 and −0.30), but the single-part effects do in this run.
  Accepted: system prompt −0.48, tool list −0.66. Rejected: system prompt
  −0.12 [−0.33, +0.09], tool list −0.16 [−0.38, +0.06]. This was one of several splits
  tried, and it is not backed by the continuous human score.
- One session, one replicate per arm (two for bare), one model, one prompt, one `dsh`
  version and lock-down. Pairwise contrasts have a standard error of 0.07–0.11, the
  additivity contrast 0.17. The intervals do not include session-to-session variation,
  which in E1-H moved the harness gap by 0.24.
- Retries: 7 connection errors in API arms (4 of them in bare), no empty-answer resamples;
  13 real-harness runs had one internal retry after a transport failure. Dropping all runs
  with a retry changes none of the six main contrasts by more than 0.02. Among the other
  contrasts, replay against the system prompt alone moves from −0.14 to −0.16 and its
  interval then just excludes zero ([−0.33, −0.00]).

## Cost

The account balance fell from ¥104.13 before the ablation work to ¥64.72 after it, about
¥39 including the request capture and three smoke tests. The token-based estimate is
¥40.46 for the main invocation (994 runs ¥35.69, cache warm-up ¥4.78) plus about ¥1.8 for
the smoke tests. Mean estimated cost per run: bare ¥0.027, sys ¥0.035, tools ¥0.033, ctx
¥0.034, replay ¥0.050, real harness ¥0.045.

## Reproduce

```
python3 src/review/run_e1_harness_ablation.py --workers 26   # the invocation used
python3 src/analysis/analyze_e1h_ablation.py                 # -> results/e1h_ablation_summary.md
```

Running the first command again would retry the four invalid-JSON runs and add a second
invocation to the log. Log: `data/processed/e1h_ablation_log.jsonl`. Raw outputs:
`data/processed/reviews_e1h_ablation/` (full reasoning text for the five API arms; the
harness event streams cap reasoning at about 8 KiB, and the full harness reasoning is in
the session logs under `.tools/dsh-home/sessions/`). All of these are git-ignored.
