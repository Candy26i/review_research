# ACL 2025 version tracking: gpt-6.1-sol vs deepseek-chat, three reviews per version

- gpt log: `data/processed/acl_reviews_gpt61sol_log.jsonl` | DeepSeek log: `data/processed/acl_reviews_log.jsonl`
- matched versions (three gpt-6.1-sol reviews and three DeepSeek reviews): 3617 versions of 1374 papers with >= 2 matched versions; position = index among the paper's matched versions. gpt versions without all three reviews: 0.
- gpt-6.1-sol mean of 3 = mean of reps 0, 1, 2 (rep 0: main run, 2026-11-16; rep 1: 500-version retest of 2026-11-16 + the three-review run; rep 2: the three-review run, 2026-11-17/18; reasoning_effort not sent (API default, medium), temperature 1.0, flex tier except 60 default-tier smoke/pilot rep-0 reviews; three API keys, see the key check).
- Primary contrast, fixed in the run notes before the extra reviews were called (after the one-review results had been seen): gpt mean of 3 vs DeepSeek mean of 3, by the within-paper (fixed-effects) slope. Everything else is secondary. DeepSeek r0 is a reference column; the one-review comparison is `results/acl_gpt61sol_vs_deepseek.md`.
- DeepSeek: requests went to the deepseek-chat alias; every raw response reports model deepseek-flash with one system_fingerprint (e72d3c7f), and all three reps were called interleaved within one session, 2026-10-23 08:51-15:14 UTC.
- 201 papers with >= 4 versions carry 61% of the weight of the within-paper slope.

## Rating change per revision step

| estimate | gpt-6.1-sol mean of 3 | DeepSeek r0 (1 review, reference) | DeepSeek mean of 3 |
|---|---|---|---|
| within-paper slope [95% CI], sign-flip p | +0.051 [+0.035, +0.067], p < 0.001 | +0.123 [+0.091, +0.155], p < 0.001 | +0.111 [+0.090, +0.132], p < 0.001 |
| same, with ln(words) covariate | +0.022 [+0.005, +0.046] | +0.089 [+0.057, +0.126] | +0.074 [+0.053, +0.100] |
| mixed model, random intercept | +0.049 [+0.036, +0.062], p < 0.001 | +0.104 [+0.073, +0.135], p < 0.001 | +0.102 [+0.083, +0.121], p < 0.001 |
| mixed model + ln(words) | +0.026 [+0.012, +0.039] | +0.068 [+0.036, +0.099] | +0.067 [+0.048, +0.087] |
| mixed model, random intercept and slope | +0.054 [+0.038, +0.070] | +0.107 [+0.075, +0.140] | +0.108 [+0.086, +0.130] |
| equal weight per paper (mean of per-paper slopes) | +0.059 [+0.038, +0.081] | +0.137 [+0.085, +0.191] | +0.124 [+0.092, +0.157] |
| SD of ratings | 0.95 | 1.19 | 0.93 |
| standardized within-paper slope (slope / SD; 3-review arms: SD of 3-review means) | +0.054 [+0.037, +0.070] | +0.103 [+0.076, +0.130] | +0.118 [+0.097, +0.141] |
| share rated >= 5 (3-review arms: at least 2 of 3 reviews >= 5) | 38.3% | 39.2% | 36.5% |
| P(rating >= 5), percentage points per step (3-review arms: at least 2 of 3 reviews >= 5) | +2.3 [+1.3, +3.2], p < 0.001 | +4.8 [+3.4, +6.1], p < 0.001 | +4.3 [+3.0, +5.6], p < 0.001 |
| conditional logit, log-odds of >= 5 per step (3-review arms: at least 2 of 3 reviews >= 5) | +0.51 [+0.32, +0.70] (165 papers with variation) | +0.32 [+0.22, +0.42] (498 papers with variation) | +0.37 [+0.26, +0.49] (389 papers with variation) |

## Slope relative to each model's own repeat-call noise (secondary, not in the plan)

Noise from the three reviews of each of the 3617 matched versions: pooled within-version variance (ddof 1) = sigma^2 of one review; a 3-review mean has sigma / sqrt(3). Noise-scaled slope = slope of the 3-review mean / sigma of one review. Joint cluster bootstrap over papers.

Read with care: (1) gpt's ratings are almost only 3 or 5, so its noise comes from the 13% of versions whose three reviews disagree, and on a near-binary scale slope / sigma depends on how many versions sit near the 3/5 cut; (2) gpt's three reviews were called on different days with three API keys, DeepSeek's within one session; (3) the 500-version retest (`results/acl_gpt61sol_retest.md`) is a subset of these reviews, not an independent check. A ratio near 1 means no detectable difference, not equal sensitivity.

| | gpt-6.1-sol | DeepSeek | gpt minus DeepSeek |
|---|---|---|---|
| repeat-call SD of one review [95% CI] | 0.41 [0.38, 0.43] | 0.88 [0.85, 0.90] | |
| SD of a 3-review mean (sigma / sqrt(3)) | 0.23 | 0.51 | |
| versions with all three reviews equal | 87.1% | 46.9% | |
| within-paper slope (mean of 3) / SD of one review | +0.125 [+0.087, +0.164] | +0.126 [+0.103, +0.150] | -0.001 [-0.042, +0.038]; ratio 0.99 [0.68, 1.31] |

## Check: each gpt-6.1-sol review on its own

| | within-paper slope [95% CI] | mean rating | share >= 5 | rating counts |
|---|---|---|---|---|
| rep 0 | +0.045 [+0.025, +0.065] | 3.779 | 38.3% | 1.0: 7, 3.0: 2224, 5.0: 1326, 6.0: 60 |
| rep 1 | +0.068 [+0.046, +0.090] | 3.786 | 38.8% | 1.0: 8, 3.0: 2205, 5.0: 1352, 6.0: 52 |
| rep 2 | +0.039 [+0.020, +0.059] | 3.770 | 38.0% | 1.0: 8, 3.0: 2234, 5.0: 1324, 6.0: 51 |
| rep 1 minus rep 0 | +0.023 [+0.002, +0.044] | +0.007 | | |
| rep 2 minus rep 0 | -0.006 [-0.028, +0.018] | -0.009 | | |
| rep 2 minus rep 1 | -0.029 [-0.054, -0.002] | -0.016 | | |

DeepSeek's reps on the same versions: slopes rep 0 +0.123 [+0.091, +0.155], rep 1 +0.097 [+0.066, +0.131], rep 2 +0.112 [+0.076, +0.148]; mean ratings 3.898 / 3.863 / 3.835.

Within-version rep-label permutation test (shuffle the three reviews' labels within each version, 10,000 permutations; statistic = largest minus smallest single-review slope): gpt spread 0.029, p = 0.047; DeepSeek spread 0.025, p = 0.506. Many rep, key and level checks were run, so a p near 0.05 is not interpreted.

Post hoc sensitivity without gpt rep 1 (mean of reps 0 and 2): slope +0.042 [+0.027, +0.059]; minus DeepSeek mean of 3 -0.068 [-0.092, -0.045].

CI width of the mean-of-3 slope relative to a single review's CI width: vs rep 0 79%, vs rep 1 72%, vs rep 2 80%.

## Check: API key

Deviation of each review from its version's 3-review mean, by the API key it was called with (key from the response's created time: first key before 2026-11-16 05:00 UTC, third key from 2026-11-17 07:30:45 UTC; run notes). The version mean includes the review itself, so a review whose key supplies n of the version's three reviews shows (1 - n/3) of that key's level shift: about 2/3 for key 1 (always one review per version), less for keys 2 and 3, and key 2's row is near 0 partly by construction (it supplies all three reviews of many versions). Key is tied to rep and call date (matched versions; reviews by key and rep: key 1 1045/0/0; key 2 2572/2588/2423; key 3 0/1029/1194 for reps 0/1/2), so this check cannot separate key, rep and date.

| key | reviews | mean deviation [95% CI] |
|---|---|---|
| 1 | 1045 | -0.013 [-0.032, +0.007] |
| 2 | 7583 | +0.000 [-0.004, +0.005] |
| 3 | 2223 | +0.005 [-0.008, +0.018] |

## Direction of changed adjacent pairs

Share of changed adjacent pairs that go up (pairs with delta != 0):

| pair class | gpt-6.1-sol mean of 3 | DeepSeek r0 (1 review, reference) | DeepSeek mean of 3 |
|---|---|---|---|
| identical text | 46% of 71 | 63% of 125 | 50% of 210 |
| cosmetic | 53% of 70 | 51% of 195 | 56% of 291 |
| minor | 60% of 85 | 52% of 192 | 59% of 287 |
| major | 72% of 223 | 64% of 359 | 65% of 589 |
| ALL | 63% of 449 | 58% of 871 | 60% of 1377 |

## Difference gpt-6.1-sol minus DeepSeek (same versions, paired by paper)

| estimate | vs DeepSeek mean of 3 (primary) | vs DeepSeek r0 (reference) |
|---|---|---|
| within-paper slope | -0.060 [-0.083, -0.039], p < 0.001 | -0.072 [-0.106, -0.039], p < 0.001 |
| with ln(words) covariate | -0.052 [-0.077, -0.026] | -0.067 [-0.104, -0.032] |
| difference of standardized slopes (each arm's slope / that arm's own SD; SD of 3-review means for the 3-review arms) | -0.065 [-0.091, -0.041] | -0.049 [-0.079, -0.021] |
| P(rating >= 5), points per step (3-review arms: at least 2 of 3 reviews >= 5) | -2.0 [-3.5, -0.6], p = 0.010 | -2.5 [-4.0, -1.0], p = 0.002 |
| stacked mixed model, position x arm (random intercept per paper and per paper x arm; common residual variance; model-based CI) | -0.053 [-0.076, -0.031], p < 0.001 | - |

## Adjacent version pairs

Delta = rating of the later version minus the earlier one. Mean delta CI: cluster bootstrap over papers.

**gpt-6.1-sol mean of 3**

| pair class | n | P(up) | P(tie) | P(down) | mean delta [95% CI] | mean abs delta |
|---|---|---|---|---|---|---|
| identical text | 370 | 8.9% | 80.8% | 10.3% | -0.02 [-0.05, +0.01] | 0.16 |
| cosmetic | 492 | 7.5% | 85.8% | 6.7% | +0.01 [-0.02, +0.04] | 0.11 |
| minor | 464 | 11.0% | 81.7% | 7.3% | +0.03 [-0.01, +0.07] | 0.19 |
| major | 917 | 17.6% | 75.7% | 6.8% | +0.13 [+0.09, +0.16] | 0.26 |
| ALL | 2243 | 12.6% | 80.0% | 7.4% | +0.06 [+0.04, +0.07] | 0.19 |

**DeepSeek r0 (1 review, reference)**

| pair class | n | P(up) | P(tie) | P(down) | mean delta [95% CI] | mean abs delta |
|---|---|---|---|---|---|---|
| identical text | 370 | 21.4% | 66.2% | 12.4% | +0.13 [+0.01, +0.25] | 0.65 |
| cosmetic | 492 | 20.1% | 60.4% | 19.5% | +0.07 [-0.03, +0.17] | 0.73 |
| minor | 464 | 21.3% | 58.6% | 20.0% | -0.01 [-0.12, +0.11] | 0.77 |
| major | 917 | 25.0% | 60.9% | 14.2% | +0.22 [+0.15, +0.30] | 0.78 |
| ALL | 2243 | 22.6% | 61.2% | 16.3% | +0.13 [+0.09, +0.17] | 0.75 |

**DeepSeek mean of 3**

| pair class | n | P(up) | P(tie) | P(down) | mean delta [95% CI] | mean abs delta |
|---|---|---|---|---|---|---|
| identical text | 370 | 28.1% | 43.2% | 28.6% | +0.01 [-0.06, +0.08] | 0.47 |
| cosmetic | 492 | 33.3% | 40.9% | 25.8% | +0.07 [+0.01, +0.13] | 0.49 |
| minor | 464 | 36.2% | 38.1% | 25.6% | +0.10 [+0.04, +0.17] | 0.52 |
| major | 917 | 42.0% | 35.8% | 22.2% | +0.20 [+0.15, +0.25] | 0.57 |
| ALL | 2243 | 36.6% | 38.6% | 24.8% | +0.12 [+0.10, +0.15] | 0.52 |

## Sensitivity: versions with text identical to the previous version dropped

3138 versions of 1265 papers.

| arm | within-paper slope [95% CI] |
|---|---|
| gpt-6.1-sol mean of 3 | +0.061 [+0.043, +0.081] |
| DeepSeek r0 (1 review, reference) | +0.131 [+0.093, +0.172] |
| DeepSeek mean of 3 | +0.133 [+0.109, +0.157] |
| gpt minus DeepSeek mean of 3 | -0.072 [-0.099, -0.045] |

## Sensitivity: versions under 1,000 words dropped (text-extraction failures)

Dropped: 2412.12567v1, 2503.15351v1 (every other matched version has >= 1,944 words): near-empty texts (a title with an empty heading; one appendix paragraph) that carry most of the leverage on the ln(words) coefficient. Their ratings: 2412.12567v1 gpt-6.1-sol mean of 3 1.00, DeepSeek r0 (1 review, reference) 1.00, DeepSeek mean of 3 1.00; 2503.15351v1 gpt-6.1-sol mean of 3 1.67, DeepSeek r0 (1 review, reference) 1.00, DeepSeek mean of 3 1.00. Papers then left with one version leave the frame: 2503.15351. Remaining: 3614 versions of 1373 papers.

| arm | within-paper slope [95% CI] | with ln(words) covariate |
|---|---|---|
| gpt-6.1-sol mean of 3 | +0.048 [+0.034, +0.063] | +0.036 [+0.020, +0.053] |
| DeepSeek r0 (1 review, reference) | +0.120 [+0.090, +0.150] | +0.098 [+0.061, +0.133] |
| DeepSeek mean of 3 | +0.108 [+0.087, +0.128] | +0.081 [+0.056, +0.105] |
| gpt minus DeepSeek mean of 3 | -0.060 [-0.082, -0.037] | -0.045 [-0.071, -0.017] |
| gpt minus DeepSeek r0 (1 review, reference) | -0.072 [-0.104, -0.039] | -0.061 [-0.099, -0.025] |

## Rating histograms (matched versions)

- DeepSeek r0 (1 review, reference): 1.0: 22, 3.0: 2176, 5.0: 974, 6.0: 441, 8.0: 4
- gpt-6.1-sol single reviews: see the per-rep check above.
