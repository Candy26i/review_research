# ACL 2025 version tracking: gpt-6.1-sol vs deepseek-chat

- gpt log: `data/processed/acl_reviews_gpt61sol_log.jsonl` | DeepSeek log: `data/processed/acl_reviews_log.jsonl`
- matched versions (scored by both arms): 3617 versions of 1374 papers with >= 2 matched versions; position = index among the paper's matched versions.
- gpt-6.1-sol: one review per version (rep 0), reasoning effort medium (API default), temperature 1.0. Its noise level is estimated below from identical-text pairs; a direct retest (a second review of 500 random versions) is in `results/acl_gpt61sol_retest.md` and agrees (SD 0.40 vs 0.41).
- DeepSeek r0 = the rep-0 review of each version (same prompt_sha and text_sha), the like-for-like comparison. DeepSeek mean of 3 is for reference.
- DeepSeek: requests went to the deepseek-chat alias; every raw response reports model deepseek-flash with one system_fingerprint (e72d3c7f), and all three reps were called interleaved within one session, 2026-10-23 08:51-15:14 UTC.
- 201 papers with >= 4 versions carry 61% of the weight of the within-paper slope.

## Rating change per revision step

| estimate | gpt-6.1-sol (1 review) | DeepSeek r0 (1 review) | DeepSeek mean of 3 (reference) |
|---|---|---|---|
| within-paper slope [95% CI], sign-flip p | +0.045 [+0.025, +0.065], p < 0.001 | +0.123 [+0.091, +0.155], p < 0.001 | +0.111 [+0.090, +0.132], p < 0.001 |
| same, with ln(words) covariate | +0.020 [-0.002, +0.048] | +0.089 [+0.057, +0.126] | +0.074 [+0.053, +0.100] |
| mixed model, random intercept | +0.042 [+0.026, +0.059], p < 0.001 | +0.104 [+0.073, +0.135], p < 0.001 | +0.102 [+0.083, +0.121], p < 0.001 |
| mixed model + ln(words) | +0.023 [+0.006, +0.041] | +0.068 [+0.036, +0.099] | +0.067 [+0.048, +0.087] |
| mixed model, random intercept and slope | +0.046 [+0.027, +0.066] | +0.107 [+0.075, +0.140] | +0.108 [+0.086, +0.130] |
| equal weight per paper (mean of per-paper slopes) | +0.057 [+0.027, +0.085] | +0.137 [+0.085, +0.191] | +0.124 [+0.092, +0.157] |
| SD of ratings | 1.01 | 1.19 | 0.93 |
| standardized within-paper slope (slope / SD; reference column: SD of 3-review means) | +0.045 [+0.025, +0.064] | +0.103 [+0.076, +0.130] | +0.118 [+0.097, +0.141] |
| share rated >= 5 (reference column: mean of 3 >= 5) | 38.3% | 39.2% | 19.4% |
| P(rating >= 5), percentage points per step (reference column: mean of 3 >= 5) | +2.0 [+1.0, +2.9], p < 0.001 | +4.8 [+3.4, +6.1], p < 0.001 | +3.6 [+2.2, +5.1], p < 0.001 |
| conditional logit, log-odds of >= 5 per step (reference column: mean of 3 >= 5) | +0.37 [+0.20, +0.54] (178 papers with variation) | +0.32 [+0.22, +0.42] (498 papers with variation) | +0.41 [+0.28, +0.55] (307 papers with variation) |

## Slope relative to each model's own repeat-call noise

Noise of a single review, from the 370 adjacent pairs whose text is identical (321 papers): sigma = sqrt(mean(delta^2) / 2). These pairs are not a random sample of papers, so treat sigma as an estimate.

| | gpt-6.1-sol | DeepSeek r0 | gpt minus DeepSeek r0 |
|---|---|---|---|
| repeat-call SD of one review [95% CI] | 0.41 [0.32, 0.48] | 0.84 [0.76, 0.92] | |
| within-paper slope / that SD | +0.111 [+0.060, +0.174] | +0.147 [+0.108, +0.190] | -0.035 [-0.096, +0.031] |

Share of changed adjacent pairs that go up (pairs with delta != 0):

| pair class | gpt-6.1-sol (1 review) | DeepSeek r0 (1 review) | DeepSeek mean of 3 (reference) |
|---|---|---|---|
| identical text | 41% of 32 | 63% of 125 | 50% of 210 |
| cosmetic | 55% of 29 | 51% of 195 | 56% of 291 |
| minor | 52% of 58 | 52% of 192 | 59% of 287 |
| major | 77% of 116 | 64% of 359 | 65% of 589 |
| ALL | 63% of 235 | 58% of 871 | 60% of 1377 |

## Difference gpt-6.1-sol minus DeepSeek (same versions, paired by paper)

| estimate | vs DeepSeek r0 | vs DeepSeek mean of 3 |
|---|---|---|
| within-paper slope | -0.077 [-0.114, -0.044], p < 0.001 | -0.065 [-0.092, -0.040], p < 0.001 |
| with ln(words) covariate | -0.070 [-0.107, -0.033] | -0.055 [-0.081, -0.024] |
| standardized slope (mean of 3: SD of 3-review means) | -0.058 [-0.089, -0.029] | -0.074 [-0.102, -0.048] |
| P(rating >= 5), points per step (mean of 3: mean >= 5) | -2.8 [-4.4, -1.3], p < 0.001 | -1.6 [-3.3, +0.0], p = 0.065 |
| stacked mixed model, position x arm (random intercept per paper and per paper x arm; common residual variance; model-based CI) | -0.066 [-0.100, -0.032], p < 0.001 | - |

## Adjacent version pairs

Delta = rating of the later version minus the earlier one. Mean delta CI: cluster bootstrap over papers.

**gpt-6.1-sol (1 review)**

| pair class | n | P(up) | P(tie) | P(down) | mean delta [95% CI] | mean abs delta |
|---|---|---|---|---|---|---|
| identical text | 370 | 3.5% | 91.4% | 5.1% | -0.03 [-0.08, +0.02] | 0.17 |
| cosmetic | 492 | 3.3% | 94.1% | 2.6% | +0.01 [-0.03, +0.04] | 0.11 |
| minor | 464 | 6.5% | 87.5% | 6.0% | +0.01 [-0.05, +0.07] | 0.24 |
| major | 917 | 9.7% | 87.4% | 2.9% | +0.13 [+0.09, +0.17] | 0.25 |
| ALL | 2243 | 6.6% | 89.5% | 3.9% | +0.05 [+0.03, +0.07] | 0.20 |

**DeepSeek r0 (1 review)**

| pair class | n | P(up) | P(tie) | P(down) | mean delta [95% CI] | mean abs delta |
|---|---|---|---|---|---|---|
| identical text | 370 | 21.4% | 66.2% | 12.4% | +0.13 [+0.01, +0.25] | 0.65 |
| cosmetic | 492 | 20.1% | 60.4% | 19.5% | +0.07 [-0.03, +0.17] | 0.73 |
| minor | 464 | 21.3% | 58.6% | 20.0% | -0.01 [-0.12, +0.11] | 0.77 |
| major | 917 | 25.0% | 60.9% | 14.2% | +0.22 [+0.15, +0.30] | 0.78 |
| ALL | 2243 | 22.6% | 61.2% | 16.3% | +0.13 [+0.09, +0.17] | 0.75 |

**DeepSeek mean of 3 (reference)**

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
| gpt-6.1-sol (1 review) | +0.059 [+0.036, +0.083] |
| DeepSeek r0 (1 review) | +0.131 [+0.093, +0.172] |
| DeepSeek mean of 3 (reference) | +0.133 [+0.109, +0.157] |
| gpt minus DeepSeek r0 | -0.072 [-0.115, -0.030] |

## Sensitivity: versions under 1,000 words dropped (text-extraction failures)

Dropped: 2412.12567v1, 2503.15351v1 (every other matched version has >= 1,944 words): near-empty texts (a title with an empty heading; one appendix paragraph) that carry most of the leverage on the ln(words) coefficient. Their ratings: 2412.12567v1 gpt-6.1-sol (1 review) 1.00, DeepSeek r0 (1 review) 1.00, DeepSeek mean of 3 (reference) 1.00; 2503.15351v1 gpt-6.1-sol (1 review) 3.00, DeepSeek r0 (1 review) 1.00, DeepSeek mean of 3 (reference) 1.00. Papers then left with one version leave the frame: 2503.15351. Remaining: 3614 versions of 1373 papers.

| arm | within-paper slope [95% CI] | with ln(words) covariate |
|---|---|---|
| gpt-6.1-sol (1 review) | +0.043 [+0.025, +0.061] | +0.033 [+0.011, +0.056] |
| DeepSeek r0 (1 review) | +0.120 [+0.090, +0.150] | +0.098 [+0.061, +0.133] |
| DeepSeek mean of 3 (reference) | +0.108 [+0.087, +0.128] | +0.081 [+0.056, +0.105] |
| gpt minus DeepSeek r0 (1 review) | -0.077 [-0.112, -0.042] | -0.065 [-0.104, -0.026] |
| gpt minus DeepSeek mean of 3 (reference) | -0.065 [-0.091, -0.037] | -0.048 [-0.080, -0.015] |

## Rating histograms (matched versions)

- gpt-6.1-sol (1 review): 1.0: 7, 3.0: 2224, 5.0: 1326, 6.0: 60
- DeepSeek r0 (1 review): 1.0: 22, 3.0: 2176, 5.0: 974, 6.0: 441, 8.0: 4
