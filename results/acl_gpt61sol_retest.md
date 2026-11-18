# gpt-6.1-sol repeat-call noise on ACL 2025 (500-version retest)

- sample: `data/processed/acl_gpt61sol_retest_sample.jsonl`, 500 versions drawn at random from the 3,618 scored versions; 500 with a second gpt-6.1-sol review (437 papers).
- gpt-6.1-sol: rep 0 (main run) vs rep 1 (retest, 2026-11-16 21:23 UTC to 2026-11-16 22:25 UTC; 11-21 h after rep 0, median 15 h). Rep 1: flex tier, max_completion_tokens 16000, explicit cache mode, no reasoning_effort sent. 490 of the 500 rep-0 reviews had the identical request; the others are default-tier smoke/pilot rows. 142 rep-0 reviews came from the first API key.
- DeepSeek: rep 0 vs rep 1 of the same versions; the mean over its three rep pairs is shown for reference.
- CIs: cluster bootstrap over papers, 2,000 draws.

| | gpt-6.1-sol r0 vs r1 | DeepSeek r0 vs r1 | gpt minus DeepSeek | DeepSeek, mean of 3 pairs |
|---|---|---|---|---|
| same rating twice | 92.0% [89.6, 94.3] | 64.4% [60.0, 68.6] | +27.6 [+23.0, +32.5] | 64.1% |
| Cohen's kappa (chance-corrected) | 0.84 [0.79, 0.88] | 0.33 [0.26, 0.41] | +0.50 [+0.42, +0.59] | 0.33 |
| same side of 5 (>= 5 or not; common cut for both scales) | 92.4% | 72.8% | +19.6 [+15.0, +24.5] | 74.0% |
| mean absolute difference | 0.16 | 0.69 | | 0.67 |
| mean difference r1 - r0 | -0.00 [-0.05, +0.05] | +0.01 [-0.09, +0.12] | | |
| repeat-call SD of one review | 0.40 [0.33, 0.45] | 0.86 [0.79, 0.93] | -0.46 [-0.56, -0.37]; ratio 0.46 [0.38, 0.54] | 0.84 |
| ICC(2,1), one review | 0.84 [0.80, 0.89] | 0.46 [0.38, 0.55] | +0.38 [+0.29, +0.48] | ICC(2,1) over 3 reps 0.49 |
| reliability of a 3-review mean (Spearman-Brown) | 0.94 | 0.72 | | 0.74 (from the 3-rep ICC) |

ICC here is reliability across versions of different papers, so it is driven mostly by between-paper differences. The within-paper slope removes those; within papers, repeat-call noise is still most of the residual variation for both models (projection section).

Earlier indirect estimate of the repeat-call SD from 370 adjacent versions with identical text: gpt-6.1-sol 0.41 [0.32, 0.48], DeepSeek r0 0.84 [0.76, 0.92]. The direct retest agrees for both models; the two estimates share some rep-0 reviews (versions in both sets), so they are not fully independent.

gpt-6.1-sol rating transitions r0 -> r1: 3->1: 1, 3->3: 287, 3->5: 19, 5->3: 19, 5->5: 166, 5->6: 1, 6->6: 7

## Version slope relative to repeat-call noise

Slope = within-paper slope on all 1,374 papers (rep 0); SD from this retest (DeepSeek: same 500 versions, r0 vs r1). Joint cluster bootstrap over all papers.

| | slope per step | repeat-call SD | slope / SD [95% CI] |
|---|---|---|---|
| gpt-6.1-sol | +0.045 | 0.40 | +0.114 [+0.063, +0.171] |
| DeepSeek r0 | +0.123 | 0.86 | +0.142 [+0.104, +0.182] |
| gpt minus DeepSeek | | | -0.028 [-0.090, +0.036] |
| gpt / DeepSeek | | | 0.80 [0.44, 1.30] |

With DeepSeek's SD from all versions' three reviews (0.88) the difference is -0.026 [-0.085, +0.038].
Scaled by each model's rating SD instead (main report), the difference is detected: -0.058 [-0.089, -0.029]. Whether gpt-6.1-sol is "less sensitive" depends on the yardstick.

## What more reviews per version would buy (projection)

Ratio of the slope CI width with K reviews per version (averaged) to the width with one. "iid" shrinks the noise part of the residual variance; "cluster-robust" shrinks only the noise part of the cluster-robust slope variance (between-paper clustering does not average away) and matches the cluster-bootstrap CIs used throughout. DeepSeek has 3 reviews of every version, so its realized ratio checks the formulas.

| | K | iid formula | cluster-robust formula | realized |
|---|---|---|---|---|
| gpt-6.1-sol | 2 | 79% (sigma CI: 72-86%) | 86% (81-90%) | not observed |
| gpt-6.1-sol | 3 | 71% (sigma CI: 60-81%) | 80% (73-86%) | not observed |
| DeepSeek (vs r0) | 2 | 73% | 73% | CI width 76%, bootstrap SD 76%, cluster-robust SE 76% |
| DeepSeek (vs r0) | 3 | 61% | 61% | CI width 65%, bootstrap SD 64%, cluster-robust SE 64%; vs r0/r1/r2 as baseline: 64/62/58% |

gpt-6.1-sol: residual variance 0.21, repeat-call noise 0.16 (74% of it); cluster-robust SE / iid SE = 1.18 (DeepSeek r0 1.00). The excess clustering is why the cluster-robust projection gains less. On DeepSeek the iid formula overstates the realized gain by a few points; expect the same or more for gpt-6.1-sol, whose sigma rests on few rating flips.
