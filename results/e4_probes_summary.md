# E4: synthetic edit probes - content sensitivity vs length sensitivity

- review log: `data/processed/probe_reviews_log.jsonl` - 740 rows, 739 parsed_ok rating rows used
- thresholds (fixed upfront): content-sensitive iff mean degradation delta < -0.30 with 95% CI excluding 0; length-sensitive iff pad_filler delta > +0.15 or drop_appendix delta < -0.15 with 95% CI excluding 0

## Probe construction context (from the probes log)

| variant | n built | n skipped | mean length ratio vs base |
|---|---|---|---|
| base | 25 | 0 | 1.00 (n=25) |
| shuffle_paragraphs | 25 | 0 | 1.00 (n=25) |
| corrupt_numbers | 25 | 0 | 1.00 (n=25) |
| typos | 25 | 0 | 1.00 (n=25) |
| pad_filler | 25 | 0 | 1.31 (n=25) |
| drop_appendix | 23 | 2 (no appendix heading: 2) | 0.60 (n=23) |

## Model: deepseek-reasoner

- papers with a scored base: 25 (mean base rating 4.88) | reps per (paper, variant) cell: 3-3

### 1. Per-variant paired effects (variant mean minus the same paper's base mean, across reps)

| variant | n papers | mean delta | 95% CI | delta<0 | delta==0 | delta>0 | mean \|delta\| |
|---|---|---|---|---|---|---|---|
| shuffle_paragraphs | 25 | +0.29 | [+0.04, +0.54] | 12.0% | 36.0% | 52.0% | 0.48 |
| corrupt_numbers | 25 | -0.27 | [-0.54, +0.01] | 48.0% | 40.0% | 12.0% | 0.48 |
| typos | 25 | -0.41 | [-0.67, -0.15] | 56.0% | 36.0% | 8.0% | 0.52 |
| pad_filler | 25 | -0.92 | [-1.27, -0.57] | 72.0% | 24.0% | 4.0% | 1.00 |
| drop_appendix | 23 | -0.01 | [-0.27, +0.24] | 30.4% | 43.5% | 26.1% | 0.39 |

Subscore deltas (same pairing; n in parentheses = papers where both sides report that subscore):

| variant | d soundness | d presentation | d contribution | largest mover | expected | right subscore? |
|---|---|---|---|---|---|---|
| shuffle_paragraphs | +0.16 (n=25) | -0.37 (n=25) | +0.08 (n=25) | presentation | - | - |
| corrupt_numbers | +0.05 (n=25) | -0.28 (n=25) | -0.07 (n=25) | presentation | soundness | NO |
| typos | -0.15 (n=25) | -0.77 (n=25) | -0.08 (n=25) | presentation | presentation | YES |
| pad_filler | -0.19 (n=25) | -1.29 (n=25) | -0.29 (n=25) | presentation | - | - |
| drop_appendix | +0.06 (n=23) | +0.01 (n=23) | +0.03 (n=23) | soundness | - | - |
- typos -> presentation should move most: **YES**
- corrupt_numbers -> soundness should move most: **NO**

### 2. Mixed model: rating ~ C(variant, base reference) + (1|paper), rep-level rows

- n rep-level rows: 444 | papers: 25 | variants: 5 + base

| variant (vs base) | coef | 95% CI | p |
|---|---|---|---|
| corrupt_numbers | -0.27 | [-0.54, +0.00] | 0.0534 |
| drop_appendix | -0.01 | [-0.29, +0.27] | 0.9526 |
| pad_filler | -0.92 | [-1.19, -0.65] | 0.0000 |
| shuffle_paragraphs | +0.29 | [+0.02, +0.56] | 0.0336 |
| typos | -0.41 | [-0.68, -0.14] | 0.0028 |

### 3. The 2x2 verdict: content sensitivity vs length sensitivity

- content sensitivity (mean paired delta over length-preserved degradations shuffle_paragraphs, corrupt_numbers, typos): -0.13 (95% CI [-0.31, +0.05], n=25 papers) | threshold < -0.30 with CI excluding 0 -> not fired
- length sensitivity: pad_filler -0.92 (95% CI [-1.27, -0.57], n=25) vs > +0.15 -> not fired | drop_appendix -0.01 (95% CI [-0.27, +0.24], n=23) vs < -0.15 -> not fired
- pad-minus-drop asymmetry (papers with both, n=23): -0.94 (95% CI [-1.30, -0.58])

|  | length-sensitive: NO | length-sensitive: YES |
|---|---|---|
| content-sensitive: YES | genuine quality signal | quality signal + length bias |
| content-sensitive: NO | insensitive / noise-level **<= this model** | length-driven reviewer |

This model is NOT content-sensitive by the mechanical rule: the mean rating delta across the length-preserved degradations is -0.13 (95% CI [-0.31, +0.05], n=25), which does not cross the -0.30 threshold with a CI excluding 0. It is NOT length-sensitive: pad_filler moved ratings by -0.92 (95% CI [-1.27, -0.57]) against the +0.15 threshold and drop_appendix by -0.01 (95% CI [-0.27, +0.24]) against the -0.15 threshold, so neither trigger fired. Verdict: neither fired: effects are below both thresholds (noise-level at this sample size).

### 4. Noise context (E0/E1 true-placebo floor)

- true-placebo noise floor for this model (from the e0 summary): mean |delta| 0.53 (n=17 adjacent cosmetic pairs, mean delta -0.18)
- the floor is a between-adjacent-version quantity from real revisions, so ratios are context, not a significance test.

| effect | mean delta | \|effect\| / noise floor | exceeds floor? |
|---|---|---|---|
| content (mean degradation delta) | -0.13 | 0.2x | no |
| shuffle_paragraphs | +0.29 | 0.6x | no |
| corrupt_numbers | -0.27 | 0.5x | no |
| typos | -0.41 | 0.8x | no |
| pad_filler | -0.92 | 1.7x | YES |
| drop_appendix | -0.01 | 0.0x | no |

## Model: gemini-3.8-flash

- papers with a scored base: 25 (mean base rating 5.68) | reps per (paper, variant) cell: 1-2

### 1. Per-variant paired effects (variant mean minus the same paper's base mean, across reps)

| variant | n papers | mean delta | 95% CI | delta<0 | delta==0 | delta>0 | mean \|delta\| |
|---|---|---|---|---|---|---|---|
| shuffle_paragraphs | 25 | -0.06 | [-0.34, +0.22] | 24.0% | 60.0% | 16.0% | 0.38 |
| corrupt_numbers | 25 | +0.08 | [-0.26, +0.42] | 16.0% | 56.0% | 28.0% | 0.48 |
| typos | 25 | -0.04 | [-0.38, +0.30] | 28.0% | 52.0% | 20.0% | 0.48 |
| pad_filler | 25 | -2.16 | [-2.69, -1.63] | 92.0% | 4.0% | 4.0% | 2.20 |
| drop_appendix | 23 | +0.04 | [-0.27, +0.36] | 21.7% | 56.5% | 21.7% | 0.43 |

Subscore deltas (same pairing; n in parentheses = papers where both sides report that subscore):

| variant | d soundness | d presentation | d contribution | largest mover | expected | right subscore? |
|---|---|---|---|---|---|---|
| shuffle_paragraphs | -0.08 (n=25) | -0.26 (n=25) | -0.08 (n=25) | presentation | - | - |
| corrupt_numbers | -0.02 (n=25) | -0.20 (n=25) | -0.04 (n=25) | presentation | soundness | NO |
| typos | -0.06 (n=25) | -0.58 (n=25) | -0.02 (n=25) | presentation | presentation | YES |
| pad_filler | -0.52 (n=25) | -1.46 (n=25) | -0.44 (n=25) | presentation | - | - |
| drop_appendix | -0.04 (n=23) | -0.02 (n=23) | +0.00 (n=23) | soundness | - | - |
- typos -> presentation should move most: **YES**
- corrupt_numbers -> soundness should move most: **NO**

### 2. Mixed model: rating ~ C(variant, base reference) + (1|paper), rep-level rows

- n rep-level rows: 295 | papers: 25 | variants: 5 + base

| variant (vs base) | coef | 95% CI | p |
|---|---|---|---|
| corrupt_numbers | +0.08 | [-0.24, +0.40] | 0.6210 |
| drop_appendix | -0.01 | [-0.34, +0.31] | 0.9432 |
| pad_filler | -2.16 | [-2.48, -1.84] | 0.0000 |
| shuffle_paragraphs | -0.07 | [-0.39, +0.25] | 0.6679 |
| typos | -0.04 | [-0.36, +0.28] | 0.8048 |

### 3. The 2x2 verdict: content sensitivity vs length sensitivity

- content sensitivity (mean paired delta over length-preserved degradations shuffle_paragraphs, corrupt_numbers, typos): -0.01 (95% CI [-0.27, +0.26], n=25 papers) | threshold < -0.30 with CI excluding 0 -> not fired
- length sensitivity: pad_filler -2.16 (95% CI [-2.69, -1.63], n=25) vs > +0.15 -> not fired | drop_appendix +0.04 (95% CI [-0.27, +0.36], n=23) vs < -0.15 -> not fired
- pad-minus-drop asymmetry (papers with both, n=23): -2.11 (95% CI [-2.71, -1.51])

|  | length-sensitive: NO | length-sensitive: YES |
|---|---|---|
| content-sensitive: YES | genuine quality signal | quality signal + length bias |
| content-sensitive: NO | insensitive / noise-level **<= this model** | length-driven reviewer |

This model is NOT content-sensitive by the mechanical rule: the mean rating delta across the length-preserved degradations is -0.01 (95% CI [-0.27, +0.26], n=25), which does not cross the -0.30 threshold with a CI excluding 0. It is NOT length-sensitive: pad_filler moved ratings by -2.16 (95% CI [-2.69, -1.63]) against the +0.15 threshold and drop_appendix by +0.04 (95% CI [-0.27, +0.36]) against the -0.15 threshold, so neither trigger fired. Verdict: neither fired: effects are below both thresholds (noise-level at this sample size).

### 4. Noise context (E0/E1 true-placebo floor)

- true-placebo noise floor for this model (from the e0 summary): mean |delta| 0.31 (n=17 adjacent cosmetic pairs, mean delta +0.04)
- the floor is a between-adjacent-version quantity from real revisions, so ratios are context, not a significance test.

| effect | mean delta | \|effect\| / noise floor | exceeds floor? |
|---|---|---|---|
| content (mean degradation delta) | -0.01 | 0.0x | no |
| shuffle_paragraphs | -0.06 | 0.2x | no |
| corrupt_numbers | +0.08 | 0.3x | no |
| typos | -0.04 | 0.1x | no |
| pad_filler | -2.16 | 7.0x | YES |
| drop_appendix | +0.04 | 0.1x | no |

